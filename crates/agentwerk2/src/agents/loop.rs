//! Multi-agent loop driver. Each agent runs in its own tokio task
//! against a shared `TicketSystem` passed in by `Runnable::run`. Also
//! defines the `Runnable` trait that `TicketSystem` implements.

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::providers::types::{ResponseStatus, StreamEvent};
use crate::providers::{AsUserMessage, ContentBlock, Message, ModelRequest};
use crate::tools::{ToolCall, ToolContext};

use super::agent::Agent;
use super::policy::PolicyConform;
use super::tickets::{Status, TicketSystem};

/// What it means to be a thing agents can be assigned to and run
/// against. Implemented by `TicketSystem`. Future types could
/// implement it the same way `PolicyConform` is implemented today.
pub trait Runnable: Sized {
    /// Stage one agent. Chainable.
    fn assign(self, agent: Agent) -> Self;

    /// Stage many agents. Default impl folds over `assign`.
    fn assign_all<I>(self, agents: I) -> Self
    where
        I: IntoIterator<Item = Agent>,
    {
        let mut s = self;
        for a in agents {
            s = s.assign(a);
        }
        s
    }

    /// Drive every staged agent. Idles on empty queue until the
    /// implementor's interrupt signal fires.
    fn run(self) -> impl Future<Output = Self> + Send;

    /// Drive every staged agent. Auto-stops once the queue settles.
    fn run_until_empty(self) -> impl Future<Output = Self> + Send;
}

const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Drive many agents against one shared `TicketSystem`. Each agent runs
/// in its own tokio task; locks on the queue are held only around the
/// queue/metric ops and never across `provider.respond().await`, so
/// model calls really do overlap.
pub(super) async fn run_main_loop(agents: Vec<Agent>, tickets: Arc<Mutex<TicketSystem>>) {
    let mut handles = Vec::with_capacity(agents.len());
    for agent in agents {
        handles.push(tokio::spawn(handle_tickets(agent, Arc::clone(&tickets))));
    }
    for h in handles {
        let _ = h.await;
    }
}

/// Per-agent loop. Picks the next `Todo` whose type the agent handles,
/// hands it to `process_ticket`, repeats. When no eligible work is
/// queued, idles on `IDLE_POLL_INTERVAL` until something arrives, the
/// cancel signal fires, or a policy hits.
pub(super) async fn handle_tickets(agent: Agent, tickets: Arc<Mutex<TicketSystem>>) {
    loop {
        let claim = {
            let mut sys = tickets.lock().unwrap();
            if sys.is_interrupted() || sys.policy_violated() {
                return;
            }
            let next = sys
                .list_by_status(Status::Todo)
                .into_iter()
                .find(|t| agent.handles(&t.r#type))
                .map(|t| t.key.clone());
            next.inspect(|key| {
                sys.record_step(agent.get_name());
                let _ = sys.update_status(key, Status::InProgress);
            })
        };

        let Some(key) = claim else {
            tokio::time::sleep(IDLE_POLL_INTERVAL).await;
            continue;
        };

        process_ticket(&agent, &tickets, &key).await;
    }
}

/// One ticket from claimed → settled. Owns the per-ticket message
/// vector. Today the inner `loop {}` always exits after one iteration;
/// when tools land, the assistant's `ToolUse` blocks will drive a
/// second pass.
async fn process_ticket(agent: &Agent, tickets: &Arc<Mutex<TicketSystem>>, key: &str) {
    let mut messages: Vec<Message> = Vec::new();
    if let Some(ctx) = agent.context_message() {
        messages.push(Message::user(ctx));
    }
    let task_msg = {
        let sys = tickets.lock().unwrap();
        sys.get(key).map(|t| t.as_user_message())
    };
    let Some(task_msg) = task_msg else {
        return;
    };
    messages.push(task_msg);

    let (max_request_tokens, interrupt_signal) = {
        let sys = tickets.lock().unwrap();
        (
            sys.policies().max_request_tokens,
            sys.interrupt_signal_handle(),
        )
    };

    loop {
        let request = ModelRequest {
            model: agent.model_str().to_string(),
            system_prompt: agent.system_prompt(),
            messages: messages.clone(),
            tools: agent.tool_definitions(),
            max_request_tokens,
            tool_choice: None,
        };
        let response = match agent
            .provider_handle()
            .respond(request, no_op_event_handler())
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let mut sys = tickets.lock().unwrap();
                sys.record_error(agent.get_name(), key, &e);
                let _ = sys.update_status(key, Status::Failed);
                return;
            }
        };

        {
            let mut sys = tickets.lock().unwrap();
            sys.record_request(agent.get_name(), &response.usage);
        }
        messages.push(Message::Assistant {
            content: response.content.clone(),
        });

        // Walk the assistant content for tool calls.
        let calls: Vec<ToolCall> = response
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect();

        if response.status != ResponseStatus::ToolUse || calls.is_empty() {
            // Inner loop terminates. Status management stays in the
            // loop until a ticket-update tool ships.
            let mut sys = tickets.lock().unwrap();
            let _ = sys.update_status(key, Status::Done);
            return;
        }

        // Dispatch the tool calls. ToolContext carries working_dir, the
        // shared cancel signal, and a registry handle so ToolSearchTool
        // can reach back in.
        let ctx = ToolContext::new(agent.working_dir_or_default())
            .interrupt_signal(Arc::clone(&interrupt_signal))
            .registry(Arc::new(agent.tool_registry().clone()));
        let outcomes = agent.tool_registry().execute(&calls, &ctx).await;

        let blocks: Vec<ContentBlock> = outcomes.into_iter().map(|(b, _)| b).collect();
        messages.push(Message::User { content: blocks });

        {
            let mut sys = tickets.lock().unwrap();
            sys.record_tool_calls(agent.get_name(), calls.len() as u64);
        }
    }
}

fn no_op_event_handler() -> Arc<dyn Fn(StreamEvent) + Send + Sync> {
    Arc::new(|_| {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::DEFAULT_BEHAVIOR;
    use crate::providers::types::{ModelResponse, ResponseStatus};
    use crate::providers::{ContentBlock, Provider, ProviderResult, TokenUsage};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};

    use std::collections::VecDeque;

    struct MockProvider {
        text: String,
        usage: TokenUsage,
        /// When non-empty, the next call pops from the front. When empty,
        /// the provider falls back to the static `text` + `usage`.
        queue: Mutex<VecDeque<ModelResponse>>,
        captured_prompt: Arc<Mutex<Option<String>>>,
        captured_max_tokens: Arc<Mutex<Option<u32>>>,
        captured_messages: Arc<Mutex<Option<Vec<Message>>>>,
    }

    struct MockHandles {
        prompt: Arc<Mutex<Option<String>>>,
        max_tokens: Arc<Mutex<Option<u32>>>,
        messages: Arc<Mutex<Option<Vec<Message>>>>,
    }

    impl MockProvider {
        fn new(text: impl Into<String>) -> (Arc<Self>, MockHandles) {
            Self::with_usage(text, TokenUsage::default())
        }

        fn with_usage(text: impl Into<String>, usage: TokenUsage) -> (Arc<Self>, MockHandles) {
            Self::build(text.into(), usage, VecDeque::new())
        }

        /// Build a provider that returns each queued response in order.
        /// After the queue empties, falls back to the static text /
        /// usage (set to `"end"` / default for convenience here).
        fn queued(responses: Vec<ModelResponse>) -> (Arc<Self>, MockHandles) {
            Self::build("end".into(), TokenUsage::default(), responses.into())
        }

        fn build(
            text: String,
            usage: TokenUsage,
            queue: VecDeque<ModelResponse>,
        ) -> (Arc<Self>, MockHandles) {
            let prompt = Arc::new(Mutex::new(None));
            let max_tokens = Arc::new(Mutex::new(None));
            let messages = Arc::new(Mutex::new(None));
            let provider = Arc::new(Self {
                text,
                usage,
                queue: Mutex::new(queue),
                captured_prompt: Arc::clone(&prompt),
                captured_max_tokens: Arc::clone(&max_tokens),
                captured_messages: Arc::clone(&messages),
            });
            (
                provider,
                MockHandles {
                    prompt,
                    max_tokens,
                    messages,
                },
            )
        }
    }

    fn tool_use_response(id: &str, name: &str, input: serde_json::Value) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input,
            }],
            status: ResponseStatus::ToolUse,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    fn text_response(text: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            status: ResponseStatus::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    impl Provider for MockProvider {
        fn respond(
            &self,
            request: ModelRequest,
            _on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
            *self.captured_prompt.lock().unwrap() = Some(request.system_prompt);
            *self.captured_max_tokens.lock().unwrap() = request.max_request_tokens;
            *self.captured_messages.lock().unwrap() = Some(request.messages.clone());

            let queued = self.queue.lock().unwrap().pop_front();
            let fallback_text = self.text.clone();
            let fallback_usage = self.usage.clone();
            Box::pin(async move {
                tokio::task::yield_now().await;
                let response = queued.unwrap_or_else(|| ModelResponse {
                    content: vec![ContentBlock::Text {
                        text: fallback_text,
                    }],
                    status: ResponseStatus::EndTurn,
                    usage: fallback_usage,
                    model: "mock".into(),
                });
                Ok(response)
            })
        }
    }

    fn agent_with(provider: Arc<dyn Provider>, name: &str) -> Agent {
        Agent::new().name(name).provider(provider).model("mock")
    }

    #[tokio::test]
    async fn single_agent_drains_queue() {
        let (provider, _) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new();
        tickets.create("first", "", "task", "tester");
        tickets.create("second", "", "task", "tester");

        let agent = agent_with(provider, "worker").role("you are a worker");
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.steps(), 2);
        assert_eq!(tickets.requests(), 2);
        assert_eq!(tickets.list_by_status(Status::Done).len(), 2);
    }

    #[tokio::test]
    async fn interrupt_signal_stops_an_idle_agent() {
        let (provider, _) = MockProvider::new("ok");
        let signal = Arc::new(AtomicBool::new(true));
        let tickets = TicketSystem::new().interrupt_signal(Arc::clone(&signal));

        let agent = agent_with(provider, "worker");
        let tickets = tickets.assign(agent).run().await;

        assert_eq!(tickets.steps(), 0);
        assert_eq!(tickets.requests(), 0);
    }

    #[tokio::test]
    async fn max_steps_stops_processing_after_threshold() {
        let (provider, _) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new().max_steps(2);
        tickets.create("a", "", "task", "tester");
        tickets.create("b", "", "task", "tester");
        tickets.create("c", "", "task", "tester");

        let agent = agent_with(provider, "worker");
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 2);
        assert_eq!(tickets.list_by_status(Status::Todo).len(), 1);
    }

    #[tokio::test]
    async fn max_input_tokens_stops_processing_after_threshold() {
        let (provider, _) = MockProvider::with_usage(
            "ok",
            TokenUsage {
                input_tokens: 60,
                output_tokens: 0,
                ..TokenUsage::default()
            },
        );
        let mut tickets = TicketSystem::new().max_input_tokens(100);
        tickets.create("a", "", "task", "tester");
        tickets.create("b", "", "task", "tester");
        tickets.create("c", "", "task", "tester");

        let agent = agent_with(provider, "worker");
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 2);
        assert_eq!(tickets.list_by_status(Status::Todo).len(), 1);
    }

    #[tokio::test]
    async fn system_prompt_includes_role_and_behavior() {
        let (provider, handles) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker")
            .role("ROLE_TEXT")
            .behavior("BEHAVIOR_TEXT")
            .context("CONTEXT_TEXT");
        let _ = tickets.assign(agent).run_until_empty().await;

        let prompt = handles
            .prompt
            .lock()
            .unwrap()
            .clone()
            .expect("prompt captured");
        let role_pos = prompt.find("ROLE_TEXT").expect("role present");
        let behavior_pos = prompt.find("BEHAVIOR_TEXT").expect("behavior present");
        assert!(role_pos < behavior_pos);
        assert!(!prompt.contains("CONTEXT_TEXT"));
        assert!(!prompt.contains("## Context"));
    }

    #[tokio::test]
    async fn context_renders_as_first_user_message() {
        let (provider, handles) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new();
        tickets.create("task summary", "task body", "task", "tester");

        let agent = agent_with(provider, "worker").context("CONTEXT_TEXT");
        let _ = tickets.assign(agent).run_until_empty().await;

        let messages = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        assert_eq!(messages.len(), 2);
        match &messages[0] {
            Message::User { content } => match content.first() {
                Some(ContentBlock::Text { text }) => {
                    assert!(text.starts_with("## Context\n\n"));
                    assert!(text.contains("CONTEXT_TEXT"));
                }
                other => panic!("expected text block, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
        match &messages[1] {
            Message::User { content } => match content.first() {
                Some(ContentBlock::Text { text }) => {
                    assert_eq!(text, "task summary\n\ntask body");
                }
                other => panic!("expected text block, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn task_user_message_uses_ticket_as_user_message_format() {
        let (provider, handles) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new();
        tickets.create("the summary", "the body", "task", "tester");

        let agent = agent_with(provider, "worker"); // no context
        let _ = tickets.assign(agent).run_until_empty().await;

        let messages = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            Message::User { content } => match content.first() {
                Some(ContentBlock::Text { text }) => {
                    assert_eq!(text, "the summary\n\nthe body");
                }
                other => panic!("expected text block, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn system_prompt_falls_back_to_default_behavior_when_unset() {
        let (provider, handles) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker").role("ROLE_TEXT");
        let _ = tickets.assign(agent).run_until_empty().await;

        let prompt = handles
            .prompt
            .lock()
            .unwrap()
            .clone()
            .expect("prompt captured");
        assert!(prompt.contains("ROLE_TEXT"));
        assert!(prompt.contains(DEFAULT_BEHAVIOR));
    }

    #[tokio::test]
    async fn max_request_tokens_is_forwarded_to_the_provider_request() {
        let (provider, handles) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new().max_request_tokens(256);
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker");
        let _ = tickets.assign(agent).run_until_empty().await;

        assert_eq!(*handles.max_tokens.lock().unwrap(), Some(256));
    }

    #[tokio::test]
    async fn run_main_loop_drives_multiple_agents_in_parallel() {
        let (provider, _) = MockProvider::new("done");
        let mut tickets = TicketSystem::new();
        tickets.create("a", "", "task", "tester");
        tickets.create("b", "", "task", "tester");
        tickets.create("c", "", "task", "tester");
        tickets.create("d", "", "task", "tester");

        let alice = agent_with(Arc::clone(&(provider.clone() as Arc<dyn Provider>)), "alice");
        let bob = agent_with(provider, "bob");
        let tickets = tickets.assign_all([alice, bob]).run_until_empty().await;

        // Both agents drained the queue together; per-agent attribution
        // isn't observable now that the loop no longer writes the model
        // reply as an agent-authored comment. The aggregate counters
        // are still meaningful.
        assert_eq!(tickets.list_by_status(Status::Done).len(), 4);
        assert_eq!(tickets.steps(), 4);
        assert_eq!(tickets.requests(), 4);
    }

    #[tokio::test]
    async fn idle_agent_picks_up_a_late_arriving_ticket() {
        let (provider, _) = MockProvider::new("done");
        let signal = Arc::new(AtomicBool::new(false));
        let tickets = TicketSystem::new().interrupt_signal(Arc::clone(&signal));

        let agent = agent_with(provider, "worker");
        // Wrap in Arc<Mutex<>> manually so we can keep mutating after the
        // task starts, then use `run_main_loop` directly. We can't use the
        // `run` consume-and-return pattern here because we need outside
        // access to the queue while the agent is parked.
        let shared = Arc::new(Mutex::new(tickets));
        let task = {
            let agents = vec![agent];
            let shared = Arc::clone(&shared);
            tokio::spawn(async move { run_main_loop(agents, shared).await })
        };

        tokio::time::sleep(Duration::from_millis(50)).await;
        shared
            .lock()
            .unwrap()
            .create("late", "", "task", "tester");

        let watcher_signal = Arc::clone(&signal);
        let watcher_shared = Arc::clone(&shared);
        let watcher = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(20)).await;
                let sys = watcher_shared.lock().unwrap();
                if sys.list_by_status(Status::Done).len() == 1 {
                    watcher_signal.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });

        let _ = task.await;
        let _ = watcher.await;

        let sys = shared.lock().unwrap();
        assert_eq!(sys.list_by_status(Status::Done).len(), 1);
        assert_eq!(sys.requests(), 1);
    }

    /// Tiny mock implementor that exercises `assign_all`'s default impl.
    struct Counter {
        names: Vec<String>,
    }

    impl Runnable for Counter {
        fn assign(mut self, agent: Agent) -> Self {
            self.names.push(agent.get_name().to_string());
            self
        }

        async fn run(self) -> Self {
            self
        }

        async fn run_until_empty(self) -> Self {
            self
        }
    }

    #[test]
    fn assign_all_folds_over_assign() {
        let counter = Counter { names: Vec::new() };
        let (provider, _) = MockProvider::new("ok");
        let counter = counter.assign_all([
            agent_with(Arc::clone(&(provider.clone() as Arc<dyn Provider>)), "alice"),
            agent_with(Arc::clone(&(provider.clone() as Arc<dyn Provider>)), "bob"),
            agent_with(provider, "carol"),
        ]);
        assert_eq!(counter.names, vec!["alice", "bob", "carol"]);
    }

    #[tokio::test]
    async fn agent_with_ticket_type_skips_other_types() {
        let (provider, _) = MockProvider::new("ok");
        let mut tickets = TicketSystem::new();
        tickets.create("real work", "", "task", "tester");
        tickets.create("a bug", "", "bug", "tester");

        let agent = agent_with(provider, "worker").ticket_type("task");
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 1);
        assert_eq!(tickets.list_by_status(Status::Todo).len(), 1);
        let done = &tickets.list_by_status(Status::Done)[0];
        assert_eq!(done.r#type, "task");
        let todo = &tickets.list_by_status(Status::Todo)[0];
        assert_eq!(todo.r#type, "bug");
    }

    use crate::tools::{Tool, ToolResult};

    fn echo_tool() -> Tool {
        Tool::new("echo", "Echo the input back")
            .schema(serde_json::json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
            }))
            .read_only(true)
            .handler(|input, _ctx| {
                Box::pin(async move {
                    let v = input["value"].as_str().unwrap_or("").to_string();
                    Ok(ToolResult::success(v))
                })
            })
    }

    #[tokio::test]
    async fn tool_use_response_dispatches_and_loops() {
        let (provider, handles) = MockProvider::queued(vec![
            tool_use_response("call-1", "echo", serde_json::json!({"value": "hi"})),
            text_response("done"),
        ]);
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker").tool(echo_tool());
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 1);
        assert_eq!(tickets.requests(), 2);

        // The captured `messages` is the second call's slice — it must
        // contain the assistant ToolUse plus the tool-result User block.
        let captured = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        let last = captured.last().expect("at least one message");
        match last {
            Message::User { content } => match content.first() {
                Some(ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                }) => {
                    assert_eq!(tool_use_id, "call-1");
                    assert_eq!(content, "hi");
                    assert!(!is_error);
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            other => panic!("expected last message to be User(ToolResult), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_calls_counter_increments_on_dispatch() {
        let (provider, _) = MockProvider::queued(vec![
            tool_use_response("c1", "echo", serde_json::json!({"value": "a"})),
            text_response("done"),
        ]);
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker").tool(echo_tool());
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.tool_calls(), 1);
    }

    #[tokio::test]
    async fn tool_error_surfaces_to_model_as_is_error_block() {
        let (provider, handles) = MockProvider::queued(vec![
            tool_use_response("c1", "broken", serde_json::json!({})),
            text_response("done"),
        ]);
        let broken = Tool::new("broken", "Always fails")
            .read_only(true)
            .handler(|_input, _ctx| {
                Box::pin(async move { Ok(ToolResult::error("boom")) })
            });
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker").tool(broken);
        let _ = tickets.assign(agent).run_until_empty().await;

        let captured = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        let last = captured.last().unwrap();
        match last {
            Message::User { content } => match content.first() {
                Some(ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                }) => {
                    assert_eq!(tool_use_id, "c1");
                    assert!(content.contains("boom"));
                    assert!(*is_error);
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }
}
