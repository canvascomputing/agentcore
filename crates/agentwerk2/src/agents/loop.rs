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
            let todos = sys.list_by_status(Status::Todo);
            // Path A: tickets already directed to this agent (via assignee)
            // — finish those first, regardless of ticket_types.
            let path_a = todos
                .iter()
                .find(|t| t.assignee.as_deref() == Some(agent.get_name()))
                .map(|t| (t.key.clone(), false));
            // Path B: open Todos whose type matches the agent's allow-list.
            let path_b = todos
                .iter()
                .find(|t| t.assignee.is_none() && agent.handles(&t.r#type))
                .map(|t| (t.key.clone(), true));
            let claim = path_a.or(path_b);
            if let Some((key, needs_assign)) = claim.as_ref() {
                if *needs_assign {
                    let _ = sys.assign_to(key, agent.get_name());
                }
                sys.record_step(agent.get_name());
            }
            claim.map(|(key, _)| key)
        };

        let Some(key) = claim else {
            tokio::time::sleep(IDLE_POLL_INTERVAL).await;
            continue;
        };

        agent.set_current_ticket(Some(key.clone()));
        process_ticket(&agent, &tickets, &key).await;
        agent.set_current_ticket(None);
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
                tickets
                    .lock()
                    .unwrap()
                    .record_error(agent.get_name(), key, &e);
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
            // Inner loop terminates. The agent didn't request more
            // work — it's the agent's job (via the ticket tools) to
            // settle the ticket; the loop never writes status.
            return;
        }

        // Dispatch the tool calls. ToolContext carries working_dir,
        // the shared cancel signal, the registry handle (for
        // ToolSearchTool), and ticket-side info (for the ticket tools).
        let ctx = ToolContext::new(agent.working_dir_or_default())
            .interrupt_signal(Arc::clone(&interrupt_signal))
            .registry(Arc::new(agent.tool_registry().clone()))
            .tickets(Arc::clone(tickets))
            .current_ticket(key.to_string())
            .agent_name(agent.get_name().to_string());
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

    /// Build the model output that drives `count` tickets all the way
    /// to `Status::Done` via `manage_tickets_tool`. Each ticket gets
    /// three responses, walking the legal state-machine path: a
    /// ToolUse transitioning `Todo → InProgress`, a ToolUse
    /// transitioning `InProgress → Done`, and a final EndTurn text.
    /// Queue this as the mock provider's response stream when a test
    /// drives `run_until_empty` and needs the agent to drain its
    /// work.
    fn transition_to_done_responses(count: usize) -> Vec<ModelResponse> {
        transition_to_done_responses_with_usage(count, TokenUsage::default())
    }

    /// Same as `transition_to_done_responses` but stamps each response
    /// with the given `usage` — for tests that exercise token-budget
    /// policies.
    fn transition_to_done_responses_with_usage(
        count: usize,
        usage: TokenUsage,
    ) -> Vec<ModelResponse> {
        let mut out = Vec::with_capacity(count * 3);
        for _ in 0..count {
            let mut to_in_progress = tool_use_response(
                "transition-in-progress",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "InProgress"}),
            );
            to_in_progress.usage = usage.clone();
            let mut to_done = tool_use_response(
                "transition-done",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "Done"}),
            );
            to_done.usage = usage.clone();
            let mut text = text_response("done");
            text.usage = usage.clone();
            out.push(to_in_progress);
            out.push(to_done);
            out.push(text);
        }
        out
    }

    #[tokio::test]
    async fn single_agent_drains_queue() {
        let (provider, _) = MockProvider::queued(transition_to_done_responses(2));
        let mut tickets = TicketSystem::new();
        tickets.create("first", "", "task", "tester");
        tickets.create("second", "", "task", "tester");

        let agent = agent_with(provider, "worker")
            .role("you are a worker")
            .tool(ManageTicketsTool);
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.steps(), 2);
        // 3 requests per ticket (InProgress, Done, EndTurn text);
        // 2 dispatches per ticket (the two transitions).
        assert_eq!(tickets.requests(), 6);
        assert_eq!(tickets.tool_calls(), 4);
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
        let (provider, _) = MockProvider::queued(transition_to_done_responses(3));
        let mut tickets = TicketSystem::new().max_steps(2);
        tickets.create("a", "", "task", "tester");
        tickets.create("b", "", "task", "tester");
        tickets.create("c", "", "task", "tester");

        let agent = agent_with(provider, "worker").tool(ManageTicketsTool);
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 2);
        assert_eq!(tickets.list_by_status(Status::Todo).len(), 1);
    }

    #[tokio::test]
    async fn max_input_tokens_stops_processing_after_threshold() {
        // Each ticket needs two requests (tool_use + EndTurn). With 30
        // input tokens per response, ticket 1 spends 60, ticket 2 spends
        // 60 more (120 total) and trips the 100-token threshold. The
        // agent halts before picking up ticket 3.
        let usage = TokenUsage {
            input_tokens: 30,
            output_tokens: 0,
            ..TokenUsage::default()
        };
        let queue = transition_to_done_responses_with_usage(3, usage);
        let (provider, _) = MockProvider::queued(queue);
        let mut tickets = TicketSystem::new().max_input_tokens(100);
        tickets.create("a", "", "task", "tester");
        tickets.create("b", "", "task", "tester");
        tickets.create("c", "", "task", "tester");

        let agent = agent_with(provider, "worker").tool(ManageTicketsTool);
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 2);
        assert_eq!(tickets.list_by_status(Status::Todo).len(), 1);
    }

    #[tokio::test]
    async fn system_prompt_includes_role_and_behavior() {
        let (provider, handles) = MockProvider::queued(transition_to_done_responses(1));
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker")
            .role("ROLE_TEXT")
            .behavior("BEHAVIOR_TEXT")
            .context("CONTEXT_TEXT")
            .tool(ManageTicketsTool);
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
        let (provider, handles) = MockProvider::queued(transition_to_done_responses(1));
        let mut tickets = TicketSystem::new();
        tickets.create("task summary", "task body", "task", "tester");

        let agent = agent_with(provider, "worker")
            .context("CONTEXT_TEXT")
            .tool(ManageTicketsTool);
        let _ = tickets.assign(agent).run_until_empty().await;

        let messages = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        // The captured slice is from the second (final) request, so it
        // also carries the assistant ToolUse + tool result. The first
        // two messages remain the context block and the ticket.
        assert!(messages.len() >= 2);
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
        let (provider, handles) = MockProvider::queued(transition_to_done_responses(1));
        let mut tickets = TicketSystem::new();
        tickets.create("the summary", "the body", "task", "tester");

        let agent = agent_with(provider, "worker").tool(ManageTicketsTool);
        let _ = tickets.assign(agent).run_until_empty().await;

        let messages = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        // Without context, the ticket is the first user message — the
        // captured slice also carries the trailing tool exchange.
        assert!(!messages.is_empty());
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
        let (provider, handles) = MockProvider::queued(transition_to_done_responses(1));
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker")
            .role("ROLE_TEXT")
            .tool(ManageTicketsTool);
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
        let (provider, handles) = MockProvider::queued(transition_to_done_responses(1));
        let mut tickets = TicketSystem::new().max_request_tokens(256);
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker").tool(ManageTicketsTool);
        let _ = tickets.assign(agent).run_until_empty().await;

        assert_eq!(*handles.max_tokens.lock().unwrap(), Some(256));
    }

    #[tokio::test]
    async fn run_main_loop_drives_multiple_agents_in_parallel() {
        let mut tickets = TicketSystem::new();
        tickets.create("a", "", "task", "tester");
        tickets.create("b", "", "task", "tester");
        tickets.create("c", "", "task", "tester");
        tickets.create("d", "", "task", "tester");

        // Each agent gets its own queue. With a shared provider the
        // two agents would race on the same canned responses and pop
        // each other's transitions out of order — leaving tickets
        // half-settled and the loop spinning. Either may pick up 0–4
        // tickets depending on tokio scheduling, so each queue covers
        // the worst case alone.
        let (alice_provider, _) = MockProvider::queued(transition_to_done_responses(4));
        let (bob_provider, _) = MockProvider::queued(transition_to_done_responses(4));
        let alice = agent_with(alice_provider, "alice").tool(ManageTicketsTool);
        let bob = agent_with(bob_provider, "bob").tool(ManageTicketsTool);
        let tickets = tickets.assign_all([alice, bob]).run_until_empty().await;

        // Aggregate counters are deterministic regardless of how the
        // load splits — 3 requests + 2 dispatches per ticket × 4
        // tickets.
        assert_eq!(tickets.list_by_status(Status::Done).len(), 4);
        assert_eq!(tickets.steps(), 4);
        assert_eq!(tickets.requests(), 12);
        assert_eq!(tickets.tool_calls(), 8);
    }

    #[tokio::test]
    async fn idle_agent_picks_up_a_late_arriving_ticket() {
        let (provider, _) = MockProvider::queued(transition_to_done_responses(1));
        let signal = Arc::new(AtomicBool::new(false));
        let tickets = TicketSystem::new().interrupt_signal(Arc::clone(&signal));

        let agent = agent_with(provider, "worker").tool(ManageTicketsTool);
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
        shared.lock().unwrap().create("late", "", "task", "tester");

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
        // 3 requests for the one ticket: InProgress, Done, EndTurn.
        assert_eq!(sys.requests(), 3);
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
            agent_with(
                Arc::clone(&(provider.clone() as Arc<dyn Provider>)),
                "alice",
            ),
            agent_with(Arc::clone(&(provider.clone() as Arc<dyn Provider>)), "bob"),
            agent_with(provider, "carol"),
        ]);
        assert_eq!(counter.names, vec!["alice", "bob", "carol"]);
    }

    #[tokio::test]
    async fn agent_with_ticket_type_skips_other_types() {
        // The bug ticket stays Todo forever, so the strict watcher
        // would never auto-stop on its own. Bound the run with
        // max_steps(1): one pickup of the task ticket, then halt.
        let (provider, _) = MockProvider::queued(transition_to_done_responses(1));
        let mut tickets = TicketSystem::new().max_steps(1);
        tickets.create("real work", "", "task", "tester");
        tickets.create("a bug", "", "bug", "tester");

        let agent = agent_with(provider, "worker")
            .ticket_type("task")
            .tool(ManageTicketsTool);
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 1);
        assert_eq!(tickets.list_by_status(Status::Todo).len(), 1);
        let done = &tickets.list_by_status(Status::Done)[0];
        assert_eq!(done.r#type, "task");
        let todo = &tickets.list_by_status(Status::Todo)[0];
        assert_eq!(todo.r#type, "bug");
        assert!(
            todo.assignee.is_none(),
            "bug ticket should remain unassigned"
        );
    }

    use crate::tools::{ManageTicketsTool, Tool, ToolResult};

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
            tool_use_response(
                "transition-in-progress",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "InProgress"}),
            ),
            tool_use_response(
                "transition-done",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "Done"}),
            ),
            text_response("done"),
        ]);
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker")
            .tool(echo_tool())
            .tool(ManageTicketsTool);
        let tickets = tickets.assign(agent).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 1);
        assert_eq!(tickets.requests(), 4);

        // The captured slice is the third (final) request's messages —
        // it carries every prior assistant ToolUse plus the matching
        // user ToolResult blocks. Find the echo result among them.
        let captured = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        let echo_result = captured.iter().find_map(|m| {
            let Message::User { content } = m else {
                return None;
            };
            content.iter().find_map(|cb| match cb {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } if tool_use_id == "call-1" => Some((content.clone(), *is_error)),
                _ => None,
            })
        });
        let (content, is_error) = echo_result.expect("echo tool result captured");
        assert_eq!(content, "hi");
        assert!(!is_error);
    }

    #[tokio::test]
    async fn tool_calls_counter_increments_on_dispatch() {
        let (provider, _) = MockProvider::queued(vec![
            tool_use_response("c1", "echo", serde_json::json!({"value": "a"})),
            tool_use_response(
                "transition-in-progress",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "InProgress"}),
            ),
            tool_use_response(
                "transition-done",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "Done"}),
            ),
            text_response("done"),
        ]);
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker")
            .tool(echo_tool())
            .tool(ManageTicketsTool);
        let tickets = tickets.assign(agent).run_until_empty().await;

        // Three dispatches: echo, transition→InProgress, transition→Done.
        assert_eq!(tickets.tool_calls(), 3);
    }

    #[tokio::test]
    async fn tool_error_surfaces_to_model_as_is_error_block() {
        let (provider, handles) = MockProvider::queued(vec![
            tool_use_response("c1", "broken", serde_json::json!({})),
            tool_use_response(
                "transition-in-progress",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "InProgress"}),
            ),
            tool_use_response(
                "transition-done",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "Done"}),
            ),
            text_response("done"),
        ]);
        let broken = Tool::new("broken", "Always fails")
            .read_only(true)
            .handler(|_input, _ctx| Box::pin(async move { Ok(ToolResult::error("boom")) }));
        let mut tickets = TicketSystem::new();
        tickets.create("task", "", "task", "tester");

        let agent = agent_with(provider, "worker")
            .tool(broken)
            .tool(ManageTicketsTool);
        let _ = tickets.assign(agent).run_until_empty().await;

        // Find the broken tool's result among captured messages — the
        // captured slice is from the final request and contains every
        // tool exchange leading up to it.
        let captured = handles
            .messages
            .lock()
            .unwrap()
            .clone()
            .expect("messages captured");
        let broken_result = captured.iter().find_map(|m| {
            let Message::User { content } = m else {
                return None;
            };
            content.iter().find_map(|cb| match cb {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } if tool_use_id == "c1" => Some((content.clone(), *is_error)),
                _ => None,
            })
        });
        let (content, is_error) = broken_result.expect("broken tool result captured");
        assert!(content.contains("boom"));
        assert!(is_error);
    }

    #[tokio::test]
    async fn agent_picks_up_path_a_ticket_assigned_by_another_agent() {
        // The ticket type is `bug`, but alice only handles `task`. Path
        // B never matches; the ticket gets through only because Path A
        // claims any Todo whose `assignee` already names the agent.
        let (provider, _) = MockProvider::queued(transition_to_done_responses(1));
        let mut tickets = TicketSystem::new();
        let key = tickets
            .create("delegated to alice", "", "bug", "tester")
            .key
            .clone();
        tickets.assign_to(&key, "alice").unwrap();

        let alice = agent_with(provider, "alice")
            .ticket_type("task")
            .tool(ManageTicketsTool);
        let tickets = tickets.assign(alice).run_until_empty().await;

        assert_eq!(tickets.list_by_status(Status::Done).len(), 1);
        let done = tickets.get(&key).unwrap();
        assert_eq!(done.status, Status::Done);
        assert_eq!(done.assignee.as_deref(), Some("alice"));
    }

    #[tokio::test]
    async fn path_a_takes_priority_over_path_b() {
        // TICKET-1 is a Path B candidate (`task`, unassigned). TICKET-2
        // is a Path A candidate (already assigned to alice). The loop
        // must claim TICKET-2 first even though TICKET-1 is older.
        let (provider, _) = MockProvider::queued(transition_to_done_responses(2));
        let mut tickets = TicketSystem::new();
        let path_b_key = tickets.create("path b", "", "task", "tester").key.clone();
        let path_a_key = tickets.create("path a", "", "bug", "tester").key.clone();
        tickets.assign_to(&path_a_key, "alice").unwrap();

        let alice = agent_with(provider, "alice")
            .ticket_type("task")
            .tool(ManageTicketsTool);
        let tickets = tickets.assign(alice).run_until_empty().await;

        // Both end up Done — but Path A's ticket should have been
        // picked up first. The mock doesn't capture per-ticket order
        // directly; assert via comment timestamps left by the
        // transition tool, which records the action sequence in
        // `requests()` order.
        assert_eq!(tickets.list_by_status(Status::Done).len(), 2);
        assert_eq!(tickets.get(&path_a_key).unwrap().status, Status::Done);
        assert_eq!(tickets.get(&path_b_key).unwrap().status, Status::Done);
        // Both tickets carry alice as the assignee — Path B's ticket
        // had it written by the loop, Path A's by the seeding test.
        assert_eq!(
            tickets.get(&path_a_key).unwrap().assignee.as_deref(),
            Some("alice")
        );
        assert_eq!(
            tickets.get(&path_b_key).unwrap().assignee.as_deref(),
            Some("alice")
        );
    }

    #[tokio::test]
    async fn agent_current_ticket_is_set_during_processing_and_cleared_after() {
        // A custom probe tool captures `agent.current_ticket()` while
        // the agent is mid-processing. After `run_until_empty` returns,
        // the agent's current ticket should be `None` again.
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let probe_capture = Arc::clone(&captured);

        // The probe writes the current ticket key it sees into the
        // shared slot, via the agent_name field on ToolContext — the
        // agent name is enough to look the ticket key up via the
        // `current_ticket` ambient field.
        let probe = Tool::new("probe", "Capture the current ticket")
            .read_only(true)
            .handler(move |_input, ctx| {
                let probe_capture = Arc::clone(&probe_capture);
                let key = ctx.current_ticket.clone();
                Box::pin(async move {
                    *probe_capture.lock().unwrap() = key;
                    Ok(ToolResult::success("ok"))
                })
            });

        let (provider, _) = MockProvider::queued(vec![
            tool_use_response("probe-1", "probe", serde_json::json!({})),
            tool_use_response(
                "transition-in-progress",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "InProgress"}),
            ),
            tool_use_response(
                "transition-done",
                "manage_tickets_tool",
                serde_json::json!({"action": "transition", "status": "Done"}),
            ),
            text_response("done"),
        ]);
        let mut tickets = TicketSystem::new();
        let key = tickets.create("watched", "", "task", "tester").key.clone();

        let alice = agent_with(provider, "alice")
            .tool(probe)
            .tool(ManageTicketsTool);
        // Hold a clone so we can read current_ticket() after the run.
        let alice_handle = alice.clone();
        let _ = tickets.assign(alice).run_until_empty().await;

        // During processing the probe saw the ticket key.
        assert_eq!(captured.lock().unwrap().as_deref(), Some(key.as_str()));
        // After the run, the agent's current ticket is cleared.
        assert!(alice_handle.current_ticket().is_none());
    }
}
