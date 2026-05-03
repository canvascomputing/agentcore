//! Multi-agent loop driver. Each agent runs in its own tokio task
//! against a shared `TicketSystemState` passed in by `Runnable::run`.
//! Also defines the `Runnable` trait that `TicketSystem` implements.

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::event::{Event, EventKind, ToolFailureKind};
use crate::providers::types::{ResponseStatus, StreamEvent};
use crate::providers::{AsUserMessage, ContentBlock, Message, ModelRequest};
use crate::tools::{ToolCall, ToolContext, ToolError};

use super::agent::Agent;
use super::tickets::{Status, TicketSystemState};

/// What it means to be a thing agents can be added to and run against.
/// Implemented by `TicketSystem`.
pub trait Runnable: Sized {
    /// Bind `agent` to this system: drain any tickets the agent had
    /// queued in its private default system into this one, then push a
    /// clone of `agent` onto this system's agents list. Returns the
    /// wired agent so the caller can keep using it (chain `.task(...)`
    /// etc.).
    fn add(&self, agent: Agent) -> Agent;

    /// Bind `agent` and additionally pin it to a label scope. The second
    /// slot mirrors the `_assigned` task-creation methods: a label, or
    /// (by convention) an agent name when delegating.
    fn add_assigned(&self, agent: Agent, assign: impl Into<String>) -> Agent {
        self.add(agent.label(assign))
    }

    /// Drive every staged agent until the implementor's interrupt
    /// signal fires.
    fn run(&self) -> impl Future<Output = ()> + Send;

    /// Drive every staged agent until the queue settles, a policy
    /// trips, or `.timeout(...)` elapses. Returns the result of the
    /// most recently created `Status::Done` ticket — convenient for
    /// single-task agents — or `None` if no ticket reached `Done`.
    fn run_dry(&self) -> impl Future<Output = Option<String>> + Send;
}

const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Drive many agents against one shared `TicketSystemState`. Each agent
/// runs in its own tokio task; locks on the shared state are held only
/// around the queue/metric ops and never across `provider.respond().await`,
/// so model calls really do overlap.
pub(super) async fn run_main_loop(
    agents: Vec<Agent>,
    state: Arc<Mutex<TicketSystemState>>,
) {
    let mut handles = Vec::with_capacity(agents.len());
    for agent in agents {
        handles.push(tokio::spawn(handle_tickets(agent, Arc::clone(&state))));
    }
    for h in handles {
        let _ = h.await;
    }
}

/// Per-agent loop. Picks the next eligible ticket, hands it to
/// `process_ticket`, repeats. When no eligible work is queued, idles on
/// `IDLE_POLL_INTERVAL` until something arrives, the cancel signal fires,
/// or a policy hits.
pub(super) async fn handle_tickets(
    agent: Agent,
    state: Arc<Mutex<TicketSystemState>>,
) {
    loop {
        let claim = {
            let mut sys = state.lock().unwrap();
            if sys.is_interrupted() {
                return;
            }
            if let Some((kind, limit)) = sys.policy_violated_kind() {
                let handler = agent.resolve_event_handler();
                handler(Event::new(
                    agent.get_name(),
                    EventKind::PolicyViolated { kind, limit },
                ));
                return;
            }
            // Path A: tickets already directed to this agent (already
            // InProgress with assignee == self), regardless of labels.
            let path_a = sys
                .list_by_status(Status::InProgress)
                .iter()
                .find(|t| t.assignee() == Some(agent.get_name()))
                .map(|t| (t.key().to_string(), false));
            // Path B: open Todos whose labels intersect the agent's
            // declared label scope.
            let path_b = sys
                .list_by_status(Status::Todo)
                .iter()
                .find(|t| t.assignee().is_none() && agent.handles(&t.labels))
                .map(|t| (t.key().to_string(), true));
            let claim = path_a.or(path_b);
            if let Some((key, needs_assign)) = claim.as_ref() {
                if *needs_assign {
                    let _ = sys.assign_to(key, agent.get_name());
                    let _ = sys.update_status(key, Status::InProgress);
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
        process_ticket(&agent, &state, &key).await;
        agent.set_current_ticket(None);
    }
}

/// One ticket from claimed → settled. Owns the per-ticket message vector.
async fn process_ticket(
    agent: &Agent,
    state: &Arc<Mutex<TicketSystemState>>,
    key: &str,
) {
    let handler = agent.resolve_event_handler();
    let emit = |kind: EventKind| handler(Event::new(agent.get_name(), kind));

    let mut messages: Vec<Message> = Vec::new();
    if let Some(ctx) = agent.context_message() {
        messages.push(Message::user(ctx));
    }
    let task_msg = {
        let sys = state.lock().unwrap();
        sys.get(key).map(|t| t.as_user_message())
    };
    let Some(task_msg) = task_msg else {
        return;
    };
    messages.push(task_msg);
    emit(EventKind::TicketClaimed {
        key: key.to_string(),
    });

    let (max_request_tokens, max_schema_retries, interrupt_signal) = {
        let sys = state.lock().unwrap();
        (
            sys.policies().max_request_tokens,
            sys.policies().max_schema_retries.unwrap_or(u32::MAX),
            sys.interrupt_signal_handle(),
        )
    };
    // Consecutive schema-validation failures since the last successful
    // schema-checked tool call. Bounded by `max_schema_retries`.
    let mut consecutive_schema_failures: u32 = 0;

    let on_stream: Arc<dyn Fn(StreamEvent) + Send + Sync> = {
        let handler = agent.resolve_event_handler();
        let name = agent.get_name().to_string();
        Arc::new(move |ev| {
            if let StreamEvent::TextDelta { text, .. } = ev {
                handler(Event::new(
                    &name,
                    EventKind::TextChunkReceived { content: text },
                ));
            }
        })
    };

    loop {
        // Settled? Stop the inner loop. The agent's `done` tool action
        // is the only way to reach Done from inside this loop.
        {
            let sys = state.lock().unwrap();
            if let Some(t) = sys.get(key) {
                if matches!(t.status(), Status::Done | Status::Failed) {
                    drop(sys);
                    emit(EventKind::TicketFinished {
                        key: key.to_string(),
                    });
                    return;
                }
            } else {
                return;
            }
        }

        emit(EventKind::RequestStarted {
            model: agent.model_str().to_string(),
        });
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
            .respond(request, Arc::clone(&on_stream))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                emit(EventKind::RequestFailed {
                    kind: e.kind(),
                    message: e.to_string(),
                });
                state
                    .lock()
                    .unwrap()
                    .record_error(agent.get_name(), key, &e);
                emit(EventKind::TicketFinished {
                    key: key.to_string(),
                });
                return;
            }
        };

        emit(EventKind::RequestFinished {
            model: response.model.clone(),
        });
        emit(EventKind::TokensReported {
            model: response.model.clone(),
            usage: response.usage.clone(),
        });

        {
            let mut sys = state.lock().unwrap();
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
            // Inner loop terminates without a `done` call — the ticket
            // stays InProgress and Path A may re-pick it. The agent must
            // call the `done` tool to settle.
            emit(EventKind::TicketFinished {
                key: key.to_string(),
            });
            return;
        }

        for call in &calls {
            emit(EventKind::ToolCallStarted {
                tool_name: call.name.clone(),
                call_id: call.id.clone(),
                input: call.input.clone(),
            });
        }

        let ctx = ToolContext::new(agent.working_dir_or_default())
            .interrupt_signal(Arc::clone(&interrupt_signal))
            .registry(Arc::new(agent.tool_registry().clone()))
            .ticket_system_state(Arc::clone(state))
            .current_ticket(key.to_string())
            .agent_name(agent.get_name().to_string());
        let outcomes = agent.tool_registry().execute(&calls, &ctx).await;

        for (block, verdict) in &outcomes {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                let call = calls.iter().find(|c| &c.id == tool_use_id);
                let tool_name = call.map(|c| c.name.clone()).unwrap_or_default();
                match verdict {
                    Ok(output) => {
                        if let Some(call) = call {
                            if is_done_call(call) {
                                consecutive_schema_failures = 0;
                            }
                        }
                        emit(EventKind::ToolCallFinished {
                            tool_name,
                            call_id: tool_use_id.clone(),
                            output: output.clone(),
                        });
                    }
                    Err(err) => {
                        if matches!(err, ToolError::SchemaValidationFailed { .. }) {
                            consecutive_schema_failures =
                                consecutive_schema_failures.saturating_add(1);
                        }
                        emit(EventKind::ToolCallFailed {
                            tool_name,
                            call_id: tool_use_id.clone(),
                            message: err.message(),
                            kind: match err {
                                ToolError::ToolNotFound { .. } => ToolFailureKind::ToolNotFound,
                                ToolError::ExecutionFailed { .. } => {
                                    ToolFailureKind::ExecutionFailed
                                }
                                ToolError::SchemaValidationFailed { .. } => {
                                    ToolFailureKind::SchemaValidationFailed
                                }
                            },
                        });
                    }
                }
            }
        }

        let blocks: Vec<ContentBlock> = outcomes.into_iter().map(|(b, _)| b).collect();
        messages.push(Message::User { content: blocks });

        {
            let mut sys = state.lock().unwrap();
            sys.record_tool_calls(agent.get_name(), calls.len() as u64);
        }

        if consecutive_schema_failures >= max_schema_retries {
            emit(EventKind::PolicyViolated {
                kind: crate::event::PolicyKind::MaxSchemaRetries,
                limit: u64::from(max_schema_retries),
            });
            // Force-fail the ticket so Path A doesn't re-pick it
            // forever. The agent demonstrably can't satisfy the schema;
            // the ticket is dead.
            {
                let mut sys = state.lock().unwrap();
                let _ = sys.force_status(key, Status::Failed);
            }
            emit(EventKind::TicketFinished {
                key: key.to_string(),
            });
            return;
        }
    }
}

/// Whether a tool call goes through `done`-side schema validation. Used
/// to reset `consecutive_schema_failures` on a successful `done` call.
fn is_done_call(call: &ToolCall) -> bool {
    matches!(
        call.name.as_str(),
        "manage_tickets_tool" | "write_tickets_tool"
    ) && call.input.get("action").and_then(|v| v.as_str()) == Some("done")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::tickets::TicketSystem;
    use crate::providers::types::{ModelResponse, ResponseStatus};
    use crate::providers::{ContentBlock, Provider, ProviderResult, TokenUsage};
    use crate::tools::ManageTicketsTool;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex as StdMutex;

    /// Mock provider that pops responses from a queue, falling back to a
    /// canned EndTurn text once empty.
    struct MockProvider {
        queue: StdMutex<VecDeque<ModelResponse>>,
        fallback_text: String,
    }

    impl MockProvider {
        fn queued(responses: Vec<ModelResponse>) -> Arc<Self> {
            Arc::new(Self {
                queue: StdMutex::new(responses.into()),
                fallback_text: "end".to_string(),
            })
        }
    }

    impl Provider for MockProvider {
        fn respond(
            &self,
            _request: crate::providers::ModelRequest,
            _on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
            let queued = self.queue.lock().unwrap().pop_front();
            let fallback = self.fallback_text.clone();
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(queued.unwrap_or_else(|| ModelResponse {
                    content: vec![ContentBlock::Text { text: fallback }],
                    status: ResponseStatus::EndTurn,
                    usage: TokenUsage::default(),
                    model: "mock".into(),
                }))
            })
        }
    }

    fn tool_use(id: &str, name: &str, input: serde_json::Value) -> ModelResponse {
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

    fn text(t: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: t.into() }],
            status: ResponseStatus::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    /// Build a response sequence that settles `count` tickets each with a
    /// single `done` call followed by an EndTurn text.
    fn done_responses(count: usize, result: &str) -> Vec<ModelResponse> {
        let mut out = Vec::with_capacity(count * 2);
        for _ in 0..count {
            out.push(tool_use(
                "done-1",
                "manage_tickets_tool",
                serde_json::json!({"action": "done", "result": result}),
            ));
            out.push(text("ok"));
        }
        out
    }

    fn agent_with(provider: Arc<dyn Provider>, name: &str) -> Agent {
        Agent::new()
            .name(name)
            .provider(provider)
            .model("mock")
            .silent()
    }

    #[tokio::test]
    async fn path_b_claims_and_auto_flips_to_inprogress() {
        let provider = MockProvider::queued(done_responses(1, "found it"));
        let tickets = TicketSystem::new();
        tickets.task_assigned("research the moon", "research");
        let alice = agent_with(provider, "alice")
            .label("research")
            .tool(ManageTicketsTool);
        tickets.add(alice);
        tickets.run_dry().await;

        let t = tickets.first().unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.assignee(), Some("alice"));
        assert_eq!(t.result(), Some("found it"));
    }

    #[tokio::test]
    async fn path_a_picks_up_assigned_inprogress_ticket() {
        let provider = MockProvider::queued(done_responses(1, "delivered"));
        let tickets = TicketSystem::new();
        // Alice is registered, so task_assigned with "alice" treats the
        // string as an agent name and births the ticket InProgress.
        let alice = agent_with(provider, "alice").tool(ManageTicketsTool);
        tickets.add(alice);
        tickets.task_assigned("specific work for alice", "alice");
        tickets.run_dry().await;

        let t = tickets.first().unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.assignee(), Some("alice"));
        assert_eq!(t.result(), Some("delivered"));
    }

    #[tokio::test]
    async fn endturn_without_done_leaves_ticket_inprogress() {
        // Drive max_steps(1) so the loop exits after one ticket pickup
        // even though the agent never settles it. The watcher would
        // otherwise spin on the still-pending ticket.
        let provider = MockProvider::queued(vec![text("done thinking")]);
        let tickets = TicketSystem::new().max_steps(1);
        tickets.task_assigned("research", "research");
        let alice = agent_with(provider, "alice")
            .label("research")
            .tool(ManageTicketsTool);
        tickets.add(alice);
        tickets.run_dry().await;

        let t = tickets.first().unwrap();
        assert_eq!(t.status(), Status::InProgress);
        assert!(t.result().is_none());
    }

    #[tokio::test]
    async fn done_with_schema_validates_result() {
        let schema = crate::schemas::Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        }))
        .unwrap();
        let provider = MockProvider::queued(vec![
            tool_use(
                "done-1",
                "manage_tickets_tool",
                serde_json::json!({
                    "action": "done",
                    "result": "{\"answer\": \"42\"}"
                }),
            ),
            text("ok"),
        ]);
        let tickets = TicketSystem::new();
        tickets.task_schema_assigned("hi", schema, "research");
        let alice = agent_with(provider, "alice")
            .label("research")
            .tool(ManageTicketsTool);
        tickets.add(alice);
        tickets.run_dry().await;

        let t = tickets.first().unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result(), Some("{\"answer\": \"42\"}"));
    }

    #[tokio::test]
    async fn schema_failure_in_done_trips_max_schema_retries() {
        let schema = crate::schemas::Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        }))
        .unwrap();
        // Three bad done calls in a row; max_schema_retries(3) trips.
        let bad_done = || {
            tool_use(
                "done-bad",
                "manage_tickets_tool",
                serde_json::json!({
                    "action": "done",
                    "result": "{\"answer\": 7}"
                }),
            )
        };
        let provider = MockProvider::queued(vec![
            bad_done(),
            bad_done(),
            bad_done(),
            text("never"),
        ]);
        let tickets = TicketSystem::new()
            .max_schema_retries(3)
            .max_steps(5);
        tickets.task_schema_assigned("hi", schema, "research");
        let alice = agent_with(provider, "alice")
            .label("research")
            .tool(ManageTicketsTool);
        tickets.add(alice);
        tickets.run_dry().await;

        let t = tickets.first().unwrap();
        assert_eq!(t.status(), Status::Failed);
    }

    #[tokio::test]
    async fn timeout_stops_run_dry_with_pending_tickets() {
        // Empty response queue → mock falls through to EndTurn "end".
        // Combined with no tools registered, the inner loop exits
        // immediately with the ticket still InProgress; Path A re-picks
        // forever. Timeout must break the spin.
        let provider = MockProvider::queued(Vec::new());
        let tickets = TicketSystem::new().timeout(Duration::from_millis(150));
        tickets.task_assigned("never settled", "research");
        let alice = agent_with(provider, "alice").label("research");
        tickets.add(alice);

        let started = std::time::Instant::now();
        tickets.run_dry().await;
        let elapsed = started.elapsed();

        assert!(elapsed < Duration::from_millis(800), "elapsed={elapsed:?}");
        let unsettled = tickets
            .tickets()
            .into_iter()
            .filter(|t| matches!(t.status(), Status::Todo | Status::InProgress))
            .count();
        assert_ne!(unsettled, 0);
    }

    #[tokio::test]
    async fn interrupt_signal_stops_idle_agent_immediately() {
        let provider = MockProvider::queued(Vec::new());
        let signal = Arc::new(AtomicBool::new(true));
        let tickets = TicketSystem::new().interrupt_signal(Arc::clone(&signal));
        let alice = agent_with(provider, "alice");
        tickets.add(alice);
        // .run() (not run_dry) — uses the supplied signal, which is
        // pre-fired so the loop exits before doing anything.
        tickets.run().await;
        assert_eq!(tickets.requests(), 0);
        assert_eq!(tickets.steps(), 0);
        // Ensure the signal isn't re-cleared by any internal logic.
        assert!(signal.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn done_resets_consecutive_schema_failure_counter() {
        let schema = crate::schemas::Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        }))
        .unwrap();
        // Two failures, one success — the counter resets, so the trip
        // doesn't fire even though four total bad calls would otherwise
        // exceed max_schema_retries(3).
        let bad = || {
            tool_use(
                "bad",
                "manage_tickets_tool",
                serde_json::json!({"action": "done", "result": "{\"answer\": 7}"}),
            )
        };
        let good = tool_use(
            "good",
            "manage_tickets_tool",
            serde_json::json!({"action": "done", "result": "{\"answer\": \"yes\"}"}),
        );
        let provider = MockProvider::queued(vec![bad(), bad(), good, text("ok")]);
        let tickets = TicketSystem::new().max_schema_retries(3);
        tickets.task_schema_assigned("hi", schema, "research");
        let alice = agent_with(provider, "alice")
            .label("research")
            .tool(ManageTicketsTool);
        tickets.add(alice);
        tickets.run_dry().await;

        let t = tickets.first().unwrap();
        assert_eq!(t.status(), Status::Done);
    }

    #[tokio::test]
    async fn run_dry_returns_last_done_result_for_single_task() {
        let provider = MockProvider::queued(done_responses(1, "the answer"));
        let tickets = TicketSystem::new();
        tickets.task_assigned("anything", "research");
        let alice = agent_with(provider, "alice")
            .label("research")
            .tool(ManageTicketsTool);
        tickets.add(alice);
        let result = tickets.run_dry().await;
        assert_eq!(result.as_deref(), Some("the answer"));
    }

    #[tokio::test]
    async fn run_dry_returns_none_when_no_ticket_reaches_done() {
        let provider = MockProvider::queued(Vec::new());
        let tickets = TicketSystem::new().timeout(Duration::from_millis(50));
        tickets.task_assigned("never", "research");
        let alice = agent_with(provider, "alice").label("research");
        tickets.add(alice);
        let result = tickets.run_dry().await;
        assert!(result.is_none());
    }
}
