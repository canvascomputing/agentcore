//! Multi-agent loop driver. Each agent runs in its own tokio task,
//! reading the shared ticket store, policies, stats, and interrupt
//! signal off its own fields (stamped at `bind_agent` time). Also
//! defines the `Runnable` trait that `TicketSystem` implements.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::event::{Event, EventKind, ToolFailureKind};
use crate::providers::types::{ResponseStatus, StreamEvent};
use crate::providers::{AsUserMessage, ContentBlock, Message, ModelRequest};
use crate::tools::{ToolCall, ToolContext, ToolError};

use super::agent::Agent;
use super::stats::LoopStats;
use super::tickets::{
    policy_violated_kind, tickets_assign_to, tickets_find, tickets_force_status, tickets_get,
    tickets_update_status, Status,
};

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
    /// signal fires. Use this when tickets keep arriving over time and
    /// the run is bounded by an external stop signal.
    fn run(&self) -> impl Future<Output = ()> + Send;

    /// Drive every staged agent until the queue settles, a policy
    /// trips, or `.timeout(...)` elapses. Returns the result of the
    /// most recently created `Status::Done` ticket, or an empty string
    /// when no ticket reached `Done`. Use this for fixed-batch runs
    /// where every ticket is enqueued up front.
    fn run_dry(&self) -> impl Future<Output = String> + Send;
}

const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Drive many agents in parallel. Each agent runs in its own tokio
/// task and reads the shared state through its own Arcs (`tickets`,
/// `stats`, `interrupt_signal`); the loop never holds those across
/// `provider.respond().await`.
pub(super) async fn run_main_loop(agents: Vec<Agent>) {
    let mut handles = Vec::with_capacity(agents.len());
    for agent in agents {
        handles.push(tokio::spawn(handle_tickets(agent)));
    }
    for h in handles {
        let _ = h.await;
    }
}

/// Future that resolves once `signal` flips. Polls on the same cadence
/// as `ToolContext::wait_for_cancel`. Pair with `tokio::select!` to
/// drop the losing branch on cancel: dropping the
/// `provider.respond(...)` future cascades to `reqwest`'s in-flight
/// request being aborted.
pub(super) async fn wait_for_signal(signal: &Arc<AtomicBool>) {
    const POLL: Duration = Duration::from_millis(50);
    loop {
        if signal.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Per-agent loop. Picks the next eligible ticket, hands it to
/// `process_ticket`, repeats. When no eligible work is queued, idles on
/// `IDLE_POLL_INTERVAL` until something arrives, the cancel signal fires,
/// or a policy hits.
pub(super) async fn handle_tickets(agent: Agent) {
    let ticket_system = agent
        .ticket_system
        .upgrade()
        .expect("Agent's TicketSystem was dropped before run() finished");
    let signal = Arc::clone(&ticket_system.interrupt_signal.lock().unwrap());
    loop {
        if signal.load(Ordering::Relaxed) {
            return;
        }
        let policies = ticket_system.policies();
        if let Some((kind, limit)) = policy_violated_kind(&policies, &ticket_system.stats) {
            let handler = agent.resolve_event_handler();
            handler(Event::new(
                agent.get_name(),
                EventKind::PolicyViolated { kind, limit },
            ));
            return;
        }
        // Path A: tickets already directed to this agent (already
        // InProgress with assignee == self), regardless of labels.
        let path_a = tickets_find(&ticket_system, |t| {
            t.is_in_progress() && t.is_assigned_to(agent.get_name())
        })
        .map(|t| (t.key().to_string(), false));
        // Path B: open Todos whose labels intersect the agent's
        // declared label scope.
        let path_b = tickets_find(&ticket_system, |t| {
            t.is_todo() && t.assignee().is_none() && agent.handles(&t.labels)
        })
        .map(|t| (t.key().to_string(), true));
        let claim = path_a.or(path_b);

        let key = match claim {
            Some((key, needs_assign)) => {
                if needs_assign {
                    let _ = tickets_assign_to(&ticket_system, &key, agent.get_name());
                    let _ = tickets_update_status(&ticket_system, &key, Status::InProgress);
                }
                ticket_system.stats.record_step();
                key
            }
            None => {
                tokio::time::sleep(IDLE_POLL_INTERVAL).await;
                continue;
            }
        };

        process_ticket(&agent, &ticket_system, &signal, &key).await;
    }
}

/// One ticket from claimed → settled. Owns the per-ticket message vector.
async fn process_ticket(
    agent: &Agent,
    ticket_system: &Arc<crate::agents::tickets::TicketSystem>,
    interrupt_signal: &Arc<std::sync::atomic::AtomicBool>,
    key: &str,
) {
    let handler = agent.resolve_event_handler();
    let emit = |kind: EventKind| handler(Event::new(agent.get_name(), kind));

    let mut messages: Vec<Message> = Vec::new();
    if let Some(ctx) = agent.context_message() {
        messages.push(Message::user(ctx));
    }
    let task_msg = tickets_get(ticket_system, key).map(|t| t.as_user_message());
    let Some(task_msg) = task_msg else {
        return;
    };
    messages.push(task_msg);
    emit(EventKind::TicketStarted {
        key: key.to_string(),
    });

    let policies = ticket_system.policies();
    let max_request_tokens = policies.max_request_tokens;
    let max_schema_retries = policies.max_schema_retries.unwrap_or(u32::MAX);
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
        if interrupt_signal.load(Ordering::Relaxed) {
            return;
        }
        // Settled? Stop the inner loop. The agent's `done` tool action
        // is the only way to reach Done from inside this loop.
        match tickets_get(ticket_system, key) {
            Some(t) if matches!(t.status(), Status::Done | Status::Failed) => {
                emit(terminal_event(t.status(), key));
                return;
            }
            Some(_) => {}
            None => return,
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
        let provider = agent.provider_handle();
        let response = tokio::select! {
            biased;
            _ = wait_for_signal(interrupt_signal) => return,
            r = provider.respond(request, Arc::clone(&on_stream)) => r,
        };
        let response = match response {
            Ok(r) => r,
            Err(e) => {
                emit(EventKind::RequestFailed {
                    kind: e.kind(),
                    message: e.to_string(),
                });
                ticket_system.stats.record_error();
                emit(EventKind::TicketFailed {
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

        ticket_system
            .stats
            .record_request(response.usage.input_tokens, response.usage.output_tokens);
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
            // call the `done` tool to mark the ticket done.
            emit(EventKind::TicketFailed {
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
            .interrupt_signal(Arc::clone(interrupt_signal))
            .registry(Arc::new(agent.tool_registry().clone()))
            .ticket_system(Arc::clone(ticket_system))
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

        for _ in 0..calls.len() {
            ticket_system.stats.record_tool_call();
        }

        if consecutive_schema_failures >= max_schema_retries {
            emit(EventKind::PolicyViolated {
                kind: crate::event::PolicyKind::MaxSchemaRetries,
                limit: u64::from(max_schema_retries),
            });
            // Force-fail the ticket so Path A doesn't re-pick it
            // forever. The agent demonstrably can't satisfy the schema;
            // the ticket is dead.
            let _ = tickets_force_status(ticket_system, key, Status::Failed);
            emit(EventKind::TicketFailed {
                key: key.to_string(),
            });
            return;
        }
    }
}

/// Whether a tool call goes through `done`-side schema validation. Used
/// to reset `consecutive_schema_failures` on a successful `done` call.
fn is_done_call(call: &ToolCall) -> bool {
    if call.name == "mark_ticket_done_tool" {
        return true;
    }
    matches!(
        call.name.as_str(),
        "manage_tickets_tool" | "write_tickets_tool"
    ) && call.input.get("action").and_then(|v| v.as_str()) == Some("done")
}

fn terminal_event(status: Status, key: &str) -> EventKind {
    match status {
        Status::Done => EventKind::TicketDone {
            key: key.to_string(),
        },
        Status::Failed => EventKind::TicketFailed {
            key: key.to_string(),
        },
        other => unreachable!("terminal_event called with non-terminal status {other:?}"),
    }
}
