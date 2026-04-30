//! Structured events the loop emits so callers can observe a run
//! without wrapping the loop itself.

use std::sync::Arc;

use crate::providers::{RequestErrorKind, TokenUsage};

/// Which configured policy a [`EventKind::PolicyViolated`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyKind {
    /// `max_steps` — the step cap across all agents.
    Steps,
    /// `max_input_tokens` — cumulative request-side token cap.
    InputTokens,
    /// `max_output_tokens` — cumulative reply-side token cap.
    OutputTokens,
    /// `max_schema_retries` — consecutive schema-validation failures
    /// while processing one ticket. Resets after every successful
    /// schema-checked tool call.
    MaxSchemaRetries,
}

/// Categorical discriminant for [`EventKind::ToolCallFailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFailureKind {
    /// The registry had no tool with that name.
    ToolNotFound,
    /// The tool was invoked but its execution raised an error.
    ExecutionFailed,
    /// A schema-checked tool rejected its input. Counted against
    /// `policies.max_schema_retries` by the loop.
    SchemaValidationFailed,
}

/// Observation emitted during an agent run.
#[derive(Debug, Clone)]
pub struct Event {
    /// Name of the agent that produced this event.
    pub agent_name: String,
    /// What happened.
    pub kind: EventKind,
}

impl Event {
    pub(crate) fn new(agent_name: impl Into<String>, kind: EventKind) -> Self {
        Self { agent_name: agent_name.into(), kind }
    }
}

/// Categorical discriminant of [`Event`].
#[derive(Debug, Clone)]
pub enum EventKind {
    /// Agent claimed a ticket and began working on it.
    TicketClaimed { key: String },
    /// Agent finished processing a ticket. Status transition is the agent's
    /// responsibility via ticket tools; this fires regardless of outcome.
    TicketFinished { key: String },
    /// Provider request began.
    RequestStarted { model: String },
    /// Provider request finished successfully.
    RequestFinished { model: String },
    /// Provider request failed. The run is about to stop for this ticket.
    RequestFailed { kind: RequestErrorKind, message: String },
    /// Provider reported token counts for the last request.
    TokensReported { model: String, usage: TokenUsage },
    /// A streamed text chunk arrived from the provider.
    TextChunkReceived { content: String },
    /// Tool invocation began.
    ToolCallStarted { tool_name: String, call_id: String, input: serde_json::Value },
    /// Tool invocation succeeded.
    ToolCallFinished { tool_name: String, call_id: String, output: String },
    /// Tool invocation failed. The error is sent back to the model as a
    /// tool-result message; the run continues.
    ToolCallFailed { tool_name: String, call_id: String, message: String, kind: ToolFailureKind },
    /// A configured policy was exceeded; the run is about to stop.
    PolicyViolated { kind: PolicyKind, limit: u64 },
}

/// Default observer. Prints ticket lifecycle, tool activity, policy
/// violations, and request failures to stderr. Quiet variants
/// (token counts, streaming chunks, request start/finish) are dropped.
pub fn default_logger() -> Arc<dyn Fn(Event) + Send + Sync> {
    Arc::new(|event: Event| {
        let agent = &event.agent_name;
        match &event.kind {
            EventKind::TicketClaimed { key } => {
                eprintln!("[{agent}] claimed {key}");
            }
            EventKind::TicketFinished { key } => {
                eprintln!("[{agent}] finished {key}");
            }
            EventKind::ToolCallStarted { tool_name, input, .. } => {
                eprintln!("[{agent}] {tool_name}({})", compact_input(input));
            }
            EventKind::ToolCallFailed { tool_name, message, kind, .. } => {
                eprintln!("[{agent}] {tool_name} failed ({kind:?}): {message}");
            }
            EventKind::RequestFailed { message, .. } => {
                eprintln!("[{agent}] request failed: {message}");
            }
            EventKind::PolicyViolated { kind, limit } => {
                eprintln!("[{agent}] policy violated: {kind:?} limit={limit}");
            }
            _ => {}
        }
    })
}

fn compact_input(input: &serde_json::Value) -> String {
    let one_line = input.to_string().replace('\n', " ");
    const MAX: usize = 80;
    if one_line.chars().count() <= MAX {
        one_line
    } else {
        let cut: String = one_line.chars().take(MAX).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::TokenUsage;

    fn all_variants() -> Vec<EventKind> {
        vec![
            EventKind::TicketClaimed { key: "T-1".into() },
            EventKind::TicketFinished { key: "T-1".into() },
            EventKind::RequestStarted { model: "m".into() },
            EventKind::RequestFinished { model: "m".into() },
            EventKind::RequestFailed {
                kind: RequestErrorKind::ConnectionFailed,
                message: "timeout".into(),
            },
            EventKind::TokensReported {
                model: "m".into(),
                usage: TokenUsage::default(),
            },
            EventKind::TextChunkReceived { content: "hello".into() },
            EventKind::ToolCallStarted {
                tool_name: "bash".into(),
                call_id: "c1".into(),
                input: serde_json::json!({"cmd": "ls"}),
            },
            EventKind::ToolCallFinished {
                tool_name: "bash".into(),
                call_id: "c1".into(),
                output: "file.txt".into(),
            },
            EventKind::ToolCallFailed {
                tool_name: "bash".into(),
                call_id: "c1".into(),
                message: "not found".into(),
                kind: ToolFailureKind::ToolNotFound,
            },
            EventKind::ToolCallFailed {
                tool_name: "manage_tickets_tool".into(),
                call_id: "c2".into(),
                message: "Schema validation failed".into(),
                kind: ToolFailureKind::SchemaValidationFailed,
            },
            EventKind::PolicyViolated {
                kind: PolicyKind::Steps,
                limit: 10,
            },
            EventKind::PolicyViolated {
                kind: PolicyKind::MaxSchemaRetries,
                limit: 10,
            },
        ]
    }

    #[test]
    fn default_logger_handles_every_variant() {
        let logger = default_logger();
        for kind in all_variants() {
            logger(Event::new("agent", kind));
        }
    }
}
