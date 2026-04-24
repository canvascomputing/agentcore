//! Agent-level run and construction errors: policy violations (budgets / caps) and spawned-task crashes.

use std::fmt;

use crate::event::PolicyKind;

/// Failures that originate in the agent runtime, independent of the
/// provider or any tool. Builder misconfiguration (missing provider / model,
/// unreadable prompt files) panics at run / builder time instead.
#[derive(Debug)]
pub enum AgentError {
    /// A spawned agent's execution blew up — either panicked inside a tool or
    /// prompt, or was externally aborted. `message` is the runtime's
    /// description of the crash (panic payload or abort reason).
    AgentCrashed { message: String },
    /// A configured policy (`max_turns`, `max_input_tokens`, `max_output_tokens`,
    /// `max_schema_retries`) was exceeded and the run terminated. `kind` says
    /// which policy tripped.
    PolicyViolated { kind: PolicyKind, limit: u64 },
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::AgentCrashed { message } => {
                write!(f, "agent crashed: {message}")
            }
            AgentError::PolicyViolated { kind, limit } => {
                let label = match kind {
                    PolicyKind::Turns => "Turn limit reached",
                    PolicyKind::InputTokens => "Input token limit reached",
                    PolicyKind::OutputTokens => "Output token limit reached",
                    PolicyKind::SchemaRetries => "Schema retry limit reached",
                };
                write!(f, "{label}: limit={limit}")
            }
        }
    }
}

impl std::error::Error for AgentError {}
