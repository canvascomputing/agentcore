//! Agent-level run errors: cancellation, internal stubs, and lifecycle misuse.

use std::fmt;

/// Failures that originate in the agent runtime, independent of the provider
/// or any tool.
#[derive(Debug)]
pub enum AgentError {
    /// External cancellation fired during the run.
    Cancelled,
    /// A code path intentionally returns this sentinel because the feature
    /// is not yet implemented (e.g. context compaction). Reserved for stubs
    /// that must fail loudly rather than silently.
    NotImplemented(&'static str),
    /// An `OutputFuture` or `AgentHandle` was polled after the run already
    /// completed — API misuse by the caller.
    PolledAfterCompletion,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::Cancelled => write!(f, "Operation cancelled"),
            AgentError::NotImplemented(what) => write!(f, "Not implemented: {what}"),
            AgentError::PolledAfterCompletion => {
                write!(f, "Agent handle polled after completion")
            }
        }
    }
}

impl std::error::Error for AgentError {}
