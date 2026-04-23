//! Tool-system errors raised via `Err(...)` — distinct from the in-band `ToolResult::Error(String)` that most tool failures use to signal recoverable problems back to the model.

use std::fmt;

/// Failures a tool reports as `Err` rather than as an in-band `ToolResult::Error`.
/// Reserved for situations the model cannot recover from by correcting its
/// arguments. Most tool failures (bad args, non-zero bash exit, file-not-found,
/// timeouts) flow through `ToolResult::Error(String)` instead.
#[derive(Debug)]
pub enum ToolError {
    /// A tool was invoked but its `ToolContext` was missing infrastructure
    /// the tool needs: `LoopRuntime`, caller spec, or command queue. A
    /// harness-configuration bug, not a model error.
    ContextUnavailable { tool_name: String, message: String },

    /// Tool arguments could not be deserialized into the tool's input type.
    /// Raised when the tool cannot produce a useful `ToolResult::Error`
    /// because the argument shape itself is invalid.
    ArgumentsRejected { tool_name: String, message: String },
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::ContextUnavailable { tool_name, message } => {
                write!(f, "Tool context unavailable ({tool_name}): {message}")
            }
            ToolError::ArgumentsRejected { tool_name, message } => {
                write!(f, "Tool arguments rejected ({tool_name}): {message}")
            }
        }
    }
}

impl std::error::Error for ToolError {}
