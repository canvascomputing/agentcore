//! Tool-system errors raised via `Err(...)` — distinct from the in-band `ToolResult::Error(String)` that most tool failures use to signal recoverable problems back to the model.

use std::fmt;

/// A tool failed for reasons the model can't fix by retrying with different
/// arguments — harness wiring missing, persistence broken, etc. Most tool
/// failures (bad args, non-zero bash exit, file-not-found, timeouts, unknown
/// tool names) flow through [`ToolResult::Error`](super::ToolResult::Error)
/// instead; those reach the model as tool-result messages.
#[derive(Debug)]
pub struct ToolError {
    /// Name of the tool that raised the error.
    pub tool_name: String,
    /// Human-readable description of what went wrong.
    pub message: String,
}

impl ToolError {
    pub fn new(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tool {} failed: {}", self.tool_name, self.message)
    }
}

impl std::error::Error for ToolError {}
