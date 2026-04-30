//! Tool-system errors raised via `Err(...)` — distinct from the in-band `ToolResult::Error(String)` that most tool failures use to signal recoverable problems back to the model.

use std::fmt;

/// A tool failed for reasons the model can't fix by retrying with different
/// arguments. Most tool failures (bad args, non-zero bash exit, file-not-found,
/// timeouts) flow through [`ToolResult::Error`](super::ToolResult::Error)
/// instead; those reach the model as tool-result messages.
#[derive(Debug)]
pub enum ToolError {
    /// Registry has no tool with this name. The loop forwards the failure to
    /// the model as a tool-result error so it can pick a different tool.
    ToolNotFound { tool_name: String },
    /// The tool was invoked but its execution raised an error. Covers harness
    /// wiring gaps, persistence/IO failures, and anything else a tool returns
    /// via `Err(...)`.
    ExecutionFailed { tool_name: String, message: String },
    /// A schema-checked tool rejected the model's payload. Distinct from
    /// `ExecutionFailed` so the loop can apply the dedicated retry budget
    /// (`policies.max_schema_retries`) and emit
    /// `ToolFailureKind::SchemaValidationFailed`.
    SchemaValidationFailed { tool_name: String, message: String },
}

impl ToolError {
    /// Name of the tool that raised the error.
    pub fn tool_name(&self) -> &str {
        match self {
            ToolError::ToolNotFound { tool_name } => tool_name,
            ToolError::ExecutionFailed { tool_name, .. } => tool_name,
            ToolError::SchemaValidationFailed { tool_name, .. } => tool_name,
        }
    }

    /// Human-readable description of what went wrong, suitable for the
    /// model-visible tool-result block.
    pub fn message(&self) -> String {
        match self {
            ToolError::ToolNotFound { tool_name } => format!("Unknown tool: {tool_name}"),
            ToolError::ExecutionFailed { message, .. } => message.clone(),
            ToolError::SchemaValidationFailed { message, .. } => message.clone(),
        }
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::ToolNotFound { tool_name } => write!(f, "Tool {tool_name} not found"),
            ToolError::ExecutionFailed { tool_name, message } => {
                write!(f, "Tool {tool_name} failed: {message}")
            }
            ToolError::SchemaValidationFailed { tool_name, message } => {
                write!(f, "Tool {tool_name} schema validation failed: {message}")
            }
        }
    }
}

impl std::error::Error for ToolError {}
