use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};
use crate::tools::util::{glob_match, run_shell_command, DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS};

/// Shell command execution tool restricted to commands matching a glob pattern.
pub struct BashSingleTool {
    pattern: String,
    tool_name: String,
    description: String,
    read_only: bool,
}

impl BashSingleTool {
    /// Create a new `BashSingleTool` with the given `name` that only permits
    /// commands matching `pattern`.
    pub fn new(name: &str, pattern: &str) -> Self {
        let pattern = pattern.trim().to_string();
        assert!(!pattern.is_empty(), "Pattern must not be empty");

        let description = format!(
            "Executes a bash command matching the pattern `{pattern}`.\n\
             Only commands that match this pattern are allowed. Other commands will be rejected.\n\n\
             The command is executed via `sh -c` in the working directory.\n\
             You may specify an optional timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS}, max: {MAX_TIMEOUT_MS})."
        );

        Self {
            pattern,
            tool_name: name.to_string(),
            description,
            read_only: false,
        }
    }

    /// Set whether this tool is considered read-only.
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
}

impl Tool for BashSingleTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": format!("The bash command to execute (must match pattern `{}`)", self.pattern)
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": format!("Optional timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS})")
                }
            },
            "required": ["command"]
        })
    }

    fn is_read_only(&self) -> bool {
        self.read_only
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let command = match input.get("command").and_then(|v| v.as_str()) {
                Some(cmd) => cmd,
                None => return Ok(ToolResult::error("Missing required field: command")),
            };

            if !glob_match(&self.pattern, command) {
                return Ok(ToolResult::error(format!(
                    "Command '{command}' does not match allowed pattern '{}'",
                    self.pattern
                )));
            }

            let timeout_ms = input
                .get("timeout_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_TIMEOUT_MS);

            Ok(run_shell_command(command, &ctx.working_directory, timeout_ms).await)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::test_tool_context;

    #[test]
    fn new_sets_name() {
        let tool = BashSingleTool::new("echo", "echo *");
        assert_eq!(tool.name(), "echo");
        assert!(!tool.is_read_only());
    }

    #[test]
    #[should_panic(expected = "Pattern must not be empty")]
    fn new_with_empty_pattern_panics() {
        BashSingleTool::new("empty", "");
    }

    #[test]
    fn read_only() {
        let tool = BashSingleTool::new("echo", "echo *").read_only(true);
        assert!(tool.is_read_only());
    }

    #[tokio::test]
    async fn rejects_non_matching_command() {
        let tool = BashSingleTool::new("echo", "echo *");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "rm -rf /" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("does not match"));
    }

    #[tokio::test]
    async fn accepts_matching_command() {
        let tool = BashSingleTool::new("echo", "echo *");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[tokio::test]
    async fn bare_command_rejects_args() {
        let tool = BashSingleTool::new("echo", "echo");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
    }
}
