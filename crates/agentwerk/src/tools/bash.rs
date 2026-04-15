use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};
use crate::tools::util::{glob_match, run_shell_command, DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS};

/// Shell command execution tool restricted to commands matching a glob pattern.
pub struct BashGlobTool {
    pattern: String,
    tool_name: String,
    description: String,
    read_only: bool,
}

impl BashGlobTool {
    /// Create a new `BashGlobTool` with the given `name` that only permits
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

    /// Override the auto-generated description.
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Set whether this tool is considered read-only.
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
}

impl Tool for BashGlobTool {
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

/// Create an unrestricted bash tool with the standard description.
#[allow(non_snake_case)]
pub fn BashTool() -> BashGlobTool {
    BashGlobTool::new("bash", "*").with_description(&format!(
        "\
Executes a bash command in the working directory and returns its output.

IMPORTANT: Avoid using this tool when a dedicated tool exists:
- File search: Use glob (NOT find or ls)
- Content search: Use grep (NOT grep or rg via bash)
- Read files: Use read_file (NOT cat/head/tail)
- Edit files: Use edit_file (NOT sed/awk)
- Write files: Use write_file (NOT echo/heredoc)

# Instructions
- Always quote file paths that contain spaces with double quotes.
- Try to maintain your current working directory by using absolute paths.
- You may specify an optional timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS}, max: {MAX_TIMEOUT_MS}).

When issuing multiple commands:
- If commands are independent, make multiple tool calls in parallel.
- If commands depend on each other, chain with && in a single call.
- Do NOT use newlines to separate commands.

# Anti-patterns
- Do not sleep between commands that can run immediately.
- Do not retry failing commands in a sleep loop — diagnose the root cause.
- Do not use interactive flags (-i) as they require input which is not supported."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::test_tool_context;

    #[test]
    fn bash_tool_defaults() {
        let tool = BashTool();
        assert_eq!(tool.name(), "bash");
        assert!(!tool.is_read_only());
    }

    #[test]
    fn glob_tool_sets_name() {
        let tool = BashGlobTool::new("echo", "echo *");
        assert_eq!(tool.name(), "echo");
        assert!(!tool.is_read_only());
    }

    #[test]
    #[should_panic(expected = "Pattern must not be empty")]
    fn glob_tool_empty_pattern_panics() {
        BashGlobTool::new("empty", "");
    }

    #[test]
    fn glob_tool_read_only() {
        let tool = BashGlobTool::new("echo", "echo *").read_only(true);
        assert!(tool.is_read_only());
    }

    #[test]
    fn glob_tool_custom_description() {
        let tool = BashGlobTool::new("git", "git *").with_description("Run git commands.");
        assert_eq!(tool.description(), "Run git commands.");
    }

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool();
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.content.contains("hello"));
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool();
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "sleep 10", "timeout_ms": 100 });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }

    #[tokio::test]
    async fn bash_bad_command() {
        let tool = BashTool();
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "nonexistent_command_xyz" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn glob_rejects_non_matching() {
        let tool = BashGlobTool::new("echo", "echo *");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "rm -rf /" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("does not match"));
    }

    #[tokio::test]
    async fn glob_accepts_matching() {
        let tool = BashGlobTool::new("echo", "echo *");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[tokio::test]
    async fn glob_bare_command_rejects_args() {
        let tool = BashGlobTool::new("echo", "echo");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
    }
}
