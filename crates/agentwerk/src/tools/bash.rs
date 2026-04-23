//! Shell access for agents. Two constructors separate the safe pattern-restricted default from the unrestricted power-tool variant.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{ToolLike, ToolContext, ToolResult};
use crate::tools::util::{glob_match, run_shell_command};

/// Execute shell commands. Two constructors:
/// [`BashTool::new`] restricts execution to commands matching a glob pattern;
/// [`BashTool::unrestricted`] allows any command. Not read-only.
///
/// # Examples
///
/// Pattern-restricted (only `git ...` commands):
///
/// ```
/// use agentwerk::tools::BashTool;
/// use agentwerk::Agent;
///
/// Agent::new().tool(BashTool::new("git", "git *"));
/// ```
///
/// Unrestricted:
///
/// ```
/// use agentwerk::tools::BashTool;
/// use agentwerk::Agent;
///
/// Agent::new().tool(BashTool::unrestricted());
/// ```
pub struct BashTool {
    pattern: String,
    tool_name: String,
    description: String,
    read_only: bool,
}

impl BashTool {
    /// Default per-command timeout when the model omits `timeout_ms`.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_millis(120_000);

    /// Maximum per-command timeout the model is allowed to request.
    pub const MAX_TIMEOUT: Duration = Duration::from_millis(600_000);

    /// Create a new `BashTool` with the given `name` that only permits
    /// commands matching `pattern`.
    pub fn new(name: &str, pattern: &str) -> Self {
        let pattern = pattern.trim().to_string();
        assert!(!pattern.is_empty(), "Pattern must not be empty");

        let description = format!(
            "Executes a bash command matching the pattern `{pattern}`.\n\
             Only commands that match this pattern are allowed. Other commands will be rejected.\n\n\
             The command is executed via `sh -c` in the working directory.\n\
             You may specify an optional timeout in milliseconds (default: {default}, max: {max}).",
            default = Self::DEFAULT_TIMEOUT.as_millis(),
            max = Self::MAX_TIMEOUT.as_millis(),
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

impl ToolLike for BashTool {
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
                    "description": format!("Optional timeout in milliseconds (default: {})", Self::DEFAULT_TIMEOUT.as_millis())
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

            let timeout = input
                .get("timeout_ms")
                .and_then(|v| v.as_u64())
                .map(Duration::from_millis)
                .unwrap_or(Self::DEFAULT_TIMEOUT);

            Ok(run_shell_command(command, timeout, ctx).await)
        })
    }
}

impl BashTool {
    /// Create an unrestricted bash tool with the standard description.
    pub fn unrestricted() -> Self {
        Self::new("bash", "*").with_description(&format!(
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
- You may specify an optional timeout in milliseconds (default: {default}, max: {max}).

When issuing multiple commands:
- If commands are independent, make multiple tool calls in parallel.
- If commands depend on each other, chain with && in a single call.
- Do NOT use newlines to separate commands.

# Anti-patterns
- Do not sleep between commands that can run immediately.
- Do not retry failing commands in a sleep loop — diagnose the root cause.
- Do not use interactive flags (-i) as they require input which is not supported.",
            default = Self::DEFAULT_TIMEOUT.as_millis(),
            max = Self::MAX_TIMEOUT.as_millis(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::test_tool_context;

    #[test]
    fn bash_tool_defaults() {
        let tool = BashTool::unrestricted();
        assert_eq!(tool.name(), "bash");
        assert!(!tool.is_read_only());
    }

    #[test]
    fn glob_tool_sets_name() {
        let tool = BashTool::new("echo", "echo *");
        assert_eq!(tool.name(), "echo");
        assert!(!tool.is_read_only());
    }

    #[test]
    #[should_panic(expected = "Pattern must not be empty")]
    fn glob_tool_empty_pattern_panics() {
        BashTool::new("empty", "");
    }

    #[test]
    fn glob_tool_read_only() {
        let tool = BashTool::new("echo", "echo *").read_only(true);
        assert!(tool.is_read_only());
    }

    #[test]
    fn glob_tool_custom_description() {
        let tool = BashTool::new("git", "git *").with_description("Run git commands.");
        assert_eq!(tool.description(), "Run git commands.");
    }

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool::unrestricted();
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(content.contains("hello"));
        assert!(matches!(result, ToolResult::Success(_)));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool::unrestricted();
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "sleep 10", "timeout_ms": 100 });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(matches!(result, ToolResult::Error(_)));
        assert!(content.contains("timed out"));
    }

    #[tokio::test]
    async fn bash_bad_command() {
        let tool = BashTool::unrestricted();
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "nonexistent_command_xyz" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(matches!(result, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn glob_rejects_non_matching() {
        let tool = BashTool::new("echo", "echo *");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "rm -rf /" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(matches!(result, ToolResult::Error(_)));
        assert!(content.contains("does not match"));
    }

    #[tokio::test]
    async fn glob_accepts_matching() {
        let tool = BashTool::new("echo", "echo *");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(matches!(result, ToolResult::Success(_)));
        assert!(content.contains("hello"));
    }

    #[tokio::test]
    async fn glob_bare_command_rejects_args() {
        let tool = BashTool::new("echo", "echo");
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(matches!(result, ToolResult::Error(_)));
    }
}
