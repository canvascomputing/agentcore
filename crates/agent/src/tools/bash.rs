use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

/// Shell command execution tool.
pub struct BashTool;

const DESCRIPTION: &str = "\
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
- You may specify an optional timeout in milliseconds (default: 120000, max: 600000).

When issuing multiple commands:
- If commands are independent, make multiple tool calls in parallel.
- If commands depend on each other, chain with && in a single call.
- Do NOT use newlines to separate commands.

# Anti-patterns
- Do not sleep between commands that can run immediately.
- Do not retry failing commands in a sleep loop — diagnose the root cause.
- Do not use interactive flags (-i) as they require input which is not supported.";

impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds (default: 120000)"
                }
            },
            "required": ["command"]
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let command = match input.get("command").and_then(|v| v.as_str()) {
                Some(cmd) => cmd,
                None => {
                    return Ok(ToolResult::error("Missing required field: command"));
                }
            };

            let timeout_ms = input
                .get("timeout_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(120_000);

            let timeout_duration = Duration::from_millis(timeout_ms);

            let result = tokio::time::timeout(
                timeout_duration,
                tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .current_dir(&ctx.working_directory)
                    .output(),
            )
            .await;

            match result {
                Err(_) => Ok(ToolResult::error(format!("Command timed out after {timeout_ms}ms"))),
                Ok(Err(e)) => Ok(ToolResult::error(format!("Failed to execute command: {e}"))),
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                    let mut content = stdout;
                    if !stderr.is_empty() {
                        content.push_str("\n--- stderr ---\n");
                        content.push_str(&stderr);
                    }

                    let is_error = !output.status.success();

                    Ok(ToolResult { content, is_error })
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::test_tool_context;

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool;
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "echo hello" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.content.contains("hello"));
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool;
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "sleep 10", "timeout_ms": 100 });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }

    #[tokio::test]
    async fn bash_bad_command() {
        let tool = BashTool;
        let ctx = test_tool_context();
        let input = serde_json::json!({ "command": "nonexistent_command_xyz" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
    }
}
