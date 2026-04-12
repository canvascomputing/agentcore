use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

/// Shell command execution tool.
pub struct BashTool;

impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Executes a bash command in the working directory and returns its output."
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
