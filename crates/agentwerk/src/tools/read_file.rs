//! The agent's eyes on the filesystem. Lets a model read a file it did not receive in the prompt.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use crate::providers::ProviderResult;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;

/// Read a file with optional line offset and limit. Returns line-numbered
/// text so the model can reference specific lines in subsequent edits.
/// Read-only.
pub struct ReadFileTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("read_file.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for ReadFileTool {
    fn name(&self) -> &str {
        &tool_file().name
    }

    fn description(&self) -> &str {
        description()
    }

    fn input_schema(&self) -> Value {
        tool_file().input_schema.clone()
    }

    fn is_read_only(&self) -> bool {
        tool_file().read_only
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let path = match input["path"].as_str() {
                Some(p) => p,
                None => {
                    return Ok(ToolResult::error("Missing required parameter: path"));
                }
            };

            let resolved = ctx.working_dir.join(path);

            let content = match std::fs::read_to_string(&resolved) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(ToolResult::error(format!("Failed to read file: {e}")));
                }
            };

            let lines: Vec<&str> = content.lines().collect();

            let offset = input["offset"].as_u64().unwrap_or(1).max(1) as usize;
            let limit = input["limit"]
                .as_u64()
                .map(|l| l as usize)
                .unwrap_or(lines.len().saturating_sub(offset - 1));

            let start = offset - 1;
            let end = (start + limit).min(lines.len());

            let mut result = String::new();
            for (i, line) in lines[start..end].iter().enumerate() {
                let line_num = start + i + 1;
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&format!("{line_num}\t{line}"));
            }

            Ok(ToolResult::success(result))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_ctx(dir: &std::path::Path) -> ToolContext {
        ToolContext::new(PathBuf::from(dir))
    }

    #[tokio::test]
    async fn read_file_cases() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "alpha\nbeta\ngamma\ndelta\n").unwrap();

        struct Case {
            name: &'static str,
            input: Value,
            expect_error: bool,
            expect_contains: &'static str,
        }

        let cases = vec![
            Case {
                name: "full file",
                input: serde_json::json!({ "path": "test.txt" }),
                expect_error: false,
                expect_contains: "1\talpha\n2\tbeta\n3\tgamma\n4\tdelta",
            },
            Case {
                name: "offset and limit",
                input: serde_json::json!({ "path": "test.txt", "offset": 2, "limit": 2 }),
                expect_error: false,
                expect_contains: "2\tbeta\n3\tgamma",
            },
            Case {
                name: "nonexistent file",
                input: serde_json::json!({ "path": "no_such_file.txt" }),
                expect_error: true,
                expect_contains: "Failed to read file",
            },
        ];

        let tool = ReadFileTool;
        let ctx = test_ctx(dir.path());

        for case in cases {
            let result = tool.call(case.input, &ctx).await.unwrap();
            let is_error = matches!(result, ToolResult::Error(_));
            let (ToolResult::Success(content)
            | ToolResult::Error(content)
            | ToolResult::SchemaError(content)) = &result;
            assert_eq!(
                is_error, case.expect_error,
                "case '{}': expected is_error={}, got is_error={}",
                case.name, case.expect_error, is_error
            );
            assert!(
                content.contains(case.expect_contains),
                "case '{}': expected content to contain {:?}, got {:?}",
                case.name,
                case.expect_contains,
                content
            );
        }
    }
}
