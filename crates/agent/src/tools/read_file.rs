use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

pub struct ReadFileTool;

const DESCRIPTION: &str = "\
Read the contents of a file, optionally returning a specific range of lines.

- Results are returned with line numbers starting at 1.
- When you already know which part of the file you need, use offset and limit to read only that part. This is important for larger files.
- This tool can only read files, not directories. To list a directory, use list_directory or bash with ls.";

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read"
                }
            },
            "required": ["path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let path = match input["path"].as_str() {
                Some(p) => p,
                None => {
                    return Ok(ToolResult::error("Missing required parameter: path"));
                }
            };

            let resolved = ctx.working_directory.join(path);

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

            let start = offset - 1; // convert to 0-based
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
            assert_eq!(
                result.is_error, case.expect_error,
                "case '{}': expected is_error={}, got is_error={}",
                case.name, case.expect_error, result.is_error
            );
            assert!(
                result.content.contains(case.expect_contains),
                "case '{}': expected content to contain {:?}, got {:?}",
                case.name,
                case.expect_contains,
                result.content
            );
        }
    }
}
