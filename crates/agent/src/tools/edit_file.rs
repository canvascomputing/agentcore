use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

pub struct EditFileTool;

const DESCRIPTION: &str = "\
Edit a file by replacing occurrences of a string.

- You must use read_file at least once before editing a file. Understand the contents before modifying.
- The edit will FAIL if old_string is not unique in the file. Provide more surrounding context to make it unique, or use replace_all to change every occurrence.
- When editing, preserve the exact indentation (tabs/spaces) as it appears in the file.
- ALWAYS prefer editing existing files over creating new ones.
- Use replace_all for renaming or replacing a string across the entire file.";

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
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
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
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

            let old_string = match input["old_string"].as_str() {
                Some(s) => s,
                None => {
                    return Ok(ToolResult::error("Missing required parameter: old_string"));
                }
            };

            let new_string = match input["new_string"].as_str() {
                Some(s) => s,
                None => {
                    return Ok(ToolResult::error("Missing required parameter: new_string"));
                }
            };

            let replace_all = input["replace_all"].as_bool().unwrap_or(false);

            let resolved = ctx.working_directory.join(path);

            let content = match std::fs::read_to_string(&resolved) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(ToolResult::error(format!("Failed to read file: {e}")));
                }
            };

            let count = content.matches(old_string).count();

            if count == 0 {
                return Ok(ToolResult::error(format!("old_string not found in {path}")));
            }

            if count > 1 && !replace_all {
                return Ok(ToolResult::error(format!(
                    "Found {count} occurrences of old_string in {path}. Use replace_all to replace all."
                )));
            }

            let new_content = if replace_all {
                content.replace(old_string, new_string)
            } else {
                content.replacen(old_string, new_string, 1)
            };

            match std::fs::write(&resolved, &new_content) {
                Ok(()) => Ok(ToolResult::success(format!("Edited {path}: replaced {count} occurrence(s)"))),
                Err(e) => Ok(ToolResult::error(format!("Failed to write file: {e}"))),
            }
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
    async fn unique_match_replaced() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "hello world").unwrap();

        let tool = EditFileTool;
        let ctx = test_ctx(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "path": "f.txt",
                    "old_string": "world",
                    "new_string": "rust"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!result.is_error, "unexpected error: {}", result.content);
        let content = std::fs::read_to_string(dir.path().join("f.txt")).unwrap();
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn non_unique_errors_without_replace_all() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "aaa bbb aaa").unwrap();

        let tool = EditFileTool;
        let ctx = test_ctx(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "path": "f.txt",
                    "old_string": "aaa",
                    "new_string": "ccc"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("2"));
    }

    #[tokio::test]
    async fn replace_all_replaces_every_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "aaa bbb aaa").unwrap();

        let tool = EditFileTool;
        let ctx = test_ctx(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "path": "f.txt",
                    "old_string": "aaa",
                    "new_string": "ccc",
                    "replace_all": true
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!result.is_error, "unexpected error: {}", result.content);
        let content = std::fs::read_to_string(dir.path().join("f.txt")).unwrap();
        assert_eq!(content, "ccc bbb ccc");
    }

    #[tokio::test]
    async fn not_found_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "hello world").unwrap();

        let tool = EditFileTool;
        let ctx = test_ctx(dir.path());

        let result = tool
            .call(
                serde_json::json!({
                    "path": "f.txt",
                    "old_string": "missing",
                    "new_string": "replacement"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }
}
