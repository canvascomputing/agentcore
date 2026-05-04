//! In-place find-and-replace on a file, so a model can modify existing code without restating the whole file.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;
use crate::providers::ProviderResult as Result;

/// In-place string replacement in an existing file. The model supplies the
/// old and new strings; the tool fails if the old string is absent or
/// matches more than once. Not read-only.
pub struct EditFileTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("edit_file.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for EditFileTool {
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

            let resolved = ctx.working_dir.join(path);

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
                Ok(()) => Ok(ToolResult::success(format!(
                    "Edited {path}: replaced {count} occurrence(s)"
                ))),
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

        let (ToolResult::Success(out) | ToolResult::Error(out) | ToolResult::SchemaError(out)) = &result;
        assert!(
            matches!(result, ToolResult::Success(_)),
            "unexpected error: {out}"
        );
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

        let (ToolResult::Success(content) | ToolResult::Error(content) | ToolResult::SchemaError(content)) = &result;
        assert!(matches!(result, ToolResult::Error(_)));
        assert!(content.contains("2"));
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

        let (ToolResult::Success(out) | ToolResult::Error(out) | ToolResult::SchemaError(out)) = &result;
        assert!(
            matches!(result, ToolResult::Success(_)),
            "unexpected error: {out}"
        );
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

        let (ToolResult::Success(content) | ToolResult::Error(content) | ToolResult::SchemaError(content)) = &result;
        assert!(matches!(result, ToolResult::Error(_)));
        assert!(content.contains("not found"));
    }
}
