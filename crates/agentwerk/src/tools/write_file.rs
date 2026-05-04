//! Lets an agent create or overwrite a file on disk. Pairs with `read_file` and `edit_file` to give a model full file-editing reach.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;
use crate::providers::ProviderResult as Result;

/// Create or overwrite a file. Destructive: existing content is replaced.
/// Not read-only, so the loop runs it serially.
pub struct WriteFileTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("write_file.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for WriteFileTool {
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

            let content = match input["content"].as_str() {
                Some(c) => c,
                None => {
                    return Ok(ToolResult::error("Missing required parameter: content"));
                }
            };

            let resolved = ctx.working_dir.join(path);

            if let Some(parent) = resolved.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return Ok(ToolResult::error(format!(
                        "Failed to create parent directories: {e}"
                    )));
                }
            }

            match std::fs::write(&resolved, content) {
                Ok(()) => Ok(ToolResult::success(format!("File written: {path}"))),
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
    async fn create_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool;
        let ctx = test_ctx(dir.path());

        let result = tool
            .call(
                serde_json::json!({ "path": "new.txt", "content": "hello world" }),
                &ctx,
            )
            .await
            .unwrap();

        let (ToolResult::Success(content) | ToolResult::Error(content) | ToolResult::SchemaError(content)) = &result;
        assert!(content.contains("File written: new.txt"));

        let written = std::fs::read_to_string(dir.path().join("new.txt")).unwrap();
        assert_eq!(written, "hello world");
    }

    #[tokio::test]
    async fn overwrite_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("existing.txt"), "old content").unwrap();

        let tool = WriteFileTool;
        let ctx = test_ctx(dir.path());

        let result = tool
            .call(
                serde_json::json!({ "path": "existing.txt", "content": "new content" }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(matches!(result, ToolResult::Success(_)));
        let written = std::fs::read_to_string(dir.path().join("existing.txt")).unwrap();
        assert_eq!(written, "new content");
    }

    #[tokio::test]
    async fn creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool;
        let ctx = test_ctx(dir.path());

        let result = tool
            .call(
                serde_json::json!({ "path": "a/b/c/deep.txt", "content": "nested" }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(matches!(result, ToolResult::Success(_)));
        let written = std::fs::read_to_string(dir.path().join("a/b/c/deep.txt")).unwrap();
        assert_eq!(written, "nested");
    }
}
