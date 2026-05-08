//! `KnowledgeTool`: the model's interface to a `Knowledge` store.
//! The store lives in `agents::knowledge`; this file only wraps it
//! with a `ToolLike` impl driven by the declarative `knowledge.tool.json`.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use serde_json::Value;

use crate::agents::knowledge::Knowledge;
use crate::providers::ProviderResult;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;

/// The model's four-action handle on a `Knowledge` store:
/// `write`, `read`, `remove`, `list`.
pub struct KnowledgeTool {
    store: Arc<Knowledge>,
}

impl KnowledgeTool {
    pub fn new(store: Arc<Knowledge>) -> Self {
        Self { store }
    }
}

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("knowledge.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for KnowledgeTool {
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
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let action = input.get("action").and_then(Value::as_str).unwrap_or("");

            match action {
                "write" => {
                    let slug = match input.get("slug").and_then(Value::as_str) {
                        Some(s) => s,
                        None => return Ok(ToolResult::error("Missing required parameter: slug")),
                    };
                    let summary = match input.get("summary").and_then(Value::as_str) {
                        Some(s) => s,
                        None => {
                            return Ok(ToolResult::error("Missing required parameter: summary"))
                        }
                    };
                    let content = match input.get("content").and_then(Value::as_str) {
                        Some(s) => s,
                        None => {
                            return Ok(ToolResult::error("Missing required parameter: content"))
                        }
                    };
                    let tags: Vec<String> = input
                        .get("tags")
                        .and_then(Value::as_array)
                        .map(|arr| {
                            arr.iter()
                                .filter_map(Value::as_str)
                                .map(String::from)
                                .collect()
                        })
                        .unwrap_or_default();

                    match self.store.write_page(slug, summary, content, &tags) {
                        Ok(out) => {
                            let pct = if out.index_char_limit > 0 {
                                (out.index_chars_used * 100) / out.index_char_limit
                            } else {
                                0
                            };
                            Ok(ToolResult::success(format!(
                                "{} ({} pages, {}% — {}/{} chars)",
                                out.message,
                                out.pages,
                                pct,
                                out.index_chars_used,
                                out.index_char_limit,
                            )))
                        }
                        Err(why) => Ok(ToolResult::error(why)),
                    }
                }

                "read" => {
                    let slug = match input.get("slug").and_then(Value::as_str) {
                        Some(s) => s,
                        None => return Ok(ToolResult::error("Missing required parameter: slug")),
                    };
                    match self.store.read_page(slug) {
                        Ok(body) => Ok(ToolResult::success(body)),
                        Err(_) => Ok(ToolResult::success(format!(
                            "No page found for `{slug}`. Check the knowledge index before reading — only slugs listed there exist."
                        ))),
                    }
                }

                "remove" => {
                    let slug = match input.get("slug").and_then(Value::as_str) {
                        Some(s) => s,
                        None => return Ok(ToolResult::error("Missing required parameter: slug")),
                    };
                    match self.store.remove_page(slug) {
                        Ok(out) => {
                            let pct = if out.index_char_limit > 0 {
                                (out.index_chars_used * 100) / out.index_char_limit
                            } else {
                                0
                            };
                            Ok(ToolResult::success(format!(
                                "{} ({} pages, {}% — {}/{} chars)",
                                out.message,
                                out.pages,
                                pct,
                                out.index_chars_used,
                                out.index_char_limit,
                            )))
                        }
                        Err(why) => Ok(ToolResult::error(why)),
                    }
                }

                "list" => {
                    let idx = self.store.index();
                    let body = if idx.is_empty() {
                        "(no pages)".to_string()
                    } else {
                        idx
                    };
                    Ok(ToolResult::success(body))
                }

                "" => Ok(ToolResult::error("Missing required parameter: action")),
                other => Ok(ToolResult::error(format!("Unknown action: {other}"))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::knowledge::Knowledge;

    fn fresh_store() -> (Arc<Knowledge>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Knowledge::open(dir.path()).unwrap();
        (store, dir)
    }

    fn ctx() -> ToolContext {
        ToolContext::new(std::env::current_dir().unwrap())
    }

    fn assert_success(result: &ToolResult, fragment: &str) {
        match result {
            ToolResult::Success(s) => {
                assert!(s.contains(fragment), "expected `{fragment}` in `{s}`")
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    fn assert_error(result: &ToolResult, fragment: &str) {
        match result {
            ToolResult::Error(s) => assert!(s.contains(fragment), "expected `{fragment}` in `{s}`"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn write_action_creates_page() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({
                    "action": "write",
                    "slug": "test",
                    "summary": "A test page",
                    "content": "# Test\n\nContent."
                }),
                &ctx(),
            )
            .await
            .unwrap();
        assert_success(&r, "page written");
        assert!(store.index().contains("test"));
    }

    #[tokio::test]
    async fn read_action_returns_page_body() {
        let (store, _dir) = fresh_store();
        store
            .write_page("test", "A test", "# Test\n\nHello.", &[])
            .unwrap();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "read", "slug": "test"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_success(&r, "Hello.");
    }

    #[tokio::test]
    async fn read_action_missing_page_returns_soft_success() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "read", "slug": "nonexistent"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_success(&r, "No page found");
    }

    #[tokio::test]
    async fn read_action_strips_frontmatter() {
        let (store, _dir) = fresh_store();
        store
            .write_page("test", "A test", "# Test\n\nHello.", &["tag".into()])
            .unwrap();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "read", "slug": "test"}),
                &ctx(),
            )
            .await
            .unwrap();
        match &r {
            ToolResult::Success(s) => {
                assert!(!s.contains("---"));
                assert!(!s.contains("updated:"));
            }
            _ => panic!("expected Success"),
        }
    }

    #[tokio::test]
    async fn remove_action_deletes_page() {
        let (store, _dir) = fresh_store();
        store
            .write_page("temp", "Temporary", "# Temp", &[])
            .unwrap();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "remove", "slug": "temp"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_success(&r, "page removed");
        assert!(store.index().is_empty());
    }

    #[tokio::test]
    async fn list_action_returns_index() {
        let (store, _dir) = fresh_store();
        store
            .write_page("config", "Config page", "# Config", &[])
            .unwrap();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(serde_json::json!({"action": "list"}), &ctx())
            .await
            .unwrap();
        assert_success(&r, "config");
    }

    #[tokio::test]
    async fn list_action_empty_store() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(serde_json::json!({"action": "list"}), &ctx())
            .await
            .unwrap();
        assert_success(&r, "(no pages)");
    }

    #[tokio::test]
    async fn write_without_slug_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "write", "summary": "s", "content": "c"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_error(&r, "slug");
    }

    #[tokio::test]
    async fn write_without_summary_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "write", "slug": "test", "content": "c"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_error(&r, "summary");
    }

    #[tokio::test]
    async fn write_without_content_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "write", "slug": "test", "summary": "s"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_error(&r, "content");
    }

    #[tokio::test]
    async fn unknown_action_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool
            .call(serde_json::json!({"action": "wat"}), &ctx())
            .await
            .unwrap();
        assert_error(&r, "Unknown action");
    }

    #[tokio::test]
    async fn missing_action_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = KnowledgeTool::new(Arc::clone(&store));
        let r = tool.call(serde_json::json!({}), &ctx()).await.unwrap();
        assert_error(&r, "action");
    }
}
