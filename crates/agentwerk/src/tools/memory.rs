//! `MemoryTool`: the model's interface to a `Memory`. The store and the
//! curator live in `agents::memory`; this file only wraps the store with a
//! `ToolLike` impl driven by the declarative `memory.tool.json`.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use serde_json::Value;

use crate::agents::memory::Memory;
use crate::providers::ProviderResult;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;

/// The model's three-action handle on a `Memory`: `add`, `replace`, `remove`.
/// Substring match for `replace` and `remove`. Registered automatically when
/// `Agent::memory(&store)` is called.
pub struct MemoryTool {
    store: Arc<Memory>,
}

impl MemoryTool {
    pub fn new(store: Arc<Memory>) -> Self {
        Self { store }
    }
}

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("memory.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for MemoryTool {
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
            let content = input.get("content").and_then(Value::as_str);
            let old_text = input.get("old_text").and_then(Value::as_str);

            let outcome = match action {
                "add" => match content {
                    Some(c) => self.store.add(c).map(|_| "entry added"),
                    None => return Ok(ToolResult::error("Missing required parameter: content")),
                },
                "replace" => match (old_text, content) {
                    (Some(o), Some(c)) => self.store.replace(o, c).map(|_| "entry replaced"),
                    _ => {
                        return Ok(ToolResult::error(
                            "Missing required parameter: replace needs both `old_text` and `content`",
                        ));
                    }
                },
                "remove" => match old_text {
                    Some(o) => self.store.remove(o).map(|_| "entry removed"),
                    None => return Ok(ToolResult::error("Missing required parameter: old_text")),
                },
                "" => return Ok(ToolResult::error("Missing required parameter: action")),
                other => return Ok(ToolResult::error(format!("Unknown action: {other}"))),
            };

            match outcome {
                Ok(msg) => Ok(ToolResult::success(msg)),
                Err(why) => Ok(ToolResult::error(why)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::memory::Memory;

    fn fresh_store() -> (Arc<Memory>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Memory::open(dir.path()).unwrap();
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
    async fn add_action_appends_entry_to_the_store() {
        let (store, _dir) = fresh_store();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "add", "content": "first"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_success(&r, "added");
        assert_eq!(store.entries().join("\n§\n"), "first");
    }

    #[tokio::test]
    async fn replace_action_swaps_unique_entry_in_place() {
        let (store, _dir) = fresh_store();
        store.add("first").unwrap();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "replace", "old_text": "first", "content": "FIRST"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_success(&r, "replaced");
        assert_eq!(store.entries().join("\n§\n"), "FIRST");
    }

    #[tokio::test]
    async fn remove_action_drops_unique_entry() {
        let (store, _dir) = fresh_store();
        store.add("one").unwrap();
        store.add("two").unwrap();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "remove", "old_text": "one"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_success(&r, "removed");
        assert_eq!(store.entries().join("\n§\n"), "two");
    }

    #[tokio::test]
    async fn add_without_content_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(serde_json::json!({"action": "add"}), &ctx())
            .await
            .unwrap();
        assert_error(&r, "content");
        assert_eq!(store.entries().join("\n§\n"), "");
    }

    #[tokio::test]
    async fn remove_without_old_text_is_rejected() {
        let (store, _dir) = fresh_store();
        store.add("seed").unwrap();
        let before = store.entries().join("\n§\n");
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(serde_json::json!({"action": "remove"}), &ctx())
            .await
            .unwrap();
        assert_error(&r, "old_text");
        assert_eq!(store.entries().join("\n§\n"), before);
    }

    #[tokio::test]
    async fn replace_without_old_text_or_content_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "replace", "content": "x"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert_error(&r, "old_text");
    }

    #[tokio::test]
    async fn unknown_action_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(serde_json::json!({"action": "wat"}), &ctx())
            .await
            .unwrap();
        assert_error(&r, "Unknown action");
        assert_eq!(store.entries().join("\n§\n"), "");
    }

    #[tokio::test]
    async fn missing_action_is_rejected() {
        let (store, _dir) = fresh_store();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool.call(serde_json::json!({}), &ctx()).await.unwrap();
        assert_error(&r, "action");
    }

    #[tokio::test]
    async fn non_string_content_is_treated_as_missing_content() {
        let (store, _dir) = fresh_store();
        let tool = MemoryTool::new(Arc::clone(&store));
        let r = tool
            .call(
                serde_json::json!({"action": "add", "content": 42}),
                &ctx(),
            )
            .await
            .unwrap();
        // The dispatch reads `content` as a `&str`; a non-string value is
        // surfaced to the model as a missing-parameter error so it can retry.
        assert_error(&r, "content");
        assert_eq!(store.entries().join("\n§\n"), "");
    }
}
