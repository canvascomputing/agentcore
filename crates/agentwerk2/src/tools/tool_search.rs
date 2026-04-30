//! Deferred-tool discovery. Lets an agent browse and surface tools that were withheld from its initial definitions block to keep the system-prompt context small.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;
use crate::providers::ProviderResult as Result;

/// Search the tool registry by query string. Pair with tools that set
/// [`Tool::should_defer`](crate::Tool::should_defer) to `true`: the
/// model sees only their names until it discovers them through this tool,
/// keeping the initial system prompt small.
pub struct ToolSearchTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("tool_search.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for ToolSearchTool {
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
            let query = match input.get("query").and_then(|v| v.as_str()) {
                Some(q) => q,
                None => {
                    return Ok(ToolResult::error("Missing required field: query"));
                }
            };

            let registry = match ctx.tool_registry.as_ref() {
                Some(r) => r,
                None => {
                    return Ok(ToolResult::error("No tool registry available"));
                }
            };

            let results = registry.search(query);

            if results.is_empty() {
                return Ok(ToolResult::success(format!(
                    "No tools found matching '{query}'."
                )));
            }

            let count = results.len();
            let mut content = format!("Found {count} tool(s) matching '{query}':\n\n");

            for def in &results {
                let pretty_schema =
                    serde_json::to_string_pretty(&def.input_schema).unwrap_or_default();

                content.push_str(&format!("## {}\n\n", def.name));
                content.push_str(&format!("{}\n\n", def.description));
                content.push_str("Input schema:\n```json\n");
                content.push_str(&pretty_schema);
                content.push_str("\n```\n\n---\n\n");
            }

            for def in &results {
                ctx.mark_tool_discovered(&def.name);
            }
            Ok(ToolResult::success(content))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;

    use crate::providers::ProviderResult;
    use crate::tools::ToolRegistry;

    /// Minimal mock tool inlined for tests (agentwerk2 has no shared
    /// testutil module).
    struct MockTool {
        name: String,
        read_only: bool,
        result: String,
    }

    impl MockTool {
        fn new(name: &str, read_only: bool, result: &str) -> Self {
            Self {
                name: name.into(),
                read_only,
                result: result.into(),
            }
        }
    }

    impl ToolLike for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "mock"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        fn is_read_only(&self) -> bool {
            self.read_only
        }
        fn call<'a>(
            &'a self,
            _input: serde_json::Value,
            _ctx: &'a ToolContext,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
            let result = self.result.clone();
            Box::pin(async move { Ok(ToolResult::success(result)) })
        }
    }

    fn registry_with_mock_tools() -> Arc<ToolRegistry> {
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("read_file", true, "file contents"));
        registry.register(MockTool::new("write_file", false, "written"));
        registry.register(MockTool::new("list_directory", true, "listing"));
        Arc::new(registry)
    }

    fn ctx_registry(registry: Arc<ToolRegistry>) -> ToolContext {
        ToolContext::new(std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")))
            .registry(registry)
    }

    #[tokio::test]
    async fn find_by_name() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "read_file" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(content.contains("read_file"));
    }

    #[tokio::test]
    async fn find_by_keyword() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "file" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(content.contains("read_file"));
        assert!(content.contains("write_file"));
    }

    #[tokio::test]
    async fn no_results() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "nonexistent_xyz" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(content.contains("No tools found"));
    }

    #[tokio::test]
    async fn returns_schema() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "read_file" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(content.contains("```json"));
        assert!(content.contains("\"type\""));
    }

    #[tokio::test]
    async fn missing_query_errors() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({});
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(matches!(result, ToolResult::Error(_)));
        assert!(content.contains("Missing required field: query"));
    }

    #[tokio::test]
    async fn no_registry_errors() {
        let ctx = ToolContext::new(
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
        );
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "anything" });
        let result = tool.call(input, &ctx).await.unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(matches!(result, ToolResult::Error(_)));
        assert!(content.contains("No tool registry available"));
    }

    #[tokio::test]
    async fn is_never_deferred() {
        let tool = ToolSearchTool;
        assert!(!tool.should_defer());
    }

    #[tokio::test]
    async fn is_read_only() {
        let tool = ToolSearchTool;
        assert!(tool.is_read_only());
    }

    #[tokio::test]
    async fn marks_found_tools_as_discovered() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_registry(Arc::clone(&registry));

        let tool = ToolSearchTool;
        tool.call(serde_json::json!({ "query": "read_file" }), &ctx)
            .await
            .unwrap();

        let discovered = registry.discovered.lock().unwrap();
        assert!(discovered.contains("read_file"));
    }
}
