//! Deferred-tool discovery. Lets an agent browse and surface tools that were withheld from its initial definitions block to keep the system-prompt context small.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{ToolContext, ToolResult, Toolable};

/// Tool that searches the tool registry by query string.
pub struct ToolSearchTool;

const DESCRIPTION: &str = "\
Search for available tools by name or keyword. Returns tool names, descriptions, and input schemas.";

impl Toolable for ToolSearchTool {
    fn name(&self) -> &str {
        "tool_search"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find tools by name or keyword"
                }
            },
            "required": ["query"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn should_defer(&self) -> bool {
        false
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
    use crate::testutil::MockTool;
    use crate::tools::tool::ToolRegistry;
    use std::sync::Arc;

    fn registry_with_mock_tools() -> Arc<ToolRegistry> {
        let mut registry = ToolRegistry::new();
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
        use crate::agent::Agent;

        let registry = registry_with_mock_tools();
        let agent = Agent::new()
            .name("t")
            .model_name("mock")
            .identity_prompt("")
            .provider(Arc::new(crate::testutil::MockProvider::text("ok")));
        let (_spec, runtime) = agent.compile(None).unwrap();
        let runtime = Arc::new(runtime);
        let ctx = ToolContext::new(
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
        )
        .registry(registry)
        .runtime(runtime.clone());

        let tool = ToolSearchTool;
        tool.call(serde_json::json!({ "query": "read_file" }), &ctx)
            .await
            .unwrap();

        let discovered = runtime.discovered_tools.lock().unwrap();
        assert!(discovered.contains("read_file"));
    }
}
