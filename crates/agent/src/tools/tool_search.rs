use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tool::{Tool, ToolContext, ToolResult};

/// Tool that searches the tool registry by query string.
pub struct ToolSearchTool;

impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "tool_search"
    }

    fn description(&self) -> &str {
        "Searches for available tools by name or keyword."
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
                    return Ok(ToolResult {
                        content: "Missing required field: query".to_string(),
                        is_error: true,
                    });
                }
            };

            let registry = match ctx.tool_registry.as_ref() {
                Some(r) => r,
                None => {
                    return Ok(ToolResult {
                        content: "No tool registry available".to_string(),
                        is_error: true,
                    });
                }
            };

            let results = registry.search(query);

            if results.is_empty() {
                return Ok(ToolResult {
                    content: format!("No tools found matching '{query}'."),
                    is_error: false,
                });
            }

            let count = results.len();
            let mut content = format!("Found {count} tool(s) matching '{query}':\n\n");

            for result in &results {
                let def = &result.definition;
                let pretty_schema =
                    serde_json::to_string_pretty(&def.input_schema).unwrap_or_default();

                content.push_str(&format!("## {}\n\n", def.name));
                content.push_str(&format!("{}\n\n", def.description));
                content.push_str("Input schema:\n```json\n");
                content.push_str(&pretty_schema);
                content.push_str("\n```\n\n---\n\n");
            }

            Ok(ToolResult {
                content,
                is_error: false,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::MockTool;
    use crate::tool::ToolRegistry;
    use std::sync::Arc;

    fn registry_with_mock_tools() -> Arc<ToolRegistry> {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("read_file", true, "file contents"));
        registry.register(MockTool::new("write_file", false, "written"));
        registry.register(MockTool::new("list_directory", true, "listing"));
        Arc::new(registry)
    }

    fn ctx_with_registry(registry: Arc<ToolRegistry>) -> ToolContext {
        ToolContext::new(std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")))
            .with_registry(registry)
    }

    #[tokio::test]
    async fn find_by_name() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_with_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "read_file" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("read_file"));
    }

    #[tokio::test]
    async fn find_by_keyword() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_with_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "file" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("read_file"));
        assert!(result.content.contains("write_file"));
    }

    #[tokio::test]
    async fn no_results() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_with_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "nonexistent_xyz" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No tools found"));
    }

    #[tokio::test]
    async fn returns_schema() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_with_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "read_file" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.content.contains("```json"));
        assert!(result.content.contains("\"type\""));
    }

    #[tokio::test]
    async fn missing_query_errors() {
        let registry = registry_with_mock_tools();
        let ctx = ctx_with_registry(registry);
        let tool = ToolSearchTool;
        let input = serde_json::json!({});
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Missing required field: query"));
    }

    #[tokio::test]
    async fn no_registry_errors() {
        let ctx =
            ToolContext::new(std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));
        let tool = ToolSearchTool;
        let input = serde_json::json!({ "query": "anything" });
        let result = tool.call(input, &ctx).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No tool registry available"));
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
}
