use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::RuntimeContext;
use crate::error::Result;
use crate::provider::types::ContentBlock;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Context passed to tool execution.
#[derive(Clone)]
pub struct ToolContext {
    pub working_directory: PathBuf,
    pub(crate) tool_registry: Option<Arc<ToolRegistry>>,
    pub(crate) runtime_context: Option<RuntimeContext>,
}

impl ToolContext {
    pub fn new(working_directory: PathBuf) -> Self {
        Self {
            working_directory,
            tool_registry: None,
            runtime_context: None,
        }
    }

    pub(crate) fn registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    pub(crate) fn runtime_context(mut self, ctx: RuntimeContext) -> Self {
        self.runtime_context = Some(ctx);
        self
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("working_directory", &self.working_directory)
            .field("tool_registry", &self.tool_registry)
            .field("has_runtime_context", &self.runtime_context.is_some())
            .finish()
    }
}

/// Definition sent to the LLM as part of the tools parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// A tool call extracted from an LLM response.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// Result returned by a tool execution.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: false }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: true }
    }
}

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

/// The core tool interface. Object-safe via boxed futures.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;

    fn is_read_only(&self) -> bool {
        false
    }

    fn should_defer(&self) -> bool {
        false
    }

    fn search_hints(&self) -> Vec<String> {
        Vec::new()
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// Registry of tools available to an agent.
pub(crate) struct ToolRegistry {
    pub(crate) tools: Vec<Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.tools.iter().map(|t| t.name()).collect();
        f.debug_struct("ToolRegistry")
            .field("tools", &names)
            .finish()
    }
}

impl ToolRegistry {
    pub(crate) fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub(crate) fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.push(Arc::new(tool));
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    /// Tool definitions for the LLM. Deferred tools that haven't been
    /// discovered yet get name-only stubs; all others get full definitions.
    pub(crate) fn definitions(&self, discovered: &HashSet<String>) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| {
                if t.should_defer() && !discovered.contains(t.name()) {
                    ToolDefinition {
                        name: t.name().to_string(),
                        description: String::new(),
                        input_schema: serde_json::json!({}),
                    }
                } else {
                    t.definition()
                }
            })
            .collect()
    }

    /// Search tools by query string. Returns matches sorted by relevance (highest first).
    pub(crate) fn search(&self, query: &str) -> Vec<ToolDefinition> {
        let query_lower = query.to_lowercase();
        let mut scored: Vec<(ToolDefinition, u32)> = self
            .tools
            .iter()
            .filter_map(|t| {
                let mut score = 0u32;
                let name = t.name().to_lowercase();
                let desc = t.description().to_lowercase();

                // Exact name match
                if name == query_lower {
                    score += 100;
                } else if name.contains(&query_lower) {
                    score += 50;
                }

                // Description match
                if desc.contains(&query_lower) {
                    score += 25;
                }

                // Search hints match
                for hint in t.search_hints() {
                    if hint.to_lowercase().contains(&query_lower) {
                        score += 30;
                    }
                }

                if score > 0 {
                    Some((t.definition(), score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.into_iter().map(|(def, _)| def).collect()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolBuilder
// ---------------------------------------------------------------------------

type ToolHandler = Box<
    dyn Fn(Value, &ToolContext) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
        + Send
        + Sync,
>;

struct BuiltTool {
    name: String,
    description: String,
    schema: Value,
    read_only: bool,
    defer: bool,
    hints: Vec<String>,
    handler: ToolHandler,
}

impl Tool for BuiltTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        self.schema.clone()
    }
    fn is_read_only(&self) -> bool {
        self.read_only
    }
    fn should_defer(&self) -> bool {
        self.defer
    }
    fn search_hints(&self) -> Vec<String> {
        self.hints.clone()
    }
    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        (self.handler)(input, ctx)
    }
}

pub struct ToolBuilder {
    name: String,
    description: String,
    schema: Value,
    read_only: bool,
    defer: bool,
    hints: Vec<String>,
    handler: Option<ToolHandler>,
}

impl ToolBuilder {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({"type": "object", "properties": {}}),
            read_only: false,
            defer: false,
            hints: Vec::new(),
            handler: None,
        }
    }

    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = schema;
        self
    }

    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    pub fn should_defer(mut self, defer: bool) -> Self {
        self.defer = defer;
        self
    }

    pub fn search_hints(mut self, hints: Vec<String>) -> Self {
        self.hints = hints;
        self
    }

    pub fn handler<F>(mut self, f: F) -> Self
    where
        F: Fn(Value, &ToolContext) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
            + Send
            + Sync
            + 'static,
    {
        self.handler = Some(Box::new(f));
        self
    }

    pub fn build(self) -> impl Tool {
        let handler = self
            .handler
            .expect("ToolBuilder requires a handler before build()");
        BuiltTool {
            name: self.name,
            description: self.description,
            schema: self.schema,
            read_only: self.read_only,
            defer: self.defer,
            hints: self.hints,
            handler,
        }
    }
}

// ---------------------------------------------------------------------------
// execute_tool_calls
// ---------------------------------------------------------------------------

enum ToolBatch {
    Concurrent(Vec<ToolCall>),
    Serial(ToolCall),
}

fn partition_tool_calls(calls: &[ToolCall], registry: &ToolRegistry) -> Vec<ToolBatch> {
    let mut batches: Vec<ToolBatch> = Vec::new();
    let mut concurrent_batch: Vec<ToolCall> = Vec::new();

    for call in calls {
        let is_read_only = registry
            .get(&call.name)
            .map_or(false, |t| t.is_read_only());

        if is_read_only {
            concurrent_batch.push(call.clone());
        } else {
            if !concurrent_batch.is_empty() {
                batches.push(ToolBatch::Concurrent(std::mem::take(&mut concurrent_batch)));
            }
            batches.push(ToolBatch::Serial(call.clone()));
        }
    }

    if !concurrent_batch.is_empty() {
        batches.push(ToolBatch::Concurrent(concurrent_batch));
    }

    batches
}

/// Execute tool calls with concurrent read-only batching and serial write execution.
pub(crate) async fn execute_tool_calls(
    calls: &[ToolCall],
    ctx: &ToolContext,
) -> Vec<ContentBlock> {
    let registry = match ctx.tool_registry.as_ref() {
        Some(r) => r,
        None => {
            return calls
                .iter()
                .map(|call| ContentBlock::ToolResult {
                    tool_use_id: call.id.clone(),
                    content: "No tool registry available".into(),
                    is_error: true,
                })
                .collect();
        }
    };

    let batches = partition_tool_calls(calls, registry);
    let mut results: Vec<ContentBlock> = Vec::new();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(10));

    for batch in batches {
        match batch {
            ToolBatch::Concurrent(calls) => {
                let mut set = tokio::task::JoinSet::new();
                for call in calls {
                    let sem = semaphore.clone();
                    let ctx = ctx.clone();
                    let tool_arc = registry.get(&call.name);
                    let call_id = call.id.clone();
                    let call_name = call.name.clone();
                    let input = call.input.clone();

                    set.spawn(async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = match tool_arc {
                            Some(t) => match t.call(input, &ctx).await {
                                Ok(r) => r,
                                Err(e) => ToolResult::error(format!("Tool error: {e}")),
                            },
                            None => ToolResult::error(format!("Unknown tool: {call_name}")),
                        };
                        (call_id, result)
                    });
                }

                while let Some(join_result) = set.join_next().await {
                    if let Ok((id, result)) = join_result {
                        results.push(ContentBlock::ToolResult {
                            tool_use_id: id,
                            content: result.content,
                            is_error: result.is_error,
                        });
                    }
                }
            }
            ToolBatch::Serial(call) => {
                let result = match registry.get(&call.name) {
                    Some(tool) => match tool.call(call.input.clone(), ctx).await {
                        Ok(r) => r,
                        Err(e) => ToolResult::error(format!("Tool error: {e}")),
                    },
                    None => ToolResult::error(format!("Unknown tool: {}", call.name)),
                };
                results.push(ContentBlock::ToolResult {
                    tool_use_id: call.id.clone(),
                    content: result.content,
                    is_error: result.is_error,
                });
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::*;

    #[test]
    fn registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        let tool = MockTool::new("read_file", true, "file contents");
        registry.register(tool);

        assert!(registry.get("read_file").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn registry_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("read", true, "ok"));
        registry.register(MockTool::new("write", false, "ok"));

        let defs = registry.definitions(&HashSet::new());
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "read");
        assert_eq!(defs[1].name, "write");
    }

    #[test]
    fn registry_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());

        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("t", true, "ok"));
        assert!(!registry.is_empty());
    }

    #[test]
    fn registry_definitions_deferred() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("always_visible", true, "ok"));
        registry.register(DeferredMockTool::new("deferred_tool"));

        // Without discovery: deferred tool has empty definition
        let discovered = HashSet::new();
        let defs = registry.definitions(&discovered);
        assert_eq!(defs.len(), 2);
        let deferred = defs.iter().find(|d| d.name == "deferred_tool").unwrap();
        assert!(deferred.description.is_empty());
        assert_eq!(deferred.input_schema, serde_json::json!({}));

        // With discovery: deferred tool has full definition
        let mut discovered = HashSet::new();
        discovered.insert("deferred_tool".to_string());
        let defs = registry.definitions(&discovered);
        let deferred = defs.iter().find(|d| d.name == "deferred_tool").unwrap();
        assert!(!deferred.description.is_empty());
    }

    #[test]
    fn registry_search_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("read_file", true, "ok"));
        registry.register(MockTool::new("write_file", false, "ok"));

        let results = registry.search("read");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "read_file");
    }

    #[test]
    fn registry_clone() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("t", true, "ok"));
        let cloned = registry.clone();
        assert_eq!(cloned.definitions(&HashSet::new()).len(), 1);
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_error() {
        let registry = ToolRegistry::new();
        let ctx = test_tool_context().registry(Arc::new(registry));
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "nonexistent".into(),
            input: serde_json::json!({}),
        }];

        let results = execute_tool_calls(&calls, &ctx).await;
        assert_eq!(results.len(), 1);
        match &results[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(is_error);
                assert!(content.contains("Unknown tool"));
            }
            other => panic!("Expected ToolResult, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn execute_read_only_tools_concurrently() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("read1", true, "result1"));
        registry.register(MockTool::new("read2", true, "result2"));
        let ctx = test_tool_context().registry(Arc::new(registry));

        let calls = vec![
            ToolCall {
                id: "c1".into(),
                name: "read1".into(),
                input: serde_json::json!({}),
            },
            ToolCall {
                id: "c2".into(),
                name: "read2".into(),
                input: serde_json::json!({}),
            },
        ];

        let results = execute_tool_calls(&calls, &ctx).await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn execute_serial_tool() {
        let mut registry = ToolRegistry::new();
        let tool = MockTool::new("write_file", false, "written");
        registry.register(tool);
        let ctx = test_tool_context().registry(Arc::new(registry));

        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "write_file".into(),
            input: serde_json::json!({"path": "/tmp/test"}),
        }];

        let results = execute_tool_calls(&calls, &ctx).await;
        assert_eq!(results.len(), 1);
        match &results[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert!(!is_error);
                assert_eq!(content, "written");
            }
            other => panic!("Expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn tool_builder_basic() {
        let tool = ToolBuilder::new("echo", "Echoes input")
            .schema(serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}}))
            .read_only(true)
            .handler(|input, _ctx| {
                Box::pin(async move {
                    let text = input["text"].as_str().unwrap_or("").to_string();
                    Ok(ToolResult::success(text))
                })
            })
            .build();

        assert_eq!(tool.name(), "echo");
        assert!(tool.is_read_only());
    }

    #[test]
    fn tool_builder_defer_and_hints() {
        let tool = ToolBuilder::new("advanced", "Advanced tool")
            .should_defer(true)
            .search_hints(vec!["analyze".into(), "inspect".into()])
            .handler(|_input, _ctx| {
                Box::pin(async { Ok(ToolResult::success("ok")) })
            })
            .build();

        assert!(tool.should_defer());
        assert_eq!(tool.search_hints().len(), 2);
    }

    #[test]
    #[should_panic(expected = "requires a handler")]
    fn tool_builder_panics_without_handler() {
        let _ = ToolBuilder::new("no_handler", "missing").build();
    }

}
