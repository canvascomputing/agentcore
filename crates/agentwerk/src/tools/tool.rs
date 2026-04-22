//! Core tool infrastructure — the `Toolable` trait, the ad-hoc `Tool` struct, and the registry the loop consults before each provider call.

use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{AgentSpec, LoopRuntime};
use crate::error::Result;
use crate::provider::types::ContentBlock;

/// Context passed to tool execution. `runtime` and `caller_spec` are ambient internals —
/// `SpawnAgentTool` and `ToolSearchTool` read them to reach the run's provider / handlers /
/// queue and to inherit the caller's resolved model; external tool authors do not use them.
#[derive(Clone)]
pub struct ToolContext {
    pub working_directory: PathBuf,
    pub(crate) tool_registry: Option<Arc<ToolRegistry>>,
    pub(crate) runtime: Option<Arc<LoopRuntime>>,
    pub(crate) caller_spec: Option<Arc<AgentSpec>>,
    pub(crate) cancel_signal: Arc<AtomicBool>,
}

impl ToolContext {
    pub fn new(working_directory: PathBuf) -> Self {
        Self {
            working_directory,
            tool_registry: None,
            runtime: None,
            caller_spec: None,
            cancel_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    pub(crate) fn runtime(mut self, runtime: Arc<LoopRuntime>) -> Self {
        self.cancel_signal = runtime.cancel_signal.clone();
        self.runtime = Some(runtime);
        self
    }

    pub(crate) fn caller_spec(mut self, spec: Arc<AgentSpec>) -> Self {
        self.caller_spec = Some(spec);
        self
    }

    /// Future that resolves once the current run is cancelled. Pair with
    /// `tokio::select!` to drop the losing branch on Ctrl-C; dropped futures
    /// cascade to `reqwest` aborts and (with `kill_on_drop(true)`) subprocess
    /// kills. When the context is not attached to a running loop, the future
    /// stays pending forever and the `select!` degrades to a plain await.
    pub async fn wait_for_cancel(&self) {
        crate::util::wait_for_cancel(&self.cancel_signal).await;
    }

    pub(crate) fn mark_tool_discovered(&self, name: &str) {
        if let Some(runtime) = self.runtime.as_ref() {
            runtime
                .discovered_tools
                .lock()
                .unwrap()
                .insert(name.to_string());
        }
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("working_directory", &self.working_directory)
            .field("tool_registry", &self.tool_registry)
            .field("has_runtime", &self.runtime.is_some())
            .field("has_caller_spec", &self.caller_spec.is_some())
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

/// Outcome of a tool execution — a success payload or a failure message.
#[derive(Debug, Clone)]
pub enum ToolResult {
    Success(String),
    Failure(String),
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self::Success(content.into())
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self::Failure(content.into())
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    pub fn is_err(&self) -> bool {
        matches!(self, Self::Failure(_))
    }

    pub fn content(&self) -> &str {
        match self {
            Self::Success(s) | Self::Failure(s) => s,
        }
    }
}

/// The core tool interface. Object-safe via boxed futures.
///
/// Implement this on any type you want an agent to be able to invoke. For ad-hoc
/// tools defined inline, use the concrete [`Tool`] struct instead — it implements
/// `Toolable` for you.
pub trait Toolable: Send + Sync {
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

/// Registry of tools available to an agent.
pub(crate) struct ToolRegistry {
    pub(crate) tools: Vec<Arc<dyn Toolable>>,
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

    pub(crate) fn register(&mut self, tool: impl Toolable + 'static) {
        self.tools.push(Arc::new(tool));
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn Toolable>> {
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

    /// Execute tool calls with concurrent read-only batching and serial write
    /// execution.
    ///
    /// Returns `(ContentBlock, ToolResult)` pairs so the caller can read both
    /// the LLM-visible result block and any `ToolResult` side-effects
    /// (structured output capture, discovered tool names) without dispatching
    /// on tool names.
    pub(crate) async fn execute(
        &self,
        calls: &[ToolCall],
        ctx: &ToolContext,
    ) -> Vec<(ContentBlock, ToolResult)> {
        let batches = partition_tool_calls(calls, self);
        let mut results: Vec<(ContentBlock, ToolResult)> = Vec::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(10));

        for batch in batches {
            match batch {
                ToolBatch::Concurrent(calls) => {
                    let mut set = tokio::task::JoinSet::new();
                    for call in calls {
                        let sem = semaphore.clone();
                        let ctx = ctx.clone();
                        let tool_arc = self.get(&call.name);
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
                            let block = content_block_for(&id, &result);
                            results.push((block, result));
                        }
                    }
                }
                ToolBatch::Serial(call) => {
                    let result = match self.get(&call.name) {
                        Some(tool) => match tool.call(call.input.clone(), ctx).await {
                            Ok(r) => r,
                            Err(e) => ToolResult::error(format!("Tool error: {e}")),
                        },
                        None => ToolResult::error(format!("Unknown tool: {}", call.name)),
                    };
                    let block = content_block_for(&call.id, &result);
                    results.push((block, result));
                }
            }
        }

        results
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
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

type ToolHandler = Box<
    dyn Fn(Value, &ToolContext) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
        + Send
        + Sync,
>;

/// Ad-hoc tool defined inline with a closure handler.
///
/// Chain builder methods to configure, then hand the tool to an agent:
/// ```ignore
/// let greet = Tool::new("greet", "Say hello")
///     .schema(serde_json::json!({"type": "object", "properties": {}}))
///     .handler(|_input, _ctx| Box::pin(async {
///         Ok(ToolResult::success("hi"))
///     }));
///
/// Agent::new().tool(greet);
/// ```
///
/// A handler is required — omitting [`Tool::handler`] causes the first invocation
/// to panic. For tools with complex state, implement [`Toolable`] directly on your
/// own type instead.
pub struct Tool {
    name: String,
    description: String,
    schema: Value,
    read_only: bool,
    defer: bool,
    hints: Vec<String>,
    handler: Option<ToolHandler>,
}

impl Tool {
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

    pub fn defer(mut self, defer: bool) -> Self {
        self.defer = defer;
        self
    }

    pub fn hints(mut self, hints: Vec<String>) -> Self {
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
}

impl Toolable for Tool {
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
        let handler = self
            .handler
            .as_ref()
            .expect("Tool requires a handler — set one via `.handler(...)` before use");
        (handler)(input, ctx)
    }
}

fn content_block_for(tool_use_id: &str, result: &ToolResult) -> ContentBlock {
    let (content, is_error) = match result {
        ToolResult::Success(s) => (s.clone(), false),
        ToolResult::Failure(s) => (s.clone(), true),
    };
    ContentBlock::ToolResult {
        tool_use_id: tool_use_id.to_string(),
        content,
        is_error,
    }
}

enum ToolBatch {
    Concurrent(Vec<ToolCall>),
    Serial(ToolCall),
}

fn partition_tool_calls(calls: &[ToolCall], registry: &ToolRegistry) -> Vec<ToolBatch> {
    let mut batches: Vec<ToolBatch> = Vec::new();
    let mut concurrent_batch: Vec<ToolCall> = Vec::new();

    for call in calls {
        let is_read_only = registry.get(&call.name).map_or(false, |t| t.is_read_only());

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
        let ctx = test_tool_context();
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "nonexistent".into(),
            input: serde_json::json!({}),
        }];

        let results = registry.execute(&calls, &ctx).await;
        assert_eq!(results.len(), 1);
        match &results[0].0 {
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
        let ctx = test_tool_context();

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

        let results = registry.execute(&calls, &ctx).await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn execute_serial_tool() {
        let mut registry = ToolRegistry::new();
        let tool = MockTool::new("write_file", false, "written");
        registry.register(tool);
        let ctx = test_tool_context();

        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "write_file".into(),
            input: serde_json::json!({"path": "/tmp/test"}),
        }];

        let results = registry.execute(&calls, &ctx).await;
        assert_eq!(results.len(), 1);
        match &results[0].0 {
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
    fn tool_basic() {
        let tool = Tool::new("echo", "Echoes input")
            .schema(
                serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            )
            .read_only(true)
            .handler(|input, _ctx| {
                Box::pin(async move {
                    let text = input["text"].as_str().unwrap_or("").to_string();
                    Ok(ToolResult::success(text))
                })
            });

        assert_eq!(tool.name(), "echo");
        assert!(tool.is_read_only());
    }

    #[test]
    fn tool_defer_and_hints() {
        let tool = Tool::new("advanced", "Advanced tool")
            .defer(true)
            .hints(vec!["analyze".into(), "inspect".into()])
            .handler(|_input, _ctx| Box::pin(async { Ok(ToolResult::success("ok")) }));

        assert!(tool.should_defer());
        assert_eq!(tool.search_hints().len(), 2);
    }

    #[tokio::test]
    #[should_panic(expected = "requires a handler")]
    async fn tool_panics_without_handler() {
        let tool = Tool::new("no_handler", "missing");
        let ctx = test_tool_context();
        let _ = tool.call(serde_json::json!({}), &ctx).await;
    }
}
