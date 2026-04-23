//! Core tool infrastructure: the `Tool` trait, the ad-hoc `Tool` struct, and the registry the loop consults before each provider call.

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
use crate::event::ToolFailureKind;
use crate::provider::types::ContentBlock;

/// Context passed to tool execution. `runtime` and `caller_spec` are ambient
/// internals: `SpawnAgentTool` and `ToolSearchTool` read them to reach the
/// run's provider, handlers, and queue, and to inherit the caller's resolved
/// model. External tool authors do not use them.
#[derive(Clone)]
pub struct ToolContext {
    /// Working directory the tool runs in. Tools that touch the filesystem
    /// should resolve relative paths against this.
    pub working_directory: PathBuf,
    pub(crate) tool_registry: Option<Arc<ToolRegistry>>,
    pub(crate) runtime: Option<Arc<LoopRuntime>>,
    pub(crate) caller_spec: Option<Arc<AgentSpec>>,
    pub(crate) cancel_signal: Arc<AtomicBool>,
}

impl ToolContext {
    /// A fresh context rooted at `working_directory`, with no runtime or
    /// caller spec. Tools that call sub-agents or search the registry need a
    /// context installed by the loop; bare contexts are for standalone use.
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

/// Tool definition sent to the provider as part of the `tools` parameter.
/// Built from a [`Tool`]'s `name`, `description`, and `input_schema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name the model uses when calling the tool.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema describing the tool's arguments.
    pub input_schema: Value,
}

/// A tool call extracted from a provider response.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Provider-assigned call id, echoed back in the tool result block.
    pub id: String,
    /// Name of the tool the model chose.
    pub name: String,
    /// Arguments the model supplied, conforming to the tool's input schema.
    pub input: Value,
}

/// Outcome of a tool execution: a success payload or an error message. Both
/// variants flow back to the model as ordinary content — [`ToolResult::Error`]
/// lets the model correct its arguments and try again.
#[derive(Debug, Clone)]
pub enum ToolResult {
    /// Tool ran and produced this text payload.
    Success(String),
    /// Tool failed; the message is shown to the model so it can recover.
    Error(String),
}

impl ToolResult {
    /// Build a successful result from a text payload.
    pub fn success(content: impl Into<String>) -> Self {
        Self::Success(content.into())
    }

    /// Build an error result. The message is visible to the model.
    pub fn error(content: impl Into<String>) -> Self {
        Self::Error(content.into())
    }
}

/// The core tool interface. Object-safe via boxed futures.
///
/// Implement this on any type you want an agent to be able to invoke. For
/// ad-hoc tools defined inline, use the [`Tool`] struct's builder
/// (`Tool::new(name, description).schema(...).handler(...)`).
pub trait ToolLike: Send + Sync {
    /// Unique name the model uses to call the tool.
    fn name(&self) -> &str;

    /// Human-readable description shown to the model.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's arguments.
    fn input_schema(&self) -> Value;

    /// Whether this tool has no side effects. Read-only tools in the same
    /// turn run concurrently; non-read-only tools run serially. Default: `false`.
    fn is_read_only(&self) -> bool {
        false
    }

    /// Whether the tool's full definition is hidden until it is discovered
    /// via [`ToolSearchTool`](crate::tools::ToolSearchTool). Deferred tools appear
    /// to the model as name-only stubs. Default: `false`.
    fn should_defer(&self) -> bool {
        false
    }

    /// Run the tool. The future is held by the agent loop and dropped on
    /// cancellation; pair long-running work with [`ToolContext::wait_for_cancel`]
    /// in a `tokio::select!` to drop the losing branch promptly.
    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;
}

/// Registry of tools available to an agent.
pub(crate) struct ToolRegistry {
    pub(crate) tools: Vec<Arc<dyn ToolLike>>,
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

    pub(crate) fn register(&mut self, tool: impl ToolLike + 'static) {
        self.tools.push(Arc::new(tool));
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn ToolLike>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    /// Tool definitions sent to the provider. Deferred tools that haven't
    /// been discovered yet get name-only stubs; all others get full definitions.
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
                    ToolDefinition {
                        name: t.name().to_string(),
                        description: t.description().to_string(),
                        input_schema: t.input_schema(),
                    }
                }
            })
            .collect()
    }

    /// Execute tool calls with concurrent read-only batching and serial write
    /// execution.
    ///
    /// Returns `(ContentBlock, ToolResult, Option<ToolFailureKind>)` triples
    /// so the caller can read both the model-visible result block and the
    /// failure classification (`InBand` for `ToolResult::Error`, `Infrastructure`
    /// for an `Err` raised by the tool or an unknown-tool lookup).
    pub(crate) async fn execute(
        &self,
        calls: &[ToolCall],
        ctx: &ToolContext,
    ) -> Vec<(ContentBlock, ToolResult, Option<ToolFailureKind>)> {
        let batches = partition_tool_calls(calls, self);
        let mut results: Vec<(ContentBlock, ToolResult, Option<ToolFailureKind>)> = Vec::new();
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
                            let (result, failure_kind) =
                                invoke(tool_arc, &call_name, input, &ctx).await;
                            (call_id, result, failure_kind)
                        });
                    }

                    while let Some(join_result) = set.join_next().await {
                        if let Ok((id, result, failure_kind)) = join_result {
                            let block = content_block_for(&id, &result);
                            results.push((block, result, failure_kind));
                        }
                    }
                }
                ToolBatch::Serial(call) => {
                    let (result, failure_kind) =
                        invoke(self.get(&call.name), &call.name, call.input.clone(), ctx).await;
                    let block = content_block_for(&call.id, &result);
                    results.push((block, result, failure_kind));
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

                if score > 0 {
                    Some((
                        ToolDefinition {
                            name: t.name().to_string(),
                            description: t.description().to_string(),
                            input_schema: t.input_schema(),
                        },
                        score,
                    ))
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
/// A handler is required — omitting [`Tool::handler`] causes the first
/// invocation to panic. For tools with complex state, implement
/// [`ToolLike`] directly on your own type instead.
pub struct Tool {
    name: String,
    description: String,
    schema: Value,
    read_only: bool,
    defer: bool,
    handler: Option<ToolHandler>,
}

impl Tool {
    /// A new tool with an empty-object input schema and no handler. Set the
    /// handler with [`Tool::handler`] before handing the tool to an agent.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({"type": "object", "properties": {}}),
            read_only: false,
            defer: false,
            handler: None,
        }
    }

    /// Replace the input schema. Defaults to an empty-object schema.
    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = schema;
        self
    }

    /// Mark the tool read-only so the loop runs it concurrently with other
    /// read-only calls in the same turn.
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Hide the tool's full definition until it is discovered via
    /// [`ToolSearchTool`](crate::tools::ToolSearchTool).
    pub fn defer(mut self, defer: bool) -> Self {
        self.defer = defer;
        self
    }

    /// Install the closure that runs when the model calls this tool.
    /// Required: omitting this causes the first invocation to panic.
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

impl ToolLike for Tool {
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

async fn invoke(
    tool: Option<Arc<dyn ToolLike>>,
    name: &str,
    input: Value,
    ctx: &ToolContext,
) -> (ToolResult, Option<ToolFailureKind>) {
    match tool {
        Some(t) => match t.call(input, ctx).await {
            Ok(r @ ToolResult::Success(_)) => (r, None),
            Ok(r @ ToolResult::Error(_)) => (r, Some(ToolFailureKind::InBand)),
            Err(e) => (
                ToolResult::error(format!("Tool error: {e}")),
                Some(ToolFailureKind::Infrastructure),
            ),
        },
        None => (
            ToolResult::error(format!("Unknown tool: {name}")),
            Some(ToolFailureKind::InBand),
        ),
    }
}

fn content_block_for(tool_use_id: &str, result: &ToolResult) -> ContentBlock {
    let (content, is_error) = match result {
        ToolResult::Success(s) => (s.clone(), false),
        ToolResult::Error(s) => (s.clone(), true),
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
    fn tool_defer_builder() {
        let tool = Tool::new("advanced", "Advanced tool")
            .defer(true)
            .handler(|_input, _ctx| Box::pin(async { Ok(ToolResult::success("ok")) }));

        assert!(tool.should_defer());
    }

    #[tokio::test]
    #[should_panic(expected = "requires a handler")]
    async fn tool_panics_without_handler() {
        let tool = Tool::new("no_handler", "missing");
        let ctx = test_tool_context();
        let _ = tool.call(serde_json::json!({}), &ctx).await;
    }
}
