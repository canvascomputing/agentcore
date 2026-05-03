//! Core tool infrastructure: the `ToolLike` trait, the ad-hoc `Tool` struct, and the registry the loop consults before each provider call.

use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agents::tickets::TicketSystemState;
use crate::providers::types::ContentBlock;
use crate::providers::{ProviderResult, ProviderToolDefinition};

use super::error::ToolError;

/// Context passed to tool execution. `tool_registry` and the ticket-side
/// fields are ambient internals — only the built-in `ToolSearchTool` and
/// the ticket tools (`Read`/`Write`/`Manage`) read them. External tool
/// authors use `working_dir`, `interrupt_signal`, and
/// `wait_for_cancel`.
#[derive(Clone)]
pub struct ToolContext {
    /// Working directory the tool runs in. Tools that touch the filesystem
    /// should resolve relative paths against this.
    pub working_dir: PathBuf,
    pub interrupt_signal: Arc<AtomicBool>,
    pub(crate) tool_registry: Option<Arc<ToolRegistry>>,
    pub(crate) ticket_system_state: Option<Arc<Mutex<TicketSystemState>>>,
    pub(crate) current_ticket: Option<String>,
    pub(crate) agent_name: Option<String>,
}

impl ToolContext {
    /// A fresh context rooted at `working_dir`, with no registry handle and
    /// a fresh never-firing cancel signal. Tools that search the registry
    /// need a context installed by the loop; bare contexts are for
    /// standalone use and tests.
    pub fn new(working_dir: PathBuf) -> Self {
        Self {
            working_dir,
            interrupt_signal: Arc::new(AtomicBool::new(false)),
            tool_registry: None,
            ticket_system_state: None,
            current_ticket: None,
            agent_name: None,
        }
    }

    /// Override the cancel signal — typically the one shared by the loop's
    /// `TicketSystem` so tools cooperate with run-level cancellation.
    pub fn interrupt_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.interrupt_signal = signal;
        self
    }

    pub(crate) fn registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    pub(crate) fn ticket_system_state(
        mut self,
        state: Arc<Mutex<TicketSystemState>>,
    ) -> Self {
        self.ticket_system_state = Some(state);
        self
    }

    pub(crate) fn current_ticket(mut self, key: String) -> Self {
        self.current_ticket = Some(key);
        self
    }

    pub(crate) fn agent_name(mut self, name: String) -> Self {
        self.agent_name = Some(name);
        self
    }

    pub(crate) fn ticket_system_state_handle(
        &self,
    ) -> Option<&Arc<Mutex<TicketSystemState>>> {
        self.ticket_system_state.as_ref()
    }

    pub(crate) fn current_ticket_key(&self) -> Option<&str> {
        self.current_ticket.as_deref()
    }

    pub(crate) fn agent_name_str(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }

    /// Future that resolves once the current run is cancelled. Pair with
    /// `tokio::select!` to drop the losing branch on cancel; dropped futures
    /// cascade to `reqwest` aborts and (with `kill_on_drop(true)`) subprocess
    /// kills. With a fresh context the future stays pending forever and the
    /// `select!` degrades to a plain await.
    pub async fn wait_for_cancel(&self) {
        const POLL: std::time::Duration = std::time::Duration::from_millis(50);
        loop {
            if self.interrupt_signal.load(Ordering::Relaxed) {
                return;
            }
            tokio::time::sleep(POLL).await;
        }
    }

    pub(crate) fn mark_tool_discovered(&self, name: &str) {
        if let Some(registry) = self.tool_registry.as_ref() {
            registry.mark_discovered(name);
        }
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("working_dir", &self.working_dir)
            .field("has_registry", &self.tool_registry.is_some())
            .finish()
    }
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

/// Outcome of a tool execution: a success payload, a generic error
/// message, or a schema-validation failure. All three flow back to
/// the model as ordinary content blocks; the [`SchemaError`] variant
/// is distinguished so the loop can apply
/// `policies.max_schema_retries` to it specifically.
///
/// [`SchemaError`]: ToolResult::SchemaError
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResult {
    /// Tool ran and produced this text payload.
    Success(String),
    /// Tool failed; the message is shown to the model so it can recover.
    Error(String),
    /// Tool rejected the input because it failed schema validation.
    /// The message lists the violations so the model can fix them.
    SchemaError(String),
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

    /// Build a schema-validation failure. Mapped into
    /// [`ToolError::SchemaValidationFailed`] by the registry and
    /// counted against `policies.max_schema_retries`.
    pub fn schema_error(content: impl Into<String>) -> Self {
        Self::SchemaError(content.into())
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
    /// step run concurrently; non-read-only tools run serially. Default: `false`.
    fn is_read_only(&self) -> bool {
        false
    }

    /// Whether the tool's full definition is hidden until it is discovered
    /// via `ToolSearchTool`. Deferred tools appear to the model as
    /// name-only stubs. Default: `false`.
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
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>>;
}

/// Registry of tools available to an agent. Also owns the set of deferred
/// tools discovered during the run: cloning the registry starts with an
/// empty set.
pub(crate) struct ToolRegistry {
    pub(crate) tools: Vec<Arc<dyn ToolLike>>,
    pub(crate) discovered: Mutex<HashSet<String>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.tools.iter().map(|t| t.name()).collect();
        f.debug_struct("ToolRegistry")
            .field("tools", &names)
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            discovered: Mutex::new(HashSet::new()),
        }
    }
}

impl ToolRegistry {
    pub(crate) fn register(&mut self, tool: impl ToolLike + 'static) {
        self.tools.push(Arc::new(tool));
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn ToolLike>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    pub(crate) fn mark_discovered(&self, name: &str) {
        self.discovered.lock().unwrap().insert(name.to_string());
    }

    /// Tool definitions sent to the provider. Deferred tools that haven't
    /// been discovered yet get name-only stubs; all others get full
    /// definitions.
    pub(crate) fn definitions(&self) -> Vec<ProviderToolDefinition> {
        let discovered = self.discovered.lock().unwrap();
        self.tools
            .iter()
            .map(|t| {
                if t.should_defer() && !discovered.contains(t.name()) {
                    ProviderToolDefinition {
                        name: t.name().to_string(),
                        description: String::new(),
                        input_schema: serde_json::json!({}),
                    }
                } else {
                    ProviderToolDefinition {
                        name: t.name().to_string(),
                        description: t.description().to_string(),
                        input_schema: t.input_schema(),
                    }
                }
            })
            .collect()
    }

    /// Search tools by query string. Returns matches sorted by relevance
    /// (highest first).
    pub(crate) fn search(&self, query: &str) -> Vec<ProviderToolDefinition> {
        let query_lower = query.to_lowercase();
        let mut scored: Vec<(ProviderToolDefinition, u32)> = self
            .tools
            .iter()
            .filter_map(|t| {
                let mut score = 0u32;
                let name = t.name().to_lowercase();
                let desc = t.description().to_lowercase();

                if name == query_lower {
                    score += 100;
                } else if name.contains(&query_lower) {
                    score += 50;
                }

                if desc.contains(&query_lower) {
                    score += 25;
                }

                if score > 0 {
                    Some((
                        ProviderToolDefinition {
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

    /// Execute tool calls with concurrent read-only batching and serial
    /// write execution.
    ///
    /// Returns `(ContentBlock, Result<String, ToolError>)` pairs so the
    /// caller can read both the model-visible result block and the typed
    /// verdict.
    pub(crate) async fn execute(
        &self,
        calls: &[ToolCall],
        ctx: &ToolContext,
    ) -> Vec<(ContentBlock, std::result::Result<String, ToolError>)> {
        let batches = partition_tool_calls(calls, self);
        let mut results: Vec<(ContentBlock, std::result::Result<String, ToolError>)> = Vec::new();
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
                            let outcome = invoke(tool_arc, &call_name, input, &ctx).await;
                            (call_id, outcome)
                        });
                    }

                    while let Some(join_result) = set.join_next().await {
                        if let Ok((id, outcome)) = join_result {
                            let block = content_block_for(&id, &outcome);
                            results.push((block, outcome));
                        }
                    }
                }
                ToolBatch::Serial(call) => {
                    let outcome =
                        invoke(self.get(&call.name), &call.name, call.input.clone(), ctx).await;
                    let block = content_block_for(&call.id, &outcome);
                    results.push((block, outcome));
                }
            }
        }

        results
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
            discovered: Mutex::new(HashSet::new()),
        }
    }
}

type ToolHandler = Box<
    dyn Fn(
            Value,
            &ToolContext,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + '_>>
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

    /// Construct a `Tool` from a `.tool.json` definition. The returned tool
    /// has its name, rendered description, input schema, and read-only flag
    /// populated from the JSON; attach a handler via [`Tool::handler`]
    /// before registering it with an agent. Panics on malformed JSON.
    pub fn from_tool_file(json: &str) -> Self {
        let tf = super::tool_file::ToolFile::parse(json);
        Tool::new(tf.name.clone(), tf.render_markdown())
            .schema(tf.input_schema.clone())
            .read_only(tf.read_only)
    }

    /// Replace the input schema (JSON Schema). Defaults to an empty-object
    /// schema.
    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = schema;
        self
    }

    /// Mark the tool read-only so the loop runs it concurrently with other
    /// read-only calls in the same step.
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Hide the tool's full definition until it is discovered via
    /// `ToolSearchTool`.
    pub fn defer(mut self, defer: bool) -> Self {
        self.defer = defer;
        self
    }

    /// Install the closure that runs when the model calls this tool.
    /// Required: omitting this causes the first invocation to panic.
    pub fn handler<F>(mut self, f: F) -> Self
    where
        F: Fn(
                Value,
                &ToolContext,
            ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + '_>>
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
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
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
) -> std::result::Result<String, ToolError> {
    let Some(t) = tool else {
        return Err(ToolError::ToolNotFound {
            tool_name: name.into(),
        });
    };
    match t.call(input, ctx).await {
        Ok(ToolResult::Success(s)) => Ok(s),
        Ok(ToolResult::Error(s)) => Err(ToolError::ExecutionFailed {
            tool_name: name.into(),
            message: s,
        }),
        Ok(ToolResult::SchemaError(s)) => Err(ToolError::SchemaValidationFailed {
            tool_name: name.into(),
            message: s,
        }),
        Err(e) => Err(ToolError::ExecutionFailed {
            tool_name: name.into(),
            message: e.to_string(),
        }),
    }
}

fn content_block_for(
    tool_use_id: &str,
    outcome: &std::result::Result<String, ToolError>,
) -> ContentBlock {
    let (content, is_error) = match outcome {
        Ok(s) => (s.clone(), false),
        Err(e) => (e.message(), true),
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
        let is_read_only = registry.get(&call.name).is_some_and(|t| t.is_read_only());

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

    /// Tiny mock used across registry tests.
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
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        fn is_read_only(&self) -> bool {
            self.read_only
        }
        fn call<'a>(
            &'a self,
            _input: Value,
            _ctx: &'a ToolContext,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
            let result = self.result.clone();
            Box::pin(async move { Ok(ToolResult::success(result)) })
        }
    }

    struct DeferredMockTool {
        name: String,
    }

    impl DeferredMockTool {
        fn new(name: &str) -> Self {
            Self { name: name.into() }
        }
    }

    impl ToolLike for DeferredMockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "deferred mock"
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        fn should_defer(&self) -> bool {
            true
        }
        fn call<'a>(
            &'a self,
            _input: Value,
            _ctx: &'a ToolContext,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
            Box::pin(async { Ok(ToolResult::success("ok")) })
        }
    }

    fn test_ctx() -> ToolContext {
        ToolContext::new(std::env::current_dir().unwrap())
    }

    #[test]
    fn registry_register_and_get() {
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("read_file", true, "file contents"));
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn from_tool_file_populates_name_description_schema_read_only() {
        let json = r#"{
            "name": "demo_tool",
            "summary": ["Do the demo thing."],
            "constraints": ["Returns nothing useful."],
            "read_only": true,
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }
        }"#;
        let tool = Tool::from_tool_file(json);
        assert_eq!(tool.name(), "demo_tool");
        assert!(tool.description().contains("Do the demo thing."));
        assert!(tool.description().contains("- Returns nothing useful."));
        assert!(tool.is_read_only());
        let schema = tool.input_schema();
        assert_eq!(schema["properties"]["x"]["type"], "string");
        assert_eq!(schema["required"][0], "x");
    }

    #[test]
    fn registry_definitions() {
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("read", true, "ok"));
        registry.register(MockTool::new("write", false, "ok"));

        let defs = registry.definitions();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "read");
        assert_eq!(defs[1].name, "write");
    }

    #[test]
    fn registry_definitions_deferred() {
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("always_visible", true, "ok"));
        registry.register(DeferredMockTool::new("deferred_tool"));

        let defs = registry.definitions();
        assert_eq!(defs.len(), 2);
        let deferred = defs.iter().find(|d| d.name == "deferred_tool").unwrap();
        assert!(deferred.description.is_empty());
        assert_eq!(deferred.input_schema, serde_json::json!({}));

        registry.mark_discovered("deferred_tool");
        let defs = registry.definitions();
        let deferred = defs.iter().find(|d| d.name == "deferred_tool").unwrap();
        assert!(!deferred.description.is_empty());
    }

    #[test]
    fn registry_search_by_name() {
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("read_file", true, "ok"));
        registry.register(MockTool::new("write_file", false, "ok"));

        let results = registry.search("read");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "read_file");
    }

    #[test]
    fn registry_clone() {
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("t", true, "ok"));
        let cloned = registry.clone();
        assert_eq!(cloned.definitions().len(), 1);
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_error() {
        let registry = ToolRegistry::default();
        let ctx = test_ctx();
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
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("read1", true, "result1"));
        registry.register(MockTool::new("read2", true, "result2"));
        let ctx = test_ctx();

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
        let mut registry = ToolRegistry::default();
        registry.register(MockTool::new("write_file", false, "written"));
        let ctx = test_ctx();

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
        let ctx = test_ctx();
        let _ = tool.call(serde_json::json!({}), &ctx).await;
    }
}
