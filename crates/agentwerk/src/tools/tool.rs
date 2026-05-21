//! Core tool infrastructure: the `ToolLike` trait, the ad-hoc `Tool` struct, and the registry the loop consults before each provider call.

use std::collections::HashSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agents::knowledge::Knowledge;
use crate::agents::tickets::TicketSystem;
use crate::providers::types::ContentBlock;
use crate::providers::{ProviderResult, ProviderToolDefinition};

use super::error::ToolError;

/// Per-tool ceiling on tool-result bytes. Outputs larger than this are
/// offloaded to `<ticket-dir>/outputs/<tool_use_id>.txt` and replaced
/// with a stub. ~12.5K tokens at the `bytes/4` estimator.
const PER_TOOL_CAP: usize = 50_000;

/// Aggregate ceiling on a single step's tool-result bytes. When parallel
/// tools each return moderate but legal sizes, this cap offloads the
/// largest until the step is under budget. ~50K tokens.
const PER_STEP_CAP: usize = 200_000;

/// Bytes of original output retained in the stub preview. Snapped to
/// the last newline within the window when one exists, otherwise
/// floored to a UTF-8 boundary so multi-byte code points are never
/// split.
const PREVIEW_CHARS: usize = 2_000;

/// Context passed to tool execution. `tool_registry` and the ticket-side
/// fields are ambient internals — only the built-in `ToolSearchTool` and
/// the ticket tools (`Read`/`Write`/`Manage`) read them. External tool
/// authors use `dir`, `interrupt_signal`, and
/// `wait_for_cancel`.
#[derive(Clone)]
pub struct ToolContext {
    /// Directory the tool runs in. Tools that touch the filesystem
    /// should resolve relative paths against this.
    pub dir: PathBuf,
    pub interrupt_signal: Arc<AtomicBool>,
    pub(crate) tool_registry: Option<Arc<ToolRegistry>>,
    pub(crate) ticket_system: Option<Arc<TicketSystem>>,
    pub(crate) agent_name: Option<String>,
    pub(crate) ticket_key: Option<String>,
    pub(crate) knowledge: Option<Arc<Knowledge>>,
}

impl ToolContext {
    /// A fresh context rooted at `dir`, with no registry handle and
    /// a fresh never-firing cancel signal. Tools that search the registry
    /// need a context installed by the loop; bare contexts are for
    /// standalone use and tests.
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            interrupt_signal: Arc::new(AtomicBool::new(false)),
            tool_registry: None,
            ticket_system: None,
            agent_name: None,
            ticket_key: None,
            knowledge: None,
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

    pub(crate) fn ticket_system(mut self, system: Arc<TicketSystem>) -> Self {
        self.ticket_system = Some(system);
        self
    }

    pub(crate) fn agent_name(mut self, name: String) -> Self {
        self.agent_name = Some(name);
        self
    }

    pub(crate) fn ticket_key(mut self, key: String) -> Self {
        self.ticket_key = Some(key);
        self
    }

    pub(crate) fn knowledge(mut self, knowledge: Arc<Knowledge>) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    pub(crate) fn ticket_system_handle(&self) -> Option<&Arc<TicketSystem>> {
        self.ticket_system.as_ref()
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
            .field("dir", &self.dir)
            .field("has_registry", &self.tool_registry.is_some())
            .field("has_ticket_system", &self.ticket_system.is_some())
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
        let name = name.trim();
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
    /// Returns `(ContentBlock, Result<String, ToolError>, Option<PathBuf>)`
    /// triples so the caller can read the model-visible result block,
    /// the typed verdict, and the offload path for oversized outputs.
    pub(crate) async fn execute(
        &self,
        calls: &[ToolCall],
        ctx: &ToolContext,
    ) -> Vec<(
        ContentBlock,
        std::result::Result<String, ToolError>,
        Option<PathBuf>,
    )> {
        let batches = partition_tool_calls(calls, self);
        let mut results: Vec<(
            ContentBlock,
            std::result::Result<String, ToolError>,
            Option<PathBuf>,
        )> = Vec::new();
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
                            let outcome = replace_empty_output(outcome, &call_name);
                            let (outcome, path) =
                                cap_oversized_result(outcome, &ctx, &call_id, PER_TOOL_CAP);
                            (call_id, outcome, path)
                        });
                    }

                    while let Some(join_result) = set.join_next().await {
                        if let Ok((id, outcome, path)) = join_result {
                            let block = content_block_for(&id, &outcome);
                            results.push((block, outcome, path));
                        }
                    }
                }
                ToolBatch::Serial(call) => {
                    let outcome =
                        invoke(self.get(&call.name), &call.name, call.input.clone(), ctx).await;
                    let outcome = replace_empty_output(outcome, &call.name);
                    let (outcome, path) =
                        cap_oversized_result(outcome, ctx, &call.id, PER_TOOL_CAP);
                    let block = content_block_for(&call.id, &outcome);
                    results.push((block, outcome, path));
                }
            }
        }

        cap_aggregate_outputs(&mut results, ctx, PER_STEP_CAP);

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
///     .handler(|_input, _ctx| async {
///         Ok(ToolResult::success("hi"))
///     });
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
    /// The closure may be a bare `async` block — the builder boxes the
    /// returned future internally. Required: omitting this causes the
    /// first invocation to panic.
    pub fn handler<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ProviderResult<ToolResult>> + Send + 'static,
    {
        self.handler = Some(Box::new(move |v, c| Box::pin(f(v, c.clone()))));
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

/// Substitute a placeholder when a tool returns an empty success
/// payload. Empty `tool_result.content` has triggered provider sampling
/// edge cases; the placeholder keeps the assistant turn well-formed.
/// Errors and non-empty successes pass through.
fn replace_empty_output(
    outcome: std::result::Result<String, ToolError>,
    tool_name: &str,
) -> std::result::Result<String, ToolError> {
    match outcome {
        Ok(s) if s.is_empty() => Ok(format!("({tool_name} completed with no output)")),
        other => other,
    }
}

fn content_block_for(
    tool_use_id: &str,
    outcome: &std::result::Result<String, ToolError>,
) -> ContentBlock {
    let (content, succeeded) = match outcome {
        Ok(s) => (s.clone(), true),
        Err(e) => (e.message(), false),
    };
    ContentBlock::ToolResult {
        tool_use_id: tool_use_id.to_string(),
        content,
        succeeded,
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

/// Replace `Ok(s)` with a stub when `s.len()` exceeds `per_tool_cap`,
/// persisting the original output under the ticket's outputs folder.
/// Returns the outcome plus the persisted file path (when offload
/// happened). `Err(_)` outcomes pass through: tool errors are short by
/// construction and never need offloading. When persistence fails (no
/// ticket bound, write error), the raw content passes through
/// unstubbed.
fn cap_oversized_result(
    outcome: std::result::Result<String, ToolError>,
    ctx: &ToolContext,
    tool_use_id: &str,
    per_tool_cap: usize,
) -> (std::result::Result<String, ToolError>, Option<PathBuf>) {
    match outcome {
        Err(e) => (Err(e), None),
        Ok(content) if content.len() <= per_tool_cap => (Ok(content), None),
        Ok(content) => match persist_output(ctx, tool_use_id, &content) {
            None => (Ok(content), None),
            Some(path) => {
                let preview = truncate_preview(&content);
                let stub = format_oversized_tool_result(content.len(), &path, preview);
                (Ok(stub), Some(path))
            }
        },
    }
}

/// Aggregate-cap pass: while the total `ContentBlock::ToolResult` bytes
/// in `results` exceed `per_step_cap`, replace the largest non-stub
/// result with a stub. Stops when no further offload would help (either
/// the cap is met or persistence has failed for every remaining
/// candidate).
fn cap_aggregate_outputs(
    results: &mut [(
        ContentBlock,
        std::result::Result<String, ToolError>,
        Option<PathBuf>,
    )],
    ctx: &ToolContext,
    per_step_cap: usize,
) {
    loop {
        let total: usize = results
            .iter()
            .map(|(b, _, _)| match b {
                ContentBlock::ToolResult { content, .. } => content.len(),
                _ => 0,
            })
            .sum();
        if total <= per_step_cap {
            return;
        }
        let mut largest: Option<(usize, usize)> = None;
        for (i, (b, _, _)) in results.iter().enumerate() {
            if let ContentBlock::ToolResult { content, .. } = b {
                if content.starts_with(OVERSIZED_STUB_TAG_OPEN) {
                    continue;
                }
                let len = content.len();
                if largest.is_none_or(|(_, max_len)| len > max_len) {
                    largest = Some((i, len));
                }
            }
        }
        let Some((i, _)) = largest else {
            return;
        };
        let ContentBlock::ToolResult {
            tool_use_id,
            content,
            succeeded,
        } = &results[i].0
        else {
            return;
        };
        let tool_use_id = tool_use_id.clone();
        let original = content.clone();
        let succeeded = *succeeded;
        let Some(path) = persist_output(ctx, &tool_use_id, &original) else {
            // Persistence failed; nothing further this pass can do.
            return;
        };
        let preview = truncate_preview(&original);
        let stub = format_oversized_tool_result(original.len(), &path, preview);
        results[i].0 = ContentBlock::ToolResult {
            tool_use_id,
            content: stub.clone(),
            succeeded,
        };
        if results[i].1.is_ok() {
            results[i].1 = Ok(stub);
        }
        results[i].2 = Some(path);
    }
}

/// Offload `content` to the ticket's outputs folder via the bound
/// `TicketSystem`. Returns the absolute path on success; `None` when no
/// ticket key is present on the context, no ticket system is bound, or
/// the write fails. Best-effort, matching the surrounding observational-
/// persistence contract.
fn persist_output(ctx: &ToolContext, tool_use_id: &str, content: &str) -> Option<PathBuf> {
    let system = ctx.ticket_system.as_ref()?;
    let key = ctx.ticket_key.as_deref()?;
    system.write_tool_output(key, tool_use_id, content)
}

const OVERSIZED_STUB_TAG_OPEN: &str = "<persisted-output>";
const OVERSIZED_STUB_TAG_CLOSE: &str = "</persisted-output>";

/// Render the stub the model sees in place of an oversized tool result.
/// Wraps the size summary, the offload path, and a leading preview in
/// `<persisted-output>...</persisted-output>` tags.
fn format_oversized_tool_result(original_len: usize, path: &Path, preview: &str) -> String {
    let size = format_bytes(original_len);
    let preview_size = format_bytes(preview.len());
    format!(
        "{OVERSIZED_STUB_TAG_OPEN}Output too large ({size}). Full output saved to: {path}\n\
         Preview (first {preview_size}):\n\
         {preview}\n\
         {OVERSIZED_STUB_TAG_CLOSE}",
        path = path.display(),
    )
}

/// Return a leading slice of `content` up to `PREVIEW_CHARS` bytes,
/// snapped to the last newline within that window. Falls back to a
/// UTF-8 boundary floor when no newline is present.
fn truncate_preview(content: &str) -> &str {
    let window = PREVIEW_CHARS.min(content.len());
    let cut = content[..window]
        .rfind('\n')
        .map(|i| i + 1)
        .unwrap_or_else(|| utf8_boundary_floor(content, window));
    &content[..cut]
}

fn format_bytes(n: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    if n < 1024 {
        format!("{n} B")
    } else if (n as f64) < MB {
        format!("{:.1} KB", n as f64 / KB)
    } else {
        format!("{:.1} MB", n as f64 / MB)
    }
}

/// Floor an index to the largest UTF-8 char boundary `<= i`. Cheap when
/// `i` already lands on a boundary; otherwise scans back at most three
/// bytes.
fn utf8_boundary_floor(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
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
                succeeded, content, ..
            } => {
                assert!(!succeeded);
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
                content, succeeded, ..
            } => {
                assert!(succeeded);
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
            .handler(|input, _ctx| async move {
                let text = input["text"].as_str().unwrap_or("").to_string();
                Ok(ToolResult::success(text))
            });

        assert_eq!(tool.name(), "echo");
        assert!(tool.is_read_only());
    }

    #[test]
    fn tool_defer_builder() {
        let tool = Tool::new("advanced", "Advanced tool")
            .defer(true)
            .handler(|_input, _ctx| async { Ok(ToolResult::success("ok")) });

        assert!(tool.should_defer());
    }

    #[tokio::test]
    #[should_panic(expected = "requires a handler")]
    async fn tool_panics_without_handler() {
        let tool = Tool::new("no_handler", "missing");
        let ctx = test_ctx();
        let _ = tool.call(serde_json::json!({}), &ctx).await;
    }

    // ---- Layer 1: result-cap helpers ----

    fn ticket_ctx() -> (ToolContext, Arc<TicketSystem>, String, crate::test_util::TempDir) {
        let dir = crate::test_util::TempDir::new().unwrap();
        let system = TicketSystem::new();
        system.dir(dir.path().to_path_buf());
        system.task("seed");
        let key = "TICKET-1".to_string();
        let ctx = test_ctx()
            .ticket_system(Arc::clone(&system))
            .ticket_key(key.clone());
        (ctx, system, key, dir)
    }

    fn tool_result_block(id: &str, content: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: content.into(),
            succeeded: true,
        }
    }

    fn outputs_path(dir: &std::path::Path, key: &str, tool_use_id: &str) -> PathBuf {
        dir.join("tickets")
            .join(key)
            .join("outputs")
            .join(format!("{tool_use_id}.txt"))
    }

    #[test]
    fn cap_oversized_result_passes_through_under_cap() {
        let ctx = test_ctx();
        let (outcome, path) = cap_oversized_result(Ok("hello".into()), &ctx, "call-1", 100);
        assert_eq!(outcome.unwrap(), "hello");
        assert!(path.is_none());
    }

    #[test]
    fn cap_oversized_result_replaces_oversized_ok_with_stub() {
        let (ctx, _system, key, dir) = ticket_ctx();
        let (outcome, path) =
            cap_oversized_result(Ok("a".repeat(500)), &ctx, "call-xyz", 100);
        let stub = outcome.unwrap();
        assert!(stub.starts_with("<persisted-output>"));
        assert!(stub.contains("Output too large"));
        assert!(stub.contains("Full output saved to:"));
        assert!(stub.contains("Preview (first"));
        assert!(stub.ends_with("</persisted-output>"));
        let path = path.expect("offload path");
        assert_eq!(path, outputs_path(dir.path(), &key, "call-xyz"));
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body, "a".repeat(500));
    }

    #[test]
    fn cap_oversized_result_passes_through_errs() {
        let ctx = test_ctx();
        let (outcome, path) = cap_oversized_result(
            Err(ToolError::ExecutionFailed {
                tool_name: "tool".into(),
                message: "boom".into(),
            }),
            &ctx,
            "call-1",
            10,
        );
        assert!(matches!(outcome, Err(ToolError::ExecutionFailed { .. })));
        assert!(path.is_none());
    }

    #[test]
    fn cap_oversized_result_returns_raw_when_no_ticket_key() {
        let ctx = test_ctx();
        let payload = "x".repeat(500);
        let (outcome, path) =
            cap_oversized_result(Ok(payload.clone()), &ctx, "call-1", 100);
        assert_eq!(outcome.unwrap(), payload);
        assert!(path.is_none(), "no ticket key means no offload");
    }

    #[test]
    fn cap_aggregate_offloads_largest_first() {
        let (ctx, _system, key, dir) = ticket_ctx();
        // Sizes chosen so the stub's own bytes (~200) don't dominate.
        let small = "a".repeat(40_000);
        let big = "b".repeat(80_000);
        let tiny = "c".repeat(30_000);
        let mut results = vec![
            (tool_result_block("c1", &small), Ok(small.clone()), None),
            (tool_result_block("c2", &big), Ok(big.clone()), None),
            (tool_result_block("c3", &tiny), Ok(tiny.clone()), None),
        ];
        cap_aggregate_outputs(&mut results, &ctx, 100_000);
        // c2 (the largest) was offloaded; the other two stayed inline.
        match &results[1].0 {
            ContentBlock::ToolResult { content, .. } => {
                assert!(content.starts_with("<persisted-output>"));
                assert!(content.contains("Full output saved to:"));
            }
            _ => panic!("expected ToolResult"),
        }
        let big_path = results[1].2.clone().expect("c2 path recorded");
        assert_eq!(big_path, outputs_path(dir.path(), &key, "c2"));
        let body = std::fs::read_to_string(&big_path).unwrap();
        assert_eq!(body, big);

        assert!(matches!(
            &results[0].0,
            ContentBlock::ToolResult { content, .. } if content.len() == 40_000
        ));
        assert!(matches!(
            &results[2].0,
            ContentBlock::ToolResult { content, .. } if content.len() == 30_000
        ));
        assert!(results[0].2.is_none());
        assert!(results[2].2.is_none());
    }

    #[test]
    fn cap_aggregate_stops_when_only_small_results_remain() {
        let (ctx, _system, _key, _dir) = ticket_ctx();
        // Many small results whose total far exceeds the cap, but
        // each is already a stub-marked block. Aggregate should bail
        // rather than spin: stubs are skipped, so no candidates.
        let mut results: Vec<(
            ContentBlock,
            std::result::Result<String, ToolError>,
            Option<PathBuf>,
        )> = (0..5)
            .map(|i| {
                let id = format!("c{i}");
                let stub = format!("<persisted-output>already stubbed {i}</persisted-output>");
                (tool_result_block(&id, &stub), Ok(stub), None)
            })
            .collect();
        let before: Vec<String> = results
            .iter()
            .map(|(b, _, _)| match b {
                ContentBlock::ToolResult { content, .. } => content.clone(),
                _ => String::new(),
            })
            .collect();
        cap_aggregate_outputs(&mut results, &ctx, 10);
        let after: Vec<String> = results
            .iter()
            .map(|(b, _, _)| match b {
                ContentBlock::ToolResult { content, .. } => content.clone(),
                _ => String::new(),
            })
            .collect();
        assert_eq!(before, after, "aggregate must be a no-op when only stubs remain");
    }

    #[test]
    fn format_oversized_tool_result_renders_template() {
        let path = PathBuf::from("/tmp/agentwerk/tickets/TICKET-1/outputs/call-1.txt");
        let stub = format_oversized_tool_result(1_048_576, &path, "preview-body");
        assert!(stub.starts_with("<persisted-output>"));
        assert!(stub.contains("Output too large (1.0 MB)."));
        assert!(stub.contains("Full output saved to: /tmp/agentwerk/tickets/TICKET-1/outputs/call-1.txt"));
        assert!(stub.contains("Preview (first 12 B):"));
        assert!(stub.contains("preview-body"));
        assert!(stub.ends_with("</persisted-output>"));
    }

    #[test]
    fn truncate_preview_snaps_at_last_newline_in_window() {
        let mut content = String::new();
        // Build a payload where the last newline within PREVIEW_CHARS is
        // at byte 1_900.
        content.push_str(&"a".repeat(1_900));
        content.push('\n');
        content.push_str(&"b".repeat(500));
        let preview = truncate_preview(&content);
        assert_eq!(preview.len(), 1_901);
        assert!(preview.ends_with('\n'));
    }

    #[test]
    fn truncate_preview_falls_back_to_utf8_boundary_when_no_newline() {
        let content = "x".repeat(3_000);
        let preview = truncate_preview(&content);
        assert_eq!(preview.len(), PREVIEW_CHARS);
        assert!(content.is_char_boundary(preview.len()));
    }

    #[test]
    fn replace_empty_output_substitutes_placeholder() {
        let outcome: std::result::Result<String, ToolError> = Ok(String::new());
        let outcome = replace_empty_output(outcome, "bash");
        assert_eq!(outcome.unwrap(), "(bash completed with no output)");
    }

    #[test]
    fn replace_empty_output_passes_non_empty_through() {
        let outcome: std::result::Result<String, ToolError> = Ok("hello".into());
        let outcome = replace_empty_output(outcome, "bash");
        assert_eq!(outcome.unwrap(), "hello");
    }

    #[test]
    fn replace_empty_output_passes_errors_through() {
        let outcome: std::result::Result<String, ToolError> = Err(ToolError::ExecutionFailed {
            tool_name: "bash".into(),
            message: "boom".into(),
        });
        let outcome = replace_empty_output(outcome, "bash");
        assert!(matches!(outcome, Err(ToolError::ExecutionFailed { .. })));
    }
}
