# Tool Trait System

## Overview

This sub-plan covers the `Tool` trait, `ToolContext`, `ToolRegistry`, tool orchestration (concurrent/serial partitioning), `ToolBuilder` for closure-based tools, and the `Toolset` trait for dynamic tool groups. After this, tools can be registered, looked up, and executed — but there is no agent loop calling them yet.

## Dependencies

- [Core types](../1-base/types.md): `AgenticError`, `Result`

## Files

```
crates/agent-core/src/tool.rs
```

## Specification

### 7.1 Tool Trait and ToolContext

```rust
use std::path::PathBuf;
use std::sync::Arc;

/// Runtime context passed to Tool::call(). Provides tools with access to
/// the working directory for path resolution.
///
/// Designed as a minimal data bag — no session mutation, no memory, no
/// artifact system. Projects needing richer context can embed their own
/// state in the tool struct at construction time.
#[derive(Clone, Debug)]
pub struct ToolContext {
    /// Working directory for resolving relative paths.
    /// File tools use this instead of `std::env::current_dir()`.
    pub working_directory: PathBuf,
    /// Tool registry reference, used by tool_search. Set by the agent loop.
    pub tool_registry: Option<Arc<ToolRegistry>>,
}

/// This is the canonical definition of ToolDefinition. Increment 1 uses a
/// temporary stub in `provider.rs` that gets replaced by this re-export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,  // JSON Schema object
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

/// Core tool trait. Object-safe via boxed futures.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;
    fn is_read_only(&self) -> bool { false }

    /// Whether this tool should be deferred (name-only definition sent to LLM
    /// until discovered via tool_search). Defaults to false.
    fn should_defer(&self) -> bool { false }

    /// Keywords that help tool_search find this tool. Scored lower than name
    /// matches but higher than description matches. Defaults to empty.
    fn search_hints(&self) -> Vec<String> { Vec::new() }

    fn call(&self, input: serde_json::Value, ctx: &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>;

    fn definition(&self) -> ToolDefinition { ... }
}

/// ToolRegistry stores `Vec<Arc<dyn Tool>>` (not `Box<dyn Tool>`) so it can
/// be cloned. Each `Tool` is wrapped in `Arc` on registration.
pub struct ToolRegistry { /* Vec<Arc<dyn Tool>> */ }
impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: impl Tool + 'static); // wraps in Arc internally
    pub fn definitions(&self) -> Vec<ToolDefinition>;
    pub fn get(&self, name: &str) -> Option<&dyn Tool>; // returns reference through Arc

    /// Full definitions for non-deferred tools + discovered deferred tools.
    /// Undiscovered deferred tools get name-only definitions.
    pub fn definitions_filtered(&self, discovered: &HashSet<String>) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| {
            if !tool.should_defer() || discovered.contains(tool.name()) {
                tool.definition()
            } else {
                ToolDefinition {
                    name: tool.name().to_string(),
                    description: String::new(),
                    input_schema: serde_json::json!({}),
                }
            }
        }).collect()
    }

    /// Search tools by query. Returns matches sorted by score (highest first).
    ///
    /// Scoring: exact name = 100, name segment = +10, search hint = +4,
    /// description substring = +2. Only returns tools with score > 0.
    pub fn search(&self, query: &str) -> Vec<ToolSearchResult> {
        let query_lower = query.to_lowercase();
        let query_parts: Vec<&str> = query_lower.split_whitespace().collect();
        let mut results: Vec<ToolSearchResult> = Vec::new();

        for tool in &self.tools {
            let name = tool.name().to_lowercase();
            let score = if name == query_lower {
                100  // exact match
            } else {
                let mut s: u32 = 0;
                let name_segments: Vec<&str> = name.split('_').collect();
                for part in &query_parts {
                    for seg in &name_segments {
                        if seg.contains(part) { s += 10; }
                    }
                    for hint in tool.search_hints() {
                        if hint.to_lowercase().contains(part) { s += 4; }
                    }
                    if tool.description().to_lowercase().contains(part) { s += 2; }
                }
                s
            };
            if score > 0 {
                results.push(ToolSearchResult { definition: tool.definition(), score });
            }
        }
        results.sort_by(|a, b| b.score.cmp(&a.score));
        results
    }

    /// True if any registered tool has should_defer() == true.
    pub fn has_deferred_tools(&self) -> bool {
        self.tools.iter().any(|t| t.should_defer())
    }
}

#[derive(Debug, Clone)]
pub struct ToolSearchResult {
    pub definition: ToolDefinition,
    pub score: u32,
}
```

**Example: Implementing a custom tool (struct-based)**

```rust
/// A tool that fetches the current weather for a city.
struct WeatherTool {
    api_key: String,
}

impl Tool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> &str { "Get the current weather for a city." }
    fn is_read_only(&self) -> bool { true }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'"
                }
            },
            "required": ["city"]
        })
    }

    fn call(&self, input: serde_json::Value, _ctx: &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
    {
        Box::pin(async move {
            let city = input.get("city")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AgenticError::Tool {
                    tool_name: "get_weather".into(),
                    message: "Missing required field: city".into(),
                })?;

            // Call weather API (user-provided HTTP client)
            let weather = fetch_weather(&self.api_key, city).await?;
            Ok(ToolResult { content: weather, is_error: false })
        })
    }
}

// Register the custom tool on an agent
let agent = AgentBuilder::new()
    .name("weather_bot")
    .model("claude-haiku-4-5-20241022")
    .system_prompt("You help users check the weather.")
    .tool(WeatherTool { api_key: "sk-...".into() })
    .build()?;
```

### 7.2 Tool Orchestration

Tool calls are partitioned into batches for execution. Consecutive read-only tools are grouped into a concurrent batch and executed via a `JoinSet` with a `Semaphore` limiting concurrency to 10. Each write tool forms its own serial batch and is awaited sequentially.

```rust
const MAX_TOOL_CONCURRENCY: usize = 10;

/// Execute tool calls and return ContentBlock results.
///
/// Each `ToolResult` is converted to `ContentBlock::ToolResult` using the `id`
/// field from the corresponding `ToolCall` as `tool_use_id`. The mapping is:
/// `ContentBlock::ToolResult { tool_use_id: call.id.clone(), content: result.content, is_error: result.is_error }`
pub async fn execute_tool_calls(
    calls: &[ToolCall], registry: &ToolRegistry, ctx: &ToolContext,
) -> Vec<ContentBlock>;

fn partition_tool_calls(calls: &[ToolCall], registry: &ToolRegistry) -> Vec<ToolBatch>;

enum ToolBatch {
    Concurrent(Vec<ToolCall>),  // consecutive read-only tools → JoinSet + Semaphore(10)
    Serial(ToolCall),           // each write tool → sequential await
}
```

Each tool in a batch receives the same `&ToolContext`. The context is immutable — tools cannot modify it during execution.

### 7.3 ToolBuilder — Build a Tool from Parts

A single builder that constructs a `Tool` from name, description, schema, and a handler closure. Covers simple tools where a full struct impl is unnecessary.

```rust
/// Type alias for the handler closure signature.
type ToolHandler = Box<
    dyn Fn(serde_json::Value, &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
    + Send + Sync,
>;

pub struct ToolBuilder {
    name: String,
    description: String,
    schema: serde_json::Value,
    read_only: bool,
    should_defer: bool,
    search_hints: Vec<String>,
    handler: Option<ToolHandler>,
}

impl ToolBuilder {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema: json!({"type": "object", "properties": {}}),
            read_only: false,
            should_defer: false,
            search_hints: Vec::new(),
            handler: None,
        }
    }

    pub fn schema(mut self, schema: serde_json::Value) -> Self {
        self.schema = schema; self
    }

    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only; self
    }

    pub fn should_defer(mut self, defer: bool) -> Self {
        self.should_defer = defer; self
    }

    pub fn search_hints(mut self, hints: Vec<String>) -> Self {
        self.search_hints = hints; self
    }

    pub fn handler<F>(mut self, f: F) -> Self
    where
        F: Fn(serde_json::Value, &ToolContext)
            -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
            + Send + Sync + 'static,
    {
        self.handler = Some(Box::new(f)); self
    }

    /// Build the tool. Panics if no handler was set.
    pub fn build(self) -> impl Tool {
        BuiltTool {
            name: self.name,
            description: self.description,
            schema: self.schema,
            read_only: self.read_only,
            should_defer: self.should_defer,
            search_hints: self.search_hints,
            handler: self.handler.expect("ToolBuilder: handler is required"),
        }
    }
}

/// Internal struct returned by ToolBuilder::build(). Not public — callers
/// interact with it through the Tool trait.
struct BuiltTool {
    name: String,
    description: String,
    schema: serde_json::Value,
    read_only: bool,
    should_defer: bool,
    search_hints: Vec<String>,
    handler: ToolHandler,
}

impl Tool for BuiltTool {
    fn name(&self) -> &str { &self.name }
    fn description(&self) -> &str { &self.description }
    fn input_schema(&self) -> serde_json::Value { self.schema.clone() }
    fn is_read_only(&self) -> bool { self.read_only }
    fn should_defer(&self) -> bool { self.should_defer }
    fn search_hints(&self) -> Vec<String> { self.search_hints.clone() }

    fn call(&self, input: serde_json::Value, ctx: &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
    {
        (self.handler)(input, ctx)
    }
}
```

**Example: Building a tool with `ToolBuilder`**

```rust
use std::time::{SystemTime, UNIX_EPOCH};

let tool = ToolBuilder::new("get_timestamp", "Returns the current UNIX timestamp.")
    .read_only(true)
    .schema(json!({
        "type": "object",
        "properties": {},
    }))
    .handler(|_input, _ctx| {
        Box::pin(async move {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH).unwrap().as_secs();
            Ok(ToolResult { content: ts.to_string(), is_error: false })
        })
    })
    .build();

let agent = AgentBuilder::new()
    .name("timestamp_bot")
    .model("claude-haiku-4-5-20241022")
    .system_prompt("You help users check timestamps.")
    .tool(tool)
    .build()?;
```

Two ways to create tools — pick whichever fits:
- **`ToolBuilder`** — quick, no boilerplate, good for simple stateless tools
- **`impl Tool for MyStruct`** — full control, good for tools with their own state (API keys, config, etc.)

### 7.4 Toolset — Dynamic Tool Groups

A collection of tools that can be resolved dynamically. Useful for MCP integrations, feature-gated tools, and plugin systems.

```rust
/// A collection of tools that can be resolved dynamically.
pub trait Toolset: Send + Sync {
    fn tools(&self) -> Vec<Box<dyn Tool>>;
}
```

`AgentBuilder` gains a `toolset()` method that registers all tools from a `Toolset`:

```rust
impl AgentBuilder {
    /// Register all tools from a toolset.
    pub fn toolset(mut self, ts: impl Toolset + 'static) -> Self {
        for tool in ts.tools() {
            self = self.tool_boxed(tool);
        }
        self
    }

    /// Register a boxed tool (used internally by toolset()).
    pub fn tool_boxed(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }
}
```

**Example: Composing toolsets**

```rust
// A downstream project can define its own toolset
struct MyProjectToolset;

impl Toolset for MyProjectToolset {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(FileStatsTool),
            Box::new(CustomLintTool::new()),
        ]
    }
}

// Compose built-in + custom tools
let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are a helpful assistant.")
    .toolset(BuiltinToolset)      // all built-in tools
    .toolset(MyProjectToolset)    // project-specific tools
    .build()?;
```

### 7.6 Convenience Registration

See Section 7.4 — `BuiltinToolset` implements `Toolset`. Use it via the builder:

```rust
let agent = AgentBuilder::new()
    .toolset(BuiltinToolset)
    .build()?;
```

### MockTool (Test Infrastructure, in `testutil.rs`)

```rust
/// A mock tool that records calls and returns a fixed result.
pub struct MockTool {
    pub name: String,
    pub read_only: bool,
    pub result: String,
    pub is_error: bool,
    pub delay: Option<Duration>,
    pub calls: Mutex<Vec<serde_json::Value>>,
}

impl MockTool {
    pub fn new(name: &str, read_only: bool, result: &str) -> Self;
    pub fn call_count(&self) -> usize;
    pub fn with_delay(name: &str, read_only: bool, result: &str, delay: Duration) -> Self;
    pub fn failing(name: &str, message: &str) -> Self;
    pub fn last_input(&self) -> Option<serde_json::Value>;
}

impl Tool for MockTool { ... }
```

### Test Tool Context Helpers (in `testutil.rs`)

```rust
/// Build a minimal ToolContext for testing.
pub fn test_tool_context() -> ToolContext {
    ToolContext {
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        tool_registry: None,
    }
}

/// Build a ToolContext with a registry for testing tool_search scenarios.
pub fn test_tool_context_with_registry(registry: Arc<ToolRegistry>) -> ToolContext {
    ToolContext {
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        tool_registry: Some(registry),
    }
}
```

## Work Items

1. **`tool.rs`** — Spec Sections 7.1, 7.2, 7.3, 7.4
   - `ToolContext` struct with `working_directory` and `tool_registry`
   - `ToolDefinition`, `ToolResult`, `ToolCall`, `ToolSearchResult` structs
   - `Tool` trait (object-safe: `name`, `description`, `input_schema`, `is_read_only`, `should_defer`, `search_hints`, `call` with `&ToolContext`)
   - `ToolBuilder` + internal `BuiltTool` struct (with `should_defer` and `search_hints` fields)
   - `Toolset` trait
   - `ToolRegistry` — register, lookup by name, list definitions, `definitions_filtered()`, `search()`, `has_deferred_tools()`
   - `execute_tool_calls()` — partition into `ToolBatch::Concurrent` (consecutive read-only) and `ToolBatch::Serial` (each write tool), execute with `JoinSet` + `Semaphore(10)` for concurrent batches, accepts `&ToolContext`
   - `partition_tool_calls()` function

2. **`testutil.rs`** (additions) — MockTool, test_tool_context, test_tool_context_with_registry
   - `MockTool` — configurable name, read_only, result, delay, error; call recording with `call_count()` and `last_input()`
   - `test_tool_context()`, `test_tool_context_with_registry()` — minimal `ToolContext` builders

3. **`lib.rs`** — `BuiltinToolset` implementing `Toolset` (registers all tools above + placeholders for spawn_agent and task tools from later increments)

## Tests

Use `tempfile` crate in dev-dependencies.

### `tool.rs` Tests

Shared helpers: `MockTool::new(name, read_only, response) -> MockTool` — mock tool returning fixed content. `MockTool::with_delay(name, read_only, response, delay) -> MockTool` — mock tool with configurable latency. `MockTool::failing(name, error_msg) -> MockTool` — mock tool returning `is_error: true`.

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `tool_context_fields_accessible` | Direct | `ToolContext` field (`working_directory`) is accessible |
| `registry_register_and_lookup` | Direct | Register a tool, look up by name (found), look up unknown (None) |
| `registry_definitions_lists_all` | Direct | Register 2 tools, `definitions()` returns both names |
| `tool_builder_basic` | Direct | Build tool with handler, call it, verify content and is_error=false |
| `tool_builder_missing_handler_panics` | `#[should_panic]` | Building without a handler panics |
| `tool_builder_read_only_default_false` | Direct | Default `is_read_only()` returns false |
| `partition_batching_table` | Table-driven (8 cases) | Batch patterns: all-read→C, all-write→S×N, mixed RWRR→C-S-C, etc. |
| `concurrent_tools_faster_than_serial` | Timing | 3 read-only tools (100ms each) complete in <250ms concurrently |
| `tool_error_result_has_is_error_true` | Direct | `MockTool::failing` returns `is_error: true` with error message |
| `builtin_toolset_returns_all_tools` | Direct | `BuiltinToolset.tools()` contains read_file, write_file, edit_file, glob, grep, list_directory, bash, tool_search |
| `tool_should_defer_default_false` | Direct | Default `should_defer()` returns false |
| `tool_search_hints_default_empty` | Direct | Default `search_hints()` returns empty Vec |
| `registry_definitions_filtered_includes_non_deferred` | Direct | Non-deferred tools always get full definitions |
| `registry_definitions_filtered_deferred_undiscovered` | Direct | Deferred undiscovered tool gets name-only (empty description + schema) |
| `registry_definitions_filtered_deferred_discovered` | Direct | Deferred discovered tool gets full definition |
| `registry_has_deferred_tools` | Direct | true with deferred tool, false without |
| `registry_search_exact_name` | Direct | Exact name match scores 100, sorted first |
| `registry_search_name_segments` | Direct | `read_file` found by query `read` (score +10) |
| `registry_search_hints_scored` | Direct | Tool with hint `"web"` found by query `web` (score +4) |
| `registry_search_description_match` | Direct | Description containing `weather` found by query `weather` (score +2) |
| `registry_search_no_match_excluded` | Direct | Unrelated tool not in results |
| `registry_search_sorted_by_score` | Direct | Multiple matches returned highest-score-first |
| `tool_builder_defer_and_hints` | Direct | `should_defer(true)` and `search_hints(...)` produce correct values |

## Done Criteria

- `cargo build -p agent-core` compiles (tool.rs module)
- All tool.rs tests pass
- Can register tools, call `execute_tool_calls()` with mock tool calls, and get results back
- Concurrent batch execution is faster than serial for 3+ read-only tools
- `MockTool` records calls and supports delay/error simulation
