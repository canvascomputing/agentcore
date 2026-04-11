# Core Types, Providers, Cost Tracking

## Overview

Foundation types that everything else depends on. After this sub-plan, the crate compiles and you can make LLM API calls and track costs — but there is no agent loop yet.

## Dependencies

**No dependencies.** All other increments depend on this one.

## Files

```
crates/agent-core/Cargo.toml
crates/agent-core/src/lib.rs
crates/agent-core/src/error.rs
crates/agent-core/src/message.rs
crates/agent-core/src/provider.rs
crates/agent-core/src/cost.rs
```

## Specification

### 2.1 Error (`error.rs`)

```rust
pub type Result<T> = std::result::Result<T, AgenticError>;

#[derive(Debug)]
pub enum AgenticError {
    Api { message: String, status: Option<u16>, retryable: bool },
    Tool { tool_name: String, message: String },
    Io(std::io::Error),
    Json(serde_json::Error),
    Aborted,
    MaxTurnsExceeded(u32),
    BudgetExceeded { spent: f64, limit: f64 },
    ContextOverflow { token_count: u64, limit: u64 },
    SchemaValidation { path: String, message: String },
    SchemaRetryExhausted { retries: u32 },
    Other(String),
}
// Manual Display + Error impls, From<io::Error>, From<serde_json::Error>
```

#### Testing `error.rs`

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `display_api_error` | Direct | ApiError variant displays status code and message |
| `from_io_error` | Direct | `std::io::Error` converts to `AgenticError::Io` via `From` |
| `from_json_error` | Direct | `serde_json::Error` converts to `AgenticError::Json` via `From` |
| `budget_exceeded_shows_amounts` | Direct | BudgetExceeded variant displays spent and limit amounts |
| `all_variants_display_non_empty` | Direct | Every AgenticError variant produces a non-empty Display string |

---

### 2.2 Messages (`message.rs`)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: Vec<ContentBlock> },
    #[serde(rename = "assistant")]
    Assistant { content: Vec<ContentBlock> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: serde_json::Value },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: String, #[serde(default)] is_error: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason { EndTurn, ToolUse, MaxTokens }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub model: String,
}
```

#### Testing `message.rs`

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `message_serde_round_trip` | Direct | User message with Text block survives serialize→deserialize |
| `tool_use_block_serde` | Direct | ToolUse ContentBlock round-trips with id, name, input intact |
| `tool_result_is_error_defaults_false` | Direct | ToolResult with is_error omitted defaults to false |
| `usage_add_accumulates` | Direct | Usage::add() sums all four token fields correctly |

---

### 2.3 LLM Provider (`provider.rs`)

Transport-agnostic. The `AnthropicProvider` accepts an injectable HTTP transport function so the core crate carries zero HTTP dependencies. Users supply their own reqwest/hyper/ureq call.

```rust
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    /// Force the model to use a specific tool (structured output enforcement).
    pub tool_choice: Option<ToolChoice>,
}

/// Controls which tool the model must use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    /// Model chooses freely (default when None).
    Auto,
    /// Force the model to call this specific tool.
    Specific { name: String },
}

/// Core LLM provider trait. Object-safe via boxed futures.
pub trait LlmProvider: Send + Sync {
    fn complete(&self, request: CompletionRequest)
        -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>>;
}

/// Injectable HTTP transport: async fn(url, headers, body) -> response_json
pub type HttpTransport = Box<
    dyn Fn(&str, Vec<(&str, String)>, serde_json::Value)
        -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send>>
        + Send + Sync,
>;

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,       // default: "https://api.anthropic.com"
    transport: HttpTransport,
}

impl AnthropicProvider {
    pub fn new(api_key: String, transport: HttpTransport) -> Self;
    pub fn base_url(self, url: String) -> Self;
    // Builds Anthropic Messages API JSON body from CompletionRequest,
    // calls transport(url, headers, body), parses response into ModelResponse.
    // Headers: x-api-key, anthropic-version, content-type
    // ToolChoice::Auto → { "type": "auto" }
    // ToolChoice::Specific { name } → { "type": "tool", "name": name }
}
impl LlmProvider for AnthropicProvider { ... }
```

The transport injection replaces the traditional multi-SDK approach (one SDK per cloud provider) with a single code path. Each provider only knows how to serialize requests and deserialize responses — the HTTP layer is the caller's concern.

**Note on `ToolDefinition`:** `CompletionRequest` references `ToolDefinition` which is fully defined in `tool.rs` (Increment 3, [`../3-tools/traits.md`](../3-tools/traits.md)). During Increment 1 development, define `ToolDefinition` as a temporary stub in `provider.rs`:

```rust
/// Temporary stub — replaced by re-export from `tool.rs` when Increment 2 lands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,  // JSON Schema object
}
```

When Increment 2 is complete, delete this stub and re-export `ToolDefinition` from `tool.rs` instead.

**Example: Creating a provider with reqwest**

```rust
// The transport closure wraps any HTTP client. Here: reqwest.
let transport: HttpTransport = Box::new(|url, headers, body| {
    Box::pin(async move {
        let client = reqwest::Client::new();
        let mut req = client.post(url).json(&body);
        for (key, value) in headers {
            req = req.header(key, value);
        }
        let resp = req.send().await.map_err(|e| AgenticError::Other(e.to_string()))?;
        let json: serde_json::Value = resp.json().await
            .map_err(|e| AgenticError::Other(e.to_string()))?;
        Ok(json)
    })
});

// Anthropic direct
let anthropic = AnthropicProvider::new(
    std::env::var("ANTHROPIC_API_KEY").unwrap(),
    transport,
);

// Or LiteLLM proxy (same transport, different provider)
let litellm = LiteLlmProvider::new("any-key".into(), transport)
    .base_url("http://localhost:4000".into());

// Both implement LlmProvider — pass to AgentBuilder or InvocationContext
let provider: Arc<dyn LlmProvider> = Arc::new(anthropic);
```

#### LiteLLM / OpenAI-Compatible Provider

LiteLLM is a proxy that exposes 100+ LLM providers (OpenAI, Azure, Bedrock, Vertex, Ollama, etc.) behind a single OpenAI-compatible API. Supporting it unlocks access to all those providers with zero additional code.

```rust
/// Provider for LiteLLM and any OpenAI-compatible API.
/// Uses the OpenAI chat completions format:
///   POST {base_url}/v1/chat/completions
///   Authorization: Bearer {api_key}
///
/// Key differences from Anthropic format:
///   - Messages use `role` + `content` (string or array), no serde tag dispatch
///   - System prompt is a message with role "system" (not a top-level field)
///   - Tool definitions wrapped in { type: "function", function: { name, description, parameters } }
///   - Tool calls in response: choices[0].message.tool_calls[{ id, function: { name, arguments } }]
///   - Tool results sent as messages with role "tool" and tool_call_id
///   - Stop reason: finish_reason = "stop" | "tool_calls" | "length"
///   - Usage: { prompt_tokens, completion_tokens, total_tokens } (no cache breakdown)
pub struct LiteLlmProvider {
    api_key: String,
    base_url: String,       // default: "http://localhost:4000" (local LiteLLM proxy)
    transport: HttpTransport,
}

impl LiteLlmProvider {
    pub fn new(api_key: String, transport: HttpTransport) -> Self;
    pub fn base_url(self, url: String) -> Self;

    // === Request serialization (CompletionRequest → OpenAI JSON) ===
    //
    // 1. System prompt → { "role": "system", "content": system_prompt }
    //    Prepended as first message (OpenAI puts system in messages, not top-level).
    //
    // 2. User/Assistant messages → translate ContentBlock variants:
    //    - ContentBlock::Text → { "type": "text", "text": "..." }
    //    - ContentBlock::ToolUse → NOT embedded in content; collected separately into
    //      assistant message's "tool_calls" array:
    //      { "id": id, "type": "function", "function": { "name": name, "arguments": input_json_string } }
    //      Note: arguments is a JSON *string*, not an object.
    //    - ContentBlock::ToolResult → separate message with role "tool":
    //      { "role": "tool", "tool_call_id": id, "content": result_string }
    //
    // 3. Tool definitions → "tools" array, each entry:
    //    { "type": "function", "function": { "name": "...", "description": "...", "parameters": {json_schema} } }
    //
    // 4. tool_choice:
    //    ToolChoice::Auto → { "type": "auto" }  (or omit)
    //    ToolChoice::Specific { name } → { "type": "function", "function": { "name": name } }
    //
    // 5. Other fields: "model", "max_tokens" mapped directly.
    //
    // === Response deserialization (OpenAI JSON → ModelResponse) ===
    //
    // Response shape:
    //   {
    //     "choices": [{
    //       "message": {
    //         "role": "assistant",
    //         "content": "text or null",
    //         "tool_calls": [{ "id": "...", "type": "function",
    //                          "function": { "name": "...", "arguments": "json_string" } }]
    //       },
    //       "finish_reason": "stop" | "tool_calls" | "length"
    //     }],
    //     "usage": { "prompt_tokens": N, "completion_tokens": N, "total_tokens": N }
    //   }
    //
    // Mapping:
    //   finish_reason "stop"       → StopReason::EndTurn
    //   finish_reason "tool_calls" → StopReason::ToolUse
    //   finish_reason "length"     → StopReason::MaxTokens
    //
    //   choices[0].message.content      → ContentBlock::Text (if non-null)
    //   choices[0].message.tool_calls[] → ContentBlock::ToolUse per entry
    //     function.arguments is a JSON string → parse to serde_json::Value for ToolUse.input
    //
    //   usage.prompt_tokens      → Usage.input_tokens
    //   usage.completion_tokens  → Usage.output_tokens
    //   usage.cache_read_input_tokens and cache_creation_input_tokens → 0
    //     (OpenAI format doesn't report cache tokens; some LiteLLM-proxied
    //      providers may include these as extension fields — parse if present, default 0)
    //
    // Headers sent:
    //   Authorization: Bearer {api_key}
    //   Content-Type: application/json
}

impl LlmProvider for LiteLlmProvider { ... }
```

#### When to use which provider

| Provider | Use when... |
|----------|-------------|
| `AnthropicProvider` | Direct Anthropic API access. Full cache token reporting. Native tool_use format. |
| `LiteLlmProvider` | Using LiteLLM proxy, OpenAI, Azure OpenAI, Ollama, or any OpenAI-compatible endpoint. Access 100+ models through a single provider. |

Both providers use the same `HttpTransport` injection, so switching between them requires no changes to the HTTP layer — only the serialization/deserialization logic differs.

#### Testing `provider.rs`

Tests use a capture transport that records the JSON body sent and returns canned JSON responses. Shared helper: `capture_transport(response) -> (HttpTransport, Arc<Mutex<Vec<Value>>>)`.

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `anthropic_serializes_system_prompt_as_top_level` | Direct | System prompt goes to top-level `system` field, not in messages |
| `litellm_serializes_system_prompt_as_message` | Direct | System prompt becomes a system-role message in messages array |
| `litellm_parses_tool_calls_from_response` | Direct | OpenAI-format tool_calls parsed into ContentBlock::ToolUse |
| `litellm_finish_reason_mapping` | Table-driven | "stop"→EndTurn, "tool_calls"→ToolUse, "length"→MaxTokens |
| `transport_error_propagated` | Direct | HttpTransport returning Err surfaces as AgenticError |
| `anthropic_empty_content_array` | Direct | Empty content array in response returns empty text |
| `litellm_null_content_field` | Direct | null content field treated as empty string |
| `anthropic_tool_choice_serialization` | Direct | ToolChoice::Auto→"auto", ToolChoice::Specific→{"type":"tool","name":...} |
| `litellm_tool_choice_serialization` | Direct | ToolChoice mapped to OpenAI format |

---

### 8. Cost Tracking (`cost.rs`)

Thread-safe via `Arc<Mutex<_>>` so sub-agents share with parent.

```rust
#[derive(Debug, Clone)]
pub struct ModelCosts {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_usd: f64,
    pub request_count: u64,
}

#[derive(Clone)]
pub struct CostTracker { /* Arc<Mutex<CostTrackerInner>> */ }

impl CostTracker {
    pub fn new() -> Self;
    pub fn model_pricing(&self, model: &str, costs: ModelCosts);
    pub fn record_usage(&self, model: &str, usage: &Usage);
    pub fn record_tool_calls(&self, count: u64);
    pub fn total_cost_usd(&self) -> f64;
    pub fn total_requests(&self) -> u64;
    pub fn total_tool_calls(&self) -> u64;
    pub fn model_usage(&self) -> HashMap<String, ModelUsage>;
    pub fn summary(&self) -> String;  // formatted multi-line summary
}
```

Default pricing for Claude Haiku 4.5, Sonnet 4, Opus 4 included. Users can add pricing for custom models via `model_pricing()`.

#### Testing `cost.rs`

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `empty_tracker_zero_cost` | Direct | New CostTracker reports $0.00 |
| `record_usage_accumulates` | Direct | Two record_usage calls sum correctly |
| `multiple_models_tracked_separately` | Direct | Different models tracked independently in per-model breakdown |
| `custom_pricing_applied` | Direct | model_pricing() overrides default per-token costs |
| `tool_calls_tracked` | Direct | record_tool_calls() increments tool call counter |
| `summary_contains_model_name` | Direct | summary() string includes the model name |
| `concurrent_recording_thread_safe` | Concurrency | 10 threads × record_usage: total matches expected sum |

**Example: Tracking costs across agents**

```rust
let tracker = CostTracker::new();

// Add pricing for a custom/self-hosted model
tracker.model_pricing("my-local-llama", ModelCosts {
    input_per_million: 0.0,
    output_per_million: 0.0,
    cache_read_per_million: 0.0,
    cache_write_per_million: 0.0,
});

// Record usage after each LLM call (done automatically by the agent loop)
tracker.record_usage("claude-sonnet-4-20250514", &Usage {
    input_tokens: 2500,
    output_tokens: 800,
    cache_read_input_tokens: 1200,
    cache_creation_input_tokens: 0,
});

// The tracker is Clone (via Arc<Mutex>), so sub-agents share the same instance
let child_tracker = tracker.clone();
// ... sub-agent records usage on child_tracker, parent sees it too ...

// Query totals
println!("Total cost: ${:.4}", tracker.total_cost_usd());
println!("Total API calls: {}", tracker.total_requests());

// Per-model breakdown
for (model, usage) in tracker.model_usage() {
    println!("{}: {} input, {} output (${:.4})",
        model, usage.input_tokens, usage.output_tokens, usage.cost_usd);
}

// Formatted summary (suitable for CLI output)
println!("{}", tracker.summary());
// Output:
//   Total cost:            $0.0123
//   claude-sonnet-4:  2.5k input, 800 output, 1.2k cache read ($0.0123)
```

---

### MockProvider (Test Infrastructure)

Pre-populated response queue with request tracking. Lives in `testutil.rs`, compiled only under `#[cfg(test)]`.

```rust
// crates/agent-core/src/testutil.rs (compiled only in #[cfg(test)])

/// A mock LLM provider that returns pre-configured responses in order.
/// Tracks all requests for assertions.
pub struct MockProvider {
    responses: Mutex<VecDeque<ModelResponse>>,
    pub requests: Mutex<Vec<CompletionRequest>>,
}

impl MockProvider {
    /// Create with a queue of responses returned in FIFO order.
    pub fn new(responses: Vec<ModelResponse>) -> Self;

    /// Convenience: single text response with end_turn.
    pub fn text(text: &str) -> Self;

    /// Convenience: tool_use response followed by end_turn response.
    pub fn tool_then_text(tool_name: &str, input: serde_json::Value, final_text: &str) -> Self;

    /// Zero responses — `.complete()` returns error immediately.
    pub fn empty() -> Self;

    /// Always returns the given error.
    pub fn error(err: AgenticError) -> Self;

    /// Returns a StructuredOutput tool_use response, then a text response.
    pub fn structured_output(input: serde_json::Value, final_text: &str) -> Self;

    /// Number of requests received.
    pub fn request_count(&self) -> usize;

    /// The most recent request, if any.
    pub fn last_request(&self) -> Option<CompletionRequest>;

    /// Extract system prompts from all recorded requests.
    pub fn system_prompts(&self) -> Vec<String>;
}

impl LlmProvider for MockProvider { ... }
```

#### Response Builders

```rust
/// Build a simple text-only ModelResponse.
pub fn text_response(text: &str) -> ModelResponse;

/// Build a tool_use ModelResponse.
pub fn tool_response(tool_name: &str, id: &str, input: serde_json::Value) -> ModelResponse;
```

---

### `lib.rs` — Public Re-exports (Increment 1 subset)

```rust
// LLM providers (Section 2)
pub use provider::{LlmProvider, AnthropicProvider, LiteLlmProvider, CompletionRequest, HttpTransport, ToolChoice};

// Messages (Section 2)
pub use message::{Message, ContentBlock, ModelResponse, StopReason, Usage};

// Cost tracking (Section 8)
pub use cost::{CostTracker, ModelCosts, ModelUsage};

// Errors (Section 2)
pub use error::{AgenticError, Result};
```

### Dependencies

**agent-core Cargo.toml:**

```toml
[dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "fs", "io-util"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
tempfile = "3"
tokio = { version = "1", features = ["test-util", "macros", "rt-multi-thread"] }
```

Stdlib alternatives for common crates:

| Need | Approach |
|------|----------|
| Async traits | Boxed futures for object safety (Rust 1.75+ native `async fn` in traits where possible) |
| Error handling | Manual `Display`/`Error` impls |
| ID generation | `std::time::SystemTime` nanos + `std::hash::RandomState` |
| HTTP transport | Injectable closure (`HttpTransport` type) — users supply their own reqwest/hyper/ureq call |
| Observability | `Event` callback — users wire their own logging |
| Cancellation | `Arc<AtomicBool>` checked at loop boundaries |
| File locking | `libc::flock` on Unix, `LockFileEx` on Windows (advisory locks with retry) |

## Work Items

1. **`error.rs`** — Spec Section 2.1
   - `AgenticError` enum with all variants
   - `Result<T>` type alias
   - `Display`, `Error` impls, `From<io::Error>`, `From<serde_json::Error>`

2. **`message.rs`** — Spec Section 2.2
   - `Message` enum (System, User, Assistant) with serde tags
   - `ContentBlock` enum (Text, ToolUse, ToolResult) with serde tags
   - `StopReason`, `Usage`, `ModelResponse`
   - `Usage::add()` for accumulation
   - `Usage::default()`

3. **`provider.rs`** — Spec Section 2.3
   - `CompletionRequest`, `ToolChoice` structs
   - `LlmProvider` trait (object-safe via boxed futures)
   - `HttpTransport` type alias
   - `AnthropicProvider` — serialize `CompletionRequest` to Anthropic Messages API JSON, deserialize response to `ModelResponse`
   - `LiteLlmProvider` — serialize to OpenAI chat completions format, deserialize response (see detailed field mapping in spec Section 2.3)
   - Both providers: `.new()`, `.base_url()`

4. **`cost.rs`** — Spec Section 8
   - `ModelCosts`, `ModelUsage` structs
   - `CostTracker` with `Arc<Mutex<CostTrackerInner>>`
   - Default pricing for Claude Haiku 4.5, Sonnet 4, Opus 4
   - `model_pricing()`, `record_usage()`, `total_cost_usd()`, `model_usage()`, `summary()`

5. **`testutil.rs`** — MockProvider, response builders
   - `MockProvider` — pre-populated response queue, request tracking, factory methods (`text`, `tool_then_text`, `empty`, `error`, `structured_output`), query methods (`request_count`, `last_request`, `system_prompts`)
   - Response builders: `text_response()`, `tool_response()`

6. **`lib.rs`** — Re-export all public types from above modules

## Tests

Test recipes are provided in the Specification sections above and summarized below by module.

### `error.rs` Tests (Section 2.1)

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `display_api_error` | Direct | API error Display includes message and status |
| `from_io_error` | Direct | `From<io::Error>` produces `AgenticError::Io` |
| `from_json_error` | Direct | `From<serde_json::Error>` produces `AgenticError::Json` |
| `budget_exceeded_shows_amounts` | Direct | Display output includes spent and limit amounts |
| `all_variants_display_non_empty` | Table-driven | Iterates all 11 `AgenticError` variants, verifies non-empty Display |

### `message.rs` Tests (Section 2.2)

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `message_serde_round_trip` | Direct | User message serializes with `"role":"user"` and deserializes back |
| `tool_use_block_serde` | Direct | ToolUse block serializes with `"type":"tool_use"` and round-trips |
| `tool_result_is_error_defaults_false` | Direct | Missing `is_error` defaults to false via `#[serde(default)]` |
| `usage_add_accumulates` | Direct | `Usage::add()` correctly sums all token fields |

### `provider.rs` Tests (Section 2.3)

Shared helper: `capture_transport(response) -> (HttpTransport, Arc<Mutex<Vec<Value>>>)` — records request bodies and returns canned responses.

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `anthropic_serializes_system_prompt_as_top_level` | Direct | System prompt is top-level field (not in messages array) |
| `litellm_serializes_system_prompt_as_message` | Direct | System prompt is first message with role "system" |
| `litellm_parses_tool_calls_from_response` | Direct | OpenAI-format tool_calls parsed into `ContentBlock::ToolUse` |
| `litellm_finish_reason_mapping` | Table-driven | 3 cases: "stop"→EndTurn, "tool_calls"→ToolUse, "length"→MaxTokens |
| `transport_error_propagated` | Direct | Transport closure returning Err propagates through provider |
| `anthropic_empty_content_array` | Direct | Empty content array in response produces empty content vec |
| `litellm_null_content_field` | Direct | null content field produces no text blocks |
| `anthropic_tool_choice_serialization` | Direct | `ToolChoice::Specific` serialized as `{ "type": "tool", "name": "..." }` |
| `litellm_tool_choice_serialization` | Direct | `ToolChoice::Specific` serialized as `{ "type": "function", "function": { "name": "..." } }` |

### `cost.rs` Tests (Section 8)

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `empty_tracker_zero_cost` | Direct | New tracker has zero cost, requests, and tool calls |
| `record_usage_accumulates` | Direct | 2 calls on same model: tokens and request count summed |
| `multiple_models_tracked_separately` | Direct | 2 models tracked in separate `ModelUsage` entries |
| `custom_pricing_applied` | Direct | Custom pricing via `model_pricing()`: 1M tokens at $1/M = $1 |
| `tool_calls_tracked` | Direct | `record_tool_calls()` increments correctly across calls |
| `summary_contains_model_name` | Direct | `summary()` output includes model identifier |
| `concurrent_recording_thread_safe` | Concurrency | 10 tokio tasks × 100 recordings: verify total = 1000 requests |

## Done Criteria

- `cargo build -p agent-core` compiles
- `cargo test -p agent-core` passes all tests above
- Can create an `AnthropicProvider`, call `.complete()` with a mock transport, and get a `ModelResponse` back
- `MockProvider` returns queued responses in FIFO order and records all requests
- Response builders produce valid `ModelResponse` values
