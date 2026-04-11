# Data Flows, Concurrency, and Public API

## Overview

This sub-plan documents the key data flows through the system, the concurrency model, and the public API surface for `agent_core`.

## Dependencies

- All previous increments (this is a summary/integration document)

## Files

```
crates/agent-core/src/lib.rs  (public re-exports)
```

## Specification

### 14. Key Data Flows

#### LLM Agent Turn

An `InvocationContext` containing the user prompt, state, provider, and cost tracker enters `agent.run(ctx)` on an LLM agent built via `AgentBuilder`. The agent first interpolates the system prompt by replacing `{key}` placeholders with values from `ctx.state`. If an `output_schema` is configured, it appends an enforcement instruction to the system prompt and injects the `StructuredOutputTool` into the tools list, optionally setting `tool_choice`. A context message (memory, instruction files) is injected if a `PromptBuilder` is configured, followed by the user message.

The agent then enters a loop. Each iteration checks guards (cancellation flag, turn limit, budget limit), then sends a `CompletionRequest` to `ctx.provider.complete()`. Usage is recorded on the cost tracker. The response is parsed: text blocks become `assistant_text`, tool_use blocks become `tool_calls`. If the stop reason is not `ToolUse`, the agent checks whether an `output_schema` was set but no `StructuredOutput` call was made — if so, it injects a retry message and continues (up to `max_schema_retries`, then returns `SchemaRetryExhausted`). Otherwise it returns `AgentOutput { content, usage, structured_output }`. If the stop reason is `ToolUse`, tools are executed (read-only tools concurrently, write tools serially), structured output is extracted from any `StructuredOutput` tool calls, and the loop continues.

#### Task Persistence Flow

An agent calls the `task_create` tool. `TaskStore.create(subject, description)` acquires a file lock (`.lock`), reads `.highwatermark` for the next ID, writes a task file `{next_id}.json`, updates `.highwatermark`, and releases the lock. It returns `Task { id, subject, status: Pending, ... }`.

Later, an agent calls `task_update(id, status: InProgress)`. `TaskStore.update(id, TaskUpdate { status: Some(InProgress) })` acquires the lock, reads the task file, validates (not completed, `blocked_by` all resolved), updates fields, writes back, and releases the lock.

### 15. Concurrency Model

- **LLM agent loop**: Sequential async. One turn at a time.
- **Tool execution**: `tokio::task::JoinSet` + `Semaphore(10)` for concurrent read-only batches. Write tools serial.
- **Background agents**: `tokio::spawn` via `spawn_agent` with `background: true`. Isolated messages. Shared `CostTracker`, `SessionStore`, `CommandQueue` via `Arc`. Notification delivery on completion.
- **Task persistence**: Advisory file locks with retry backoff for concurrent disk access.
- **Command queue**: `Arc<Mutex<VecDeque>>` + `tokio::sync::Notify` for consumer wakeup.
- **Cancellation**: `Arc<AtomicBool>` in `InvocationContext`, checked at each loop turn boundary.

### 16. Public API Surface

```rust
// agent_core re-exports

// Agent system
pub use agent::{Agent, AgentBuilder, AgentOutput, Event, InvocationContext};

// Command queue
pub use agent::{CommandQueue, QueuePriority, QueuedCommand, CommandSource};

// LLM providers
pub use provider::{LlmProvider, AnthropicProvider, LiteLlmProvider, CompletionRequest, HttpTransport, ToolChoice};

// Messages
pub use message::{Message, ContentBlock, ModelResponse, StopReason, Usage};

// Tools
pub use tool::{Tool, ToolCall, ToolDefinition, ToolRegistry, ToolResult,
               ToolContext, ToolBuilder, Toolset};

// Structured output
pub use agent::{OutputSchema, validate_value};

// Prompt construction
pub use prompt::{PromptBuilder, PromptSection, EnvironmentContext};

// Persistence
pub use session::{SessionStore, SessionMetadata, TranscriptEntry};
pub use task::{Task, TaskStatus, TaskStore, TaskUpdate};

// Cost tracking
pub use cost::{CostTracker, ModelCosts, ModelUsage};

// Errors
pub use error::{AgenticError, Result};
```

## Work Items

1. **`lib.rs`** — Update public re-exports to include all types from all increments

## Tests

No additional tests — this is a documentation and integration document. All types are tested in their respective modules.

## Done Criteria

- `lib.rs` re-exports all public types listed above
- `cargo doc` generates documentation for the full public API
