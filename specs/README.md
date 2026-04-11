# Agent — Rust Agentic Core Library

A minimal, dependency-light Rust library crate that other projects can embed to add agentic LLM behavior. Provides the full lifecycle: LLM integration, tool execution, task tracking, session persistence, agent orchestration (via `spawn_agent` with LLM-driven composition), and cost tracking.

---

## Architecture Overview

The system is organized in layers:

- **Agent trait** (core) — One interface for all agent types: `fn run(&self, ctx: InvocationContext) -> AgentOutput`. Implementations: LlmAgent (LLM loop + tools) and custom user-defined agents. Multi-agent composition is handled by the LLM via `spawn_agent` tool calls.
- **Shared services** (bottom layer) — Provider (LLM), Tool Registry, Session Store, Cost Tracker.

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **One `Agent` trait** | All agents compose identically — LLM agents and custom agents implement the same trait. |
| **Build-time vs runtime separation** | `AgentBuilder` configures static properties (model, tools, prompt). `InvocationContext` carries runtime data (input, state). |
| **Agent owns lifecycle** | Agents record their own transcripts, poll the command queue between turns, and manage background sub-agent state. `SessionStore`, `CommandQueue`, `CostTracker` are passed via `InvocationContext` as `Arc`-wrapped values. |
| **`PromptBuilder` separate from `InvocationContext`** | `PromptBuilder` builds the system prompt string (static, build-time). `InvocationContext` carries runtime state (dynamic, per-invocation). |

## Crate Structure

```
agent/
  Cargo.toml                    # workspace root
  crates/
    agent-core/               # library: traits, agent loop, cost tracking, persistence
      Cargo.toml
      src/
        lib.rs                  # public re-exports
        error.rs                # AgenticError, Result alias
        message.rs              # Message, ContentBlock, Usage, StopReason
        provider.rs             # LlmProvider trait, AnthropicProvider, LiteLlmProvider
        tool.rs                 # Tool trait, ToolRegistry, tool orchestration
        agent.rs                # Agent trait, AgentBuilder, agent loop
        orchestration.rs        # public re-exports
        prompt.rs               # PromptBuilder: system prompt assembly, memory, instructions
        cost.rs                 # CostTracker, ModelCosts, ModelUsage
        session.rs              # SessionStore, transcript persistence
        task.rs                 # TaskStore, task persistence, dependencies
        testutil.rs             # MockProvider, MockTool, TestHarness (#[cfg(test)])
    agent-tools/              # library: built-in file/folder tools
      Cargo.toml
      src/
        lib.rs                  # BuiltinToolset
        read_file.rs
        write_file.rs
        edit_file.rs
        list_directory.rs
        glob.rs                 # fast glob pattern file matching
        grep.rs                 # content search with context lines
        bash.rs                 # shell command execution
        spawn_agent.rs          # spawn a single agent (foreground or background)
  examples/
    code-review/
      Cargo.toml
      src/
        main.rs         # code review tool: agent + structured output
        file_stats.rs   # custom tool: file extension statistics
```

## Dependencies

### agent-core

```toml
[dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "fs", "io-util"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
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

### agent-tools

```toml
[dependencies]
agent-core = { path = "../agent-core" }
tokio = { version = "1", features = ["fs", "process"] }
serde_json = "1"
```

### code-review (example)

```toml
[dependencies]
agent-core = { path = "../../crates/agent-core" }
agent-tools = { path = "../../crates/agent-tools" }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["json"] }
```

Arg parsing via `std::env::args()` with manual matching.

### Dependency Summary

| Crate | agent-core | agent-tools | code-review |
|-------|:---:|:---:|:---:|
| `tokio` | yes | yes | yes |
| `serde` | yes | — | — |
| `serde_json` | yes | yes | yes |
| `reqwest` | — | — | yes |
| **Total** | **3** | **2** | **3** |

Everything else is stdlib: `std::io`, `std::fs`, `std::path`, `std::collections::HashMap`, `std::sync::{Arc, Mutex}`, `std::sync::atomic::AtomicBool`, `std::future::Future`, `std::pin::Pin`, `std::time`.

## Increment Dependencies

```
1 (core types + test infra) -> 2 (agent) -> 3 (tool system) -> { 4 (orchestration), 5 (persistence) } -> 6 (examples)
```

| Increment | Depends on | Parallelizable with |
|-----------|-----------|---------------------|
| 1. Core types + providers | — | — |
| 2. Agent | 1 | — |
| 3. Tool system | 1, 2 | — |
| 4. Orchestration | 2, 3 | 5 |
| 5. Persistence | 1 | 4 |
| 6. Examples | All | — |

### Integration Points Between Parallel Increments

- Increment 3 (tools) uses `AgenticError` and `Result` from Increment 1 — use a stub `error.rs` until Increment 1 merges
- Increment 2 (agent) uses `ToolRegistry` and `execute_tool_calls` from Increment 3 — needs Increment 3 complete
- Increment 5 (persistence) uses `Message` and `Usage` from Increment 1 — use stub `message.rs` until Increment 1 merges
- Increment 4 integrates agent + persistence + orchestration — this is the merge point where everything connects

## Increment Files

| Folder | Contents |
|--------|----------|
| [`1-base/`](./1-base/) | Core types, providers, cost tracking, prompt construction, test infrastructure |
| [`2-agent/`](./2-agent/) | Agent trait, AgentBuilder, agent loop, structured output |
| [`3-tools/`](./3-tools/) | Tool trait, registry, built-in tools |
| [`4-orchestration/`](./4-orchestration/) | spawn_agent tool, data flows, public API surface |
| [`5-persistence/`](./5-persistence/) | Task store, session store |
| [`6-examples/`](./6-examples/) | Example CLIs: code review, research assistant, data pipeline |

## Public API Surface

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

> The canonical Public API Surface is defined in [`4-orchestration/api.md`](./4-orchestration/api.md).

## Naming Conventions

| Convention | Example | When to use |
|-----------|---------|-------------|
| `get(key)` | `ToolRegistry::get(name)`, `TaskStore::get(id)` | Direct ID/name lookup (O(1) or O(n) scan by exact key) |
| `find_*(criteria)` | `Agent::find_agent(name)` | Search operations that may involve filtering or traversal |
| Bare method name | `AgentBuilder::tool()`, `.sub_agent()`, `.section()` | Fluent builder-style "add item" methods |
| `register(item)` | `ToolRegistry::register(tool)` | Adding items to standalone collection types |
| `open(path)` | `TaskStore::open(base_dir, list_id)` | Opening/loading an existing resource from disk |
| `create(fields)` | `TaskStore::create(subject, description)` | Creating a new entity within a resource |
| `*_at` suffix | `created_at`, `updated_at`, `recorded_at`, `last_active_at` | All timestamp fields |
