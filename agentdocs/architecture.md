# Architecture

The invariants that shape how code fits together. Layout says where code lives; this file says why the seams are where they are.

## 1. Builder, spec, loop

**A run has three stages: build the `Agent`, compile an `AgentSpec`, drive it with `run_loop`.**

- The builder carries per-run fields (provider, instruction, cancel signal) and a copy-on-write `Arc<AgentSpec>`.
- `Agent::compile` resolves the model, fills externals, and hands the loop a frozen `(Arc<AgentSpec>, LoopRuntime)` pair.
- `run_loop` owns the turn-by-turn state machine and is the only code that mutates `LoopState`.
- The loop never sees the builder; the builder never sees the loop's state.

## 2. Runtime versus state

**Each run has two buckets: externals in `LoopRuntime`, mutation in `LoopState`.**

- `LoopRuntime` holds the provider, event handler, cancel signal, command queue, session store, tool registry, and working directory.
- `LoopState` holds messages, token counters, turn number, schema retries, and collected errors.
- `LoopRuntime` is shared behind an `Arc`; `LoopState` is owned by the loop and threaded through by `&mut`.
- Sub-agents reuse the parent's runtime; they never share a state.

## 3. Copy-on-write configuration

**`AgentSpec` is shared across clones. Builder methods mutate through `Arc::make_mut`.**

- Cloning an `Agent` clones a handful of small per-run fields and bumps one `Arc` refcount.
- A builder method mutating the spec forks the `Arc` only if other clones exist; otherwise it mutates in place.
- The template (tools, sub-agents, prompts, limits) is shared; the per-run fields (instruction, template variables, handlers) are not.
- `AgentSpec` is `pub(crate)`; callers never touch it directly.

## 4. Sub-agent inheritance

**A sub-agent is an `Agent` compiled against a parent's runtime. Inheritance is per-field.**

- `Agent::compile(Some(parent))` fills the child's `model: None` with the parent's resolved model and reuses the parent's externals.
- Child per-run fields (cancel signal, event handler, working directory) override the inherited values when set.
- Tools and sub-agents are not inherited: each `AgentSpec` declares its own registry.
- `SpawnAgentTool` is the only path that calls `Agent::run_child`; the public API never exposes the parent slot.

## 5. One error at the crate boundary

**`Result<T, Error>` is the one fallible surface. Domain sub-enums live with their domain.**

- `Error::Provider`, `Error::Agent`, `Error::Tool` each wrap a domain-specific enum defined in that module.
- `Error::is_retryable` and `Error::retry_delay` delegate to the provider variant; all other categories are terminal.
- Tool failures flow two ways: `Err` bubbles as `Error::Tool`, `Ok(ToolResult::Error(...))` goes back to the model as text.
- No blanket `From<io::Error>` or `From<serde_json::Error>`; each mapping is explicit.

## 6. Events observe, Output returns

**Observation and return use separate channels. `Event` is out-of-band; `Output` is the run's value.**

- Every lifecycle transition emits an `Event` through `runtime.event_handler`.
- `AgentFinished` fires on every `Ok` termination path (`Completed`, `Cancelled`, `Failed`) and matches `Output.outcome`.
- Pre-flight failures (missing provider, unreadable prompt, unset model) return `Err(...)` without ever starting the loop.
- Handlers are cheap, non-blocking closures; the loop does not await them.

## 7. Providers own their client

**Each concrete provider owns a `reqwest::Client` directly. There is no transport abstraction.**

- The `Provider` trait has two methods: `respond` (drive one turn) and `prewarm` (warm TCP+TLS).
- `ModelRequest` and `ModelResponse` are the wire-shaped types every provider converts to and from.
- HTTP error mapping is shared through `map_http_errors` + a provider-specific `classify` closure.
- Retry and compaction are shared seams (`provider::retry::compute_delay`, `agent::compact`); vendor code does not retry.

## 8. Cancellation is cooperative

**A run is cancelled by setting one shared `Arc<AtomicBool>`. Every waiter polls it.**

- `check_guards` reads the flag at every turn boundary; tools read it via `ToolContext::wait_for_cancel`.
- `tokio::select!` pairs provider calls and tool futures with `wait_for_cancel` so a cancel drops the losing branch promptly.
- `AgentHandle` sets the flag explicitly; dropping the last handle sets it via `CancelGuard::drop`.
- `Batch` installs its own signal on every submitted agent so `BatchHandle::cancel` reaches in-flight runs.

## 9. Persistence stays internal

**`persistence/` is `pub(crate)`. Sessions and tasks are opt-in behaviors, never part of the public type surface.**

- `SessionStore` appends JSONL transcripts when `.session_directory(...)` is set; otherwise the loop writes nothing.
- `TaskStore` is reached through `TaskTool`; agents coordinate through it without importing it.
- No persistence type appears in `Output`, `Event`, or any public signature.
- Swapping the backend is a crate-internal change; callers do not break.
