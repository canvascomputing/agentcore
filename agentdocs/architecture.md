# Architecture

The invariants that shape how code fits together. Layout says where code lives; this file says why the seams are where they are.

## Builder, spec, loop

**A run has three stages: build the `Agent`, compile an `AgentSpec`, drive it with `run_loop`.**

- The builder carries per-run fields (provider, instruction, cancel signal) and a copy-on-write `Arc<AgentSpec>`.
- `Agent::compile` resolves the model, fills externals, and hands the loop a frozen `(Arc<AgentSpec>, LoopRuntime)` pair.
- `run_loop` owns the step-by-step state machine and is the only code that mutates `LoopState`.
- The loop never sees the builder; the builder never sees the loop's state.

## Runtime versus state

**Each run has two buckets: externals in `LoopRuntime`, mutation in `LoopState`.**

- `LoopRuntime` holds the provider, event handler, cancel signal, work inbox, session store, tool registry, and working directory.
- `LoopState` holds messages, token counters, step number, schema retries, and collected errors.
- `LoopRuntime` is shared behind an `Arc`; `LoopState` is owned by the loop and threaded through by `&mut`.
- Sub-agents reuse the parent's runtime; they never share a state.

## Copy-on-write configuration

**`AgentSpec` is shared across clones. Builder methods mutate through `Arc::make_mut`.**

- Cloning an `Agent` clones a handful of small per-run fields and bumps one `Arc` refcount.
- A builder method mutating the spec forks the `Arc` only if other clones exist; otherwise it mutates in place.
- The template (tools, sub-agents, prompts, limits) is shared; the per-run fields (instruction, template variables, handlers) are not.
- `AgentSpec` is `pub(crate)`; callers never touch it directly.

## Sub-agent inheritance

**A sub-agent is an `Agent` compiled against a parent's runtime. Inheritance is per-field.**

- A parent declares its sub-agents with `.staff(...)` / `.staff_more(...)`; the resulting `AgentSpec.staff` is a `Vec<Agent>`.
- `Agent::compile(Some(parent))` fills the child's `model: None` with the parent's resolved model and reuses the parent's externals.
- Child per-run fields (cancel signal, event handler, working directory) override the inherited values when set.
- Tools and `staff` are not inherited: each `AgentSpec` declares its own registry and worker list.
- `AgentTool` is the only path that calls `Agent::execute_child`; the public API never exposes the parent slot.

## AgentTool registration

**`AgentTool` is registered exactly when the spec carries `staff`. The slot is `"agent"`.**

- `build_tools(spec)` clones `spec.tool_registry`; if `spec.staff` is non-empty and no tool already occupies `"agent"`, it appends `AgentTool`.
- A caller that registers a custom tool under `"agent"` keeps theirs: explicit registration wins.
- An agent with no staff never sees `AgentTool` in its registry, even after a child is added later, because each compile rebuilds the registry.
- The sub-agent registry is rebuilt on every compile, so per-run mutation of `staff` is observed by the next call to `execute`.

## One error at the crate boundary

**`Result<T, Error>` is the one fallible surface. Domain sub-enums live with their domain.**

- `Error::Provider`, `Error::Agent`, `Error::Tool` each wrap a domain-specific enum defined in that module.
- `Error::is_retryable` and `Error::retry_delay` delegate to the provider variant; all other categories are terminal.
- Tool failures flow two ways: `Err` bubbles as `Error::Tool`, `Ok(ToolResult::Error(...))` goes back to the model as text.
- IMPORTANT: no blanket `From<io::Error>` or `From<serde_json::Error>`. Each mapping MUST be explicit.

## Events observe state, errors return failures

**`Event` reports state. `Error` reports a failed contract. The two channels carry independent information.**

- State transitions exist only as `Event` (`AgentStarted`, `StepStarted`, `ContextCompacted`); failures exist as `Error` first (`ProviderError`, `AgentError`, `ToolError`).
- An observable failure fires both: the `Error` is the machine-readable truth, the matching `Event` mirrors its kind and message (`RequestFailed`, `ToolCallFailed`, `PolicyViolated`).
- `Output.errors` is emission-ordered; on `Outcome::Failed` the last entry is the terminal cause, earlier entries are retried transients. Tool failures never land there: they go to the model and fire `ToolCallFailed`.
- IMPORTANT: pre-flight failures (missing provider, unreadable prompt, unset model) return `Err(...)` without starting the loop. No `AgentStarted` or `AgentFinished` fires.
- Handlers MUST be cheap, non-blocking closures; the loop does not await them.

## New observables pick a channel

**Each new signal lands on `Event`, `Error`, or both. Pick by what the signal describes.**

- Reached a state: `Event` only.
- Could not fulfil a contract: `Error` in the matching domain sub-enum.
- Both at once (retry, terminal request failure, policy trip): define both. Share the payload type when observer-friendly (`PolicyKind`); introduce a stripped `Kind` enum when the `Error` carries observer-hostile detail (`RequestErrorKind` for `ProviderError`, `ToolFailureKind` for `ToolError`).
- Model-fixable failure (wrong args, non-zero exit, file missing): use `ToolResult::Error(String)`; it still fires `ToolCallFailed` but stays out of `Output.errors`.

## Providers own their client

**Each concrete provider owns a `reqwest::Client` directly. There is no transport abstraction.**

- The `Provider` trait has two methods: `respond` (drive one step) and `prewarm` (warm TCP+TLS).
- `ModelRequest` and `ModelResponse` are the wire-shaped types every provider converts to and from.
- HTTP error mapping is shared through `map_http_errors` + a provider-specific `classify` closure.
- Retry and compaction are shared seams (`util::Retry`, `agent::compact`); vendor code does not retry.

## Cancellation is cooperative

**A run is cancelled by setting one shared `Arc<AtomicBool>`. Every waiter polls it.**

- `check_guards` reads the flag at every step boundary; tools read it via `ToolContext::wait_for_cancel`.
- `tokio::select!` pairs provider calls and tool futures with `wait_for_cancel` so a cancel drops the losing branch promptly.
- `AgentWorking::interrupt` sets the flag explicitly; dropping the last handle sets it via `CancelGuard::drop`.
- `Werk` installs its own signal on every dispatched agent so `Werking::interrupt` reaches in-flight runs.

## keep_working hands the loop a background task

**`Agent::keep_working` spawns the loop on tokio and returns `(AgentWorking, OutputFuture)`. The loop idles between instructions while any handle is alive.**

- `keep_working` flips `keep_alive: true` on the spec and installs a fresh `Work` inbox plus `interrupt_signal` before calling `execute` on a `tokio::spawn`.
- `AgentWorking::work(s)` posts a follow-up task; the loop picks it up at its next idle poll or step boundary. `work_more(...)` queues several at once.
- `OutputFuture` resolves once the loop exits; awaiting it does not keep the loop alive: only `AgentWorking` clones do.
- `CancelGuard` flips the cancel flag when the last `AgentWorking` clone drops, so an abandoned handle still unblocks the loop.

## Werk runs a workshop

**`Werk` runs many `Agent`s under a fixed line cap. `work` waits for a fixed cohort; `keep_working` returns a handle and a stream that accept new staff.**

- The dispatcher is a single `tokio::spawn` task that owns a `FuturesUnordered` of in-flight agent tasks bounded by `lines`.
- Each agent's name is captured at submission time and travels with it through the dispatcher, so results pair `(String, Result<Output>)`. `Werk::work` collects those into a `HashMap<String, Result<Output>>` keyed by name.
- `Werking` is `Clone`: the workshop accepts new staff while any clone is alive; dropping the last one closes it gracefully.
- `Werk::interrupt_signal` lets the caller share an external signal; otherwise the workshop owns one and overrides any per-agent signal.

## Incoming work carries dynamic tasks

**`Work` is a per-run inbox of `Task`s. The loop drains it between steps; `AgentWorking` and orchestration tools post into it.**

- The inbox lives on `LoopRuntime` and is shared by the parent and every sub-agent in the run-tree.
- `AgentWorking::work` posts with `WorkPriority::Next` so the next step picks it up before any backlog.
- Background sub-agents use the inbox to post notifications back to the parent; routing by `agent_name` keeps the inbox per-agent.
- The inbox is `pub(crate)` only: the public API exposes it through `AgentWorking` and the orchestration tools, never directly.

## Persistence stays internal

**`persistence/` MUST stay `pub(crate)`. Sessions and todo lists are opt-in behaviors, never part of the public type surface.**

- `SessionStore` appends JSONL transcripts when `.session_dir(...)` is set; otherwise the loop writes nothing.
- `TodoList` is reached through `TodoListTool`; agents coordinate through it without importing it.
- No persistence type appears in `Output`, `Event`, or any public signature.
- Swapping the backend is a crate-internal change; callers do not break.
