# Architecture

The invariants that shape how code fits together. Layout says where code lives; this file says why the seams are where they are.

## Builder, system, loop

**A run has three stages: build the `Agent`, bind it to a `TicketSystem`, drive the system with `run` (long-lived) or `run_dry` (drain the fixed batch).**

- The `Agent` builder carries identity, prompt parts, provider/model, tools, working dir, event handler, and a `Weak<TicketSystem>` (dangling by default).
- `TicketSystem::add(agent)` (or `agent.ticket_system(&shared)`) stamps the system's `Weak<Self>` onto the agent, drains any tickets the agent had queued in its private default system into the shared one, and pushes a clone of the agent onto the system's agents list.
- `TicketSystem::run` / `run_dry` spawn one tokio task per registered agent; each task upgrades its `Weak` once at the start and reads the shared store, policies, stats, and interrupt signal from the resulting `Arc<TicketSystem>`.

## Shared system, per-agent task

**Agents read shared state through one `Arc<TicketSystem>`. Locks are held only around queue and metric operations, never across `provider.respond().await`.**

- The ticket store, policies, stats, interrupt signal, and registered-agent list live on `TicketSystem`.
- The per-agent loop in `agents/loop.rs` claims one ticket, drives it through one or more provider/tool steps, and releases locks before each await.
- Multiple agents share one queue; a ticket is claimed exactly once.
- Sub-systems are not nested: a single `TicketSystem` is the unit of orchestration.

## Path A and Path B routing

**Tickets reach agents either by direct assignment (Path A) or by label scope (Path B).**

- A ticket built with `Ticket::new(...).assign_to(name)` is born `Status::InProgress` and pinned to the named agent; only that agent can pick it up.
- A ticket built with `.label(...)` (or via `task_labeled(value, label)`) is `Status::Todo` and picked up by any agent whose `label` scope intersects.
- An agent with empty labels handles only tickets with no labels; that is the "default scope".
- The system never auto-resolves a name against the registered-agent set: callers know which routing path they want.

## Finishing is a tool call

**Agents finish tickets by calling `write_result_tool` with a `result` (a non-empty string for tickets without a schema, or a JSON value for schema-bound tickets). The framework owns the lifecycle stamps and transitions the ticket to `Done`.**

- `Status` transitions go through tickets-side helpers; the agent never writes status directly. `Failed` is reserved for system-driven outcomes (schema-retry trip, policy violations).
- `task_schema*` attaches a `Schema` to the ticket; the tool validates the result and the loop applies `max_schema_retries` on mismatch.
- A successful call appends one NDJSON record `{agent, ticket, result}` to `<workspace>/results.jsonl` (configured via `TicketSystem::workspace(dir)`; falls back to the agent's working directory) and attaches the same `ResultRecord` to the ticket. The record is surfaced through `Ticket::result()`; `run_dry` returns its serialized form for the most recent `Done` ticket.
- When a workspace is set, the system also appends one JSON line to `<workspace>/tickets.jsonl` per lifecycle event (`created`, `started`, `done`, `failed`). The log is observational: errors are swallowed and a `None` workspace skips writes entirely. The result payload stays in `results.jsonl`; `tickets.jsonl` carries only the transition.

## Memory is opt-in and shareable across agents

**An agent can carry durable facts across every ticket it handles via `Agent::memory(&store)`, including across separate `run` / `run_dry` calls and across process restarts. Off by default; each ticket starts without a memory section.**

Two layers of state exist. The intra-ticket message vector inside `process_ticket` is always present; it carries a multi-step tool turn through and is dropped when the ticket reaches a terminal status. `Agent::memory(&store)` adds a separate cross-ticket layer: a `Memory` rooted at a caller-supplied directory, surfaced to the model through `MemoryTool` and rendered into the system prompt under `## Memory`.

- The store is constructed via `Memory::open(memory_dir)` and passed to one or more agents through `Agent::memory(&store)`. Two agents bound to the same `Arc<Memory>` share `memory.jsonl` (one entry per line: `{"content": "...", "added_at": <ms>}`); two agents bound to different stores see independent memory. The pattern mirrors `Agent::ticket_system(&Arc<TicketSystem>)`. Pointing `Memory::open` at the same directory as `TicketSystem::workspace` co-locates `memory.jsonl` with `results.jsonl` and `tickets.jsonl`.
- The loop reads `Memory::entries()` once at the top of `process_ticket`, concatenates them, and feeds the result to `Agent::system_prompt(memory: Option<&str>)`. The system prompt stays byte-stable across every turn of the ticket so the provider's prefix cache survives mid-ticket memory writes; cross-ticket and cross-agent writes become visible at the top of the next ticket.
- Memory is purely model-driven. The model calls `memory_tool` with `add` / `replace` / `remove`; the tool description carries the policy (work-feedback triggers, utility-ranked priority, do NOT save task progress / completed-work logs / TODOs). A hard char limit (default 2200) rejects writes that would push the rendered prompt section past the cap and tells the model to consolidate first. Callers that want to drive their own consolidation outside the model loop call `Memory::rewrite(new_entries)` directly.

## One observer, one error path

**`Event` reports state. `ProviderError` and `ToolError` report failed contracts. The two channels carry independent information.**

- State transitions exist only as `Event` (`TicketClaimed`, `TicketFinished`, `RequestStarted`, `RequestFinished`, `TextChunkReceived`, `TokensReported`).
- An observable failure fires both the typed error (`ProviderError`, `ToolError`) and a matching `Event` (`RequestFailed`, `ToolCallFailed`, `PolicyViolated`).
- A model-fixable failure (wrong arguments, schema mismatch, missing file) goes back to the model as a `ToolResult::Error` content block; it still fires `ToolCallFailed` but does not stop the run.
- Handlers MUST be cheap, non-blocking closures; the loop does not await them.

## New observables pick a channel

**Each new signal goes on `Event`, on a typed error, or on both. Pick by what the signal describes.**

- Reached a state: `Event` only.
- Could not fulfil a contract: typed error in the matching domain.
- Both at once (terminal request failure, policy trip): define both. Share the payload type when observer-friendly (`PolicyKind`); introduce a stripped `Kind` enum when the error carries observer-hostile detail (`RequestErrorKind`, `ToolFailureKind`).
- Model-fixable failure: `ToolResult::Error(String)`; still fires `ToolCallFailed` but is recoverable.

## Providers own their client

**Each concrete provider owns a `reqwest::Client` directly. There is no transport abstraction.**

- The `Provider` trait fulfils one contract: `respond` (drive one step) plus per-vendor metadata.
- `ModelRequest`, `Message`, `ContentBlock`, and `TokenUsage` are the wire-shaped types every provider converts to and from.
- HTTP error mapping is shared through `providers::map_http_errors` plus a provider-specific `classify` closure; SSE parsing lives in `providers::stream`.
- Retry happens at the request level using `Policies::max_request_retries` and `request_retry_delay`; vendor code does not retry.

## Cancellation is cooperative

**A run is cancelled by setting one shared `Arc<AtomicBool>`. Every waiter polls it.**

- The signal lives behind `TicketSystem::interrupt_signal`; each agent's loop reads it via the upgraded `Arc<TicketSystem>`.
- Tools observe it through `ToolContext::interrupt_signal` and `wait_for_cancel`; pair with `tokio::select!` so cancel drops the losing branch promptly.
- Dropping the `TicketSystem` while agents still reference it via `Weak` is the public way to abort: the upgrade fails and each task panics out cleanly.

## Stats are per-system, write-only-by-domain

**`Stats` is one struct of atomic counters; each domain interacts through its own write-only protocol.**

- `LoopStats` is what the per-agent loop sees: `record_step`, `record_request`, `record_tool_call`, `record_error`.
- `TicketStats` is what the ticket lifecycle sees: `record_created`, `record_started`, `record_done`, `record_failed`.
- Reads happen on `Stats` directly through inherent accessors (`steps()`, `tickets_done()`, `run_duration()`, `success_rate()`, ...), never through the recorder traits.
- Lock-free for increments; readers do one atomic load per call.
- `Stats::stats_for_label(label)` returns a nested `Stats` slice scoped to one label. The loop and ticket lifecycle bump each slice alongside the global counters; `run_duration()` is `None` on a slice (elapsed run duration stays global).

## Policies are per-system, checked at step boundaries

**A run stops cleanly when any limit on `Policies` trips. The check fires `EventKind::PolicyViolated` and exits the per-agent task.**

- The loop calls `policy_violated_kind` at each iteration; a non-`None` return walks the agent off the queue.
- Token budgets read from `Stats`; `max_time` reads from `Policies` and is checked separately by the `run_dry` watcher (graceful stop, not a `PolicyViolated` event).
- Schema-retry budget is applied per-ticket inside the result-writing path, not at the top of the loop.
