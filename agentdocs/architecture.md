# Architecture

The invariants that shape how code fits together. Layout says where code lives; this file says why the seams are where they are.

## Builder, system, loop

**A run has three stages: build the `Agent`, bind it to a `TicketSystem`, drive the system with `start` (long-lived) or `finish` (process a fixed batch and return).**

- The `Agent` builder carries identity, prompt parts, provider/model, tools, working dir, event handler, and a `Weak<TicketSystem>` (dangling by default).
- `TicketSystem::add(agent)` (or `agent.ticket_system(&shared)`) stamps the system's `Weak<Self>` onto the agent, drains any tickets the agent had queued in its private default system into the shared one, and pushes a clone of the agent onto the system's agents list.
- `TicketSystem::start` / `finish` spawn one tokio task per registered agent; each task upgrades its `Weak` once at the start and reads the shared store, policies, stats, and interrupt signal from the resulting `Arc<TicketSystem>`.
- `tickets.task(value)` and `tickets.task_labeled(value, label)` create a new ticket and return its key as `String`. `tickets.comment(&key, content)` appends a user-side text comment to an existing ticket: the agent loop's wait-for-input branch picks the comment up and drives the next turn on the same transcript. Use `task` to start a conversation, `comment` to continue it; this is how multi-turn chat is built on top of one ticket.

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

**Agents finish tickets through one of two finisher tools: `write_result_tool` (terminal) or `write_handover_tool` (terminal + spawn child). Both route through the same internal `write_result` helper, which owns the result-validation contract and the `Done` transition. The loop enforces the rule: a turn that ends without a finisher tool call is rejected and retried.**

- `write_result_tool` writes a `result`, attaches it to the current ticket, and transitions to `Done`. Use it for terminal work.
- `write_handover_tool` does the same and then inserts a child ticket pinned to a target agent or label with the current ticket recorded as its `parent`. Use it to atomically finish-and-chain; the alternative — `write_result_tool` followed by `manage_tickets_tool::create` — is order-sensitive and leaves the current ticket re-picked when the order is wrong.
- When a turn ends without a finisher tool call (no `write_result_tool`, no `write_handover_tool`, no result attached), the loop pushes a corrective directive and retries. This is the same retry path used for schema-validation failures, bounded by `max_schema_retries`; exhaustion emits `PolicyViolated { MaxSchemaRetries, .. }` and `TicketFailed`.
- `Status` transitions go through tickets-side helpers; the agent never writes status directly. `Failed` is reserved for system-driven outcomes (schema-retry trip, missing-finisher exhaustion, policy violations).
- `Ticket::schema(...)` attaches a `Schema` to the ticket; the finisher tool validates the result and the loop applies `max_schema_retries` on mismatch. A schema mismatch in `write_handover_tool` aborts both the parent's finish AND the child insert — the operation is atomic.
- A successful finish appends one NDJSON record `{agent, ticket, result}` to `<dir>/results.jsonl` (configured via `TicketSystem::dir(d)`; defaults to `./.agentwerk`) and attaches the same `ResultRecord` to the ticket. The record is surfaced through `Ticket::result()`; `last_result()` returns its serialized form for the most recent `Done` ticket.
- The system also appends one JSON line to `<dir>/tickets.jsonl` per lifecycle event (`created`, `started`, `done`, `failed`) and writes the full ticket state to `<dir>/tickets/<key>/ticket.<ts>.json`. The `created` event carries the optional `parent` key when set, giving the log a complete handover audit trail. The log is observational: errors are swallowed. The result payload stays in `results.jsonl`; `tickets.jsonl` carries only the transition.

## Knowledge is opt-in and shareable across agents

**An agent can carry durable facts across every ticket it handles via `Agent::knowledge(&store)`, including across separate `start` / `finish` calls and across process restarts. Off by default; each ticket starts without a knowledge section.**

Two layers of state exist. The per-ticket transcript lives on `Ticket::comments`: every message the loop sends to the provider is appended as a `Comment`, and the loop derives the request's `Vec<Message>` from those comments via `to_messages` each turn. `Agent::knowledge(&store)` adds a separate cross-ticket layer: a `Knowledge` store rooted at a caller-supplied directory, surfaced to the model through `KnowledgeTool` and rendered into the system prompt under `## Knowledge`.

- The store is constructed via `Knowledge::open(knowledge_dir)` and passed to one or more agents through `Agent::knowledge(&store)`. Two agents bound to the same `Arc<Knowledge>` share the same `index.md` and `pages/` directory; two agents bound to different stores see independent knowledge. The pattern mirrors `Agent::ticket_system(&Arc<TicketSystem>)`. Pointing `Knowledge::open` at the same directory as `TicketSystem::dir` co-locates knowledge pages with `results.jsonl` and `tickets.jsonl`.
- The store uses a page-based layout: `<dir>/index.md` holds one-line summaries, and `<dir>/pages/<slug>.md` holds full page content with frontmatter. Only the compact index is injected into the system prompt; the agent reads full pages on demand via the `read` action.
- The loop reads `Knowledge::index()` once at the top of `process_ticket` and feeds the result to `Agent::system_prompt(knowledge: Option<&str>)`. The system prompt stays byte-stable across every turn of the ticket so the provider's prefix cache survives mid-ticket knowledge writes; cross-ticket and cross-agent writes become visible at the top of the next ticket.
- Knowledge is purely model-driven. The model calls `knowledge_tool` with `write` / `read` / `remove` / `list`; the tool description carries the policy (durable facts only, do NOT save task progress / TODOs). A hard char limit on the rendered index rejects writes that would push the prompt section past the cap and tells the model to consolidate first. The limit defaults to 4000 and is configurable via `Knowledge::open(dir)?.index_char_limit(n)`.

## One observer, one error path

**`Event` reports state. `ProviderError` and `ToolError` report failed contracts. The two channels carry independent information.**

- State transitions exist only as `Event` (`TicketClaimed`, `TicketFinished`, `RequestStarted`, `RequestFinished`, `TextChunkReceived`).
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

## Persistence routes through two traits

**Every read and write in the crate goes through `Persist` (state files) or `Append` (jsonl logs) in `persistence`. No domain module hand-rolls file IO; no module knows its file's name except the implementer.**

- `Persist` defines `save(&self, dir) -> io::Result<()>` and `load(dir, &Self::Key) -> io::Result<Self>`. `Stats`, `Ticket`, and `Page` implement it; each owns its own path layout (`stats.json`, `tickets/<key>/ticket.<ts>.json`, `pages/<slug>.md`). Service bootstrap (`TicketSystem::load`, `Knowledge::load`) uses the same `load` verb for its dir-to-`Arc<Self>` entry by convention.
- `Append` defines `append(dir, &Self::Record) -> io::Result<()>`. `Results` writes `results.jsonl`; `TicketEvents` writes `tickets.jsonl`. The wrong type cannot reach the wrong file: each implementer's `append` body hardcodes the filename.
- Crate-internal helpers `write_atomic` (tmp+rename) and `append_line` (`O_APPEND` + newline) are the only places that touch the filesystem. They are `pub(crate)` so trait impls colocated with their types can call them; by convention nothing outside a `Persist` or `Append` impl reaches for them — the one documented exception is `TicketSystem::write_tool_output`, which writes single-shot flat files that don't fit either trait.
- Vocabulary is fixed: `save`, `load`, `append`. Bootstrap verbs other than `load` (e.g. `open`) are not used. Domain words (`checkpoint`, `snapshot`, `counter`, `persist`) do not appear in identifiers or test names.

## Policies are per-system, checked at step boundaries

**A run stops cleanly when any limit on `Policies` trips. The check fires `EventKind::PolicyViolated` and exits the per-agent task.**

- The loop calls `policy_violated_kind` at each iteration; a non-`None` return walks the agent off the queue.
- Token budgets read from `Stats`; `max_time` reads from `Policies` and is checked separately by the `finish` watcher (graceful stop, not a `PolicyViolated` event).
- Schema-retry budget is applied per-ticket inside the result-writing path, not at the top of the loop.
