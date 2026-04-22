# CLAUDE.md

> Keep this file up to date when the project structure or conventions change.
> Update `README.md` when the public API changes.

## Vision

> agentwerk makes agents as simple as function calls. Build one in a few lines, call it, get a result — yet behind that surface, the same agent can run shell commands, search the web, read files, or orchestrate a fleet of sub-agents. The API stays small; the capabilities don't.

## Design philosophy

1. **Library, not framework.** agentwerk provides building blocks — you compose them in your own application. No runtime to boot, no traits to implement to get started, no opinions about how you structure your code.
2. **Minimal by design.** Few dependencies, no transport abstractions, no plugin registries. Every abstraction must earn its place by removing real complexity, not by adding indirection.
3. **Composable.** Clone a template, tweak one field, run it. This is how you get from one agent to a pool of fifty — no registration, no global state.
4. **Provider-agnostic.** The same agent code works across Anthropic, OpenAI, Mistral. Swap one line, keep everything else.
5. **Observe, don't prescribe.** The execution loop emits events. The caller decides what to do with them — log, stream, ignore. No built-in UI, no forced logging.
6. **Correctness over convenience.** Zero warnings. Schema validation with retries. Typed errors. No silent fallbacks.

## Build and test

```bash
make          # build (warnings are errors)
make test              # run unit tests (warnings are errors)
make test_integration  # run integration tests (requires LLM provider)
make fmt      # format code
make clean    # clean build artifacts
make bump              # test, bump patch version, commit, and tag (part=minor or part=major); then git push --tags
```

All code must compile with zero warnings (`RUSTFLAGS="-D warnings"`).

## Project structure

```
crates/agentwerk/src/
  lib.rs                  public re-exports
  error.rs                AgenticError, Result

  provider/
    mod.rs                re-exports
    trait.rs              Provider trait, CompletionRequest, ToolChoice, prewarm_with
    types.rs              Message, ContentBlock, TokenUsage, ResponseStatus, CompletionResponse, StreamEvent
    model.rs              Model (compact_threshold, RESERVED_RESPONSE_TOKENS, COMPACTION_HEADROOM_TOKENS)
    anthropic.rs          AnthropicProvider (with SSE streaming)
    openai.rs             OpenAiProvider (with SSE streaming; includes litellm/mistral constructors)
    litellm.rs            LiteLlmProvider
    mistral.rs            MistralProvider
    environment.rs        from_env (provider auto-detection from env vars)
    stream.rs             StreamParser, SseEvent (streaming response parser)
    retry.rs              compute_delay (associated consts live on Agent)

  agent/
    mod.rs                re-exports
    agent.rs              Agent (Agent::DEFAULT_MAX_REQUEST_RETRIES / DEFAULT_BACKOFF_MS, Arc<AgentSpec> + inlined per-run fields, Agent::interpolate, Agent::compile)
    spec.rs               AgentSpec (model accessor, system_prompt method, Default + DEFAULT_MAX_REQUEST_RETRIES/DEFAULT_BACKOFF_MS), interpolate, build_context_prompt
    loop.rs               LoopRuntime (environment), LoopState (initial), run_loop
    spawn.rs              AgentHandle, OutputFuture, impl Agent { spawn }
    event.rs              Event struct + EventKind enum (AgentStarted carries description for spawned children)
    output.rs             Output, Status, Statistics, OutputSchema (validate, retry_message)
    prompts.rs            DEFAULT_BEHAVIOR_PROMPT and structured-output constants
    batch.rs              Batch (builder: .run waits for all, .spawn returns BatchHandle + BatchOutputStream for dynamic pools)
    queue.rs              CommandQueue, QueuePriority, QueuedCommand (internal)

  tools/
    mod.rs                re-exports
    tool.rs               Toolable trait, Tool struct (ad-hoc, closure-handler), ToolRegistry (ToolRegistry::execute), ToolContext
    read_file.rs          ReadFileTool
    write_file.rs         WriteFileTool
    edit_file.rs          EditFileTool
    glob.rs               GlobTool
    grep.rs               GrepTool
    list_directory.rs     ListDirectoryTool
    bash.rs               BashTool (DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS associated consts; pattern-restricted via new(), unrestricted via unrestricted())
    util.rs               glob_match, run_shell_command
    tool_search.rs        ToolSearchTool
    spawn_agent.rs        SpawnAgentTool
    send_message.rs       SendMessageTool (peer agent messaging via CommandQueue)
    task_tools.rs         TaskTool

  persistence/ (internal)
    mod.rs                re-exports
    session.rs            SessionStore (JSONL transcripts)
    task.rs               TaskStore (file-based with locking)

  testutil.rs             MockProvider, MockTool, TestHarness, AgentEventCollector

crates/use-cases/src/
  project_scanner/
    main.rs                 project scanning CLI
  deep_research/
    main.rs                 multi-agent deep research with web search
```

Tests in `crates/agentwerk/tests/` are split by whether they hit a live provider. Mock-driven tests live in `tests/unit/` (bundled by `tests/unit.rs`) and run with `make test`. Real-LLM tests live in `tests/integration/` (bundled by `tests/integration.rs`) and run with `make test_integration`. Shared integration helpers (provider setup, event handler, JSON output) are in `tests/integration/common.rs`.
Use cases are in `crates/use-cases/src/cli/`. Run with `make use_case name=<name>`.

## Key conventions

- **No new dependencies without asking.** The crate is intentionally minimal (tokio, serde, serde_json, libc, reqwest, futures-util). Providers own a `reqwest::Client` directly — no transport abstraction.
- **No ad-hoc changes to critical types without a plan.** These types form the public API and are used across the entire codebase: `Agent`, `ToolContext`, `Event`, `Toolable` trait, `CompletionRequest`, `Output`, `Batch`. Propose changes in a plan first.
- **Tools capture dependencies at construction time** via closures or struct fields. The internal `ToolContext` handles (`runtime: Arc<LoopRuntime>`, `caller_spec: Arc<AgentSpec>`) exist solely for the agent loop to give `SpawnAgentTool` / `ToolSearchTool` read access to loop state — do not use them for new tools.
- **`tools/tool.rs` vs `tools/`**: `tool.rs` defines the trait and infrastructure (`Toolable` trait, `Tool` struct for ad-hoc tools, `ToolRegistry`, `ToolContext`). Other files in `tools/` are concrete implementations.
- **`agent/` vs `provider/` vs `persistence/`**: `agent/` contains the agent's builder surface (`Agent` in `werk.rs`) and everything the execution loop consumes (`AgentSpec` / `LoopRuntime` / `LoopState` / `run_loop` in `loop.rs`), events, output, and prompts. `provider/` contains LLM communication and estimated costs. `persistence/` contains internal disk storage (session transcripts, tasks).
- **`_file` variants**: All prompt builder methods (`identity_prompt`, `instruction_prompt`, `behavior_prompt`, `context_prompt`) and `output_schema` have `_file` counterparts (e.g. `identity_prompt_file(path)`, `output_schema_file(path)`) that load content from disk. File-read errors are collected on the `Agent` and surfaced when `run()` is called.
- **Tests live inline** in each module as `#[cfg(test)] mod tests`. Use `MockProvider` and `TestHarness` from `testutil.rs`.

## Naming conventions

### Type names

**The rule: headline types re-exported at the crate root use bare, single-word names. Everything else is qualified enough to read self-documenting on its own — either an inherently specific compound, a vendor-prefixed variant, or an internal type whose module path disambiguates it.**

- **Crate-root headline types use bare single-word names.** `Agent`, `Tool`, `Event`, `Message`, `Result`, `Output`, `Status`, `Statistics`, `OutputFuture` — all re-exported at `agentwerk::`, and their bare names read unambiguously from that crate path. New headline types follow the same pattern.
- **Inherently specific compounds stand alone.** `LoopState`, `LoopRuntime`, `AgentSpec`, `AgentHandle`, `EventKind`, `OutputSchema`, `CompactReason`, `CompletionRequest`, `CompletionResponse`, `TokenUsage`, `ContentBlock`, `StreamEvent`, `ToolCall`, `ToolRegistry`, `ResponseStatus`, `CommandQueue`, `SessionStore`, `TaskStore`, `ModelLookup`, `ProviderError`, `ProviderResult` already say what they are.
- **Vendor-prefixed types** follow the same logic — `AnthropicProvider`, `OpenAiProvider`, `MistralProvider`, `LiteLlmProvider`, `BashTool`, `ReadFileTool`. The prefix disambiguates which thing.
- **Acronyms follow Rust API guidelines**: `LiteLlmProvider`, not `LiteLLMProvider`. Already consistent: `OpenAiProvider`.
- **Two structs may not share a bare name in one module.** When that would happen, keep both qualified — don't pick one to prefix as a tiebreaker.

**Variant naming in `EventKind`.** State-transition events use past-participle form (`AgentStarted`, `AgentFinished`, `ToolCallStarted`, `ToolCallFinished`, `RequestRetried`, `ContextCompacted`, `OutputTruncated`, `InputBudgetExhausted`). Terminal-failure variants use the `Error` suffix (`RequestError`, `ToolCallError`) — the variant names the outcome, and its `error: String` payload echoes the variant name. Don't mix `Error` / `Failed`.

### Builder methods

- Bare nouns or compound nouns. No `with_` prefix.
  Exception: when the method name clashes with a trait method (e.g. `with_description` on BashTool).
  Examples: `.name()`, `.model()`, `.tool()`, `.sub_agents()`, `.read_only()`.

### Constructors

- `new()` for the primary/simple constructor.
- Named constructors for semantics: `open()`, `unrestricted()`, `success()`, `error()`, `empty()`, `from_id()`, `from_env()`.

### Getters / setters on mutable refs

- `set_` / `get_` prefix to distinguish from builder methods. Example: `set_extension()`, `get_extension()`.

### Free functions

- snake_case, reserved for cases without a natural receiver type. Most operations live as methods on their owning type — `OutputSchema::validate`, `ToolRegistry::execute`, `Model::compact_threshold`, `Agent::interpolate`.

### Tool structs

- `{Name}Tool`. Example: `ReadFileTool`, `BashTool`, `SpawnAgentTool`.

## Comment conventions

Comments explain purpose that cannot be derived from the code. Default to no comment: each one must earn its place. There is no doc-coverage requirement; document only when useful.

### Doc comments (`///`)

- First line: one sentence declaring the item's purpose. Noun phrase for types/fields, verb for functions. Never "This function…" / "Returns…": the signature already says so.
- Additional paragraphs only for a constraint, invariant, or non-obvious semantic (e.g. `None` means inherit, "must be called after `compile`", an ordering guarantee).
- Do not add docs to trivial getters, `Default::default`, `From` impls, or enum variants whose names are self-explanatory.
- **Coverage within a type is all-or-none.** Within a single enum / struct / trait declaration, either every variant / field / method has a `///` carrying real info, or none does. Partial coverage (one variant documented, the rest bare) is worse than none: it implies the undocumented members have been considered and found unimportant, which is almost always wrong.

### Module docs (`//!`)

- Every source file starts with a `//!` that places the file in the bigger picture and names the problem it solves. A reader landing on the first line should learn what job this file does for the rest of the crate, and what would be missing if the file did not exist.
- One sentence, crisp. Two only when the second adds real context the first cannot carry. Aim for something a reader can absorb in a glance.
- Do not inventory the file: no bulleted list of the types defined, no structural breakdown, no "contains X, Y, Z". The contents speak for themselves.
- Keep the `//!` even when the filename already says it. It is the entry point a reader lands on, and consistency across files matters.

### Line comments (`//`)

Allowed reasons only:

- Order-dependency / crash-safety: `Write mark BEFORE task file — crash-safe`.
- API quirk or workaround: `serde_json::Map is sorted alphabetically, so we format manually`.
- Non-obvious constraint: `Newest first so 'gpt-4' doesn't shadow 'gpt-4.1'`.

Not allowed:

- Restating what the code does.
- Task, PR, issue, or changelog references ("added for X flow", "fixes #123").
- Commented-out code: delete it; git remembers.
- Stub or aspirational markers. If a function is not implemented, return `AgenticError::NotImplemented("...")`; do not leave a promise in a comment.
- `TODO` / `FIXME` / `NOTE` markers. Track work in GitHub issues; a marker without a plan is noise.
- **Section dividers of any kind** (`// -----`, `// ====`, `// ####`, ASCII boxes). If a file has several concerns, split it. Rely on `impl` blocks, module boundaries, and type structure to organize code.

### Tests

- Test names carry intent; do not narrate setup. A comment is justified only to pin an architectural invariant the test guards.
- Module-level `//!` describing the test file's scope is fine where it exists today.

### Examples

**File intro (`//!`)** — every file, situating it in the crate and naming the problem it solves, in one crisp line:

```rust
// GOOD: places the file in the bigger picture and states the problem it solves
//! The execution kernel. Runs a compiled `Agent` turn by turn until it yields an `Output`.

//! Single error type every fallible API returns, so callers match one `Result` surface.

//! Shared backoff policy for transient provider failures: one implementation, every provider waits the same.

//! On-disk JSONL transcript of an agent run. Keeps runs inspectable and replayable after the process exits.

//! The agent's eyes on the filesystem. Lets a model read a file it did not receive in the prompt.

// BAD: inventories the file's contents
//! Agent execution loop.
//!
//! Three internal structs `run_loop` consumes:
//! - `AgentSpec` — compiled agent definition (name, model, prompts, tools).
//! - `LoopRuntime` — externals (provider, event handler, queue, session).
//! - `LoopState` — mutable per-run state (messages, counters).

// BAD: narrates implementation instead of purpose
//! This file contains the error type. It has an enum with many variants
//! and Display / Error / From implementations. See below for details.

// BAD: too bare; doesn't place the file in the crate or name a problem
//! Error type.
```

**Type / field doc comments (`///`)** — purpose and invariants, never restatement:

```rust
// GOOD: states purpose, flags the non-obvious semantic
/// The agent's compiled definition. `model: None` at template level means
/// "inherit from parent"; after `Agent::compile` it is always `Some`.
pub(crate) struct AgentSpec { ... }

/// `None` means inherit from parent.
pub model: Option<Model>,

// BAD: restates the field name
/// The model field.
pub model: Option<Model>,

// BAD: partial coverage on an enum; drop all or document all with real content
pub enum AgenticError {
    /// Anything raised by a Provider call.
    Provider(ProviderError),
    Tool { ... },           // no doc
    Io(std::io::Error),     // no doc
}
```

**Function doc comments (`///`)** — verb, one line, no "Returns…":

```rust
// GOOD
/// Build the environment metadata block for prepending to the first user message.
pub(crate) fn environment(working_directory: &Path) -> String { ... }

// BAD: signature already says it returns a String
/// This function returns a String containing the environment metadata.
pub(crate) fn environment(working_directory: &Path) -> String { ... }
```

**Line comments (`//`)** — only for order, quirk, or non-obvious constraint:

```rust
// GOOD: order-dependency / crash-safety
// Write mark BEFORE task file — crash-safe.
fs::write(&mark_path, b"")?;
fs::write(&task_path, body)?;

// GOOD: API quirk
// serde_json::Map is sorted alphabetically, so we format manually.
let mut entries = Vec::with_capacity(map.len());

// GOOD: non-obvious constraint
// Newest first so 'gpt-4' doesn't shadow 'gpt-4.1'.
models.sort_by(|a, b| b.released.cmp(&a.released));

// BAD: restates the code
// Increment the counter.
counter += 1;

// BAD: history / task reference
// Added for the streaming refactor (see PR #142).
fn drain_stream(...) { ... }

// BAD: stale marker
// TODO: implement compaction
fn compact(...) { Err(AgenticError::NotImplemented("context compaction")) }

// BAD: section divider
// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------
```

## README conventions

The README is the public face of the library. Keep it terse, example-driven, and scannable.

- **Section order is fixed**: Installation → Quick Start → Use Cases → API → Development. Each top-level section maps to a TOC link in the centered header.
- **Code before prose.** Lead every subsection with a minimal working example, then explain. A one-sentence intro is enough — don't over-narrate.
- **Tables for enumerations.** Use tables to list methods, events, built-in tools, guardrails, env variables, and inheritance rules. Prefer a table over a bulleted list whenever items share the same shape (name + description, or method + default + effect).
- **Shape tables with grouping columns.** When entries cluster (Agent / LLM Provider / Tool Usage for events; File / Search / Shell / Web / Utility for tools), use a leading empty-header column with bold group labels spanning multiple rows.
- **Use `>` blockquotes for callouts** — tips, prerequisites, cross-references. Example: `> Consider configuring your LLM provider (see [Environment](#environment)).`
- **Use cases show real output.** Every entry in Use Cases includes the invocation (`make use_case ...`) and a realistic JSON output block. No placeholder results.
- **Cross-link, don't repeat.** Reference the Environment section from anywhere that mentions provider setup; reference Inheritance when discussing sub-agents. Keep each fact in one place.
- **Headers**: `#` title, `##` top-level (Installation, API, Development), `###` features (Agent, Prompting, Sub-agents, Guardrails, Batch, Tools), `####` nested topics (Inheritance).
- **Voice**: direct, imperative, no marketing. "Give your agent access to simple tools" — not "empower your application with…". Keep the one-sentence tagline style ("A minimal Rust crate that…").
- **Keep examples minimal.** Show the smallest snippet that demonstrates the feature. Elide unrelated setup with `...` or obvious imports. Use `claude-haiku-4-5-20251001` / `claude-sonnet-4-20250514` as example models to stay consistent.
- **Update triggers**: a new builder method, a new tool, a new event kind, a new env variable, or a changed default all require a README edit in the matching table. Structural/internal changes do not.
