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
make bump              # bump patch version (part=minor or part=major)
make publish           # publish to crates.io (runs tests first)
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
    werk.rs               Agent (Agent::DEFAULT_MAX_REQUEST_RETRIES / DEFAULT_BACKOFF_MS, Arc<AgentSpec> + AgentRuntime split, AgentRuntime::interpolate), Agent::compile
    loop.rs               AgentSpec (model accessor, system_prompt method, Default + DEFAULT_MAX_REQUEST_RETRIES/DEFAULT_BACKOFF_MS), LoopRuntime (environment), LoopState (initial), build_context_prompt, run_loop
    event.rs              AgentEvent enum (AgentStart carries description for spawned children)
    output.rs             AgentOutput, AgentStatus, OutputSchema (validate, retry_message)
    prompts.rs            DEFAULT_BEHAVIOR_PROMPT and structured-output constants
    pool.rs               AgentPool, AgentPoolStrategy, AgentJobId (dynamic execution with concurrency control)
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
- **No ad-hoc changes to critical types without a plan.** These types form the public API and are used across the entire codebase: `Agent`, `ToolContext`, `AgentEvent`, `Toolable` trait, `CompletionRequest`, `AgentOutput`, `AgentPool`. Propose changes in a plan first.
- **Tools capture dependencies at construction time** via closures or struct fields. The internal `ToolContext` handles (`runtime: Arc<LoopRuntime>`, `caller_spec: Arc<AgentSpec>`) exist solely for the agent loop to give `SpawnAgentTool` / `ToolSearchTool` read access to loop state — do not use them for new tools.
- **`tools/tool.rs` vs `tools/`**: `tool.rs` defines the trait and infrastructure (`Toolable` trait, `Tool` struct for ad-hoc tools, `ToolRegistry`, `ToolContext`). Other files in `tools/` are concrete implementations.
- **`agent/` vs `provider/` vs `persistence/`**: `agent/` contains the agent's builder surface (`Agent` in `werk.rs`) and everything the execution loop consumes (`AgentSpec` / `LoopRuntime` / `LoopState` / `run_loop` in `loop.rs`), events, output, and prompts. `provider/` contains LLM communication and estimated costs. `persistence/` contains internal disk storage (session transcripts, tasks).
- **`_file` variants**: All prompt builder methods (`identity_prompt`, `instruction_prompt`, `behavior_prompt`, `context_prompt`) and `output_schema` have `_file` counterparts (e.g. `identity_prompt_file(path)`, `output_schema_file(path)`) that load content from disk. File-read errors are collected on the `Agent` and surfaced when `run()` is called.
- **Tests live inline** in each module as `#[cfg(test)] mod tests`. Use `MockProvider` and `TestHarness` from `testutil.rs`.

## Naming conventions

### Type names

**The rule: domain-prefix any type whose bare name would be too generic to read self-documenting.** Visibility (pub vs pub(crate)) does NOT change this — both crate users and crate authors benefit from self-documenting names.

- **Generic single-word nouns always get the prefix.** `Status`, `Output`, `Pool`, `Spec`, `Runtime`, `Statistics`, `Event`, `EventKind` are too vague on their own. They become `AgentStatus`, `AgentOutput`, `AgentPool`, `AgentSpec`, `AgentRuntime`, `LoopRuntime`, `AgentStatistics`, `AgentEvent`, `AgentEventKind`.
- **Inherently specific compounds stand alone.** `LoopState`, `OutputSchema`, `CompactReason`, `CompletionRequest`, `CompletionResponse`, `TokenUsage`, `ContentBlock`, `StreamEvent`, `ToolCall`, `ToolRegistry`, `ResponseStatus`, `CommandQueue`, `SessionStore`, `TaskStore`, `ModelLookup`, `ProviderError`, `ProviderResult` already say what they are.
- **Vendor-prefixed types** follow the same logic — `AnthropicProvider`, `OpenAiProvider`, `MistralProvider`, `LiteLlmProvider`, `BashTool`, `ReadFileTool`. The prefix disambiguates which thing.
- **Acronyms follow Rust API guidelines**: `LiteLlmProvider`, not `LiteLLMProvider`. Already consistent: `OpenAiProvider`.
- **Two structs may not share a bare name in one module.** When that would happen (e.g. `LoopRuntime` next to `AgentRuntime`), keep both qualified — don't use a domain prefix as a tiebreaker for one.

**Documented exceptions** (kept against the rule, deliberately):
- `Message` (re-exported at root) — Anthropic API convention; "message" in an LLM crate is unambiguous and rename cost is high.
- `Result` (re-exported at root) — Rust idiom: crate-level `Result` aliases (`std::io::Result`, `anyhow::Result`) are standard; `agentwerk::Result` follows suit.
- `Agent` (re-exported at root) — THE central type; carries the crate name as its noun, no prefix needed.
- `Tool` (re-exported at root) — concrete struct for ad-hoc tools, mirrors the `Agent` shape (`Tool::new()` + builder methods, no `.build()` terminator). The companion trait is named `Toolable` to free the `Tool` name for the struct.

### Builder methods

- Bare nouns or compound nouns. No `with_` prefix.
  Exception: when the method name clashes with a trait method (e.g. `with_description` on BashTool).
  Examples: `.name()`, `.model()`, `.tool()`, `.sub_agents()`, `.read_only()`.

### Constructors

- `new()` for the primary/simple constructor.
- `with_client()` for custom-client variants.
- Named constructors for semantics: `open()`, `unrestricted()`, `success()`, `error()`, `empty()`, `from_id()`, `from_env()`.

### Getters / setters on mutable refs

- `set_` / `get_` prefix to distinguish from builder methods. Example: `set_extension()`, `get_extension()`.

### Free functions

- snake_case, reserved for cases without a natural receiver type. Most operations live as methods on their owning type — `OutputSchema::validate`, `ToolRegistry::execute`, `Model::compact_threshold`, `AgentRuntime::interpolate`.

### Tool structs

- `{Name}Tool`. Example: `ReadFileTool`, `BashTool`, `SpawnAgentTool`.

## README conventions

The README is the public face of the library. Keep it terse, example-driven, and scannable.

- **Section order is fixed**: Installation → Quick Start → Use Cases → API → Development. Each top-level section maps to a TOC link in the centered header.
- **Code before prose.** Lead every subsection with a minimal working example, then explain. A one-sentence intro is enough — don't over-narrate.
- **Tables for enumerations.** Use tables to list methods, events, built-in tools, guardrails, env variables, and inheritance rules. Prefer a table over a bulleted list whenever items share the same shape (name + description, or method + default + effect).
- **Shape tables with grouping columns.** When entries cluster (Agent / LLM Provider / Tool Usage for events; File / Search / Shell / Web / Utility for tools), use a leading empty-header column with bold group labels spanning multiple rows.
- **Use `>` blockquotes for callouts** — tips, prerequisites, cross-references. Example: `> Consider configuring your LLM provider (see [Environment](#environment)).`
- **Use cases show real output.** Every entry in Use Cases includes the invocation (`make use_case ...`) and a realistic JSON output block. No placeholder results.
- **Cross-link, don't repeat.** Reference the Environment section from anywhere that mentions provider setup; reference Inheritance when discussing sub-agents. Keep each fact in one place.
- **Headers**: `#` title, `##` top-level (Installation, API, Development), `###` features (Agent, Prompting, Sub-agents, Guardrails, AgentPool, Tools), `####` nested topics (Inheritance).
- **Voice**: direct, imperative, no marketing. "Give your agent access to simple tools" — not "empower your application with…". Keep the one-sentence tagline style ("A minimal Rust crate that…").
- **Keep examples minimal.** Show the smallest snippet that demonstrates the feature. Elide unrelated setup with `...` or obvious imports. Use `claude-haiku-4-5-20251001` / `claude-sonnet-4-20250514` as example models to stay consistent.
- **Update triggers**: a new builder method, a new tool, a new event kind, a new env variable, or a changed default all require a README edit in the matching table. Structural/internal changes do not.
