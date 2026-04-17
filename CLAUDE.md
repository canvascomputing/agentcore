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
    trait.rs              LlmProvider trait, CompletionRequest, ToolChoice, prewarm_connection
    types.rs              Message, ContentBlock, TokenUsage, StopReason, ModelResponse, StreamEvent
    model.rs              ModelSpec (Exact, Inherit)
    anthropic.rs          AnthropicProvider (with SSE streaming)
    openai.rs             OpenAiProvider (with SSE streaming; includes litellm/mistral constructors)
    stream.rs             StreamParser, SseEvent (streaming response parser)
    retry.rs              compute_delay, DEFAULT_MAX_REQUEST_RETRIES, DEFAULT_BACKOFF_MS

  agent/
    mod.rs                re-exports
    werk.rs               Agent (config+runtime split), Runtime, AgentSpec, LoopState, run_loop
    event.rs              Event enum (AgentStart carries description for spawned children)
    output.rs             AgentOutput, OutputSchema, StructuredOutputTool, validate_value
    prompts.rs            BehaviorPrompt (TaskExecution, ToolUsage, SafetyConcerns, Communication), interpolate, environment context helper
    pool.rs               AgentPool, PoolOrdering, JobId (dynamic execution with concurrency control)
    queue.rs              CommandQueue, QueuePriority, QueuedCommand (internal)

  tools/
    mod.rs                re-exports
    tool.rs               Tool trait, ToolRegistry, ToolContext, ToolBuilder, execute_tool_calls
    read_file.rs          ReadFileTool
    write_file.rs         WriteFileTool
    edit_file.rs          EditFileTool
    glob.rs               GlobTool
    grep.rs               GrepTool
    list_directory.rs     ListDirectoryTool
    bash.rs               BashTool (pattern-restricted via new(), unrestricted via unrestricted())
    util.rs               glob_match, run_shell_command, DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS
    tool_search.rs        ToolSearchTool
    spawn_agent.rs        SpawnAgentTool
    task_tools.rs         TaskTool

  persistence/ (internal)
    mod.rs                re-exports
    session.rs            SessionStore (JSONL transcripts)
    task.rs               TaskStore (file-based with locking)

  testutil.rs             MockProvider, MockTool, TestHarness, EventCollector

crates/use-cases/src/
  lib.rs                    shared provider detection, env helpers
  project_scanner/
    main.rs                 project scanning CLI
  deep_research/
    main.rs                 multi-agent deep research with web search
```

Integration tests are in `crates/agentwerk/tests/`. Shared helpers (provider setup, event handler, JSON output) are in `tests/common/mod.rs`. Run with `make test_integration`.
Use cases are in `crates/use-cases/src/cli/`. Run with `make use_case name=<name>`.

## Key conventions

- **No new dependencies without asking.** The crate is intentionally minimal (tokio, serde, serde_json, libc, reqwest, futures-util). Providers own a `reqwest::Client` directly — no transport abstraction.
- **No ad-hoc changes to critical types without a plan.** These types form the public API and are used across the entire codebase: `Agent`, `ToolContext`, `Event`, `Tool` trait, `CompletionRequest`, `AgentOutput`, `AgentPool`. Propose changes in a plan first.
- **Tools capture dependencies at construction time** via closures or struct fields. The internal `ToolContext` handles (`runtime: Arc<Runtime>`, `caller_spec: Arc<AgentSpec>`) exist solely for the agent loop to give `SpawnAgentTool` / `ToolSearchTool` read access to loop state — do not use them for new tools.
- **`tools/tool.rs` vs `tools/`**: `tool.rs` defines the trait and infrastructure (Tool, ToolRegistry, ToolBuilder, execute_tool_calls). Other files in `tools/` are concrete implementations.
- **`agent/` vs `provider/` vs `persistence/`**: `agent/` contains the agent definition (`Agent`), execution loop (`Runtime` / `AgentSpec` / `LoopState` / `run_loop`), events, output, and prompts. `provider/` contains LLM communication and estimated costs. `persistence/` contains internal disk storage (session transcripts, tasks).
- **Prompt `_file` variants**: All prompt builder methods (`identity_prompt`, `instruction_prompt`, `behavior_prompt`, `context_prompt`) have `_file` counterparts (e.g. `identity_prompt_file(path)`) that load the prompt from disk. File-read errors are collected on the `Agent` and surfaced when `run()` is called.
- **Tests live inline** in each module as `#[cfg(test)] mod tests`. Use `MockProvider` and `TestHarness` from `testutil.rs`.

## Naming conventions

- **Builder methods**: bare nouns or compound nouns. No `with_` prefix.
  Exception: when the method name clashes with a trait method (e.g. `with_description` on BashTool).
  Examples: `.name()`, `.model()`, `.tool()`, `.sub_agents()`, `.read_only()`.
- **Constructors**: `new()` for the primary/simple constructor. `with_client()` for custom-client variants. Named constructors for semantics: `open()`, `unrestricted()`, `success()`, `error()`, `empty()`.
- **Getters/setters on mutable refs**: `set_`/`get_` prefix to distinguish from builder methods.
  Example: `set_extension()`, `get_extension()`.
- **Free functions**: snake_case. Example: `execute_tool_calls()`, `prewarm_connection()`.
- **Tool structs**: `{Name}Tool`. Example: `ReadFileTool`, `BashTool`, `SpawnAgentTool`.
