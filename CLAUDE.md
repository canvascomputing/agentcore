# CLAUDE.md

> Keep this file up to date when the project structure or conventions change.
> Update `README.md` when the public API changes.

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
crates/agentcore/src/
  lib.rs                  public re-exports
  error.rs                AgenticError, Result

  provider/
    mod.rs                re-exports
    trait.rs              LlmProvider trait, CompletionRequest, ToolChoice, prewarm_connection
    types.rs              Message, ContentBlock, TokenUsage, StopReason, ModelResponse, StreamEvent
    model.rs              ModelSpec (Exact, Inherit)
    anthropic.rs          AnthropicProvider (with SSE streaming)
    openai.rs             OpenAiProvider, LiteLlmProvider, MistralProvider (with SSE streaming)
    stream.rs             StreamParser, SseEvent (streaming response parser)

  agent/
    mod.rs                re-exports
    trait.rs              Agent trait
    builder.rs            AgentBuilder
    loop.rs               AgentLoop struct, impl Agent, execute(), helpers, tests
    context.rs            InvocationContext (internal)
    event.rs              Event enum
    output.rs             AgentOutput, OutputSchema, StructuredOutputTool, validate_value
    prompts.rs            BehaviorPrompt (TaskExecution, ToolUsage, SafetyConcerns, Communication), ContextBuilder, EnvironmentContext
    pipeline.rs           Pipeline (batch execution with concurrency control)
    queue.rs              CommandQueue, QueuePriority, QueuedCommand (internal)

  tools/
    mod.rs                BuiltinToolset, re-exports
    tool.rs               Tool trait, ToolRegistry, ToolContext, ToolBuilder, execute_tool_calls
    read_file.rs          ReadFileTool
    write_file.rs         WriteFileTool
    edit_file.rs          EditFileTool
    glob.rs               GlobTool
    grep.rs               GrepTool
    list_directory.rs     ListDirectoryTool
    bash.rs               BashGlobTool (pattern-restricted), BashTool() (unrestricted)
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

Integration tests are in `crates/agentcore/tests/`. Shared helpers (provider setup, event handler, JSON output) are in `tests/common/mod.rs`. Run with `make test_integration`.
Use cases are in `crates/use-cases/src/cli/`. Run with `make use_case name=<name>`.

## Key conventions

- **No new dependencies without asking.** The crate is intentionally minimal (tokio, serde, serde_json, libc, reqwest, futures-util). Providers own a `reqwest::Client` directly — no transport abstraction.
- **No ad-hoc changes to critical types without a plan.** These types form the public API and are used across the entire codebase: `Agent`, `ToolContext`, `Event`, `Tool` trait, `AgentBuilder`, `CompletionRequest`, `AgentOutput`. Propose changes in a plan first.
- **Tools capture dependencies at construction time** via closures or struct fields. Do not use type-erased extension bags on context objects.
- **`tools/tool.rs` vs `tools/`**: `tool.rs` defines the trait and infrastructure (Tool, ToolRegistry, ToolBuilder, execute_tool_calls). Other files in `tools/` are concrete implementations.
- **`agent/` vs `provider/` vs `persistence/`**: `agent/` contains the agent loop, builder, context, events, output, and prompts. `provider/` contains LLM communication and estimated costs. `persistence/` contains internal disk storage (session transcripts, tasks).
- **Tests live inline** in each module as `#[cfg(test)] mod tests`. Use `MockProvider` and `TestHarness` from `testutil.rs`.

## Naming conventions

- **Builder methods**: bare nouns or compound nouns. No `with_` prefix.
  Exception: when the method name clashes with a trait method (e.g. `with_description` on BashGlobTool).
  Examples: `.name()`, `.model()`, `.tool()`, `.sub_agents()`, `.read_only()`.
- **Constructors**: `new()` for the primary constructor. Named constructors for variants: `open()`, `success()`, `error()`, `empty()`.
- **Getters/setters on mutable refs**: `set_`/`get_` prefix to distinguish from builder methods.
  Example: `set_extension()`, `get_extension()`.
- **Free functions**: snake_case. Example: `execute_tool_calls()`, `prewarm_connection()`.
- **Tool structs**: `{Name}Tool`. Example: `ReadFileTool`, `BashGlobTool`, `SpawnAgentTool`.
