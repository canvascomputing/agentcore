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
```

All code must compile with zero warnings (`RUSTFLAGS="-D warnings"`).

## Project structure

```
crates/agent/src/
  lib.rs                  public re-exports
  error.rs                AgenticError, Result

  provider/
    mod.rs                re-exports
    trait.rs              LlmProvider trait, CompletionRequest, ToolChoice, prewarm_connection
    types.rs              Message, ContentBlock, TokenUsage, StopReason, ModelResponse, StreamEvent
    model.rs              ModelSpec (Exact, Inherit)
    anthropic.rs          AnthropicProvider (with SSE streaming)
    openai.rs             OpenAiProvider, LiteLlmProvider, MistralProvider (with SSE streaming)
    sse.rs                SseParser, SseEvent (shared SSE line parser)
    cost.rs               CostTracker, ModelCosts, ModelUsage

  agent/
    mod.rs                re-exports
    trait.rs              Agent trait
    builder.rs            AgentBuilder
    loop.rs               AgentLoop struct, impl Agent, execute(), helpers, tests
    context.rs            InvocationContext (internal)
    event.rs              Event enum
    output.rs             AgentOutput, OutputSchema, StructuredOutputTool, validate_value
    prompts.rs            BehaviorPrompt, ContextBuilder, EnvironmentContext, prompt constants
    queue.rs              CommandQueue, QueuePriority, QueuedCommand

  tools/
    mod.rs                BuiltinToolset, re-exports
    tool.rs               Tool trait, ToolRegistry, ToolContext, ToolBuilder, execute_tool_calls
    read_file.rs          ReadFileTool
    write_file.rs         WriteFileTool
    edit_file.rs          EditFileTool
    glob.rs               GlobTool
    grep.rs               GrepTool
    list_directory.rs     ListDirectoryTool
    bash.rs               BashTool
    tool_search.rs        ToolSearchTool
    spawn_agent.rs        SpawnAgentTool
    task_tools.rs         task_create_tool, task_update_tool, task_list_tool, task_get_tool

  persistence/
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

Integration tests are in `crates/agent/tests/`. Run with `make test_integration`.
Use cases are in `crates/use-cases/src/cli/`. Run with `make use-case name=<name>`.

## Key conventions

- **No new dependencies without asking.** The crate is intentionally minimal (tokio, serde, serde_json, libc, reqwest, futures-util). Providers own a `reqwest::Client` directly — no transport abstraction.
- **No ad-hoc changes to critical types without a plan.** These types form the public API and are used across the entire codebase: `Agent`, `ToolContext`, `Event`, `Tool` trait, `AgentBuilder`, `CompletionRequest`, `AgentOutput`. Propose changes in a plan first.
- **Tools capture dependencies at construction time** via closures or struct fields. Do not use type-erased extension bags on context objects.
- **`tools/tool.rs` vs `tools/`**: `tool.rs` defines the trait and infrastructure (Tool, ToolRegistry, ToolBuilder, execute_tool_calls). Other files in `tools/` are concrete implementations.
- **`agent/` vs `provider/` vs `persistence/`**: `agent/` contains the agent loop, builder, context, events, output, and prompts (behavior defaults, constants). `provider/` contains LLM communication and cost tracking. `persistence/` contains disk storage. Tool descriptions live in their respective tool files.
- **Tests live inline** in each module as `#[cfg(test)] mod tests`. Use `MockProvider` and `TestHarness` from `testutil.rs`.
