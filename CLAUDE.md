# CLAUDE.md

> Keep this file up to date when the project structure or conventions change.

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
    types.rs              Message, ContentBlock, Usage, StopReason, ModelResponse
    provider.rs           LlmProvider trait, CompletionRequest, ToolChoice, HttpTransport
    anthropic.rs          AnthropicProvider
    litellm.rs            LiteLlmProvider
    mistral.rs            MistralProvider
    cost.rs               CostTracker, ModelCosts, ModelUsage

  agent/
    mod.rs                re-exports
    agent.rs              Agent trait, AgentLoop struct
    builder.rs            AgentBuilder
    context.rs            InvocationContext, generate_agent_id
    event.rs              Event enum
    loop.rs               AgentLoop.execute(), extracted helpers, tests
    output.rs             AgentOutput, OutputSchema, StructuredOutputTool, validate_value
    queue.rs              CommandQueue, QueuePriority, QueuedCommand
    prompt.rs             PromptBuilder, PromptSection, EnvironmentContext, prompt constants
    prompt.rs             PromptBuilder, PromptSection, EnvironmentContext

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
  lib.rs                    shared transport/provider utilities
  project_scanner/
    main.rs                 project scanning CLI
  deep_research/
    main.rs                 multi-agent deep research with web search
```

Integration tests are in `crates/agent/tests/`. Run with `make test_integration`.
Use cases are in `crates/use-cases/src/cli/`. Run with `make use-case name=<name>`.

## Key conventions

- **No new dependencies without asking.** The crate is intentionally minimal (tokio, serde, serde_json, libc). HTTP transport is injected by the caller.
- **No ad-hoc changes to critical types without a plan.** These types form the public API and are used across the entire codebase: `Agent`, `InvocationContext`, `ToolContext`, `Event`, `Tool` trait, `AgentBuilder`, `CompletionRequest`, `AgentOutput`. Propose changes in a plan first.
- **Tools capture dependencies at construction time** via closures or struct fields. Do not use type-erased extension bags on context objects.
- **`tools/tool.rs` vs `tools/`**: `tool.rs` defines the trait and infrastructure (Tool, ToolRegistry, ToolBuilder, execute_tool_calls). Other files in `tools/` are concrete implementations.
- **`agent/` vs `provider/` vs `persistence/`**: `agent/` contains the agent loop, builder, context, events, output, and prompt. `provider/` contains LLM communication and cost tracking. `persistence/` contains disk storage.
- **Tests live inline** in each module as `#[cfg(test)] mod tests`. Use `MockProvider` and `TestHarness` from `testutil.rs`.
