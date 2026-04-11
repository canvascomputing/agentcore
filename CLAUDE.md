# CLAUDE.md

> Keep this file up to date when the project structure or conventions change.

## Build and test

```bash
make          # build (warnings are errors)
make test     # run tests (warnings are errors)
make fmt      # format code
make clean    # clean build artifacts
```

All code must compile with zero warnings (`RUSTFLAGS="-D warnings"`).

## Project structure

```
crates/agent/src/
  lib.rs              public re-exports
  error.rs            AgenticError, Result
  message.rs          Message, ContentBlock, Usage, StopReason, ModelResponse
  provider.rs         LlmProvider trait, AnthropicProvider, LiteLlmProvider, HttpTransport
  tool.rs             Tool trait, ToolRegistry, ToolContext, ToolBuilder, execute_tool_calls
  cost.rs             CostTracker, ModelCosts, ModelUsage
  prompt.rs           PromptBuilder, PromptSection, EnvironmentContext
  session.rs          SessionStore (JSONL transcripts)
  task.rs             TaskStore (file-based with locking)
  testutil.rs         MockProvider, MockTool, TestHarness, EventCollector
  agent/
    mod.rs            Agent trait, re-exports
    builder.rs        AgentBuilder
    context.rs        InvocationContext, generate_agent_id
    event.rs          Event enum
    loop.rs           LlmAgent, run_loop (core agent loop)
    output.rs         AgentOutput, OutputSchema, StructuredOutputTool, validate_value
    queue.rs          CommandQueue, QueuePriority, QueuedCommand
  tools/
    mod.rs            BuiltinToolset, re-exports
    read_file.rs      ReadFileTool
    write_file.rs     WriteFileTool
    edit_file.rs      EditFileTool
    glob.rs           GlobTool
    grep.rs           GrepTool
    list_directory.rs ListDirectoryTool
    bash.rs           BashTool
    tool_search.rs    ToolSearchTool
    spawn_agent.rs    SpawnAgentTool
    task_tools.rs     task_create_tool, task_update_tool, task_list_tool, task_get_tool
```

Examples are in `crates/agent/examples/`. Run with `make example name=<name>`.

## Key conventions

- **No new dependencies without asking.** The crate is intentionally minimal (tokio, serde, serde_json, libc). HTTP transport is injected by the caller.
- **No ad-hoc changes to critical types without a plan.** These types form the public API and are used across the entire codebase: `Agent`, `InvocationContext`, `ToolContext`, `Event`, `Tool` trait, `AgentBuilder`, `CompletionRequest`. Propose changes in a plan first.
- **Tools capture dependencies at construction time** via closures or struct fields. Do not use type-erased extension bags on context objects.
- **`tool.rs` vs `tools/`**: `tool.rs` defines the trait and infrastructure (Tool, ToolRegistry, ToolBuilder, execute_tool_calls). `tools/` contains concrete tool implementations.
- **`agent/` vs top-level modules**: `agent/` contains the agent loop, builder, context, events, and output. Top-level modules (provider.rs, message.rs, etc.) are shared infrastructure.
- **Tests live inline** in each module as `#[cfg(test)] mod tests`. Use `MockProvider` and `TestHarness` from `testutil.rs`.
