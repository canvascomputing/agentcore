# Layout

Where code lives and the rules that govern placement.

## Crates

**Two crates: one library, one example set.**

- `crates/agentwerk/` is the library.
- `crates/use-cases/` holds runnable example agents that depend on the library.
- Nothing in `use-cases` is re-exported by the library.

## Top-level files

**Each top-level source file is one concern the caller observes directly.**

- `lib.rs` holds public re-exports only.
- `error.rs` defines the categorical `Error` and the `Result` alias.
- `event.rs` and `output.rs` hold the observation and result types the loop emits.
- `werk.rs` holds `Werk`, `WerkProducing`, and `WerkOutputStream`: the parallel-execution surface.
- `util.rs` and `testutil.rs` hold shared helpers and the mock provider plus test harness.

## The `agent/` module

**Contains the builder, the compiled form, and the execution loop.**

- `agent.rs` holds the `Agent` builder, including `.hire(...)`, `.work()`, and the `compile` step.
- `spec.rs` holds the compiled `AgentSpec`, including the `hires: Vec<Agent>` field.
- `loop.rs` holds `run_loop`, `LoopRuntime`, and `LoopState`.
- `retain.rs` holds `AgentWorking`, `OutputFuture`, and the `Agent::retain` entry point.
- `prompts.rs`, `compact.rs`, and `work.rs` hold prompt constants, the compaction hook, and the `IncomingWork`.

## The `provider/` module

**Contains every LLM provider plus the shared transport code.**

- `trait.rs` defines `Provider`, `ModelRequest`, and `ToolChoice`.
- `types.rs` defines `Message`, `ContentBlock`, `TokenUsage`, `ModelResponse`, and `StreamEvent`.
- `anthropic.rs`, `openai.rs`, `litellm.rs`, and `mistral.rs` are concrete providers.
- `environment.rs` implements `Provider::from_env()`.
- `stream.rs` holds the SSE parser; the shared retry strategy lives in `util::Retry`.

## The `tools/` module

**`tool.rs` holds the trait and registry; every other file is one built-in tool.**

- `tool.rs` defines `ToolLike`, `Tool`, `ToolRegistry`, and `ToolContext`.
- `read_file.rs`, `write_file.rs`, `edit_file.rs`, `glob.rs`, `grep.rs`, and `list_directory.rs` are filesystem tools.
- `bash.rs` is the shell tool (restricted via `new()`, unrestricted via `unrestricted()`).
- `agent.rs`, `send_message.rs`, `task_tools.rs`, and `tool_search.rs` are orchestration tools.
- `web_fetch.rs` is the web fetch tool.

## Internal modules

**`persistence/` MUST stay `pub(crate)` and never be exposed to callers.**

- `session.rs` holds `SessionStore`, which appends JSONL transcripts.
- `task.rs` holds `TaskStore`, which stores file-locked tasks.
- `error.rs` holds `PersistenceError`.

## Tests

**Unit tests, integration tests, and inline tests live in three separate locations.**

- `crates/agentwerk/tests/unit/` holds mock-provider tests, bundled by `tests/unit.rs`.
- `crates/agentwerk/tests/integration/` holds real-provider tests, bundled by `tests/integration.rs`.
- `tests/integration/common.rs` holds shared integration helpers.
- Every module also carries its own `#[cfg(test)] mod tests`.
