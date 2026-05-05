# Layout

Where code lives and the rules that govern placement.

## Crates

**Two crates: one library, one example set.**

- `crates/agentwerk/` is the library.
- `crates/use-cases/` holds runnable example binaries that depend on the library.
- Nothing in `use-cases` is re-exported by the library.

## Top-level files

**Each top-level source file is one concern the caller observes directly.**

- `lib.rs` holds public re-exports only. The canonical caller surface (`Agent`, `Runnable`, `TicketSystem`, `Ticket`, `Status`, `Stats`, `Tool`, `ToolContext`, `ToolLike`, `ToolResult`, `Event` family, `Schema` family) lands at the crate root so callers do not reach into leaf modules.
- `event.rs` defines `Event`, `EventKind`, `PolicyKind`, `ToolFailureKind`, and `default_logger`.
- The `agents/`, `prompts/`, `providers/`, `schemas/`, and `tools/` modules each own their domain. The `agents/` and `tools/` modules also re-export their headline types so `use agentwerk::agents::{Agent, Runnable, TicketSystem}` and `use agentwerk::tools::BashTool` work without descending into leaf files.

## The `agents/` module

**Holds the per-agent builder, the ticket system, and the multi-agent loop.**

- `agent.rs` holds the `Agent` builder and ticket-dispatch helpers; an `Agent` carries a `Weak<TicketSystem>` stamped at `bind_agent` time.
- `tickets.rs` holds `Ticket`, `Status`, `TicketError`, and `TicketSystem`: the orchestrator that owns the shared queue, registered agents, policies, interrupt signal, and stats.
- `loop.rs` holds the `Runnable` trait (implemented by `TicketSystem`) and the per-agent loop driver.
- `memory.rs` holds `Memory`: the file-backed cross-ticket store. Mutations go through `add` / `replace` / `remove` / `rewrite`.
- `policy.rs` holds `Policies` and the limit checks the loop applies on each step.
- `stats.rs` holds `Stats`, `LoopStats`, and the run-wide counters and timings.

## The `providers/` module

**Holds every concrete provider plus the shared transport types.**

- `provider.rs` defines `Provider`, `ModelRequest`, `ProviderToolDefinition`, and `ToolChoice`.
- `types.rs` defines `Message`, `ContentBlock`, `TokenUsage`, `AsUserMessage`, `ResponseStatus`, and `StreamEvent`.
- `anthropic.rs`, `openai.rs`, `mistral.rs`, and `litellm.rs` are concrete providers.
- `environment.rs` implements `from_env()` and `model_from_env()`.
- `stream.rs` holds the SSE parser; `error.rs` holds `ProviderError`, `ProviderResult`, and `RequestErrorKind`.

## The `tools/` module

**`tool.rs` holds the trait and registry; every other file is one built-in tool or a helper.**

- `tool.rs` defines `ToolLike`, `Tool`, `ToolRegistry`, `ToolContext`, and `ToolCall`.
- `read_file.rs`, `write_file.rs`, `edit_file.rs`, `glob.rs`, `grep.rs`, and `list_directory.rs` are filesystem tools.
- `bash.rs` is the shell tool (restricted via `new()`, unrestricted via `unrestricted()`).
- `tickets/` holds `ManageTicketsTool`, `ReadTicketsTool`, and `WriteTicketsTool`.
- `memory.rs` is the model-facing wrapper around `Memory` (the store and curator live in `agents::memory`).
- `tool_search.rs` is the discovery surface for deferred tools.
- `web_fetch.rs` is the web fetch tool.
- `tool_file.rs` and `util.rs` are shared helpers; `error.rs` holds `ToolError`.

## The `prompts/` and `schemas/` modules

**Composable prompt assembly and JSON-Schema validation.**

- `prompts/builder.rs` and `prompts/section.rs` hold `PromptBuilder` and `Section`, which assemble role/context blocks.
- `schemas/mod.rs` holds `Schema`, `SchemaParseError`, `SchemaViolation`, and `format_violations`.

## Tests

**Integration tests live in their own directory; everything else is inline.**

- `crates/agentwerk/tests/integration/` holds real-provider tests, bundled by `tests/integration.rs`.
- `tests/integration/common.rs` holds shared integration helpers.
- Every module also carries its own `#[cfg(test)] mod tests` for mock-free unit coverage.
