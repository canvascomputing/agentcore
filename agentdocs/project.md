# Project

agentwerk is a Rust crate for building LLM agents. An agent reads input, calls a provider, optionally invokes tools, and returns an output. The six sections below list the design principles the rest of the crate is measured against.

## Library, not framework

**The crate provides building blocks. The caller composes them.**

- No runtime to boot.
- No traits the caller must implement to get started.
- No required structure for the consuming application.
- Every feature is optional.

## Minimal surface

**Each abstraction must remove more complexity than it adds.**

- Dependencies are limited to tokio, serde, serde_json, reqwest, and futures-util.
- No transport abstractions, no plugin registries.
- Providers own a `reqwest::Client` directly.
- Indirection without a concrete benefit is not added.

## Composable

**Agents are cloned and modified, then bound to a `TicketSystem`.**

- No global registration step, no implicit state.
- Multiple agents share one `TicketSystem` and pick up tickets via label scope (Path B) or direct assignment (Path A).
- A ticket carries a `Schema`; the framework validates the agent's `done` result against it.

## Provider-agnostic

**The same agent code runs against any supported provider.**

- Anthropic, OpenAI, Mistral, and LiteLLM are supported.
- Switching providers changes only the `.model(...)` call.
- All providers share one retry policy.
- `from_env()` and `model_from_env()` (in `providers::environment`) select a provider and model from environment variables.

## Observe, do not prescribe

**The loop emits events. The caller decides what to do with them.**

- No built-in UI.
- No required logging.
- The event handler receives `Event { kind, ... }` at every lifecycle boundary.
- The handler may log, stream, store, or discard each event.

## Correctness over convenience

**Zero warnings, typed errors, no silent fallbacks.**

- The build MUST pass with `RUSTFLAGS="-D warnings"`: any warning fails it.
- Schema validation retries on mismatch, then fails explicitly.
- IMPORTANT: no blanket `From<io::Error>` or `From<serde_json::Error>`. Every conversion is an explicit mapping into a typed variant.
- Misconfigured builders panic at build time.
