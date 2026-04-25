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

- Dependencies are limited to tokio, serde, serde_json, libc, reqwest, and futures-util.
- No transport abstractions, no plugin registries.
- Providers own a `reqwest::Client` directly.
- Indirection without a concrete benefit is not added.

## Composable

**Agents are cloned and modified, not registered.**

- No registration step, no global state.
- Child agents inherit configuration from the parent by default.
- `Werk` runs many clones of one template against different inputs.
- `AgentTool` lets a running agent launch another.

## Provider-agnostic

**The same agent code runs against any supported provider.**

- Anthropic, OpenAI, Mistral, and LiteLLM are supported.
- Switching providers changes only the `.model(...)` call.
- All providers share one retry policy.
- `Provider::from_env()` selects a provider from environment variables.

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
