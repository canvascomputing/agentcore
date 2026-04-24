# Project

agentwerk is a Rust crate for building LLM agents. An agent reads input, calls a provider, optionally invokes tools, and returns an output. The six sections below list the design principles the rest of the crate is measured against.

## 1. Library, not framework

**The crate provides building blocks. The caller composes them.**

- No runtime to boot.
- No traits the caller must implement to get started.
- No required structure for the consuming application.
- Every feature is optional.

## 2. Minimal surface

**Each abstraction must remove more complexity than it adds.**

- Dependencies are limited to tokio, serde, serde_json, libc, reqwest, and futures-util.
- No transport abstractions, no plugin registries.
- Providers own a `reqwest::Client` directly.
- Indirection without a concrete benefit is not added.

## 3. Composable

**Agents are cloned and modified, not registered.**

- No registration step, no global state.
- Child agents inherit configuration from the parent by default.
- `Batch` runs many clones of one template against different inputs.
- `SpawnAgentTool` lets a running agent launch another.

## 4. Provider-agnostic

**The same agent code runs against any supported provider.**

- Anthropic, OpenAI, Mistral, and LiteLLM are supported.
- Switching providers changes only the `.model(...)` call.
- All providers share one retry policy.
- `Provider::from_env()` selects a provider from environment variables.

## 5. Observe, do not prescribe

**The loop emits events. The caller decides what to do with them.**

- No built-in UI.
- No required logging.
- The event handler receives `Event { kind, ... }` at every lifecycle boundary.
- The handler may log, stream, store, or discard each event.

## 6. Correctness over convenience

**Zero warnings, typed errors, no silent fallbacks.**

- The build fails on any warning (`RUSTFLAGS="-D warnings"`).
- Schema validation retries on mismatch, then fails explicitly.
- No blanket `From<io::Error>` or `From<serde_json::Error>`.
- Misconfigured builders panic at build time.
