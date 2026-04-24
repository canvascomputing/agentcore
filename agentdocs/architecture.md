# Architecture

Core data structures, how they interact, and the decisions that constrain them.

## 1. Core types

**Every agent run routes through the same set of types.**

- `Agent` is the mutable builder the caller configures.
- `AgentSpec` (held as `Arc`) is the immutable compiled snapshot of an agent.
- `LoopRuntime` holds the externals the loop needs: provider, event handler, session store, command queue.
- `LoopState` holds the mutable per-run state: messages, token usage, turn counter, pending tool calls.
- `run_loop` is the turn-by-turn execution function; it takes a runtime and a state.

## 2. Transport and tools

**Two traits every call goes through.**

- `Provider` completes a `ModelRequest` into a `ModelResponse` or a `StreamEvent` stream.
- `ToolLike` is the interface for any function the model can call.
- `ToolRegistry` dispatches a tool call by name.
- `ToolContext` is a read-only handle into the loop, used only by `SpawnAgentTool` and `ToolSearchTool`.

## 3. Observation and result

**The loop speaks to the caller through events and one final `Output`.**

- `Event { kind }` is emitted at every lifecycle boundary.
- `EventKind` lists what happened (`AgentStarted`, `ToolCallFinished`, `RequestRetried`, and so on).
- `Output` carries `outcome`, `errors`, `statistics`, and `messages`.
- `Batch`, `BatchHandle`, and `BatchOutputStream` run many agents against different inputs.
- `CommandQueue` carries messages that other agents push through `SendMessageTool`.

## 4. Running one agent

**The lifecycle of one run is always the same.**

- `.run()` or `.spawn()` invokes `Agent::compile`, producing `Arc<AgentSpec>`.
- `run_loop` builds a `LoopRuntime` and an initial `LoopState`.
- Each turn sends the current messages to the provider and appends the response.
- Tool calls in the response go through `ToolRegistry::execute` and their results are appended as messages.
- The loop returns `Output` when the model ends, a limit is reached, or the caller cancels.

## 5. Spawning sub-agents

**`SpawnAgentTool` is the only consumer of `ToolContext` that may read the runtime.**

- It inherits model, prompts, and tool set from `caller_spec` unless overridden.
- It shares `runtime` so provider, event handler, and command queue are the same.
- It returns an `AgentHandle` whose `OutputFuture` resolves to an `Output`.
- `.spawn()` catches panics at the tokio boundary and surfaces them as `AgentError::AgentCrashed`.

## 6. Peer messaging

**Agents send messages by pushing onto each other's `CommandQueue`.**

- `SendMessageTool` pushes a `QueuedCommand` with a priority.
- Each `run_loop` drains its queue at turn boundaries.
- The queue sits on `LoopRuntime`, so any tool with access to the runtime can push.

## 7. Running many agents

**`Batch` clones one template per input and runs them together.**

- `.run(...)` awaits every clone and returns `Vec<Output>`.
- `.spawn(...)` returns a `BatchHandle` and a `BatchOutputStream` that yields each `Output` as it completes.
- The template is cloned, not shared, so per-input modifications are safe.

## 8. Error model

**`Error` is categorical. Three variants wrap the domain sub-enums.**

- `Provider(ProviderError)` covers transport failures.
- `Agent(AgentError)` covers run-lifecycle and builder failures.
- `Tool(ToolError)` is a flat struct (`tool_name`, `message`) for infrastructure-level tool failures.
- Most tool failures surface as `ToolResult::Error` on the message channel, not as `ToolError`.
- Internal-only errors (`PersistenceError`, `SchemaViolation`) stay `pub(crate)` and are routed into the public error types by their consumer.

## 9. Termination contract

**Once the loop starts, every termination returns `Ok(Output)`.**

- `Output.outcome` is `Completed`, `Cancelled`, or `Failed`.
- `Output.errors` logs every failure seen during the run; on `Failed` the last entry is the cause.
- Builder misconfiguration panics at build time, never inside `.run()`.
- Budget and turn limits land as `AgentError::PolicyViolated { kind, usage, limit }` where `kind` is `Turns`, `InputTokens`, `OutputTokens`, or `SchemaRetries`.

## 10. Persistence

**Both stores are `pub(crate)` and support observability, not callers.**

- `SessionStore` appends a JSONL entry per turn; runs remain inspectable after the process exits.
- `TaskStore` uses file locks per task.
- `TaskStore` writes the mark-file before the payload-file so partial writes are recoverable.

## 11. Critical decisions

**Non-obvious choices. Propose a plan before deviating.**

- No new dependencies without asking.
- No ad-hoc changes to `Agent`, `ToolContext`, `Event`, `ToolLike`, `ModelRequest`, `Output`, or `Batch`.
- Tools capture dependencies at construction; `ToolContext` is not used by new tools.
- Prompt and schema files resolve eagerly; missing files panic at build time.
- No blanket `From<io::Error>` or `From<serde_json::Error>`; each domain declares its own conversion.
