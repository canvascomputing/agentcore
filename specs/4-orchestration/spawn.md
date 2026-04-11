# Agent Spawning

## Overview

This sub-plan implements multi-agent composition via the `spawn_agent` tool for runtime orchestration, plus task management tools. The LLM composes agents dynamically — sequential, parallel, or iterative — by calling `spawn_agent` at runtime.

## Dependencies

- [Agent system](../2-agent/loop.md): `Agent`, `AgentBuilder`, `AgentOutput`, `InvocationContext`, `Event`, `CommandQueue`
- [Persistence](../5-persistence/): `SessionStore`, `TaskStore`
- Transitively: [Core types](../1-base/types.md), [Tool system](../3-tools/traits.md)

## Files

```
crates/agent-tools/src/spawn_agent.rs
crates/agent-tools/src/task_create.rs
crates/agent-tools/src/task_update.rs
crates/agent-tools/src/task_list.rs
crates/agent-tools/src/task_get.rs
```

## Specification

### 10. Agent Spawning (`spawn_agent.rs`)

The `spawn_agent` tool spawns a single agent — either a named sub-agent or an ad-hoc agent with a prompt. It supports foreground (blocking) and background (`tokio::spawn`) execution modes.

The LLM composes agents dynamically:
- **Sequential**: Multiple `spawn_agent` calls across turns
- **Parallel**: Multiple `spawn_agent` calls in one message (with `background: true`)
- **Loop**: Keep calling `spawn_agent` until satisfied

#### 10.1 Tool Input Schema

```rust
#[derive(Deserialize)]
struct SpawnAgentInput {
    description: String,
    prompt: String,
    agent: Option<String>,
    model: Option<String>,
    max_turns: Option<u32>,
    background: Option<bool>,
}
```

#### 10.2 Execution

```rust
impl SpawnAgentTool {
    async fn execute(
        &self, input: SpawnAgentInput, ctx: InvocationContext,
    ) -> Result<AgentOutput> {
        let agent: Arc<dyn Agent> = if let Some(name) = &input.agent {
            self.find_agent(name)?
        } else {
            Arc::new(AgentBuilder::new()
                .name(&input.description)
                .model(input.model.as_deref().unwrap_or(&self.default_model))
                .system_prompt(&input.prompt)
                .max_turns(input.max_turns.unwrap_or(10))
                .build()?)
        };

        let child_ctx = ctx.child(&input.description).with_input(&input.prompt);

        if input.background.unwrap_or(false) {
            let agent_id = child_ctx.agent_id.clone();
            let queue = ctx.command_queue.clone();

            tokio::spawn(async move {
                let result = agent.run(child_ctx).await;
                if let Some(q) = queue {
                    match result {
                        Ok(output) => q.enqueue_notification(&agent_id, &output.content),
                        Err(e) => q.enqueue_notification(&agent_id, &format!("Failed: {e}")),
                    }
                }
            });

            Ok(AgentOutput {
                content: format!("Background agent '{}' started (id: {agent_id})", input.description),
                ..AgentOutput::empty(Usage::default())
            })
        } else {
            agent.run(child_ctx).await
        }
    }

    fn find_agent(&self, name: &str) -> Result<Arc<dyn Agent>> {
        self.sub_agents.iter()
            .find(|a| a.name() == name)
            .cloned()
            .ok_or_else(|| AgenticError::Tool {
                tool_name: "spawn_agent".into(),
                message: format!("No sub-agent named '{name}'"),
            })
    }
}
```

#### 10.3 Execution Modes

**Foreground (default):** The parent blocks until the agent completes. The tool returns the agent's output as a `ToolResult`.

**Background (`background: true`):** The agent is spawned via `tokio::spawn`. The tool returns immediately with `"Background agent '{description}' started (id: {agent_id})"`. When complete, a notification is enqueued via `command_queue.enqueue_notification()` at `Later` priority.

#### 10.4 Registering Sub-Agents

```rust
let orchestrator = AgentBuilder::new()
    .name("orchestrator")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Coordinate research tasks. Use spawn_agent to delegate work.")
    .tool(SpawnAgentTool::new())
    .sub_agent(web_searcher)
    .sub_agent(db_analyst)
    .build()?;
```

#### 10.5 What Gets Shared vs Isolated

| Resource | Shared? | Mechanism |
|----------|---------|-----------|
| Message history | **Isolated** | Each agent has its own `Vec<Message>` |
| Cost tracker | **Shared** | `InvocationContext.cost_tracker` via `Arc<Mutex>` |
| LLM provider | **Shared** | `InvocationContext.provider` via `Arc<dyn LlmProvider>` |
| Tool registry | **Isolated** | Each agent has its own tools (build-time config) |
| Cancellation | **Shared** | `InvocationContext.cancelled` via `Arc<AtomicBool>` |
| Session store | **Shared** | `InvocationContext.session_store` via `Arc<Mutex<SessionStore>>` |
| Command queue | **Shared** | `InvocationContext.command_queue` via `Arc<CommandQueue>` |

## Work Items

1. **`spawn_agent.rs`** — Spec Section 10
   - `SpawnAgentTool` — implements Tool trait
   - `SpawnAgentInput` — description, prompt, optional agent/model/max_turns/background
   - `execute()` — foreground (blocking) and background (`tokio::spawn`) execution
   - `find_agent()` — lookup registered sub-agents by name
   - Background notification delivery via `command_queue.enqueue_notification()`

2. **Task tools** — `task_create.rs`, `task_update.rs`, `task_list.rs`, `task_get.rs`
   - Each wraps the `TaskStore` from [Persistence](../5-persistence/task.md)
   - Implements `Tool` trait with appropriate input schemas

3. **Update `agent-tools/src/lib.rs`** — `BuiltinToolset` includes SpawnAgentTool and all task tools

## Tests

All tests use `MockProvider` from [`../1-base/types.md`](../1-base/types.md) and `TestHarness`/`EventCollector` from [`../2-agent/loop.md`](../2-agent/loop.md).

### `spawn_agent.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `spawn_agent_foreground` | Direct | `spawn_agent` without `background` blocks and returns agent output |
| `spawn_agent_background_delivers_notification` | Direct | `spawn_agent` with `background: true` returns immediately; notification arrives via command queue |
| `spawn_agent_named_sub_agent` | Direct | `spawn_agent` with `agent: "web_searcher"` runs the registered sub-agent |
| `spawn_agent_unknown_agent_errors` | Direct | `spawn_agent` with `agent: "nonexistent"` returns error |

### Task Tool Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `task_create_tool_returns_id` | Direct | Call with subject+description, verify task ID in result |
| `task_list_tool_returns_json` | Direct | Create 2 tasks, call list, verify JSON array |
| `task_get_tool_returns_details` | Direct | Create task, call get by ID, verify subject in result |
| `task_update_tool_changes_status` | Direct | Create, update to InProgress, get, verify status |

## Done Criteria

- `cargo build -p agent-core` compiles
- `cargo test -p agent-core` passes all spawn_agent tests
- `spawn_agent` foreground and background modes work correctly
- Background agent completion delivers notification via command queue
