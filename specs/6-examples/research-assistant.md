# Example: Research Assistant

## Overview

A multi-turn conversational agent that demonstrates session persistence (resume), sub-agent spawning via `spawn_agent`, task tracking, and prompt builder with memory. Unlike the code-review example which is one-shot, this example runs in a REPL loop and supports `--resume` to continue previous sessions.

## Dependencies

- All library increments (core types, agent, tools, orchestration, persistence)

## Files

```
examples/research-assistant/Cargo.toml
examples/research-assistant/src/main.rs
```

## Specification

### Usage

```
research-assistant [OPTIONS]

OPTIONS:
    --resume <SESSION_ID>  Resume a previous session
    --model <MODEL>        Model (default: claude-sonnet-4-20250514)
    --provider <PROVIDER>  "anthropic" (default) or "litellm"
    --api-key <KEY>        API key (or ANTHROPIC_API_KEY env var)
    --base-url <URL>       Override provider base URL
```

### What This Example Demonstrates

| Library feature | How shown |
|---|---|
| Agent execution | Multi-turn with session resume |
| Output | Plain text conversation |
| Tools | Built-in + spawn_agent |
| Persistence | Session + Task stores |
| Orchestration | Sub-agent spawning (foreground) |
| Prompt construction | PromptBuilder with environment context and memory |

### Architecture

```rust
// Main REPL loop
let mut session_store = if let Some(id) = resume_id {
    // Resume: load existing transcript
    let entries = SessionStore::load(&base_dir, &id)?;
    Arc::new(Mutex::new(SessionStore::new(&base_dir, &id)))
} else {
    let id = generate_session_id();
    Arc::new(Mutex::new(SessionStore::new(&base_dir, &id)))
};

// Build prompt with memory and environment context
let mut prompt_builder = PromptBuilder::new(
    "You are a research assistant. Help users explore topics in depth. \
     Use spawn_agent to delegate research tasks to specialized sub-agents. \
     Track your research plan using task_create and task_update.".into(),
);
prompt_builder.environment_context(&EnvironmentContext::collect(&cwd));
prompt_builder.memory(&base_dir.join("memory"))?;

// Build sub-agents
let summarizer = AgentBuilder::new()
    .name("summarizer")
    .description("Summarize a document or topic")
    .model("claude-haiku-4-5-20241022")
    .system_prompt("Summarize the provided content concisely.")
    .build()?;

let fact_checker = AgentBuilder::new()
    .name("fact_checker")
    .description("Verify claims and check facts")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Verify the provided claims. Identify any inaccuracies.")
    .tool(ReadFileTool)
    .tool(GrepTool)
    .build()?;

// Build main agent with sub-agents and task tools
let agent = AgentBuilder::new()
    .name("researcher")
    .model("claude-sonnet-4-20250514")
    .prompt_builder(prompt_builder)
    .toolset(BuiltinToolset)
    .sub_agent(summarizer)
    .sub_agent(fact_checker)
    .build()?;

// REPL loop
loop {
    let input = read_user_input().await;
    if input == "/quit" { break; }

    let ctx = InvocationContext {
        input,
        state: HashMap::new(),
        working_directory: cwd.clone(),
        provider: provider.clone(),
        cost_tracker: cost_tracker.clone(),
        on_event: on_event.clone(),
        cancelled: cancelled.clone(),
        session_store: Some(session_store.clone()),
        command_queue: Some(queue.clone()),
        agent_id: generate_agent_id("researcher"),
    };

    match agent.run(ctx).await {
        Ok(output) => println!("{}", output.content),
        Err(e) => eprintln!("Error: {e}"),
    }
}

// Print session summary
println!("\nSession: {session_id}");
println!("Resume with: research-assistant --resume {session_id}");
println!("{}", cost_tracker.summary());
```

### Key Patterns Shown

1. **Session resume**: Uses `SessionStore::load()` to restore conversation history, then continues the agent loop from where it left off.

2. **Sub-agent spawning**: The main agent uses `spawn_agent` to delegate to `summarizer` and `fact_checker` sub-agents. The LLM decides when to delegate.

3. **Task tracking**: The agent uses `task_create` and `task_update` to maintain a research plan, tracking which topics have been investigated.

4. **Prompt builder with memory**: `PromptBuilder` loads `MEMORY.md` files so the agent remembers project context across sessions.

## Work Items

1. **`Cargo.toml`** — workspace example with dependencies
2. **`main.rs`** — full implementation
   - Session resume via `--resume` flag
   - REPL loop with multi-turn conversation
   - Sub-agent registration (summarizer, fact_checker)
   - PromptBuilder with environment context and memory
   - Task tracking integration
   - Event handler for streaming output

## Tests

| Test | Pattern | What it verifies |
|------|---------|--------------------|
| `session_resume_restores_history` | Direct | Create session with 2 entries, resume, verify history loaded |
| `sub_agent_spawning_works` | Direct | MockProvider returns spawn_agent tool call; verify sub-agent executed |

## Done Criteria

- `cargo build -p research-assistant` compiles
- `cargo test -p research-assistant` passes
- Multi-turn conversation works with session persistence
- `--resume` flag correctly restores previous session
- Sub-agent delegation works via spawn_agent
