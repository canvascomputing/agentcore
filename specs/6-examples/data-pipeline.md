# Example: Data Pipeline

## Overview

A parallel data processing pipeline that demonstrates background agent spawning via `spawn_agent(background: true)`, `ToolBuilder` closure-based tools, plain text output, and event streaming. An orchestrator agent spawns multiple background workers to process data in parallel, collecting results via the command queue.

## Dependencies

- All library increments (core types, agent, tools, orchestration, persistence)

## Files

```
examples/data-pipeline/Cargo.toml
examples/data-pipeline/src/main.rs
```

## Specification

### Usage

```
data-pipeline [OPTIONS] <INPUT_DIR>

OPTIONS:
    --workers <N>          Max parallel workers (default: 3)
    --model <MODEL>        Model (default: claude-haiku-4-5-20241022)
    --provider <PROVIDER>  "anthropic" (default) or "litellm"
    --api-key <KEY>        API key (or ANTHROPIC_API_KEY env var)
    --output <PATH>        Output file (default: results.json)
```

### What This Example Demonstrates

| Library feature | How shown |
|---|---|
| Agent execution | Parallel background agents |
| Output | Aggregated results |
| Tools | ToolBuilder closures |
| Persistence | Task store only |
| Orchestration | Sub-agent spawning (background) |

### Architecture

```rust
// Build closure-based tools with ToolBuilder
let list_files_tool = ToolBuilder::new("list_input_files", "List all input files to process")
    .read_only(true)
    .schema(json!({
        "type": "object",
        "properties": {
            "directory": { "type": "string" }
        },
        "required": ["directory"]
    }))
    .handler(|input, ctx| {
        Box::pin(async move {
            let dir = ctx.working_directory.join(
                input["directory"].as_str().unwrap_or(".")
            );
            let files: Vec<String> = std::fs::read_dir(&dir)
                .map_err(|e| AgenticError::Tool {
                    tool_name: "list_input_files".into(),
                    message: e.to_string(),
                })?
                .flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.path().display().to_string())
                .collect();
            Ok(ToolResult {
                content: serde_json::to_string(&files).unwrap(),
                is_error: false,
            })
        })
    })
    .build();

// Build a worker agent for processing individual files
let worker = AgentBuilder::new()
    .name("file_processor")
    .description("Process a single data file and extract key information")
    .model(&config.model)
    .system_prompt("You are a data processor. Read the given file, extract key \
                    information, and return a summary. Be concise.")
    .tool(ReadFileTool)
    .build()?;

// Build orchestrator that spawns workers in parallel
let orchestrator = AgentBuilder::new()
    .name("pipeline")
    .model(&config.model)
    .system_prompt(format!(
        "You are a data pipeline orchestrator. Process all files in the input directory.\n\n\
         Steps:\n\
         1. Use list_input_files to discover all files\n\
         2. For each file, spawn a file_processor agent with background: true\n\
         3. Wait for all background agents to complete (results arrive as notifications)\n\
         4. Aggregate the results and provide a final summary\n\n\
         Spawn up to {workers} agents in parallel by making multiple spawn_agent calls \
         in a single message.",
        workers = config.max_workers,
    ))
    .tool(list_files_tool)
    .tool(SpawnAgentTool::new())
    .sub_agent(worker)
    .build()?;

// Set up task store for tracking work items
let task_store = TaskStore::open(&base_dir, "pipeline");

// Event handler for progress streaming
let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| {
    match event {
        Event::AgentStart { agent } => eprintln!("[started] {agent}"),
        Event::AgentEnd { agent, turns } => eprintln!("[done] {agent} ({turns} turns)"),
        Event::ToolStart { agent, tool, .. } => eprintln!("[{agent}] {tool}"),
        _ => {}
    }
});

// One-shot execution — orchestrator manages parallelism internally
let ctx = InvocationContext {
    input: format!("Process all files in {}", config.input_dir),
    state: HashMap::new(),
    working_directory: PathBuf::from(&config.input_dir),
    provider: provider.clone(),
    cost_tracker: cost_tracker.clone(),
    on_event,
    cancelled: Arc::new(AtomicBool::new(false)),
    session_store: None,
    command_queue: Some(Arc::new(CommandQueue::new())),
    agent_id: "pipeline".into(),
};

let output = orchestrator.run(ctx).await?;

// Write aggregated results
std::fs::write(&config.output, &output.content)?;
eprintln!("Results written to {}", config.output);
eprintln!("{}", cost_tracker.summary());
```

### Key Patterns Shown

1. **Parallel background agents**: The orchestrator spawns multiple `file_processor` agents with `background: true` in a single message. Each runs concurrently via `tokio::spawn`.

2. **ToolBuilder closures**: The `list_input_files` tool is built using `ToolBuilder` with a closure handler, demonstrating the lightweight alternative to struct-based tools.

3. **Command queue for results**: Background agents deliver results via `command_queue.enqueue_notification()`. The orchestrator picks them up between turns.

4. **Event streaming**: The `on_event` callback logs agent start/end and tool execution for real-time progress monitoring.

5. **Task store**: Used to track which files have been processed, enabling restart/resume of partial runs.

## Work Items

1. **`Cargo.toml`** — workspace example with dependencies
2. **`main.rs`** — full implementation
   - ToolBuilder-based `list_input_files` tool
   - Worker sub-agent for file processing
   - Orchestrator with background agent spawning
   - Event handler for progress streaming
   - Result aggregation and output

## Tests

| Test | Pattern | What it verifies |
|------|---------|--------------------|
| `background_agents_complete_and_notify` | Direct | Spawn 3 background agents, verify all 3 notifications arrive via command queue |
| `toolbuilder_closure_tool_works` | Direct | Build a tool with ToolBuilder, call it, verify result |

## Done Criteria

- `cargo build -p data-pipeline` compiles
- `cargo test -p data-pipeline` passes
- Parallel background agents process files concurrently
- Results aggregated correctly from command queue notifications
- Event handler shows real-time progress
