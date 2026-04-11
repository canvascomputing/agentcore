
# 🤖 `agent` - Minimal Agentic Framework

```
                          __   
 .---.-.-----.-----.-----|  |_ 
 |  _  |  _  |  -__|     |   _|
 |___._|___  |_____|__|__|____|
       |_____|                 
                    
A minimal Rust framework for building agentic applications.
```

**Features**:
- LLM providers (Anthropic, OpenAI-compatible)
- Tool execution (file ops, search, shell, custom)
- Structured output (JSON Schema)
- Multi-agent orchestration
- Session and task persistence
- Cost tracking

## Quick Start

```rust
use agent::*;
use std::sync::Arc;

let provider = Arc::new(AnthropicProvider::new(api_key, transport));

// Build an agent with tools
let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are a helpful coding assistant.")
    .tool(ReadFileTool)
    .tool(GrepTool)
    .tool(BashTool)
    .max_turns(10)
    .build()?;

// Run it
let ctx = InvocationContext {
    input: "Find all TODO comments in this project".into(),
    provider,
    cost_tracker: CostTracker::new(),
    on_event: Arc::new(|event| match &event {
        Event::Text { text, .. } => print!("{text}"),
        Event::ToolStart { tool, .. } => eprintln!("[tool] {tool}"),
        _ => {}
    }),
    ..  // working_directory, cancelled, state, etc.
};

let output = agent.run(ctx).await?;
println!("{}", cost_tracker.summary());
```

## Development

```bash
make
make test
make fmt
make clean
```

Examples (require `ANTHROPIC_API_KEY`):

```bash
make example
make example name=llm_provider_call
make example name=agent_with_tools
make example name=multi_agent_spawn
make example name=task_and_session_store
make example name=code_review
```