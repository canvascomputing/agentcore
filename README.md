
# 🤖 `agent` - Minimal Agentic Framework

```
                          __   
 .---.-.-----.-----.-----|  |_ 
 |  _  |  _  |  -__|     |   _|
 |___._|___  |_____|__|__|____|
       |_____|                 
                    
  A minimal Rust framework for
  building agentic applications
```

- **Providers:** Anthropic, OpenAI-compatible (LiteLLM)
- **Tools:** read, write, edit, glob, grep, list, bash, tool search, custom
- **Output:** structured JSON Schema enforcement
- **Orchestration:** multi-agent spawning
- **Persistence:** session transcripts, task store
- **Tracking:** per-model cost breakdowns

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
make          # build
make test     # test
make fmt      # format
make example  # list and run examples
make litellm  # start LiteLLM proxy
```

### Examples

Examples auto-detect the provider: `ANTHROPIC_API_KEY`, `LITELLM_API_URL`, or a running proxy at `localhost:4000`.

```bash
make example name=llm_provider_call           # direct API call
make example name=agent_with_tools            # agent with custom tool
make example name=multi_agent_spawn           # multi-agent orchestration
make example name=task_and_session_store      # persistence
make example name=code_review                 # code review CLI
```

### LiteLLM

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Use Anthropic directly |
| `ANTHROPIC_BASE_URL` | API URL (default: `https://api.anthropic.com`) |
| `ANTHROPIC_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
| `LITELLM_API_KEY` | Auth key (optional) |
| `LITELLM_API_URL` | Use LiteLLM proxy (default: `http://localhost:4000`) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |

```bash
make litellm                     # default: anthropic
make litellm provider=openai     # uses OPENAI_API_KEY
```