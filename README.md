<div align="center">

# 🤖 `agent`

```
                             __   
    .---.-.-----.-----.-----|  |_ 
    |  _  |  _  |  -__|     |   _|
    |___._|___  |_____|__|__|____|
          |_____|                 

  A minimal Rust crate that gives any
  application agentic capabilities.
```

</div>

<p align="center"><code>Agentic execution loop</code> · <code>Basic tool implementations</code> · <code>Sub-agent orchestration</code> · <code>Anthropic, Mistral, OpenAI integration</code> · <code>Schema-based output</code> · <code>Cost tracking</code></p>

## Use Cases

Every agentic application, like OpenClaw or Claude Code, reimplements the same core functionality. This crate extracts that shared foundation into a minimal, dependency-light library.

Here are example applications built with this project.

> Consider setting your LLM provider's environment variables for key, model or base URL.

### [Project Scanner](crates/use-cases/src/project_scanner/)

Scans a directory and outputs a JSON summary with project description and languages used.

```bash
make use-case name=project-scanner -- ./
```

Output:
```json
{
  "summary": "A minimal Rust framework for building agentic LLM applications with tool use",
  "languages": ["Rust"]
}
```

### [Deep Research](crates/use-cases/src/deep_research/)

Spawns three researcher sub-agents in parallel, then aggregates their findings into a structured decision. Requires `BRAVE_API_KEY` for web search.

```bash
make use-case name=deep-research args="What constitutes a good life?"
```

Output:
```json
{
  "title": "What Constitutes a Good Life: A Multi-Perspective Analysis",
  "research": "A good life emerges from the convergence of philosophical wisdom, scientific research, and cultural understanding. Key elements include meaningful relationships and social connections, a sense of purpose and personal growth, physical and mental well-being, contributing to something beyond oneself, and living in accordance with personal values. While cultural contexts vary, common themes across traditions emphasize virtue, balance, gratitude, and the cultivation of both inner fulfillment and positive impact on others."
}
```

## Quick Start

```rust
use std::sync::Arc;
use agent::{AgentBuilder, AnthropicProvider, Event, ReadFileTool, GlobTool};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = Arc::new(AnthropicProvider::from_api_key(
        std::env::var("ANTHROPIC_API_KEY")?,
    ));

    let output = AgentBuilder::new()
        .model("claude-sonnet-4-20250514")
        .system_prompt("You are a helpful assistant that reads and explains code.")
        .tool(ReadFileTool)
        .tool(GlobTool)
        .provider(provider)
        .prompt("Find all Rust source files and describe what this project does.")
        .event_handler(Arc::new(|event| match &event {
            Event::RequestStart { model, .. } => eprintln!("[requesting {model}...]"),
            Event::ToolCallStart { tool_name, .. } => eprintln!("[tool] {tool_name}"),
            Event::AgentEnd { turns, .. } => eprintln!("[done in {turns} turns]"),
            _ => {}
        }))
        .run()
        .await?;

    eprintln!("\n\nDone in {} turns, ${:.4}", output.statistics.turns, output.statistics.costs);
    Ok(())
}
```

## API

Configure an `AgentBuilder` with a provider, model, tools, and prompt, then call `.run()` to get an `AgentOutput`. Stream `Event`s during execution.

### LlmProvider

Connect to any LLM. Providers own a `reqwest::Client` for connection pooling and SSE streaming.

```rust
use agent::{AnthropicProvider, MistralProvider, LiteLlmProvider};

let provider = AnthropicProvider::from_api_key(key);
let provider = MistralProvider::from_api_key(key);
let provider = LiteLlmProvider::from_api_key(key);

let client = reqwest::Client::new();                        // share a connection pool
let provider = AnthropicProvider::new(key, client);
```

### AgentBuilder

One builder for everything — agent definition, runtime context, and execution.

```rust
use agent::AgentBuilder;

let output = AgentBuilder::new()
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are a helpful assistant.")
    .tool(ReadFileTool)
    .provider(provider)
    .prompt("What does src/main.rs do?")
    .run()
    .await?;
```

`system_prompt` defines who the agent is (same every run). `prompt` is the task (changes per run). Use `{key}` placeholders in the system prompt and fill them with `template_var`:

```rust
AgentBuilder::new()
    .name("scanner")                      // agent identity (auto-generated if omitted)
    .description("Scans projects")        // human-readable description
    .system_prompt("Analyze {project}")
    .template_var("project", json!("my-app"))
    .working_directory(PathBuf::from("./src"))
    .user_context("Additional context injected into the prompt")
    .session_dir(PathBuf::from("./sessions"))  // persist transcripts to disk
```

#### Sub-agents

Use `.build()` to get `Arc<dyn Agent>` for registration as a sub-agent. Without `.model()`, a sub-agent inherits its parent's model at runtime. Clone the builder to create multiple similar agents:

```rust
let researcher_base = AgentBuilder::new()
    .model("claude-haiku-4-5-20251001")
    .system_prompt("Research this topic.")
    .tool(brave_search_tool())
    .read_only()                          // minimal prompts, lower max_tokens
    .max_turns(3);

let r1 = researcher_base.clone().name("researcher_1").build()?;
let r2 = researcher_base.clone().name("researcher_2").build()?;

let output = AgentBuilder::new()
    .name("orchestrator")
    .system_prompt("Coordinate research.")
    .sub_agent(r1)
    .sub_agent(r2)
```

#### Guardrails

```rust
AgentBuilder::new()
    .max_turns(10)         // stop after N agentic turns
    .max_budget(1.0)       // USD spend limit
    .max_tokens(4096)      // max output tokens per request
    .cancel_signal(cancel) // Arc<AtomicBool> for external abort
```

#### Behavior prompts

Agents include defaults for task execution, tool usage, action safety, and output efficiency. Override any:

```rust
use agent::BehaviorPrompt;

AgentBuilder::new()
    .behavior_prompt(BehaviorPrompt::TaskExecution, "Follow instructions exactly.")
```

### Event

Emitted via `.event_handler()` during execution.

| Event | Description |
|-------|-------------|
| `AgentStart` | Agent begins execution |
| `AgentEnd` | Agent finishes with turn count |
| `AgentError` | Agent encountered an error |
| `TurnStart` / `TurnEnd` | Turn boundaries |
| `RequestStart` / `RequestEnd` | LLM request lifecycle |
| `TextChunk` | Streamed text token |
| `ToolCallStart` / `ToolCallEnd` | Tool execution lifecycle |
| `TokenUsage` | Token counts for a request |
| `BudgetUsage` | Cost tracking update |

### Tools

Define what the agent can do. Read-only tools run concurrently.

```rust
use agent::{ToolBuilder, ToolResult};

let tool = ToolBuilder::new("greet", "Say hello")
    .schema(json!({...}))
    .read_only(true)
    .handler(|input, ctx| Box::pin(async move {
        Ok(ToolResult::success("Hello!"))
    }))
    .build();
```

Built-in tools:

- `ReadFileTool`, `WriteFileTool`, `EditFileTool` — file operations
- `GlobTool`, `GrepTool` — search by pattern or content
- `ListDirectoryTool` — directory listing with type and size
- `BashTool` — shell command execution
- `ToolSearchTool` — discover available tools by keyword
- `SpawnAgentTool` — delegate work to a sub-agent

### AgentOutput

The result of `.run()`.

```rust
output.response_raw            // free-form LLM text
output.statistics.costs        // total USD spent
output.statistics.input_tokens // total input tokens
output.statistics.output_tokens// total output tokens
output.statistics.requests     // number of LLM calls
output.statistics.tool_calls   // number of tool executions
output.statistics.turns        // number of agentic turns
```

With `.output_schema()`, the agent returns validated JSON in `output.response`:

```rust
let output = AgentBuilder::new()
    .output_schema(json!({
        "type": "object",
        "properties": { "category": { "type": "string" } },
        "required": ["category"]
    }))
    .max_schema_retries(3)  // retry if agent doesn't comply (default: 3)

    .run().await?;

output.response.unwrap()["category"]  // "billing"
```

## Development

```bash
make                   # build
make test              # unit tests
make test_integration  # integration tests (requires LLM provider)
make fmt               # format
make use-case          # list use cases
make litellm           # start LiteLLM proxy
```

### Environment

Auto-detect the provider from environment variables:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Use Anthropic directly |
| `ANTHROPIC_BASE_URL` | API URL (default: `https://api.anthropic.com`) |
| `ANTHROPIC_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
| `MISTRAL_API_KEY` | Use Mistral directly |
| `MISTRAL_BASE_URL` | API URL (default: `https://api.mistral.ai`) |
| `MISTRAL_MODEL` | Model (default: `mistral-medium-2508`) |
| `LITELLM_API_KEY` | Auth key (optional) |
| `LITELLM_API_URL` | Use LiteLLM proxy (default: `http://localhost:4000`) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
