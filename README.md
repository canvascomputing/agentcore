<p align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentcore/main/logo.png" width="200" />
</p>

<h1 align="center">agentcore</h1>

<p align="center">
  <strong>A minimal Rust crate that gives any application agentic capabilities.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#api">API</a> •
  <a href="#development">Development</a>
</p>

<p align="center">Every agentic application, like OpenClaw or Claude Code, reimplements the same core functionality. This crate extracts that shared foundation into a minimal, dependency-light library.</p>

<p align="center">
  <a href="crates/agentcore/src/agent/loop.rs">Agentic execution loop</a> ·
  <a href="crates/agentcore/src/tools">Built-in tools</a> ·
  <a href="crates/agentcore/src/tools/spawn_agent.rs">Sub-agent orchestration</a> ·
  <a href="crates/agentcore/src/provider">Anthropic, Mistral, OpenAI integration</a> ·
  <a href="crates/agentcore/src/agent/output.rs">Schema-based output</a> ·
  <a href="crates/agentcore/src/provider/costs.rs">Cost Estimations</a>
</p>

---

## Quick Start

```rust
use std::sync::Arc;
use agentcore::{AgentBuilder, AnthropicProvider, ReadFileTool, GlobTool};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = Arc::new(AnthropicProvider::from_api_key(
        std::env::var("ANTHROPIC_API_KEY")?,
    ));

    let output = AgentBuilder::new()
        .provider(provider)
        .model("claude-sonnet-4-20250514")
        .instruction_prompt("Find all Rust source files and describe what this project does.")
        .tool(ReadFileTool)
        .tool(GlobTool)
        .run()
        .await?;

    println!("{}", output.response_raw);
    Ok(())
}
```

## Use Cases

Example applications built with this project.

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

## API

Configure an `AgentBuilder` with a provider, model, tools, and prompt, then call `.run()` to get an `AgentOutput`. Stream `Event`s during execution.

### LlmProvider

Connect to any LLM. Providers own a `reqwest::Client` for connection pooling and SSE streaming.

```rust
use agentcore::{AnthropicProvider, MistralProvider, LiteLlmProvider};

let provider = AnthropicProvider::from_api_key(key);
let provider = MistralProvider::from_api_key(key);
let provider = LiteLlmProvider::from_api_key(key);

let client = reqwest::Client::new();                        // share a connection pool
let provider = AnthropicProvider::new(key, client);
```

### AgentBuilder

One builder for everything — agent definition, runtime context, and execution.

```rust
use agentcore::AgentBuilder;

let output = AgentBuilder::new()
    .identity_prompt("You are a helpful assistant.")
    .instruction_prompt("What does src/main.rs do?")
    .model("claude-sonnet-4-20250514")
    .tool(ReadFileTool)
    .provider(provider)
    .run()
    .await?;
```

#### Prompt types

| Prompt | Purpose |
|--------|---------|
| `instruction_prompt` | The task for this run (required for `.run()`) |
| `context_prompt` | Additional context alongside the instruction |
| `behavior_prompt` | How the agent behaves — [defaults provided](#behavior-prompts) |
| `identity_prompt` | Who the agent is — persistent across runs |

Use `{key}` placeholders in the identity prompt and fill them with `template_variable` (or `template_variables`).

#### Sub-agents

Use `.build()` to get `Arc<dyn Agent>` for registration as a sub-agent. Without `.model()`, a sub-agent inherits its parent's model at runtime. Clone the builder to create multiple similar agents:

```rust
let researcher_base = AgentBuilder::new()
    .model("claude-haiku-4-5-20251001")
    .identity_prompt("Research this topic.")
    .tool(brave_search_tool())
    .max_turns(3);

let r1 = researcher_base.clone().name("researcher_1").build()?;
let r2 = researcher_base.clone().name("researcher_2").build()?;

let output = AgentBuilder::new()
    .name("orchestrator")
    .identity_prompt("Coordinate research.")
    .sub_agent(r1)
    .sub_agent(r2)
```

#### Guardrails

Limit agent execution to prevent runaway estimated costs or duration.

| Method | What it does |
|--------|-------------|
| `.max_turns(10)` | Stop after N agentic loop iterations |
| `.max_tokens(4096)` | Cap output tokens per LLM request |
| `.max_estimated_costs(5.0)` | Abort when estimated costs exceed budget |
| `.cancel_signal(signal)` | Abort via an external `Arc<AtomicBool>` (e.g., on Ctrl+C) |

Estimated costs require `.model_with_costs()` to provide token rates:

```rust
AgentBuilder::new()
    .model_with_costs(
        "claude-sonnet-4-20250514",
        ModelCosts::new(3.0, 15.0),  // $3/M input tokens, $15/M output tokens
    )
    .max_estimated_costs(5.0)
```

Costs are estimated as `(input_tokens × input_rate + output_tokens × output_rate) / 1,000,000`. Cache tokens are not included. Actual provider billing may differ.

#### Behavior prompts

Agents ship with default behavior prompts appended to the identity prompt. Override any:

| Variant | Default behavior |
|---------|-----------------|
| `TaskExecution` | Read before modifying, don't add unrequested features, diagnose failures |
| `ToolUsage` | Use dedicated tools over bash, parallelize independent calls |
| `ActionSafety` | Consider reversibility, confirm destructive operations |
| `OutputEfficiency` | Be concise, lead with the answer, skip filler |

```rust
use agentcore::BehaviorPrompt;

AgentBuilder::new()
    .behavior_prompt(BehaviorPrompt::TaskExecution, "Follow instructions exactly.")
    .behavior_prompt(BehaviorPrompt::OutputEfficiency, "Always respond in JSON.")
```

#### Sessions

Record every message exchanged during a run to disk as JSONL. The agent is not aware of this — recording happens transparently in the framework.

```rust
AgentBuilder::new()
    .session_dir(PathBuf::from("./data"))
```

Each run writes `<dir>/sessions/<id>/transcript.jsonl` — one JSON line per user message, assistant response, and tool result, with timestamps and token usage.

### Events

Emitted via `AgentBuilder.event_handler()` during execution.

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
| `EstimatedCostsUpdate` | Estimated costs update |

### Tools

Define what the agent can do. Read-only tools run concurrently.

```rust
use agentcore::{ToolBuilder, ToolResult};

let tool = ToolBuilder::new("greet", "Say hello")
    .schema(json!({...}))
    .read_only(true)
    .handler(|input, ctx| Box::pin(async move {
        Ok(ToolResult::success("Hello!"))
    }))
    .build();
```

Built-in tools:

| Tool | Description |
|------|-------------|
| `ReadFileTool` | Read a file with line numbers, offset, and limit |
| `WriteFileTool` | Create or overwrite a file |
| `EditFileTool` | Find-and-replace in a file |
| `GlobTool` | Find files by pattern (e.g., `**/*.rs`) |
| `GrepTool` | Search file contents by substring |
| `ListDirectoryTool` | List directory entries with type and size |
| `BashTool` | Execute a shell command |
| `ToolSearchTool` | Discover available tools by keyword |
| `SpawnAgentTool` | Delegate work to a sub-agent |
| `TaskTool` | Persistent task management (create, update, list, get) |
| `WebFetchTool` | Fetch a URL and return its content as text |

### AgentOutput

The result of `.run()`.

```rust
output.response_raw            // free-form LLM text
output.response                // validated JSON if output_schema was set
output.statistics.estimated_costs // estimated USD (requires .model_with_costs())
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
make fmt               # format code
make use-case          # list use cases
make bump              # bump patch version (part=minor or part=major)
make publish           # publish to crates.io (runs tests first)
make litellm           # start LiteLLM proxy
```

### Environment

Use cases and integration tests pick up the LLM provider from these environment variables:

**Anthropic**
| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key (required) |
| `ANTHROPIC_BASE_URL` | API URL (default: `https://api.anthropic.com`) |
| `ANTHROPIC_MODEL` | Model (default: `claude-sonnet-4-20250514`) |

**Mistral**
| Variable | Description |
|----------|-------------|
| `MISTRAL_API_KEY` | API key (required) |
| `MISTRAL_BASE_URL` | API URL (default: `https://api.mistral.ai`) |
| `MISTRAL_MODEL` | Model (default: `mistral-medium-2508`) |

**LiteLLM proxy**
| Variable | Description |
|----------|-------------|
| `LITELLM_API_URL` | Proxy URL (default: `http://localhost:4000`) |
| `LITELLM_API_KEY` | Auth key (optional) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
