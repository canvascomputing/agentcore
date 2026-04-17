<p align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/logo.png" width="200" />
</p>

<h1 align="center">agentwerk</h1>

<p align="center">
  <strong>A minimal Rust crate that gives any application agentic capabilities.</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#api">API</a> •
  <a href="#development">Development</a>
</p>

<p align="center">Most agentic applications reimplement the same core: an execution loop, tool dispatch, and provider integration. This crate provides that foundation as a library.</p>

<p align="center">
  <a href="crates/agentwerk/src/agent/loop.rs">Agentic execution loop</a> ·
  <a href="crates/agentwerk/src/tools">Built-in tools</a> ·
  <a href="crates/agentwerk/src/tools/spawn_agent.rs">Agent orchestration</a> ·
  <a href="crates/agentwerk/src/provider">Anthropic, Mistral, OpenAI integration</a> ·
  <a href="crates/agentwerk/src/agent/output.rs">Schema-based output</a> ·
  <a href="crates/agentwerk/src/provider/retry.rs">Retry Mechanisms</a>
</p>

---

## Installation

```bash
cargo add agentwerk
```

## Quick Start

```rust
use std::sync::Arc;
use agentwerk::{AgentBuilder, AnthropicProvider, GlobTool};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = AnthropicProvider::from_env()?;

    let output = AgentBuilder::new()
        .provider(Arc::new(provider))
        .model(model)
        .instruction_prompt("Find all Rust source files.")
        .tool(GlobTool)
        .run()
        .await?;

    println!("{}", output.response_raw);
    Ok(())
}
```

## Use Cases

Example applications built with this project.

> Consider configuring your LLM provider (see [Environment](#environment)).

### [Project Scanner](crates/use-cases/src/project_scanner/)

Two-phase pipeline: a discovery agent finds files worth reading, then a pipeline of agents summarizes each file in parallel.

```bash
make use_case name=project-scanner -- ./
```

Output:
```json
{
  "languages": ["config", "docs", "rust"],
  "files": [
    {
      "file": "README.md",
      "summary": "Project documentation for agentwerk, a Rust crate for building agentic LLM applications.",
      "language": "docs"
    },
    {
      "file": "crates/agentwerk/src/agent/loop.rs",
      "summary": "Implements the main agent loop that calls an LLM iteratively and executes tool calls.",
      "language": "rust"
    }
  ]
}
```

### [Deep Research](crates/use-cases/src/deep_research/)

Spawns three researcher sub-agents in parallel, then aggregates their findings into a structured decision. Requires `BRAVE_API_KEY` for web search.

```bash
make use_case name=deep-research args="What constitutes a good life?"
```

Output:
```json
{
  "title": "What Constitutes a Good Life: A Multi-Perspective Analysis",
  "research": "A good life emerges from the convergence of philosophical wisdom, scientific research, and cultural understanding. Key elements include meaningful relationships and social connections, a sense of purpose and personal growth, physical and mental well-being, contributing to something beyond oneself, and living in accordance with personal values. While cultural contexts vary, common themes across traditions emphasize virtue, balance, gratitude, and the cultivation of both inner fulfillment and positive impact on others."
}
```

### [Model Pricing Tracker](crates/use-cases/src/model_pricing_tracker/)

Spawns a model checker and pricing researcher in parallel to gather current model pricing from provider websites, then outputs structured JSON.

```bash
make use_case name=model-pricing-tracker
```

Output:
```json
{
  "models": [
    {
      "input_per_million": 3.0,
      "model_id": "claude-sonnet-4-20250514",
      "output_per_million": 15.0,
      "provider": "anthropic"
    },
    {
      "input_per_million": 1.0,
      "model_id": "claude-haiku-4-5-20251001",
      "output_per_million": 5.0,
      "provider": "anthropic"
    }
  ]
}
```

## API

An agent is configured with a provider, model, tools, and prompt. Running it returns an output with the response and statistics. Events are emitted during execution for streaming and observability.

### LlmProvider

Providers for Anthropic, OpenAI-compatible, Mistral, and LiteLLM. Each owns a `reqwest::Client` for connection pooling and SSE streaming.

```rust
use agentwerk::{AnthropicProvider, OpenAiProvider};

let provider = AnthropicProvider::new(key);
let provider = OpenAiProvider::new(key);

// share a connection pool
let client = reqwest::Client::new();
let provider = AnthropicProvider::with_client(key, client);
```

### AgentBuilder

Configures the agent's identity, tools, provider, and runtime options.

```rust
use agentwerk::AgentBuilder;

let output = AgentBuilder::new()
    .identity_prompt("You are a helpful assistant.")
    .instruction_prompt("What does src/main.rs do?")
    .model("claude-sonnet-4-20250514")
    .tool(ReadFileTool)
    .provider(provider)
    .run()
    .await?;
```

#### Prompting

| Method | File variant | Purpose |
|--------|-------------|---------|
| `identity_prompt` | `identity_prompt_file` | Persistent identity of the agent |
| `instruction_prompt` | `instruction_prompt_file` | Task for the current run |
| `context_prompt` | `context_prompt_file` | Additional context alongside the instruction |
| `environment_prompt` | `environment_prompt_file` | Working directory, platform, date |
| `behavior_prompt` | `behavior_prompt_file` | Behavioral directives appended to the system prompt |

```rust
AgentBuilder::new()
    .identity_prompt_file("prompts/identity.md")
    .instruction_prompt("Summarize the project.")
    .behavior_prompt_file(BehaviorPrompt::Communication, "prompts/style.md")
```

Use `{key}` placeholders in the identity prompt and fill them with `template_variable`:

```rust
AgentBuilder::new()
    .identity_prompt("You are {role}. Respond in {language}.")
    .template_variable("role", json!("a code reviewer"))
    .template_variable("language", json!("German"))
```

#### Sub-agents

Building an agent returns a shareable handle for registration as a sub-agent. Without an explicit model, a sub-agent inherits its parent's model at runtime. Clone the builder to create multiple similar agents:

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

Set limits for agentic execution. You can set `UNLIMITED` to disable a limit.

| Method | Default | What it does |
|--------|---------|-------------|
| `.max_turns(10)` | `UNLIMITED` | Stop after N agentic loop iterations |
| `.max_tokens(4096)` | `UNLIMITED` | Cap output tokens per LLM request |
| `.max_schema_retries(3)` | 10 | Retry structured output compliance |
| `.max_request_retries(5)` | 3 | Retry on transient API errors (429, 529, 5xx) |
| `.request_retry_backoff_ms(2000)` | 10,000 | Base delay for exponential backoff (`ms * 2^attempt`) |
| `.cancel_signal(signal)` | — | Abort execution from outside the agent |

#### Behavior prompts

Agents ship with default behavior prompts appended to the identity prompt. Override any:

| Variant | Default | Description |
|---------|---------|-------------|
| `TaskExecution` | `DEFAULT_TASK_EXECUTION` | Read before modifying, don't add unrequested features, diagnose failures |
| `ToolUsage` | `DEFAULT_TOOL_USAGE` | Use dedicated tools over bash, parallelize independent calls |
| `SafetyConcerns` | `DEFAULT_SAFETY_CONCERNS` | Consider reversibility and impact, prefer reversible operations |
| `Communication` | `DEFAULT_COMMUNICATION_STYLE` | Be direct, concise, lead with the answer |

```rust
use agentwerk::BehaviorPrompt;

AgentBuilder::new()
    .behavior_prompt(BehaviorPrompt::TaskExecution, "Follow instructions exactly.")
    .behavior_prompt(BehaviorPrompt::Communication, "Always respond in JSON.")
```

### Pipelines

Execute multiple agents with controlled parallelism. Each agent is fully configured with its own provider, prompts, and tools. Results are returned in push order. Individual failures do not abort the pipeline.

```rust
use agentwerk::{Pipeline, AgentBuilder, ReadFileTool};

let mut pipeline = Pipeline::new()
    .batch_size(10)
    .max_request_retries(5);

pipeline.push(
    AgentBuilder::new()
        .provider(provider.clone())
        .instruction_prompt("Summarize document A")
        .tool(ReadFileTool)
);
pipeline.push(
    AgentBuilder::new()
        .provider(provider.clone())
        .instruction_prompt("Summarize document B")
        .tool(ReadFileTool)
);

let results = pipeline.run().await;
```

### Events

Emitted via `AgentBuilder.event_handler()` during execution.

| | Event | Description |
|-|-------|-------------|
| **Agent** | `AgentStart` | Agent begins execution |
| | `AgentEnd` | Agent finishes with turn count |
| | `TurnStart` / `TurnEnd` | Turn boundaries |
| **LLM Provider** | `RequestStart` / `RequestEnd` | LLM request lifecycle |
| | `ResponseTextChunk` | Streamed text token |
| | `TokenUsage` | Token counts for a request |
| **Tool Usage** | `ToolCallStart` / `ToolCallEnd` | Tool execution lifecycle |


### Tools

Tools are functions the agent can call. Implement the `Tool` trait or use `ToolBuilder` for closures. Read-only tools run concurrently.

```rust
use agentwerk::{ToolBuilder, ToolResult};

let tool = ToolBuilder::new("greet", "Say hello")
    .schema(json!({...}))
    .read_only(true)
    .handler(|input, ctx| Box::pin(async move {
        Ok(ToolResult::success("Hello!"))
    }))
    .build();
```

Built-in tools:

| | Tool | Description |
|-|------|-------------|
| **File** | `ReadFileTool` | Read a file with line numbers, offset, and limit |
| | `WriteFileTool` | Create or overwrite a file |
| | `EditFileTool` | Find-and-replace in a file |
| **Search** | `GlobTool` | Find files by pattern (e.g., `**/*.rs`) |
| | `GrepTool` | Search file contents by substring |
| | `ListDirectoryTool` | List directory entries with type and size |
| **Shell** | `BashTool::unrestricted()` | Execute any shell command |
| | `BashTool::new(name, pattern)` | Execute shell commands matching a glob pattern |
| **Web** | `WebFetchTool` | Fetch a URL and return its content as text |
| **Utility** | `SpawnAgentTool` | Delegate work to a sub-agent |
| | `TaskTool` | Persistent task management (create, update, list, get) |
| | `ToolSearchTool` | Discover available tools by keyword |

### AgentOutput

The result of running an agent.

```rust
output.response_raw            // free-form LLM text
output.response                // validated JSON if output_schema was set

output.statistics.input_tokens // total input tokens
output.statistics.output_tokens// total output tokens
output.statistics.requests     // number of LLM calls
output.statistics.tool_calls   // number of tool executions
output.statistics.turns        // number of agentic turns
```

With an output schema, the agent returns validated JSON:

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

### LLM Request Composition

Each LLM request is assembled from four parts. Fields are listed in the order they appear in the request.

| Part | Type | Parameters | Description |
|------|------|--------|-------------|
| **model** | `String` | `model()` | The LLM model that processes the request |
| **max_tokens** | `Number` | `max_tokens()` | The maximum number of tokens the model can output |
| **tool_choice** | `ToolChoice` | `output_schema()` | A constraint that forces the model to call a specific tool |
| **system_prompt** | `String` | `identity_prompt()`<br>`behavior_prompt(TaskExecution)`<br>`behavior_prompt(ToolUsage)`<br>`behavior_prompt(SafetyConcerns)`<br>`behavior_prompt(Communication)` | Persistent instructions that define who the agent is and how it behaves |
| **message** | `Message[]` | `environment_prompt()`<br>`context_prompt()`<br>`instruction_prompt()` | The conversation history between user and assistant, starting with environment, context, and the task |
| **tools** | `ToolDefinition[]` | `tool()` | The functions the model can call during execution |

## Development

### Building and testing

```bash
make                # build (warnings are errors)
make test           # unit tests
make fmt            # format code
make clean          # remove build artifacts
make update         # update dependencies
```

### Integration tests

> Consider configuring your LLM provider (see [Environment](#environment)).

```bash
make test_integration                     # run all
make test_integration name=bash_usage     # run one
```

### Use cases

```bash
make use_case                                              # list available
make use_case name=project-scanner -- ./                   # run one
make use_case name=deep-research args="What is a good life?"  # with arguments
```

### Publishing

```bash
make bump                  # bump patch version
make bump part=minor       # bump minor version
make publish               # publish to crates.io (runs tests first)
```

### LiteLLM proxy

Start a local LiteLLM proxy on port 4000 that forwards to a provider. Requires Docker.

```bash
make litellm                       # default: anthropic
make litellm LITELLM_PROVIDER=openai       # use OpenAI
make litellm LITELLM_PROVIDER=mistral      # use Mistral
```

### Environment

Use cases and integration tests pick up the LLM provider from these environment variables:

| Variable | Description |
|----------|-------------|
| `LITELLM_PROVIDER` | Explicit provider selection (`anthropic`, `mistral`, `openai`, `litellm`). Skips auto-detection |

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

**OpenAI**
| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key (required) |
| `OPENAI_BASE_URL` | API URL (default: `https://api.openai.com`) |
| `OPENAI_MODEL` | Model (default: `gpt-4o`) |

**LiteLLM proxy**
| Variable | Description |
|----------|-------------|
| `LITELLM_BASE_URL` | Proxy URL (default: `http://localhost:4000`) |
| `LITELLM_API_KEY` | Auth key (optional) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
| `LITELLM_PROVIDER` | LLM provider (default: `anthropic`, options: `anthropic`, `mistral`, `openai`) |

