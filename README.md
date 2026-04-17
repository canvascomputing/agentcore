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

<p align="center">Most agentic applications reimplement the same core. This crate provides that foundation: execution loop, built-in tools, agent orchestration, multi-provider support, schema-based output, and retry mechanisms.</p>

<p align="center"><em>Agentwerk combines "agent" with the German "Werk" (factory) — machinery for building agentic systems.</em></p>

---

## Installation

```bash
cargo add agentwerk
```

## Quick Start

```rust
use agentwerk::{Agent, GlobTool, provider_from_env};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = provider_from_env()?;

    let output = Agent::new()
        .provider(provider)
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

Two phases: a discovery agent finds files worth reading, then a pool of agents summarizes each file in parallel.

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

### Agent

Configures the agent's identity, tools, provider, and runtime options.

```rust
use agentwerk::Agent;

let output = Agent::new()
    .identity_prompt("You are a helpful assistant.")
    .instruction_prompt("What does src/main.rs do?")
    .model("claude-sonnet-4-20250514")
    .tool(ReadFileTool)
    .provider(provider)
    .run()
    .await?;
```

Configure a template once, clone it for each task, and override the fields
that vary per run. The `AgentPool` example below shows this.

#### Prompting

| Method | File variant | Purpose |
|--------|-------------|---------|
| `identity_prompt` | `identity_prompt_file` | Persistent identity of the agent |
| `instruction_prompt` | `instruction_prompt_file` | Task for the current run |
| `context_prompt` | `context_prompt_file` | Additional context appended after environment metadata (working directory, platform, OS version, date) |
| `behavior_prompt` | `behavior_prompt_file` | Override the default behavioral directives (`DEFAULT_BEHAVIOR_PROMPT`) |

```rust
Agent::new()
    .identity_prompt_file("prompts/identity.md")
    .instruction_prompt("Summarize the project.")
    .behavior_prompt_file("prompts/behavior.md")
```

Use `{key}` placeholders in the identity prompt and fill them with `template_variable`:

```rust
Agent::new()
    .identity_prompt("You are {role}. Respond in {language}.")
    .template_variable("role", json!("a code reviewer"))
    .template_variable("language", json!("German"))
```

#### Sub-agents

An agent can register other agents as sub-agents. The LLM can then call them
by name.

```rust
let researcher_base = Agent::new()
    .model("claude-haiku-4-5-20251001")
    .identity_prompt("Research this topic.")
    .tool(brave_search_tool())
    .max_turns(3);

let r1 = researcher_base.clone().name("researcher_1");
let r2 = researcher_base.clone().name("researcher_2");

let output = Agent::new()
    .name("orchestrator")
    .identity_prompt("Coordinate research.")
    .sub_agents([r1, r2])
```

Registered sub-agents are available to the LLM by name. The LLM can also
spawn ad-hoc agents at call time, supplying the prompt for that spawn.

##### Inheritance

A sub-agent is just an `Agent`; configure it with the normal builder methods.
Three rules are specific to running as a sub-agent:

| Behavior | Fields |
|---|---|
| Inherited | `provider`, `model`, `working_directory`, `event_handler`, `cancel_signal` |
| Shared | `command_queue`, `session_store` |
| Per-agent | `identity_prompt`, `behavior_prompt`, `context_prompt`, `tools`, `output_schema`, `max_turns`, `max_tokens`, `max_schema_retries`, `max_request_retries`, `request_retry_backoff_ms` |
| Per-invocation | `instruction_prompt` |

#### Guardrails

Per-agent limits for agentic execution. Omit a setter to use the default
(most default to "no limit"). When the parent's LLM spawns a sub-agent, it
can override any of these for a single spawn.

| Method | Default | What it does |
|--------|---------|-------------|
| `.max_turns(10)` | no limit | Stop after N agentic loop iterations |
| `.max_tokens(4096)` | provider default | Cap output tokens per LLM request |
| `.max_schema_retries(3)` | 10 | Retry structured output compliance |
| `.max_request_retries(5)` | 3 | Retry on transient API errors (429, 529, 5xx) |
| `.request_retry_backoff_ms(2000)` | 10,000 | Base delay for exponential backoff (`ms * 2^attempt`) |

To abort from outside the agent, use `.cancel_signal(signal)` — see
[Inheritance](#inheritance) for how it propagates across sub-agents.

### AgentPool

Executes agents concurrently with a configurable concurrency limit. Each
agent is configured independently with its own provider, prompts, and tools.
Submit work with `spawn()` and collect results with `next()` or `drain()`.

Result strategy:

- `CompletionOrder` (default) — results are returned as each agent finishes.
- `SpawnOrder` — results are returned in the order agents were spawned.

```rust
use agentwerk::{Agent, AgentPool, PoolStrategy, ReadFileTool};

let template = Agent::new()
    .model("claude-haiku-4-5-20251001")
    .tool(ReadFileTool);

let pool = AgentPool::new()
    .batch_size(10)
    .ordering(PoolStrategy::SpawnOrder);

for doc in ["document A", "document B"] {
    pool.spawn(
        template
            .clone()
            .provider(provider.clone())
            .instruction_prompt(format!("Summarize {doc}"))
    )
    .await;
}

let results = pool.drain().await; // Vec<(JobId, Result<AgentOutput>)>
```

`spawn()` can be called after the pool has started processing. If the pool
is at capacity, it waits for a free slot.

### Events

Emitted via `Agent.event_handler()` during execution. Each `Event` carries an `agent_name: String` plus an `EventKind` variant.

```rust
use agentwerk::{Event, EventKind};

let handler = Arc::new(|event: Event| match &event.kind {
    EventKind::ToolCallStart { tool_name, .. } => eprintln!("[{}] {}", event.agent_name, tool_name),
    EventKind::AgentEnd { turns } => eprintln!("[{}] done in {} turns", event.agent_name, turns),
    _ => {}
});
```

| | Kind | Description |
|-|-------|-------------|
| **Agent** | `AgentStart { description }` | Agent begins execution. `description` is set when the LLM spawned this agent as a sub-agent (the label it gave), and `None` for the top-level run. |
| | `AgentEnd { turns }` | Agent finishes with turn count |
| | `TurnStart` / `TurnEnd` | Turn boundaries |
| **LLM Provider** | `RequestStart` / `RequestEnd` | LLM request lifecycle |
| | `ResponseTextChunk` | Streamed text token |
| | `TokenUsage` | Token counts for a request |
| **Tool Usage** | `ToolCallStart` / `ToolCallEnd` | Tool execution lifecycle |


### Tools

Tools are functions the agent can call. Implement the `Tool` trait or use `ToolBuilder` for closures.

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

> Mark a tool as `.read_only(true)` when it has no side effects. When the LLM
> calls several tools in a single response, read-only calls run in parallel;
> everything else runs serially in order. Default is `false`.

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
let output = Agent::new()
    .output_schema(json!({
        "type": "object",
        "properties": { "category": { "type": "string" } },
        "required": ["category"]
    }))
    .max_schema_retries(3)  // retry if agent doesn't comply (default: 3)

    .run().await?;

output.response.unwrap()["category"]  // "billing"
```

Or load the schema from a file:

```rust
let output = Agent::new()
    .output_schema_file("schemas/category.json")
    .run().await?;
```

### LLM Request Composition

Each LLM request is assembled from four parts. Fields are listed in the order they appear in the request.

| Part | Type | Parameters | Description |
|------|------|--------|-------------|
| **model** | `String` | `model()` | The LLM model that processes the request |
| **max_tokens** | `Number` | `max_tokens()` | The maximum number of tokens the model can output |
| **tool_choice** | `ToolChoice` | `output_schema()` | A constraint that forces the model to call a specific tool |
| **system_prompt** | `String` | `identity_prompt()`<br>`behavior_prompt()` | Persistent instructions that define who the agent is and how it behaves |
| **message** | `Message[]` | `context_prompt()`<br>`instruction_prompt()` | The conversation history between user and assistant, starting with metadata, context, and the task |
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

