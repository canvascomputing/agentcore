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

<p align="center">This crate provides a core implementation for agentic applications: execution loop, built-in tools, agent orchestration, multi-provider support, schema-based output, and retry mechanisms.</p>

<p align="center"><em>agentwerk combines "agent" with the German "Werk" (factory), thus machinery for building agentic systems.</em></p>

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

Example applications built with this project:

- [Project Scanner](crates/use-cases/src/project_scanner/): scan and analyze local files
- [Deep Research](crates/use-cases/src/deep_research/): multi-agent research with web search (requires `BRAVE_API_KEY`)
- [Model Pricing Tracker](crates/use-cases/src/model_pricing_tracker/): check model prices

```bash
make use_case                # list available names
make use_case name=<name>    # run one
```

> Consider configuring your LLM provider (see [Environment](#environment)).

## API

- [Providers](#providers): multi-provider support
- [Agent](#agent): the base interface
- [Models](#models): context window auto-detection
- [Prompting](#prompting): identity, instruction, context, behavior
- [Sub-agents](#sub-agents): orchestrate nested workers
- [Guardrails](#guardrails): retries, token caps, and turn limits
- [AgentPool](#agentpool): parallel execution
- [Events](#events): inspect agent and provider activity
- [Tools](#tools): built-in file, search, shell, and web tools
- [AgentOutput](#agentoutput): validated, schema-based responses
- [LLM Request Composition](#llm-request-composition): how a request is assembled
- [Todo](#todo): planned work

### Providers

You can integrate your agentic application with the following providers:

```rust
use agentwerk::{MistralProvider, AnthropicProvider, OpenAiProvider, LiteLLMProvider};

let provider = MistralProvider::new(key);
let provider = AnthropicProvider::new(key);
let provider = OpenAiProvider::new(key);
let provider = LiteLLMProvider::new(key);
```

### Agents

The `Agent` interface is the main entry point. Build with `Agent::new()`, chain configurations, then call `.run()`:

```rust
let output = Agent::new()
    .provider(provider)
    .model("claude-sonnet-4-20250514")
    .instruction_prompt("Summarize src/main.rs")
    .tool(ReadFileTool)
    .run()
    .await?;
```

### Models

You can configure each agent to use a single model:

```rust
Agent::new().model("claude-sonnet-4-20250514")
Agent::new().model_with_context_window_size("custom-model", 100_000)
```

`.model(id)` detects the context window size for models from supported providers. Use `.model_with_context_window_size` in case you need to set the context window size explicitly.

### Prompting

Prompts are the core ingredient of every agentic application. Here are different prompt types which can be used to drive your agent's behavior.

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

The following prompts can be configured:

| Method | Purpose |
|--------|---------|
| `identity_prompt(_file)` | Persistent identity of the agent |
| `instruction_prompt(_file)` | Task for the current run |
| `context_prompt(_file)` | Additional context appended after environment metadata (working directory, platform, OS version, date) |
| `behavior_prompt(_file)` | Override the default behavioral directives (`DEFAULT_BEHAVIOR_PROMPT`) |

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

### Sub-agents

Sub-agents allow orchestrator agents to launch her own workers. 
Orchestrator agents automatically have access to the `SpawnAgentTool`.

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

#### Inheritance

The following fields are inherited, shared or owned by the sub-agents:

| Behavior | Fields |
|---|---|
| Inherited | `provider`, `model`, `working_directory`, `event_handler`, `cancel_signal` |
| Shared | `command_queue`, `session_store` |
| Per sub-agent | `identity_prompt`, `instruction_prompt`, `behavior_prompt`, `context_prompt`, `tools`, `output_schema`, `max_turns`, `max_tokens`, `max_schema_retries`, `max_request_retries`, `request_retry_backoff_ms` |

### Guardrails

For protecting your budget or data, you can define clear execution rules for typical LLM failures:

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

Orchestrate complex workflows in parallel. Use different execution strategies:

- `CompletionOrder`: results are returned as each agent finishes (default).
- `SpawnOrder`: results are returned in the order agents were spawned.

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

You can inspect what your agent is doing and how the LLM provider API is used:

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
| **Agent** | `AgentStart` | Agent begins execution |
| | `AgentEnd` | Agent finishes execution |
| | `TurnStart` / `TurnEnd` | Turn boundaries |
| | `CompactTriggered` | Context approached maximum window length and compaction was invoked |
| **Provider** | `RequestStart` / `RequestEnd` | LLM request lifecycle |
| | `ResponseTextChunk` | Streamed text token arrived |
| | `TokenUsage` | Update on token counts due to provider requests |
| | `OutputTruncated` | Response was cut off because it exceeded allowed length |
| **Tool Usage** | `ToolCallStart` / `ToolCallEnd` | Tool execution lifecycle |


### Tools

Give your agent access to simple tools for driving tasks:

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

> Use `.read_only(true)` when a tool has no side effects. 
> If set, the the execution loop will run tools in parallel.

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

```rust
use agentwerk::{ReadFileTool, BashTool, TaskTool};

let agent = Agent::new()
    .tool(ReadFileTool)
    .tool(BashTool::new("git", "git *"))
    .tool(TaskTool::new(Path::new("/tmp/tasks")))
    .run().await?;
```

### AgentOutput

The result of running an agent.

```rust
output.response_raw            // Raw LLM output
output.response                // validated with schema

output.statistics.input_tokens // total input tokens
output.statistics.output_tokens// total output tokens
output.statistics.requests     // number of LLM requests
output.statistics.tool_calls   // number of tool calls
output.statistics.turns        // number of loop turns
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

Each LLM request is assembled from the following parts:

| Part | Type | Parameters | Description |
|------|------|--------|-------------|
| **model** | `String` | `model()` | The LLM model that processes the request |
| **max_tokens** | `Number` | `max_tokens()` | The maximum number of tokens the model can output |
| **tool_choice** | `ToolChoice` | `output_schema()` | A constraint that forces the model to call a specific tool |
| **system_prompt** | `String` | `identity_prompt()`<br>`behavior_prompt()` | Persistent instructions that define who the agent is and how it behaves |
| **message** | `Message[]` | `context_prompt()`<br>`instruction_prompt()` | The conversation history between user and assistant, starting with metadata, context, and the task |
| **tools** | `ToolDefinition[]` | `tool()` | The functions the model can call during execution |

### Todo

Planned additions to the crate:

- Context compression: summarize older messages when a conversation exceeds the LLM context window
- Session state handling: resume and persist agent sessions across runs

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
make use_case                                                 # list available
make use_case name=project-scanner -- ./                      # run one
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
make litellm                               # default: anthropic
make litellm LITELLM_PROVIDER=openai       # use OpenAI
make litellm LITELLM_PROVIDER=mistral      # use Mistral
```

### Environment

Use cases and integration tests use the following environment variables:

**General**
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

