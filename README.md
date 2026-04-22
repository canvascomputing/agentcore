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

<p align="center"><em>agentwerk pairs "agent" with the German "Werk," a word that means both factory and artwork; machinery for building agentic systems, engineered like a craft.</em></p>

---

## Installation

```bash
cargo add agentwerk
```

## Quick Start

```rust
use agentwerk::{Agent, GlobTool};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = Agent::new()
        .provider_from_env()?
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

- [Terminal REPL](crates/use-cases/src/terminal_repl/): interactive terminal chat with less than 100 lines of code
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
- [Agents](#agents): the base interface
- [Models](#models): context window auto-detection
- [Prompting](#prompting): identity, instruction, context, behavior
- [Tools](#tools): built-in file, search, shell, and web tools
- [Events](#events): agent and provider activity
- [Guardrails](#guardrails): retries, token caps, and turn limits
- [AgentOutput](#agentoutput): validated, schema-based responses
- [Sub-agents](#sub-agents): nested workers
- [Batch](#batch): parallel execution
- [Todo](#todo): planned work

### Providers

You can integrate your agentic application with the following providers:

```rust
use agentwerk::{MistralProvider, AnthropicProvider, OpenAiProvider, LiteLlmProvider};

let provider = MistralProvider::new(key);
let provider = AnthropicProvider::new(key);
let provider = OpenAiProvider::new(key);
let provider = LiteLlmProvider::new(key);
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

#### Keep Agents Alive

Use `.spawn()` when you want to keep sending instructions to your agent:

```rust
let (agent, output) = Agent::new()
    .provider(provider)
    .model("claude-sonnet-4-20250514")
    .identity_prompt("Answer questions about the codebase.")
    .tool(ReadFileTool)
    .spawn();

agent.send("What does src/main.rs do?");
agent.send("Now summarize src/lib.rs.");

agent.cancel();

let output = output.await?;
```

The agent waits for the next `send` after each reply. Call `cancel()` to stop it.

Methods on the spawned agent:

| Method | Description |
|--------|-------------|
| `.send(instruction)` | Send a new instruction |
| `.cancel()` | Stop the agent |
| `.is_cancelled()` | Check if the agent was cancelled |
| `.is_stopped()` | Check if the agent has finished |
| `.clone()` | Get another handle to the same agent |


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
    .provider(provider)
    .model("claude-sonnet-4-20250514")
    .identity_prompt("You are a helpful assistant.")
    .instruction_prompt("What does src/main.rs do?")
    .tool(ReadFileTool)
    .run()
    .await?;
```

The following methods on `Agent` configure prompts:

| Method | Description |
|--------|-------------|
| `.identity_prompt(_file)` | Persistent identity of the agent |
| `.instruction_prompt(_file)` | Task for the current run |
| `.context_prompt(_file)` | Additional context appended after environment metadata (working directory, platform, OS version, date) |
| `.behavior_prompt(_file)` | Override the default behavioral directives (`DEFAULT_BEHAVIOR_PROMPT`) |

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

### Tools

Give your agent access to simple tools for driving tasks:

```rust
use agentwerk::{Tool, ToolResult};

let tool = Tool::new("greet", "Say hello")
    .schema(json!({...}))
    .read_only(true)
    .handler(|input, ctx| Box::pin(async move {
        Ok(ToolResult::success("Hello!"))
    }));
```

> Use `.read_only(true)` when a tool has no side effects. 
> If set, the the execution loop will run tools in parallel.

#### Built-in tools

| | Tool | Description |
|-|------|-------------|
| **File** | `ReadFileTool` | Read a file with line numbers, offset, and limit |
| | `WriteFileTool` | Create or overwrite a file |
| | `EditFileTool` | Find-and-replace in a file |
| **Search** | `GlobTool` | Find files by pattern |
| | `GrepTool` | Search file contents by substring |
| | `ListDirectoryTool` | List directory entries with type and size |
| **Web** | `WebFetchTool` | Fetch a URL and return its content as text |
| **Utility** | `BashTool` | Execute shell commands matching a glob pattern |
| | `SpawnAgentTool` | Delegate work to a sub-agent |
| | `SendMessageTool` | Send messages to other agents |
| | `TaskTool` | Perform task management |
| | `ToolSearchTool` | Discover available tools by keyword |

```rust
use agentwerk::{
    ReadFileTool, WriteFileTool, EditFileTool,
    GlobTool, GrepTool, ListDirectoryTool,
    WebFetchTool, SpawnAgentTool, BashTool,
    SendMessageTool, TaskTool, ToolSearchTool,
};

let agent = Agent::new()
    .tool(ReadFileTool)
    .tool(WriteFileTool)
    .tool(EditFileTool)
    .tool(GlobTool)
    .tool(GrepTool)
    .tool(ListDirectoryTool)
    .tool(WebFetchTool)
    .tool(SpawnAgentTool)
    .tool(BashTool::new("git", "git *"))
    .tool(SendMessageTool)
    .tool(TaskTool::new(Path::new("/tmp/tasks")))
    .tool(ToolSearchTool)
    .run().await?;
```

### Events

You can inspect what your agent is doing and how the LLM provider API is used:

```rust
use agentwerk::{Event, EventKind};

let handler = Arc::new(|event: Event| match &event.kind {
    EventKind::ToolCallStarted { tool_name, .. } => {
        eprintln!("[{}] → {tool_name}", event.agent_name);
    }
    EventKind::ToolCallError { tool_name, error, .. } => {
        eprintln!("[{}] ✗ {tool_name}: {error}", event.agent_name);
    }
    EventKind::AgentFinished { turns, status } => {
        eprintln!("[{}] done in {turns} turns ({status:?})", event.agent_name);
    }
    _ => {}
});
```

> When `.event_handler(...)` is not set, agents log tool activity and lifecycle events to
> stderr via `Event::default_logger()`. You can call `.silent()` on the agent to silence the output.

| | Kind | Description |
|-|------|-------------|
| **Agent** | `AgentStarted` | Agent run began |
| | `AgentFinished` | Agent run finished |
| | `TurnStarted` | Agentic loop turn began |
| | `TurnFinished` | Agentic loop turn finished |
| | `AgentPaused` | Keep-alive agent is waiting for new input |
| | `AgentResumed` | Keep-alive agent resumed after being paused |
| **Provider** | `RequestStarted` | Provider request began |
| | `RequestFinished` | Provider request finished |
| | `RequestRetried` | Transient provider error triggered a retry |
| | `RequestError` | Provider request failed after exhausting retries |
| | `TextChunkReceived` | Streamed text token arrived |
| | `TokensReported` | Provider reported token counts for the last request |
| **Context** | `OutputTruncated` | Response was cut off at the configured length cap |
| | `ContextCompacted` | Conversation history was compacted to stay within the model's window |
| | `InputBudgetExhausted` | Cumulative input tokens crossed `max_input_tokens` |
| | `OutputBudgetExhausted` | Cumulative output tokens crossed `max_output_tokens` |
| **Tool** | `ToolCallStarted` | Tool invocation began |
| | `ToolCallFinished` | Tool invocation succeeded |
| | `ToolCallError` | Tool invocation failed |

### Guardrails

For protecting your budget or data, you can define clear execution rules for typical LLM failures. You can configure the following on your `Agent`:

| Method | Default | Description |
|--------|---------|-------------|
| `.max_turns(10)` | no limit | Stop after N agentic loop iterations |
| `.max_request_tokens(4096)` | provider default | Cap output tokens per LLM request |
| `.max_input_tokens(200_000)` | no limit | Cap cumulative input tokens across the whole run |
| `.max_output_tokens(50_000)` | no limit | Cap cumulative output tokens across the whole run |
| `.max_schema_retries(3)` | 10 | Retry structured output compliance |
| `.max_request_retries(5)` | 10 | Retry on API errors (429, 529, 5xx) |
| `.request_retry_delay(2000)` | 500 | Base delay in milliseconds for exponential backoff between request retries |

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
    .max_schema_retries(3)
    .run().await?;

output.response.unwrap()["category"]
```

Or load the schema from a file:

```rust
let output = Agent::new()
    .output_schema_file("schemas/category.json")
    .run().await?;
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
| Per sub-agent | `identity_prompt`, `instruction_prompt`, `behavior_prompt`, `context_prompt`, `tools`, `output_schema`, `max_turns`, `max_request_tokens`, `max_input_tokens`, `max_output_tokens`, `max_schema_retries`, `max_request_retries`, `request_retry_delay` |

### Batch

Run many agents in parallel with a fixed concurrency cap. Yields results as
each agent finishes; correlate each result to its input via
[`AgentOutput.name`](#agentoutput).

```rust
use agentwerk::{batch, Agent, ReadFileTool};
use futures_util::StreamExt;

let template = Agent::new()
    .model("claude-haiku-4-5-20251001")
    .tool(ReadFileTool)
    .provider(provider);

let agents = ["document A", "document B"].iter().map(|doc| {
    template
        .clone()
        .name(format!("summarize-{doc}"))
        .instruction_prompt(format!("Summarize {doc}"))
});

let results: Vec<_> = batch(agents, 10).collect().await;

for r in results {
    let out = r?;
    println!("{}: {}", out.name, out.response_raw);
}
```

`batch` accepts any `IntoIterator<Item = Agent>` and returns an
`impl Stream<Item = Result<AgentOutput>>`. Use `StreamExt::collect` to gather
every result, or combine with `for_each_concurrent` / `map` to process results
as they arrive.

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

### Local inference servers

agentwerk relies on server-side tool calling. You can enable it through the following flags:

| Server | Flag |
|---|---|
| vLLM | `--enable-auto-tool-choice --tool-call-parser <parser>` |
| SGLang | `--tool-call-parser <parser>` |
| llama.cpp `llama-server` | `--jinja` (enables tool calling) |
| Ollama | tool calling enabled by default |

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

