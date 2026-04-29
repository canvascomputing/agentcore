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
use agentwerk::tools::GlobTool;
use agentwerk::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = Agent::new()
        .provider_from_env()?
        .model("mistral-large-2512")
        .tool(GlobTool)
        .work("Find all Rust source files.")
        .await?;

    println!("{}", output.response_raw);
    Ok(())
}
```

## Use Cases

Example applications built with this project:

- [Terminal REPL](crates/use-cases/src/terminal_repl/): interactive terminal chat with less than 100 lines of code
- [Security Scanner](crates/use-cases/src/project_scanner/): deeply analyzes a directory for suspicious code
- [Divide and Conquer](crates/use-cases/src/divide_and_conquer/): partition a math problem across a Werk of workers
- [Deep Research](crates/use-cases/src/deep_research/): multi-agent research with web search (requires `BRAVE_API_KEY`)
- [Model Pricing Tracker](crates/use-cases/src/model_pricing_tracker/): check model prices

```bash
make use_case                # list available names
make use_case name=<name>    # run one
```

> Consider configuring your LLM provider (see [Environment](#environment)).

## API

- [Providers](#providers): the LLM services agentwerk connects to
- [Agents](#agents): the unit that carries out a task
- [Models](#models): the reasoning engine driving an agent
- [Prompting](#prompting): the role, behavior, and context given to an agent
- [Tools](#tools): the functions an agent can call
- [Events](#events): the stream of activity emitted during a run
- [Policies](#policies): the rules and limits applied to a run
- [Output](#output): the result returned at the end of a run
- [Coworkers](#coworkers): the subordinate agents a parent can delegate to
- [Werk](#werk): a set of agents executing in parallel
- [Todo](#todo): planned features and roadmap

### Providers

You can integrate your agentic application with the following providers:

```rust
use agentwerk::provider::{MistralProvider, AnthropicProvider, OpenAiProvider, LiteLlmProvider};

let provider = MistralProvider::new(key);
let provider = AnthropicProvider::new(key);
let provider = OpenAiProvider::new(key);
let provider = LiteLlmProvider::new(key);
```

Point an agent at a custom endpoint with a custom timeout:

```rust
let provider = AnthropicProvider::new(key)
    .base_url("http://localhost:8000")
    .timeout(Duration::from_secs(120));
```

Or pick a provider for your agent from environment variables (see [Environment](#environment)):

```rust
let output = Agent::new()
    .provider_from_env()?
    .model("mistral-small-2603")
    .work("...")
    .await?;
```

### Agents

An `Agent` is a single worker. Equip them with what they need, then hand them a task:

```rust
let output = Agent::new()
    .provider(provider)
    .model("mistral-medium-2508")
    .tool(ReadFileTool)
    .work("Summarize src/main.rs")
    .await?;
```

| Method | Description |
|--------|-------------|
| `Agent::new()` | Start a new agent builder |
| `Agent::provider(...)` | Attach a provider |
| `Agent::model(...)` | Pick a model by name or pass a configured `Model` |
| `Agent::tool(...)` | Register a tool the agent can call |
| `Agent::work(...)` | Set the task and run; awaiting returns the `Output` |

### Models

Each agent runs on one model:

```rust
Agent::new().model_from_env()?
Agent::new().model("mistral-medium-2508")
```

Models from Anthropic, OpenAI, and Mistral come with a default context window size. Override it when needed, e.g. for custom model setups. The context window size is only relevant for calculating when messages need to be compacted:

```rust
let model = Model::from_name("custom-model")
    .context_window_size(100_000);

let agent = Agent::new().model(model);
```

### Prompting

Prompts are how you brief the worker. Here are the prompt types that drive their work:

```rust
use agentwerk::Agent;

let output = Agent::new()
    .provider(provider)
    .model("mistral-medium-2508")
    .role("You are a helpful assistant.")
    .tool(ReadFileTool)
    .work("What does src/main.rs do?")
    .await?;
```

The following methods on `Agent` configure prompts. Each accepts inline text (`&str` / `String`) or a path (`&Path` / `PathBuf`); a path is read from disk:

| Method | Description |
|--------|-------------|
| `Agent::role(...)` | Persistent identity of the agent |
| `Agent::work(...)` | Instruct the agent to perform a task; awaiting drives the loop |
| `Agent::context(...)` | Override the context prompt (default: `Agent::default_context()`, containing working directory, platform, OS version, date) |
| `Agent::behavior(...)` | Override the default behavioral directives (`DEFAULT_BEHAVIOR`) |

Compose a custom context prompt:

```rust
let default = Agent::default_context();
Agent::new().context(format!("{default}\n\nExtra notes."));
```

Read identity prompt from a file:

```rust
use std::path::Path;

Agent::new()
    .role(Path::new("prompts/identity.md"))
```

Use `{key}` placeholders in the identity prompt and fill them with template variables:

```rust
Agent::new()
    .role("You are {role}. Respond in {language}.")
    .template("role", json!("a code reviewer"))
    .template("language", json!("German"))
```

#### Workers on Standby

Use `keep_working` when you want to keep sending instructions to your agent in the background:

```rust
let (handle, output) = Agent::new()
    .provider(provider)
    .model("mistral-medium-2508")
    .role("You are a codebase Q&A assistant.")
    .tool(ReadFileTool)
    .keep_working(); // start the agent and send more instructions later

handle.work("What does src/main.rs do?");
handle.work("Now summarize src/lib.rs.");

handle.interrupt(); // stop the agent when there are no more instructions to send
```

The worker stays on standby for the next task. Call `interrupt()` to send them home, or just drop the last handle.

| Method | Description |
|--------|-------------|
| `AgentWorking::work(...)` | Hand the agent a new task |
| `AgentWorking::work_more(...)` | Queue several tasks at once |
| `AgentWorking::interrupt()` | Stop the agent |
| `AgentWorking::is_interrupted()` | Check if the agent was interrupted |
| `AgentWorking::clone()` | Get another handle to the same agent |


### Tools

Give your agent access to simple tools for driving tasks:

```rust
use agentwerk::tools::ToolResult;
use agentwerk::Tool;

let greet = Tool::new("greet", "Say hello")
    .contract(json!({...}))
    .read_only(true)
    .handler(|input, ctx| Box::pin(async move {
        Ok(ToolResult::success("Hello!"))
    }));
```

You can configure `.read_only(true)` when a tool has no side effects as an optional optimization. If set, the tool will run parallelized.

> Every registered tool, description, and input schema are sent to the model on each request.
> You do not need to list, describe, or even mention the tools in your `.work(...)` prompt.

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
| **Utility** | `BashTool` | Run shell commands matching a glob pattern |
| | `AgentTool` | Hand a job off to a coworker |
| | `SendMessageTool` | Pass a memo to coworkers |
| | `ReadOutcomeTool` | Check or wait on a backgrounded coworker's result |
| | `TodoListTool` | Keep a persistent todo list of items |
| | `ToolSearchTool` | Browse the toolbox by keyword |

```rust
use agentwerk::tools::{
    ReadFileTool, WriteFileTool, EditFileTool,
    GlobTool, GrepTool, ListDirectoryTool,
    WebFetchTool, AgentTool, BashTool,
    SendMessageTool, ReadOutcomeTool, TodoListTool, ToolSearchTool,
};

let output = Agent::new()
    .tool(ReadFileTool)
    .tool(WriteFileTool)
    .tool(EditFileTool)
    .tool(GlobTool)
    .tool(GrepTool)
    .tool(ListDirectoryTool)
    .tool(WebFetchTool)
    .tool(AgentTool)
    .tool(BashTool::new("git", "git *"))
    .tool(SendMessageTool)
    .tool(ReadOutcomeTool)
    .tool(TodoListTool::new(Path::new("/tmp/todos")))
    .tool(ToolSearchTool)
    .work("Explore the repo and summarize what you find.")
    .await?;
```

### Events

You can inspect what your agent is doing through events:

```rust
use agentwerk::event::EventKind;
use agentwerk::Event;

let handler = Arc::new(|event: Event| match &event.kind {
    EventKind::ToolCallStarted { tool_name, .. } => {
        eprintln!("[{}] → {tool_name}", event.agent_name);
    }
    EventKind::ToolCallFailed { tool_name, message, .. } => {
        eprintln!("[{}] ✗ {tool_name}: {message}", event.agent_name);
    }
    EventKind::AgentFinished { steps, outcome } => {
        eprintln!("[{}] done in {steps} steps ({outcome:?})", event.agent_name);
    }
    _ => {}
});
```

| | Kind | Description |
|-|------|-------------|
| **Agent** | `AgentStarted` | Worker clocked in |
| | `AgentFinished` | Worker clocked out |
| | `StepStarted` | Agentic loop step began |
| | `StepFinished` | Agentic loop step finished |
| | `AgentPaused` | Worker on standby, waiting for new orders |
| | `AgentResumed` | Worker back on the clock |
| **Provider** | `RequestStarted` | Provider request began |
| | `RequestFinished` | Provider request finished |
| | `RequestRetried` | Transient provider error triggered a retry |
| | `RequestFailed` | Provider request failed after exhausting retries |
| | `TextChunkReceived` | Streamed text token arrived |
| | `TokensReported` | Provider reported token counts for the last request |
| **Context** | `OutputTruncated` | Response was cut off at the configured length cap |
| | `ContextCompacted` | Conversation history was compacted to stay within the model's window |
| | `PolicyViolated` | A configured policy (`max_steps`, `max_input_tokens`, `max_output_tokens`, `max_contract_retries`) was exceeded |
| | `ContractMissed` | Structured-output validation failed and the loop is asking the model to retry |
| **Tool** | `ToolCallStarted` | Tool invocation began |
| | `ToolCallFinished` | Tool invocation succeeded |
| | `ToolCallFailed` | Tool invocation failed; the run continues |

> When `.event_handler(...)` is not set, agents log tool activity and lifecycle events to
> stderr via `Event::default_logger()`. You can call `.silent()` on the agent to silence the output.

### Policies

Working rules cap the agent's run, protecting your company against typical LLM failures:

| Method | Default | Description |
|--------|---------|-------------|
| `Agent::max_steps(...)` | no limit | Stop after N agentic loop iterations |
| `Agent::max_request_tokens(...)` | provider default | Cap output tokens per LLM request |
| `Agent::max_input_tokens(...)` | no limit | Cap cumulative input tokens across the whole run |
| `Agent::max_output_tokens(...)` | no limit | Cap cumulative output tokens across the whole run |
| `Agent::max_contract_retries(...)` | 10 | Retry structured output compliance |
| `Agent::max_request_retries(...)` | 10 | Retry on API errors (429, 529, 5xx) |
| `Agent::request_retry_delay(...)` | 500ms | Base delay for exponential backoff between request retries |

### Output

The `Output` is the worker's finished result:

```rust
let output = agent.work("Summarize src/main.rs").await?;
println!("{}", output.response_raw);
```

You can enforce a contract (JSON Schema) on the agent's structured response:

```rust
let output = Agent::new()
    .contract(json!({ "type": "object", "properties": { "category": { "type": "string" } } }))
    .work("Classify this issue.")
    .await?;

println!("{}", output.response.unwrap()["category"]);
```

You can also load the contract from a file by passing a path:

```rust
use std::path::Path;

Agent::new().contract(Path::new("contracts/category.json"))
```

#### Statistics

Each shift reports a tally of what happened:

```rust
output.statistics.input_tokens   // total input tokens
output.statistics.output_tokens  // total output tokens
output.statistics.requests       // number of provider requests
output.statistics.tool_calls     // number of tool calls
output.statistics.steps          // number of agent steps
```

### Coworkers

A worker can staff their own colleagues. Once you staff at least one coworker, the parent picks up an `AgentTool` to hand work off:

```rust
let researcher_base = Agent::new()
    .model("mistral-medium-2508")
    .role("You are a research assistant.")
    .tool(brave_search_tool())
    .max_steps(3);

let r1 = researcher_base.clone().name("researcher_1");
let r2 = researcher_base.clone().name("researcher_2");

let output = Agent::new()
    .name("orchestrator")
    .role("You are a research orchestrator.")
    .staff(r1)
    .staff(r2)
    .work("Research the economic impact of quantum computing.")
    .await?;
```

| Method | Description |
|--------|-------------|
| `Agent::staff(...)` | Register one coworker to hand off to |
| `Agent::staff_more(...)` | Register many coworkers at once |

#### Inheritance

The following fields are inherited, shared or owned by the coworkers:

| Behavior | Fields |
|---|---|
| Inherited | `provider`, `model`, `working_dir`, `event_handler`, `interrupt_signal` |
| Shared | `incoming_work`, `session_store` |
| Per coworker | `role`, `task`, `behavior`, `context`, `tools`, `contract`, `max_steps`, `max_request_tokens`, `max_input_tokens`, `max_output_tokens`, `max_contract_retries`, `max_request_retries`, `request_retry_delay` |

### Werk

Run many agents in parallel with a `Werk`. Wait for the execution of all workers on a fixed number of production lines. Results arrive as a `HashMap` keyed by each agent's name:

```rust
use agentwerk::tools::ReadFileTool;
use agentwerk::{Agent, Werk};

let template = Agent::new()
    .provider(provider)
    .model("mistral-medium-2508")
    .tool(ReadFileTool);

let docs = ["document A", "document B"];
let workers = docs.iter().map(|doc| {
    template
        .clone()
        .name(*doc)
        .work(format!("Summarize {doc}"))
});

let results = Werk::new()
    .lines(10)
    .work(workers)
    .await;

for doc in &docs {
    let out = results.get(*doc).unwrap().as_ref().unwrap();
    println!("{doc}: {}", out.response_raw);
}
```

| Method | Description |
|--------|-------------|
| `Werk::lines(n)` | Number of assembly lines |
| `Werk::work(workers)` | Put every worker on the line; returns `HashMap<String, Result<Output>>` |
| `Werk::keep_working(initial)` | Open the Werk for staffing over time; returns `(Werking, WerkOutputStream)` |
| `Werk::interrupt_signal(...)` | Share an external cancel signal with the Werk |

#### Staffing on the Fly

Start a Werk that takes on workers over time. Results are reported in *completion order*:

```rust
let (handle, mut results) = Werk::new()
    .lines(10)
    .keep_working(std::iter::empty());

let docs = ["document A", "document B"];
for doc in &docs {
    handle.staff(template.clone().name(*doc).work(format!("Summarize {doc}")));
}

drop(handle); // graceful: stream ends after backlog drains

while let Some((name, result)) = results.next().await {
    let out = result?;
    println!("{name}: {}", out.response_raw);
}
```

| Method | Description |
|--------|-------------|
| `Werking::staff(...)` | Take on another worker |
| `Werking::staff_more(...)` | Take on many workers at once |
| `Werking::clone()` | Get another handle to the same Werk |
| `Werking::interrupt()` | Pull everyone off the line and shut the Werk |
| `Werking::is_interrupted()` | Check if the Werk was interrupted |

### Todo

Planned additions to the crate:

- **Context compression**: summarize older messages when a conversation exceeds the model's context window size
- **Session handling**: resume and persist agent sessions across runs

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
| llama.cpp | `--jinja` (enables tool calling) |

### Environment

Use cases and integration tests use the following environment variables:

**General**
| Variable | Description |
|----------|-------------|
| `MODEL` | Generic model override for `.model_from_env()` |

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
