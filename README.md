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

<p align="center">This crate provides a core implementation for agentic applications: execution loop based on ticketing system, built-in tools, agent orchestration, multi-provider support, schema-based output, and retry mechanisms.</p>

<p align="center"><em>agentwerk pairs "agent" with the German "Werk," a word that means both factory and artwork; machinery for building agentic systems, engineered like a craft.</em></p>

---

## Installation

```bash
cargo add agentwerk
```

## Quick Start

```rust
use agentwerk::providers::{from_env, model_from_env};
use agentwerk::tools::ReadFileTool;
use agentwerk::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let answer = Agent::new()
        .provider(from_env()?)
        .model(&model_from_env()?)
        .role("Answer questions about this repository.")
        .tool(ReadFileTool)
        .task("What does Cargo.toml describe?")
        .run()
        .await;

    println!("{answer}");
    Ok(())
}
```

## Use Cases

Example applications living under `crates/use-cases/`:

- [Terminal REPL](crates/use-cases/src/terminal_repl/): minimal interactive chat
- [Divide and Conquer](crates/use-cases/src/divide_and_conquer/): arithmetic problem shared across agents
- [Deep Research](crates/use-cases/src/deep_research_v2/): agentic web research pipeline (requires `BRAVE_API_KEY`).

Run one with:

```bash
make use_case                # list available names
make use_case name=<name>    # run one
```

> Configure your LLM provider first (see [Environment](#environment)).

## API

- [Providers](#providers): LLM providers agents can use
- [Agents](#agents): the workers that pick up tickets and produce results
- [Ticket Systems](#ticket-systems): queue for organizing work
- [Tools](#tools): tools agents can use to finish a task
- [Schemas](#schemas): schemas used for result validation
- [Events](#events): lifecycle events of agentic work
- [Stats](#stats): insight into the work of agents

### Providers

```rust
use agentwerk::providers::{AnthropicProvider, MistralProvider, OpenAiProvider, LiteLlmProvider};

let provider = AnthropicProvider::new(key);
let provider = MistralProvider::new(key);
let provider = OpenAiProvider::new(key);
let provider = LiteLlmProvider::new(key);
```

Custom endpoint and timeout:

```rust
use std::time::Duration;

let provider = AnthropicProvider::new(key)
    .base_url("http://localhost:8000")
    .timeout(Duration::from_secs(120));
```

Pick a provider from environment variables (see [Environment](#environment)):

```rust
use agentwerk::providers::{from_env, model_from_env};

let provider = from_env()?;
let model = model_from_env()?;
```

### Agents

An `Agent` is a single worker.

```rust
use agentwerk::Agent;

let agent = Agent::new()
    .name("worker_0")
    .provider(provider)
    .model(&model)
    .role("You are an arithmetic worker.")
    .label("worker")
    .tool(BashTool::new("ls", "ls *"));

agent.task("Compute 2+2.");

let answer = agent.run().await;
```

Builder methods (each returns the `Agent` for chaining):

| Method | Description |
|--------|-------------|
| `Agent::new()` | Returns a new agent builder. |
| `provider(p)` | Sets the provider the agent uses. |
| `model(m)` | Sets the model the provider runs. |
| `role(text)` | Sets the agent's role prompt. |
| `context(text)` | Sets a context block prepended to the first message of every ticket. |
| `label(l)` / `labels([..])` | Restricts the agent to tickets carrying matching labels. |
| `tool(t)` / `tools([..])` | Registers tools the agent may call. `MarkTicketDoneTool` is registered automatically. |
| `working_dir(p)` | Sets the directory tools resolve paths against. |
| `event_handler(fn)` | Sets a custom observer for the agent's events. |
| `silent()` | Drops every event instead of using the default logger. |
| `remember_history()` | Persists the agent's conversation across tickets and across `run_dry` calls. Off by default. |

Methods called after the agent is built:

| Method | Description |
|--------|-------------|
| `task(value)` | Creates a new task. |
| `task_assigned(value, label)` | Creates a labelled task. |
| `task_schema(value, schema)` | Creates a task whose result must match `schema`. |
| `task_schema_assigned(value, schema, label)` | Creates a labelled task whose result must match `schema`. |
| `create(ticket)` | Adds a `Ticket` constructed by the caller. A preset `assignee` starts it as `InProgress`. |
| `run().await` | Processes every queued task and returns the last result. |
| `get_name()` | Returns the configured name. |
| `get_labels()` | Returns the configured label scope. |
| `handles(labels)` | Returns true when the agent's label scope overlaps the given labels. |
| `history()` | Returns a clone of the stored conversation history, or `None` when memory is off. |
| `clear_history()` | Clears the stored conversation history, if any. |

### Ticket Systems

A `TicketSystem` is the shared form: one queue, several registered agents, run policies, timeout, interrupt signal, and the run-time `Stats`. Tickets carry the unit of work; agents pick them up by label scope (Path B) or direct assignment (Path A).

```rust
use agentwerk::{Runnable, TicketSystem};

let tickets = TicketSystem::new()
    .max_steps(20)
    .timeout(std::time::Duration::from_secs(60));

tickets.task("Summarise the Cargo.toml of this project.");
tickets.add(agent);
let result = tickets.run_dry().await;
```

| Method | Description |
|--------|-------------|
| `TicketSystem::new()` | Returns an `Arc<TicketSystem>`; multiple agents can share one queue. |
| `add(agent)` | Binds an `Agent` to this system and returns it for chaining. |
| `task(value)` / `task_assigned(value, label)` | Creates a new task, optionally labelled. |
| `task_schema(value, schema)` / `task_schema_assigned(...)` | Creates a task whose result must match `schema`, optionally labelled. |
| `create(ticket)` | Adds a `Ticket` constructed by the caller. A preset `assignee` starts it as `InProgress`. |
| `run().await` | Runs continuously, processing tickets as they arrive, until the interrupt signal fires. |
| `run_dry().await` | Processes every queued ticket and returns the most recent finished ticket's result, or an empty string when nothing finished. |
| `get(key)` / `tickets()` / `first()` | Returns finished tickets. |
| `stats()` | Returns the run's counters and timings. |

Configure run limits on the system:

```rust
let tickets = TicketSystem::new()
    .max_steps(40)
    .max_input_tokens(200_000)
    .max_output_tokens(50_000)
    .max_request_tokens(8_000)
    .max_schema_retries(3)
    .max_request_retries(3)
    .request_retry_delay(std::time::Duration::from_millis(500))
    .timeout(std::time::Duration::from_secs(300));
```

| Method | Description |
|--------|-------------|
| `max_steps(n)` | Caps the total step count across the run. |
| `max_input_tokens(n)` | Caps total input tokens across the run. |
| `max_output_tokens(n)` | Caps total output tokens across the run. |
| `max_request_tokens(n)` | Caps the input tokens of any single request. |
| `timeout(d)` | Caps the duration of the run. |
| `max_schema_retries(n)` | Caps schema-validation retry attempts. |
| `max_request_retries(n)` | Caps recoverable provider-error retry attempts. |
| `request_retry_delay(d)` | Sets the base delay between request retries. |

A breach fires `EventKind::PolicyViolated` and stops the run.

### Tools

```rust
use agentwerk::{Tool, ToolResult};
use serde_json::json;

let greet = Tool::new("greet", "Say hello")
    .schema(json!({
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    }))
    .read_only(true)
    .handler(|input, _ctx| Box::pin(async move {
        let name = input["name"].as_str().unwrap_or("world");
        Ok(ToolResult::success(format!("Hello, {name}!")))
    }));
```

`.read_only(true)` lets the loop run a tool concurrently with other read-only calls in the same step.

#### Built-in tools

| | Tool | Description |
|-|------|-------------|
| **File** | `ReadFileTool` | Reads a file with line numbers, offset, and limit. |
| | `WriteFileTool` | Creates or overwrites a file. |
| | `EditFileTool` | Replaces text in a file. |
| **Search** | `GlobTool` | Finds files by pattern. |
| | `GrepTool` | Searches file contents. |
| | `ListDirectoryTool` | Lists files and folders. |
| **Shell** | `BashTool` | Runs shell commands matching an allowed pattern; `BashTool::unrestricted()` allows any command. |
| **Web** | `WebFetchTool` | Fetches a URL and returns its body. |
| **Tickets** | `MarkTicketDoneTool` | Marks the current ticket done. Auto-registered on every agent; the optional `result` is validated against the ticket's `schema`. |
| | `ManageTicketsTool` | Creates, claims, and finishes tickets. |
| | `ReadTicketsTool` / `WriteTicketsTool` | Read-only and write-only halves of `ManageTicketsTool`. |
| **Discovery** | `ToolSearchTool` | Discovers tools registered with `Tool::defer(true)`. |

### Schemas

`Schema::parse` accepts a JSON-Schema document. Attach it to a ticket so the agent's `done` result must validate against it.

```rust
use agentwerk::Schema;

let schema = Schema::parse(json!({
    "type": "object",
    "properties": {
        "title":    { "type": "string", "minLength": 1 },
        "research": { "type": "string", "maxLength": 500 }
    },
    "required": ["title", "research"]
}))?;
```

Via the `task_schema*` shorthands:

```rust
tickets.task_schema_assigned(body, schema, "report");
```

Via a fully-built `Ticket` — chain `.schema(...)` with `.label(...)` for Path B routing or `.assign_to(...)` for Path A:

```rust
use agentwerk::Ticket;

// Path B: any agent labelled "report" picks it up.
tickets.create(Ticket::new(body.clone()).schema(schema.clone()).label("report"));

// Path A: pinned directly to a named agent, born InProgress.
tickets.create(Ticket::new(body).schema(schema).assign_to("report_writer"));
```

### Events

You can inspect what your agent is doing through events:

```rust
use std::sync::Arc;
use agentwerk::{Event, EventKind};

let handler = Arc::new(|event: Event| match &event.kind {
    EventKind::ToolCallStarted { tool_name, .. } => {
        eprintln!("[{}] → {tool_name}", event.agent_name);
    }
    EventKind::ToolCallFailed { tool_name, message, .. } => {
        eprintln!("[{}] ✗ {tool_name}: {message}", event.agent_name);
    }
    EventKind::TicketDone { key } => {
        eprintln!("[{}] done {key}", event.agent_name);
    }
    EventKind::TicketFailed { key } => {
        eprintln!("[{}] failed {key}", event.agent_name);
    }
    _ => {}
});
```

| | Kind | Description |
|-|------|-------------|
| **Ticket** | `TicketStarted` | An agent claimed a ticket and started work. |
| | `TicketDone` | A ticket reached `Status::Done`. |
| | `TicketFailed` | A ticket reached `Status::Failed`. |
| **Provider** | `RequestStarted` | A provider request started. |
| | `RequestFinished` | A provider request finished successfully. |
| | `RequestFailed` | A provider request failed; the ticket will not continue. |
| | `TextChunkReceived` | A streamed text chunk arrived from the provider. |
| | `TokensReported` | The provider reported token counts for the last request. |
| **Tool** | `ToolCallStarted` | A tool invocation started. |
| | `ToolCallFinished` | A tool invocation succeeded. |
| | `ToolCallFailed` | A tool invocation failed; the error is returned to the model and the ticket continues. |
| **Run** | `PolicyViolated` | A configured policy limit was reached and the run is stopping. |

> When `.event_handler(...)` is not set, agents log ticket lifecycle, tool activity, request failures, and policy violations to stderr via `default_logger()`. Call `.silent()` on the agent to drop every event.

### Stats

Run-wide counters and timings. Read after `run()` / `run_dry()` (or any time during a run).

| | Method | Description |
|-|--------|-------------|
| **Run** | `run_duration()` | Returns the duration from the first ticket start to the end of the run, or `None` while the run is still active. |
| | `work_time()` | Returns the sum of every finished ticket's `started → terminal` span; may exceed `run_duration` with concurrent agents. |
| **Tickets** | `tickets_created()` | Returns the total number of tickets created during the run. |
| | `tickets_done()` | Returns the number of tickets that reached `Status::Done`. |
| | `tickets_failed()` | Returns the number of tickets that reached `Status::Failed`. |
| | `success_rate()` | Returns `tickets_done / (tickets_done + tickets_failed)`, or `None` until at least one ticket finishes. |
| | `run_time()` | Returns the sum of every finished ticket's `creation → terminal` span. |
| | `avg_run_time()` | Returns the mean of every finished ticket's `creation → terminal` span, or `None` until at least one ticket finishes. |
| **Tokens** | `input_tokens()` | Returns the total input tokens across all provider responses. |
| | `output_tokens()` | Returns the total output tokens across all provider responses. |
| **Activity** | `steps()` | Returns the total ticket-claim iterations across all agents. |
| | `requests()` | Returns the total number of provider responses received. |
| | `tool_calls()` | Returns the total number of tool calls made. |
| | `errors()` | Returns the total number of provider errors. |

```rust
let s = tickets.stats();
println!("Duration:  {:?}", s.run_duration().unwrap_or_default());
println!("Work time: {:?}", s.work_time());
println!(
    "Tickets:   {} done, {} failed ({:.0}%)",
    s.tickets_done(),
    s.tickets_failed(),
    s.success_rate().map(|r| r * 100.0).unwrap_or(0.0),
);
println!("Avg time:  {:?}", s.avg_run_time().unwrap_or_default());
println!("Tokens:    {} in, {} out", s.input_tokens(), s.output_tokens());
println!(
    "Activity:  {} requests · {} tool calls · {} errors",
    s.requests(),
    s.tool_calls(),
    s.errors(),
);
```

## Development

### Workspace

- `crates/agentwerk/`: the library.
- `crates/use-cases/`: runnable example binaries that depend on the library.

### Building and testing

```bash
make                # build (warnings are errors)
make test           # unit tests bundled by tests/unit (workspace --lib)
make fmt            # format code
make clean          # remove build artifacts
make update         # update dependencies
```

### Integration tests

> Configure your LLM provider first (see [Environment](#environment)).

```bash
make test_integration                     # run all
make test_integration name=bash_usage     # run one
```

### Use cases

```bash
make use_case                                                 # list available
make use_case name=terminal-repl                              # run one
make use_case name=deep-research-v2 args="What is a good life?"  # with arguments
```

### Publishing

```bash
make bump                  # bump patch version, run tests, commit, tag
make bump part=minor       # bump minor version
make bump part=major       # bump major version
```

GitHub Actions handles the crates.io publish via trusted publishing once you push the new tag (`git push --tags`).

### Documentation

```bash
make doc                   # cargo doc --no-deps -p agentwerk (strict rustdoc)
```

### LiteLLM proxy

Start a local LiteLLM proxy on port 4000 that forwards to a provider. Requires Docker.

```bash
make litellm                               # default: anthropic
make litellm LITELLM_PROVIDER=openai       # use OpenAI
make litellm LITELLM_PROVIDER=mistral      # use Mistral
```

### Local inference servers

agentwerk relies on server-side tool calling. Enable it through the following flags:

| Server | Flag |
|---|---|
| vLLM | `--enable-auto-tool-choice --tool-call-parser <parser>` |
| llama.cpp | `--jinja` (enables tool calling) |

### Environment

Use cases and integration tests use the following environment variables:

**General**

| Variable | Description |
|----------|-------------|
| `MODEL` | Generic model override for `model_from_env()`. |
| `BRAVE_API_KEY` | Required by the `deep-research-v2` example. |

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
| `LITELLM_API_KEY` | Auth key (required to select via `from_env()`) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
| `LITELLM_PROVIDER` | LLM provider (`anthropic`, `mistral`, `openai`, `litellm`) — explicit selection that overrides API-key auto-detection |
