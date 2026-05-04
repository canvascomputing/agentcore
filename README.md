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

| Method | Description |
|--------|-------------|
| `Agent::new()` | Start an agent builder. |
| `provider(p)` | Attach the LLM provider the agent talks to. |
| `model(m)` | Pick the model the provider should run. |
| `role(text)` | Set the persistent role prompt the agent identifies with. |
| `context(text)` | Set a per-run context block prepended as the first user message. |
| `label(l)` / `labels([..])` | Restrict ticket pickup to matching labels (Path B). |
| `tool(t)` / `tools([..])` | Register tools the agent may call. `MarkTicketDoneTool` is registered by default so every agent can mark its current ticket done. |
| `working_dir(p)` | Working directory tools resolve filesystem paths against. |
| `event_handler(fn)` | Install a custom observer for the agent's events. |
| `silent()` | Drop every event, opting out of the default stderr logger. |
| `remember_history()` | Carry the agent's conversation across tickets, including across separate `run_dry` calls. Off by default. |
| `task(value)` / `task_assigned(...)` / `create(ticket)` | Enqueue work for the agent. |
| `run().await` | Process every queued task until the queue is empty and return the last answer (empty string if nothing settled). For long-lived runs that keep accepting work, drop down to `TicketSystem::run`. |

Inspect or reset the conversation slot at runtime (only meaningful when `remember_history()` is set):

| Method | Description |
|--------|-------------|
| `history()` | Read the agent's stored conversation history (clone). Returns `None` when memory is off. |
| `clear_history()` | Drop the agent's stored history. No-op when memory is off. |

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
| `TicketSystem::new()` | Construct a fresh system. |
| `task(value)` / `task_assigned(value, label)` | Enqueue a ticket carrying `value`, optionally label-routed (Path B). |
| `task_schema(value, schema)` / `task_schema_assigned(...)` | Enqueue with a `Schema` the agent's `done` result must validate against. |
| `create(ticket)` | Enqueue a fully-built `Ticket`. Setting `assignee` births it `InProgress` (Path A). |
| `add(agent)` | Bind an `Agent` to this system; returns the wired `Agent` for further chaining. |
| `run().await` | Stay alive until the interrupt signal fires, processing tickets as they arrive. Use this when work keeps coming in. |
| `run_dry().await` | Process every queued ticket until the queue is empty, then return the most recent `Done` ticket's `result` (empty string if none settled). Use this when the batch is fixed up front. |
| `get(key)` / `tickets()` / `first()` | Read settled tickets. |
| `stats()` | Run-time counters and timings. |

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
| `max_steps(n)` | Cap the per-ticket step count. |
| `max_input_tokens(n)` | Cap total input tokens across the run. |
| `max_output_tokens(n)` | Cap total output tokens across the run. |
| `max_request_tokens(n)` | Cap the input tokens of any single request. |
| `max_schema_retries(n)` | Cap how many times the loop asks the model to retry after a schema-validation failure. |
| `max_request_retries(n)` | Cap how many times a transient provider error is retried. |
| `request_retry_delay(d)` | Base delay between request retries. |
| `timeout(d)` | Wall-clock cap on the whole run. |

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
| **File** | `ReadFileTool` | Read a file with line numbers, offset, and limit. |
| | `WriteFileTool` | Create or overwrite a file. |
| | `EditFileTool` | Find-and-replace in a file. |
| **Search** | `GlobTool` | Find files by pattern. |
| | `GrepTool` | Search file contents. |
| | `ListDirectoryTool` | List files and folders. |
| **Shell** | `BashTool` | Pattern-restricted shell access; `BashTool::unrestricted()` for the full shell. |
| **Web** | `WebFetchTool` | Fetch a URL and return its body. |
| **Tickets** | `MarkTicketDoneTool` | Mark the current ticket done; takes an optional `result` validated against the ticket's `schema`. Auto-registered on every `Agent`. |
| | `ManageTicketsTool` | Create, claim, and settle tickets (`done` / `failed`). |
| | `ReadTicketsTool` / `WriteTicketsTool` | Read-only and write-only ticket operations. |
| **Discovery** | `ToolSearchTool` | Pair with `Tool::defer(true)` to keep tools hidden until discovered. |

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
| **Ticket** | `TicketStarted` | Agent claimed a ticket and began working on it |
| | `TicketDone` | Ticket reached `Status::Done` |
| | `TicketFailed` | Ticket reached `Status::Failed` |
| **Provider** | `RequestStarted` | Provider request began |
| | `RequestFinished` | Provider request finished successfully |
| | `RequestFailed` | Provider request failed; the run is about to stop for this ticket |
| | `TextChunkReceived` | Streamed text chunk arrived from the provider |
| | `TokensReported` | Provider reported token counts for the last request |
| **Tool** | `ToolCallStarted` | Tool invocation began |
| | `ToolCallFinished` | Tool invocation succeeded |
| | `ToolCallFailed` | Tool invocation failed; the error is sent back to the model and the run continues |
| **Run** | `PolicyViolated` | A configured policy (`max_steps`, `max_input_tokens`, `max_output_tokens`, `max_schema_retries`, `max_request_retries`) was exceeded |

> When `.event_handler(...)` is not set, agents log ticket lifecycle, tool activity, request failures, and policy violations to stderr via `default_logger()`. Call `.silent()` on the agent to drop every event.

### Stats

Run-wide counters and timings. Read after `run()` / `run_dry()` (or any time during a run).

| | Method | Description |
|-|--------|-------------|
| **Run** | `run_duration()` | Wall-clock duration from first ticket start to run-watcher firing. `None` while the run hasn't started or is still going. |
| | `work_time()` | Sum of all finished tickets' `started → terminal` spans. With concurrent agents this can exceed `run_duration`. |
| **Tickets** | `tickets_created()` | Total tickets enqueued during the run. |
| | `tickets_done()` | Tickets settled with `Status::Done`. |
| | `tickets_failed()` | Tickets settled with `Status::Failed`. |
| | `success_rate()` | `tickets_done / (tickets_done + tickets_failed)`. `None` until at least one ticket finishes. |
| | `run_time()` | Sum of finished tickets' `creation → terminal` spans. |
| | `avg_run_time()` | Mean of finished tickets' `creation → terminal` spans. `None` until at least one ticket finishes. |
| **Tokens** | `input_tokens()` | Total input tokens across all provider responses. |
| | `output_tokens()` | Total output tokens across all provider responses. |
| **Activity** | `steps()` | Total ticket-claim iterations across all agents. |
| | `requests()` | Total provider responses received. |
| | `tool_calls()` | Total tool dispatches. |
| | `errors()` | Total provider errors. |

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
| `MODEL` | Generic model override for `model_from_env()` |
| `BRAVE_API_KEY` | Required by the `deep-research-v2` example |

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
