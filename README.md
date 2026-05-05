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

<p align="center">A Rust crate for agentic workflows: a ticket-driven execution loop, agent orchestration, built-in tools, durable memory, schema-validated results, retry and token policies, and multi-provider support.</p>

<p align="center"><em>agentwerk pairs "agent" with the German "Werk," a word that means both factory and artwork; machinery for building agentic systems, engineered like a craft.</em></p>

---

## Installation

```bash
cargo add agentwerk
```

## Quick Start

```rust
use agentwerk::providers::{model_from_env, provider_from_env};
use agentwerk::tools::ReadFileTool;
use agentwerk::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let answer = Agent::new()
        .provider(provider_from_env()?)
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
- [Deep Research](crates/use-cases/src/deep_research_v2/): agentic web research pipeline (requires `BRAVE_API_KEY`)
- [Malware Scanner](crates/use-cases/src/malware_scanner/): identify indicators of compromise in a software package

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
- [Memory](#memory): durable facts the model curates and reuses across tickets
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
use agentwerk::providers::{model_from_env, provider_from_env};

let provider = provider_from_env()?;
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

| | Method | Description |
|-|--------|-------------|
| **Construct** | `Agent::new()` | New builder. |
| **Identity** | `name(s)` | Identifier used by routing and events. |
| | `label(l)` / `labels([..])` | Restrict to tickets carrying matching labels. |
| **Provider** | `provider(p)` | LLM provider. |
| | `model(m)` | Model the provider runs. |
| **Prompt** | `role(text)` | Role prompt. |
| | `context(text)` | Prepended to the first message of every ticket. |
| **Tools** | `tool(t)` / `tools([..])` | Register a tool; `WriteResultTool` is auto-registered. |
| | `working_dir(p)` | Directory tools resolve paths against. |
| **Events** | `event_handler(fn)` | Custom event observer. |
| | `silent()` | Drop every event instead of routing to the default logger. |
| **Memory** | `memory(&store)` | Bind a shared `Memory`; durable facts persist across tickets and process restarts. Off by default. |

Methods called after the agent is built:

| | Method | Description |
|-|--------|-------------|
| **Tasks** | `task(value)` | Create a task. |
| | `task_labeled(value, label)` | Create a task tagged with `label` for label-scoped routing. |
| | `task_schema(value, schema)` | Create a task whose result must validate against `schema`. |
| | `task_schema_labeled(value, schema, label)` | Create a labelled task whose result must validate against `schema`. |
| | `create(ticket)` | Add a caller-built `Ticket`; a preset `assignee` starts it `InProgress`. |
| **Run** | `run().await` | Process every queued task and return the last result. |
| **Results** | `results()` | Every `Done` ticket's `ResultRecord`, in creation order. |
| | `last_result()` | Most recent `Done` ticket's `ResultRecord`, or `None`. |
| **Inspect** | `get_name()` | Configured name. |
| | `get_labels()` | Configured label scope. |
| | `handles(labels)` | True when the label scope overlaps. |

### Ticket Systems

A `TicketSystem` is the shared form: one queue, several registered agents, run policies, interrupt signal, and the run-time `Stats`. Tickets carry the unit of work; agents pick them up by label scope (Path B) or direct assignment (Path A).

```rust
use agentwerk::{Runnable, TicketSystem};

let tickets = TicketSystem::new()
    .max_steps(20)
    .max_time(std::time::Duration::from_secs(60));

tickets.task("Summarise the Cargo.toml of this project.");
tickets.add(agent);
let result = tickets.run_dry().await;
```

| | Method | Description |
|-|--------|-------------|
| **Construct** | `TicketSystem::new()` | `Arc<TicketSystem>`; agents share one queue. |
| | `add(agent)` | Bind an `Agent` and return it for chaining. |
| **Tasks** | `task(value)` | Create a task. |
| | `task_labeled(value, label)` | Create a task tagged with `label` for label-scoped routing. |
| | `task_schema(value, schema)` | Create a task whose result must validate against `schema`. |
| | `task_schema_labeled(value, schema, label)` | Create a labelled task whose result must validate against `schema`. |
| | `create(ticket)` | Add a caller-built `Ticket`; a preset `assignee` starts it `InProgress`. |
| **Run** | `run().await` | Run continuously until the interrupt signal fires. |
| | `run_dry().await` | Process every queued ticket; returns last finished result or `""`. |
| **Results** | `results()` | Every `Done` ticket's `ResultRecord`, in creation order. |
| | `last_result()` | Most recent `Done` ticket's `ResultRecord`, or `None`. |
| **Inspect** | `get(key)` / `tickets()` / `first()` | Finished tickets. |
| | `stats()` | Run counters and timings. |

#### Policies

Configure execution limits on a ticket system. A breach fires `EventKind::PolicyViolated` and stops the run.

```rust
let tickets = TicketSystem::new()
    .max_steps(40)
    .max_input_tokens(200_000)
    .max_output_tokens(50_000)
    .max_request_tokens(8_000)
    .max_schema_retries(3)
    .max_request_retries(3)
    .request_retry_delay(std::time::Duration::from_millis(500))
    .max_time(std::time::Duration::from_secs(300));
```

| Method | Description |
|--------|-------------|
| `max_steps(n)` | Total number of steps. |
| `max_input_tokens(n)` | Total input tokens. |
| `max_output_tokens(n)` | Total output tokens. |
| `max_request_tokens(n)` | Input tokens per request. |
| `max_schema_retries(n)` | Schema-validation retry attempts. |
| `max_request_retries(n)` | Recoverable provider-error retry attempts. |
| `request_retry_delay(d)` | Base delay between request retries. |
| `max_time(d)` | Run's elapsed duration. |

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
| **Tickets** | `WriteResultTool` | Writes the agent's result for the current ticket. Validates against the ticket's `schema` (when set), appends an NDJSON record to `<results_dir>/results.jsonl`, attaches the record to the ticket, and transitions it to `Done`. Auto-registered on every agent. |
| | `ManageTicketsTool` | Reads the ticket queue and creates or edits tickets. |
| | `ReadTicketsTool` / `WriteTicketsTool` | Read-only and write-only halves of `ManageTicketsTool`. |
| **Memory** | `MemoryTool` | Adds, replaces, or removes entries in the agent's `Memory`. Auto-registered when `Agent::memory(&store)` is set. |
| **Discovery** | `ToolSearchTool` | Discovers tools registered with `Tool::defer(true)`. |

### Memory

A `Memory` is a file-backed store of durable facts the model curates itself via `MemoryTool`. The current entries are rendered into the system prompt under `## Memory` at the top of every ticket; mid-ticket writes land on disk and become visible at the next ticket. Multiple agents share one store the same way they share a `TicketSystem`.

```rust
use agentwerk::{Agent, Memory};

let memory = Memory::open("./.agentwerk-memory")?;

let alice = Agent::new()
    .name("alice")
    .provider(provider.clone())
    .model(&model)
    .memory(&memory);

let bob = Agent::new()
    .name("bob")
    .provider(provider)
    .model(&model)
    .memory(&memory);
```

Both agents read and write the same `memory.md`. Bind two agents to two different stores for independent memory.

Methods on `Memory`:

| | Method | Description |
|-|--------|-------------|
| **Open** | `Memory::open(dir)` | Open or create a store at `dir`; returns `Arc<Memory>`. |
| **Read** | `entries()` | Clone of the current entries, in insertion order. |
| **Mutate** | `add(content)` | Append an entry. Rejects empty content, duplicates, and over-limit content. |
| | `replace(old_text, content)` | Swap the unique entry containing `old_text`. |
| | `remove(old_text)` | Drop the unique entry containing `old_text`. |
| | `rewrite(entries)` | Replace every entry in one shot. |

### Schemas

`Schema::parse` accepts a JSON-Schema document. Attach it to a ticket so the agent's result (written via `write_result_tool`) must validate against it.

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
tickets.task_schema_labeled(body, schema, "report");
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
| **Ticket** | `TicketStarted` | An agent claimed a ticket. |
| | `TicketDone` | A ticket reached `Status::Done`. |
| | `TicketFailed` | A ticket reached `Status::Failed`. |
| **Provider** | `RequestStarted` | A provider request started. |
| | `RequestFinished` | A provider request finished. |
| | `RequestFailed` | A provider request failed; the ticket stops. |
| | `TextChunkReceived` | A streamed text chunk arrived. |
| | `TokensReported` | The provider reported token counts for the last request. |
| **Tool** | `ToolCallStarted` | A tool invocation started. |
| | `ToolCallFinished` | A tool invocation finished. |
| | `ToolCallFailed` | A tool invocation failed; the error returns to the model and the ticket continues. |
| **Run** | `PolicyViolated` | A policy limit was breached; the run stops. |

> When `.event_handler(...)` is not set, agents log ticket lifecycle, tool activity, request failures, and policy violations to stderr via `default_logger()`. Call `.silent()` on the agent to drop every event.

### Stats

```rust
let s = tickets.stats();
println!("{} done, {} requests, {} in / {} out tokens",
    s.tickets_done(), s.requests(), s.input_tokens(), s.output_tokens());

// Same accessors, scoped to one ticket label.
let scan = s.stats_for_label("scan");
println!("[scan] {} done, {} tokens", scan.tickets_done(), scan.input_tokens());
```

Run-wide counters and timings. Read after `run()` / `run_dry()` (or any time during a run). `stats_for_label(label)` returns a slice with the same accessors, scoped to tickets carrying that label; `run_duration()` is `None` on a slice (elapsed run duration stays global).

| | Method | Description |
|-|--------|-------------|
| **Run** | `run_duration()` | First ticket start to run end, or `None` while the run is active. |
| | `total_work_duration()` | Sum of every finished ticket's `started → terminal` span; may exceed `run_duration` with concurrent agents. |
| **Tickets** | `tickets_created()` | Total tickets created. |
| | `tickets_done()` | Tickets that reached `Status::Done`. |
| | `tickets_failed()` | Tickets that reached `Status::Failed`. |
| | `success_rate()` | `tickets_done / (tickets_done + tickets_failed)`, or `None` until a ticket finishes. |
| | `total_ticket_duration()` | Sum of every finished ticket's `creation → terminal` span. |
| | `avg_ticket_duration()` | Mean of the same span, or `None` until a ticket finishes. |
| **Tokens** | `input_tokens()` | Total input tokens across all provider responses. |
| | `output_tokens()` | Total output tokens across all provider responses. |
| **Activity** | `steps()` | Total ticket-claim iterations across agents. |
| | `requests()` | Total provider responses received. |
| | `tool_calls()` | Total tool calls. |
| | `errors()` | Total provider errors. |
| **Labels** | `stats_for_label(label)` | Nested `Stats` slice scoped to tickets carrying `label`. |

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
