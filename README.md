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

<p align="center">agentwerk builds agentic workflows around a ticket-driven execution loop, with built-in tools, memory, schema validation, retries, budget policies, and multi-provider support.</p>

<p align="center"><em>agentwerk pairs "agent" with the German "Werk", a word for both factory and artwork: machinery for building agentic systems.</em></p>

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
async fn main() {
    let results = Agent::new()
        .provider(provider_from_env().unwrap())
        .model(&model_from_env().unwrap())
        .role("You are a Rust developer who reads source files to answer questions.")
        .tool(ReadFileTool)
        .task("What does Cargo.toml describe?")
        .run_dry()
        .await;

    let answer = results.last().unwrap().result_string();
    println!("{answer}");
}
```

## Use Cases

Example applications living under `crates/use-cases/`:

- [Terminal REPL](crates/use-cases/src/terminal_repl/): minimal interactive chat
- [Divide and Conquer](crates/use-cases/src/divide_and_conquer/): arithmetic problem shared across agents
- [Deep Research](crates/use-cases/src/deep_research/): deep research pipeline (requires `BRAVE_API_KEY`)
- [Malware Scanner](crates/use-cases/src/malware_scanner/): identify indicators of compromise in a software package

Run one with:

```bash
make use_case                # list available names
make use_case name=<name>    # run one
```

> Configure an LLM provider first (see [Environment](#environment)).

# API

- [Providers](#providers): LLM backends agents send requests to.
- [Agents](#agents): Workers that pick up tickets and produce results.
- [Prompting](#prompting): How role, context, and task shape the model's input.
- [Tickets](#tickets): Shared queues that route tickets to agents.
- [Tools](#tools): Capabilities agents call to take action.
- [Memory](#memory): Durable facts the model curates across tickets.
- [Schemas](#schemas): JSON schemas that validate ticket results.
- [Events](#events): Lifecycle signals emitted while agents work.
- [Stats](#stats): Counters and timings recorded while agents work.

## Providers

A `Provider` connects an agent to an LLM service. The crate ships providers for Anthropic, OpenAI, Mistral, and a LiteLLM proxy. The same agent code runs against any of them.

```rust
use std::time::Duration;
use agentwerk::providers::{AnthropicProvider, model_from_env, provider_from_env};

// Other providers: MistralProvider, OpenAiProvider, LiteLlmProvider.
let provider = AnthropicProvider::new(key);

// Override the endpoint and request timeout.
let provider = AnthropicProvider::new(key)
    .base_url("http://localhost:8000")
    .timeout(Duration::from_secs(120));

// Pick a provider and model from environment variables, see #Environment
let provider = provider_from_env()?;
let model = model_from_env()?;
```

## Agents

An `Agent` is a worker that solves one task at a time. A **task** is the work assigned to the agent: a question, an instruction, or a structured request. Each task becomes a **ticket**: a record that tracks the work from start to finish and holds the final answer. The agent picks up a ticket, invokes the tools it has been registered with (read a file, run a command, search memory), and writes the result back onto the ticket.

### Identity

Assign the agent a name and the labels it accepts work for.

```rust
let agent = Agent::new()
    .name("worker_0")
    .label("math");
```

| Method | Description |
|--------|-------------|
| `Agent::new()` | Create a new agent builder. |
| `name(s)` | Set the identifier used for routing and events. |
| `label(l)` / `labels([..])` | Restrict the agent to tickets carrying matching labels. |

### Provider

Select the LLM service and model the agent uses.

```rust
let agent = Agent::new()
    .provider(provider)
    .model(&model);
```

| Method | Description |
|--------|-------------|
| `provider(p)` | Set the LLM provider. |
| `model(m)` | Set the model the provider runs. |

### Tools

Register the tools the agent may call, and set the directory those tools resolve paths against.

```rust
use agentwerk::tools::{ReadFileTool, GrepTool};

let agent = Agent::new()
    .tool(ReadFileTool)
    .tool(GrepTool)
    .working_dir("./src");
```

| Method | Description |
|--------|-------------|
| `tool(t)` / `tools([..])` | Register a tool the agent may call. |
| `working_dir(p)` | Set the directory tools resolve paths against. |

### Events

Observe agent activity. When no handler is set the default logger prints lifecycle events to stderr.

```rust
use std::sync::Arc;

let agent = Agent::new()
    .event_handler(Arc::new(|event| eprintln!("{event:?}")));
```

| Method | Description |
|--------|-------------|
| `event_handler(fn)` | Set a custom event observer. |

### Memory

Bind the agent to a memory store so facts persist across tickets and restarts.

```rust
use agentwerk::Memory;

// Pass a path to open a fresh store under that directory:
let agent = Agent::new().memory("./.agentwerk");

// Or share one store across multiple agents:
let memory = Memory::open("./.agentwerk")?;
let alice = Agent::new().memory(&memory);
let bob = Agent::new().memory(&memory);
```

| Method | Description |
|--------|-------------|
| `memory(into)` | Bind the agent to a memory store for facts that persist across tickets. Accepts a path (opens a fresh store) or an `&Arc<Memory>` (shares an existing store across agents). |

### Run

Start the agent with `run()`, submit tasks while it works, then drain the queue with `run_dry()` and read the results.

```rust
let agent = Agent::new()
    .provider(provider)
    .model(&model)
    .run();

agent.task("Compute 2+2.");
agent.task_labeled("Compute 3+3.", "math");

let results = agent.run_dry().await;
```

| Method | Description |
|--------|-------------|
| `task(task)` | Create a task for the agent. |
| `task_labeled(task, label)` | Create a task tagged with `label` for label-scoped routing. |
| `task_schema(task, schema)` | Create a task whose result must validate against `schema`. |
| `task_schema_labeled(task, schema, label)` | Create a labeled task whose result must validate against `schema`. |
| `ticket(ticket)` | Add a caller-built `Ticket` to the agent's queue. |
| `run()` | Start working and wait for incoming tickets. |
| `run_dry().await` | Work until every queued task finishes and return every `TicketResult`. |

## Prompting

Every prompt has three parts: `role` (who the agent is), `context` (the situation it operates in), and `task` (the specific work it must perform). The structure follows the [prompting framework](https://github.com/canvascomputing/prompting).

| Method | Description |
|--------|-------------|
| `role(text)` | Define who the agent is to narrow tone and domain. |
| `context(text)` | Provide the runtime facts that ground the agent in its session. |
| `task(task)` | State the work the agent must do, with an action verb and observable success criteria. |

### Role

The role is the agent's identity and operating rules. It is set once at build time and reused on every ticket the agent handles.

```rust
let agent = Agent::new().role("You are an arithmetic worker. Show your work.");
```

### Context

The context is the briefing the agent reads at the start of every ticket. When `context(text)` is not set, agentwerk supplies a default block of runtime facts:

```markdown
- Working directory: /Users/me/code/repo
- Platform: darwin
- OS version: 25.1.0
- Date: 2026-05-06
- Steps remaining: 8
- Input tokens remaining: 95000
- Output tokens remaining: 12000
- Time remaining: 240s
```

Override when the agent needs runtime facts the default block does not carry, such as a target file or a session identifier:

```rust
let agent = Agent::new().context("- Repo: example/widgets\n- Branch: main");
```

### Task

The task is the work itself: plain text, or a structured object the agent reads.

```rust
agent.task("Compute 2+2.");
agent.task(serde_json::json!({ "file": "Cargo.toml", "find": "version" }));
```

## Tickets

A `Ticket` is a task plus the metadata the system uses to route, schedule, and track it: labels, schema, status. The `TicketSystem` is agentwerk's core data structure: a shared queue that distributes tickets across one or more agents. Labels route work to a pool of matching agents; labelling a ticket with an agent's name pins it to that agent.

### Construct

Create a system and register the agents that will work it.

```rust
use agentwerk::TicketSystem;

let tickets = TicketSystem::new().workspace("./.agentwerk");
tickets.agent(agent);
```

| Method | Description |
|--------|-------------|
| `TicketSystem::new()` | Create a new ticket system that agents can share. |
| `agent(agent)` | Register an agent with the system. |
| `interrupt_signal(signal)` | Override the cancel signal shared across agents. |
| `workspace(dir)` | Set the workspace directory where memory, ticket results, and the ticket logs are persisted. |

### Tasks

Submit tasks to the queue. Use the `task*` shortcuts for common shapes, or build a `Ticket` and pass it to `ticket`.

```rust
use agentwerk::Ticket;

tickets.task("Compute 2+2.");
tickets.task_labeled("Write a report.", "report");

let built = Ticket::new("Summarise the Cargo.toml of this project.")
    .label("summary");
tickets.ticket(built);
```

| Method | Description |
|--------|-------------|
| `task(task)` | Create a task. |
| `task_labeled(task, label)` | Create a task tagged with `label` for label-scoped routing. |
| `task_schema(task, schema)` | Create a task whose result must validate against `schema`. |
| `task_schema_labeled(task, schema, label)` | Create a labeled task whose result must validate against `schema`. |
| `ticket(ticket)` | Add a caller-built `Ticket` to the queue. |

### Ticket

A `Ticket` packages the task body with caller-set metadata (labels, schema) and system-managed fields stamped during the run (key, status, timestamps, result). Build one to queue with `tickets.ticket(...)`, and read its state with the accessors below.

**Build**

| Method | Description |
|--------|-------------|
| `Ticket::new(task)` | Create a ticket carrying `task` as its body. |
| `label(l)` / `labels([..])` | Add labels for label-scoped routing. To pin the ticket to a specific agent, include the agent's name as a label. |
| `schema(s)` | Attach a result schema the answer must validate against. |

**Read**

```rust
let ticket = tickets.find(|t| t.status() == "done" && t.has_label("report")).unwrap();
let answer = ticket.result_string().unwrap();
```

| Method | Description |
|--------|-------------|
| `key()` | Return the system-assigned ticket key. |
| `status()` | Return the current status as `"todo"`, `"in_progress"`, `"done"`, or `"failed"`. |
| `result()` | Return the recorded `TicketResult`, if any. |
| `result_string()` | Return the result payload rendered as a string. |
| `elapsed()` | Return the elapsed time from creation to terminal status. |
| `has_label(l)` | Return whether the ticket carries label `l`. To check whether a ticket is pinned to an agent, pass the agent's name. |

### Run

Start in the background with `run()`, or block until every queued task finishes with `run_dry()`.

```rust
let results = tickets.run_dry().await;
```

| Method | Description |
|--------|-------------|
| `run()` | Start a background run and return a `Running` handle. |
| `run_dry().await` | Run until every queued task finishes and return every `TicketResult`. |

### Inspect

Look up finished tickets, filter the queue, or read aggregate statistics.

```rust
let report = tickets.find(|t| t.status() == "done" && t.has_label("report"));
let s = tickets.stats();
```

| Method | Description |
|--------|-------------|
| `get(key)` | Look up a finished ticket by key. |
| `tickets()` | List every finished ticket. |
| `first()` | Look up the first finished ticket. |
| `search(query)` | Find tickets whose task body matches `query`, case-insensitively. |
| `filter(predicate)` | Select tickets matching `predicate`, in creation order. |
| `find(predicate)` | Find the first ticket matching `predicate`. |
| `count(predicate)` | Count tickets matching `predicate`. |
| `stats()` | Read current statistics. |

### Status

Move a ticket through the state machine with `update_status`.

```rust
use agentwerk::Status;

tickets.update_status("TICKET-1", Status::InProgress)?;
```

| Method | Description |
|--------|-------------|
| `update_status(key, status)` | Transition a ticket through the state machine. |

### Policies

Configure execution policies on a ticket system. A breach fires `EventKind::PolicyViolated` and halts execution.

```rust
let tickets = TicketSystem::new()
    .max_steps(40)
    .max_time(std::time::Duration::from_secs(300))
    .max_input_tokens(200_000)
    .max_output_tokens(50_000)
    .max_request_tokens(8_000)
    .max_schema_retries(3)
    .max_request_retries(3)
    .request_retry_delay(std::time::Duration::from_millis(500));
```

| Method | Description |
|--------|-------------|
| `max_steps(n)` | Cap the total number of steps. |
| `max_time(d)` | Cap the total elapsed duration. |
| `max_input_tokens(n)` | Cap the total input tokens. |
| `max_output_tokens(n)` | Cap the total output tokens. |
| `max_request_tokens(n)` | Cap the input tokens per request. |
| `max_schema_retries(n)` | Cap the schema-validation retry attempts. |
| `max_request_retries(n)` | Cap the retry attempts on recoverable provider errors. |
| `request_retry_delay(d)` | Set the base delay between request retries. |

## Tools

Tools enable an agent to take action while solving a ticket: read a file, run a shell command, fetch a URL. The crate provides a set of built-in tools, and you can register your own.

### Built-in tools

| | Tool | Description |
|-|------|-------------|
| **File** | `ReadFileTool` | Reads a file with line numbers, offset, and limit. |
| | `WriteFileTool` | Creates or overwrites a file. |
| | `EditFileTool` | Replaces text in a file. |
| **Search** | `GlobTool` | Finds files by pattern. |
| | `GrepTool` | Searches file contents. |
| | `ListDirectoryTool` | Lists files and directories. |
| **Shell** | `BashTool` | Runs shell commands matching an allowed pattern. |
| **Web** | `WebFetchTool` | Fetches a URL and returns its body. |
| **Tickets** | `WriteResultTool` | Writes the agent's result for the current ticket and marks it done. |
| | `ManageTicketsTool` | Reads the ticket queue and creates or edits tickets. |
| | `ReadTicketsTool` | Reads the ticket queue. |
| | `WriteTicketsTool` | Creates or edits tickets in the queue. |
| **Memory** | `MemoryTool` | Adds, replaces, or removes entries in the agent's memory. |
| **Discovery** | `ToolSearchTool` | Discovers tools registered with `Tool::defer(true)`. |

### Custom tools

Each tool declares a JSON-Schema for its inputs and a handler the agent runs when it picks the tool.

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

`.read_only(true)` allows the agent to run a tool concurrently with other read-only calls in the same step.

## Memory

`Memory` allows an agent to retain knowledge from one task to the next, even across process restarts. It is a file-backed store the agent curates through `MemoryTool`.

```rust
use agentwerk::{Agent, Memory};

let memory = Memory::open("./.agentwerk")?;

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

Both agents read and write the same `memory.jsonl` (one entry per line: `{"content": "...", "added_at": <ms>}`). Bind two agents to two different stores for independent memory.

Methods on `Memory`:

| | Method | Description |
|-|--------|-------------|
| **Open** | `Memory::open(dir)` | Open or create a store at `dir`. |
| **Read** | `entries()` | List the current entries, in insertion order. |
| **Mutate** | `add(content)` | Append a new entry. |
| | `replace(old_text, content)` | Swap the unique entry containing `old_text`. |
| | `remove(old_text)` | Drop the unique entry containing `old_text`. |
| | `rewrite(entries)` | Replace every entry in a single call. |

## Schemas

A `Schema` says what a ticket's result must look like. Attach a JSON-Schema document to a ticket and the agent's answer must validate against it before the ticket can finish.

```rust
use agentwerk::Schema;

let schema = Schema::parse(json!({
    "type": "object",
    "properties": { "title": { "type": "string" } },
    "required": ["title"]
}))?;

tickets.task_schema("Write a report.", schema);
```

## Events

Events report everything that happens while agents work: which tickets they pick up, which tools they call, when provider requests start, finish, or fail. Register an event handler to log them, report progress, or measure throughput.

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
| | `TicketDone` | A ticket finished successfully. |
| | `TicketFailed` | A ticket failed. |
| **Provider** | `RequestStarted` | A provider request started. |
| | `RequestFinished` | A provider request finished. |
| | `RequestFailed` | A provider request failed and stopped the ticket. |
| | `TextChunkReceived` | A streamed text chunk arrived. |
| | `TokensReported` | The provider reported token counts for the last request. |
| **Tool** | `ToolCallStarted` | A tool invocation started. |
| | `ToolCallFinished` | A tool invocation finished. |
| | `ToolCallFailed` | A tool invocation failed but the ticket continues. |
| **Run** | `PolicyViolated` | A policy limit was breached and execution stopped. |

## Stats

`Stats` summarise what happened while agents worked: tickets finished, tokens consumed, tool calls made, time spent. Read them while agents are working, or after they finish. Labels group tickets into categories, and filtering by label scopes the figures to a single category.

```rust
let s = tickets.stats();
println!("{} done, {} tokens", s.tickets_done(), s.input_tokens());

let scan = s.stats_for_label("scan");
println!("scan: {} done", scan.tickets_done());
```

| | Method | Description |
|-|--------|-------------|
| **Run** | `elapsed()` | Return the elapsed time since the first ticket started — live while agents work, frozen at the run total once the loop finishes. `None` until the first ticket starts. |
| | `total_work_duration()` | Return the sum of every finished ticket's start-to-end span. |
| **Tickets** | `tickets_created()` | Return the count of tickets created. |
| | `tickets_done()` | Return the count of tickets that finished successfully. |
| | `tickets_failed()` | Return the count of tickets that failed. |
| | `success_rate()` | Return `done / (done + failed)`, or `None` until a ticket finishes. |
| | `total_ticket_duration()` | Return the sum of every finished ticket's creation-to-end span. |
| | `avg_ticket_duration()` | Return the mean of the same span, or `None` until a ticket finishes. |
| **Tokens** | `input_tokens()` | Return the total input tokens across all provider responses. |
| | `output_tokens()` | Return the total output tokens across all provider responses. |
| **Activity** | `steps()` | Return the total ticket-claim iterations across agents. |
| | `requests()` | Return the total provider responses received. |
| | `tool_calls()` | Return the total tool calls. |
| | `errors()` | Return the total provider errors. |
| **Labels** | `stats_for_label(label)` | Return a nested `Stats` slice scoped to tickets carrying `label`. |

# Development

## Workspace

- `crates/agentwerk/`: the library.
- `crates/use-cases/`: runnable example binaries that depend on the library.

## Building and testing

```bash
make                # build (warnings are errors)
make test           # unit tests bundled by tests/unit (workspace --lib)
make fmt            # format code
make clean          # remove build artifacts
make update         # update dependencies
```

## Integration tests

> Configure an LLM provider first (see [Environment](#environment)).

```bash
make test_integration                     # run all
make test_integration name=bash_usage     # run one
```

## Use cases

```bash
make use_case                                                 # list available
make use_case name=terminal-repl                              # run one
make use_case name=deep-research args="What is a good life?"  # with arguments
```

## Publishing

```bash
make bump                  # bump patch version, run tests, commit, tag
make bump part=minor       # bump minor version
make bump part=major       # bump major version
```

GitHub Actions handles the crates.io publish via trusted publishing once the new tag is pushed (`git push --tags`).

## Documentation

```bash
make doc                   # cargo doc --no-deps -p agentwerk (strict rustdoc)
```

## LiteLLM proxy

Start a local LiteLLM proxy on port 4000 that forwards to a provider. Requires Docker.

```bash
make litellm                               # default: anthropic
make litellm LITELLM_PROVIDER=openai       # use OpenAI
make litellm LITELLM_PROVIDER=mistral      # use Mistral
```

## Local inference servers

agentwerk relies on server-side tool calling. Enable it through the following flags:

| Server | Flag |
|---|---|
| vLLM | `--enable-auto-tool-choice --tool-call-parser <parser>` |
| llama.cpp | `--jinja` (enables tool calling) |

## Environment

Use cases and integration tests use the following environment variables:

**General**

| Variable | Description |
|----------|-------------|
| `MODEL` | Generic model override for `model_from_env()`. |
| `BRAVE_API_KEY` | Required by the `deep-research` example. |

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
| `LITELLM_PROVIDER` | LLM provider (`anthropic`, `mistral`, `openai`, `litellm`): explicit selection that overrides API-key auto-detection. |
