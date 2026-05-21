<p align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/logo.png" width="200" />
</p>

<h1 align="center">agentwerk</h1>

<p align="center">
  <strong>A minimal Rust crate for ticket-driven agentic workflows at scale.</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#api">API</a> •
  <a href="#development">Development</a>
</p>

<p align="center">agentwerk lets you create agentic workflows around a ticket-driven execution loop, with built-in tools, a knowledge store, schema validation, retry mechanisms, budget policies, and multi-provider support.</p>

<p align="center"><em>agentwerk pairs "agent" with the German "Werk", a word for both factory and artwork: machinery for building agentic systems.</em></p>

---

## Installation

```bash
cargo add agentwerk
```

## Quick Start

```rust
use agentwerk::tools::ReadFileTool;
use agentwerk::Agent;

#[tokio::main]
async fn main() {
    let work = Agent::new()
        .from_env()
        .role("You are a Rust developer who reads source files to answer questions.")
        .tool(ReadFileTool)
        .task("What does Cargo.toml describe?")
        .finish()
        .await;

    let answer = work.last_result().unwrap();
    println!("{answer}");
}
```

## Use Cases

Example projects built with agentwerk:

- [Terminal REPL](crates/use-cases/src/terminal_repl/): minimal interactive chat
- [Divide and Conquer](crates/use-cases/src/divide_and_conquer/): arithmetic problem shared across agents
- [Deep Research](crates/use-cases/src/deep_research/): deep research pipeline (requires `BRAVE_API_KEY`)
- [Malware Scanner](crates/use-cases/src/malware_scanner/): identify indicators of compromise in a software package

> Configure an LLM provider first (see [Environment](#environment)).

```bash
make use_case                # list available names
make use_case name=<name>    # run one
```

# API

- [Agents](#agents): Workers that pick up tickets and produce results.
- [Tickets](#tickets): Ticket system allowing to orchestrate complex work.
- [Prompting](#prompting): Role, context, and task shaping the work of an agent.
- [Tools](#tools): Capabilities agents use to solve a ticket.
- [Knowledge](#knowledge): Knowledge base an agent creates during a run.
- [Schemas](#schemas): Schemas for validating ticket results.
- [Compaction](#compaction): Automatic context window summarization.
- [Events](#events): Lifecycle events emitted while agents work.
- [Stats](#stats): Metrics about tickets, tokens and time.

## Agents

An `Agent` picks up **tickets**, uses assigned tools to solve them, and writes the result back onto each ticket.

```rust
use agentwerk::tools::ReadFileTool;

let agent = Agent::new()
    .name("worker_0")
    .label("math")
    .tool(ReadFileTool)
```

| Method | Description |
|--------|-------------|
| `name(s)` | Set an identifier for assigning tickets. |
| `label(l)` / `labels([..])` | Restrict the agent to tickets carrying matching labels. |
| `tool(t)` / `tools([..])` | Register a tool the agent may call. |
| `dir(p)` | Set the directory accessible for tool calls. |

### Providers

A `Provider` connects the agent to an LLM service. agentwerk ships providers for Anthropic, OpenAI, Mistral, and a LiteLLM proxy.

```rust
use agentwerk::providers::AnthropicProvider;

let agent = Agent::new()
    .provider(AnthropicProvider::new(key))
    .model("claude-sonnet-4-20250514");

// Or pick from environment variables (see Environment).
let agent = Agent::new().from_env();
```

Each provider exposes `.base_url(url)` and `.timeout(duration)` to override the endpoint and request timeout.

| Method | Description |
|--------|-------------|
| `provider(p)` | Set the LLM provider. |
| `provider_from_env()` | Detect the provider from environment variables. |
| `model(m)` | Set the model the provider runs. |
| `model_from_env()` | Read the model name from environment variables. |
| `from_env()` | Detect provider and model in one call. |

## Tickets

<p align="left">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/tickets.jpg" width="400" />
</p>

The `TicketSystem` is the core data structure of agentwerk to orchestrate complex collaboration between agents. A `task` is the work itself, a `ticket` wraps it with additional metadata, like labels and schemas. Labels route work to matching agents.

```rust
use agentwerk::{Ticket, TicketSystem};

let tickets = TicketSystem::new();

tickets.agent(security_reviewer);
tickets.agent(performance_reviewer);

for path in ["src/auth.rs", "src/session.rs", "src/tokens.rs"] {
    tickets.task_labeled(
        format!("Audit {path} for input-validation gaps."),
        "security",
    );
}

let audit = Ticket::new("Audit src/api/ for blocking I/O in async handlers.")
    .label("performance")
    .schema(finding_schema);

tickets.ticket(audit);
```

| Method | Description |
|--------|-------------|
| `agent(agent)` | Register an agent with the system. |
| `pool(n, build)` | Register `n` agents built by `build(i)`, where `i` is the 0-based worker index. |
| `dir(d)` | Set the directory where knowledge, results, and ticket logs are persisted. |
| `task(t)` | Submit a task. |
| `task_labeled(t, l)` | Submit a task tagged with `l` for label-scoped routing. |
| `ticket(t)` | Submit a caller-built `Ticket`. Compose schemas, labels, and parent links via `Ticket::new(...).schema(...).label(...)`. |

### Execution

Start, wait, and cancel a run:

```rust
tickets.start();
tickets.finish().await;
let answer = tickets.last_result();
```

| Method | Description |
|--------|-------------|
| `start()` | Begin processing tickets in the background. |
| `finish().await` | Process every queued ticket and return. |
| `cancel()` | Cancel the run from anywhere: async code, ctrl-c handlers, drop guards. |
| `stop().await` | Cancel the run and wait for it to wind down. |
| `cancel_on_ctrl_c()` | Cancel the run on the first ctrl-c. |

### Reading results

Query the system after `finish().await` returns:

```rust
tickets.finish().await;

if let Some(answer) = tickets.last_result() {
    println!("{answer}");
}

for ticket in tickets.tickets() {
    println!("{}: {}", ticket.key(), ticket.status());
}
```

| Method | Description |
|--------|-------------|
| `last_result()` | Return the most recent finished ticket's payload as a string. |
| `all_results()` | Return every finished ticket's payload as a string. |
| `tickets()` | Return every ticket in creation order, with status, payload, and metadata. |
| `is_cancelled()` | Return `true` once a cancel has been requested. |

### Policies

Configure execution policies on a ticket system. A breach fires `EventKind::PolicyViolated` and halts execution.

```rust
let tickets = TicketSystem::new();
tickets
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

## Prompting

Every prompt has three parts: `role` (who the agent is), `context` (the situation it operates in), and `task` (work it should perform). The structure follows the [prompting guide](https://github.com/canvascomputing/prompting).

```rust
let agent = Agent::new()
    .role("You are an arithmetic worker. Compute step by step and show your work.")
    .context("- Stage 2 of a math-tutor pipeline.\n- Attempts remaining: 2.")
    .task("Compute (47 * 92) / {divisor}, then round to the nearest integer.")
    .template_variable("divisor", "8");
```

When `context(...)` is not set, agentwerk supplies a default block:

```markdown
- Working directory: /Users/caro
- Platform: darwin
- OS version: 25.1.0
- Date: 2026-05-06
- Steps remaining: 8
- Input tokens remaining: 95000
- Output tokens remaining: 12000
- Time remaining: 240s
```

At each step the agent sends:

- **system**: `role(...)`, with the `knowledge(...)` index
- **tools**: tools registered with `tool(...)`
- **messages**: a list that grows step by step

`messages` begins with `context(...)` and the body of the current ticket. On each later step the agent appends the model's reply.

## Tools

Give agents access to tools helping them to solve a given task. Each tool exposes an action the agent can choose to take. agentwerk provides minimal baseline tools:

| | Tool | Description |
|-|------|-------------|
| **File** | `ReadFileTool` | Read a file with line numbers, offset, and limit. |
| | `WriteFileTool` | Create or overwrite a file. |
| | `EditFileTool` | Replace text in a file. |
| **Search** | `GlobTool` | Find files by pattern. |
| | `GrepTool` | Search file contents. |
| | `ListDirectoryTool` | List files and directories. |
| **Shell** | `BashTool` | Run a shell command matching an allowed pattern. |
| **Web** | `WebFetchTool` | Fetch a URL and read its body. |
| **Tickets** | `WriteResultTool` | Write the result for the current ticket and mark it done. |
| | `WriteHandoverTool` | Write the result, mark the ticket done, and hand follow-up work to another agent. |
| | `ManageTicketsTool` | Read the ticket queue and create or edit tickets. |
| | `ReadTicketsTool` | Read the ticket queue. |
| **Knowledge** | `KnowledgeTool` | Write, read, remove, or list pages in the agent's knowledge store. |
| **Discovery** | `ToolSearchTool` | Discover tools registered with `Tool::defer(true)`. |

### Custom tools

Define custom tools for specific needs. Each tool declares a JSON-Schema for its inputs:

```rust
use agentwerk::tools::{Tool, ToolResult};
use serde_json::json;

let greet = Tool::new("greet", "Say hello")
    .schema(json!({
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    }))
    .read_only(true)
    .handler(|input, _ctx| async move {
        let name = input["name"].as_str().unwrap_or("world");
        Ok(ToolResult::success(format!("Hello, {name}!")))
    });
```

`.read_only(true)` allows the agent to run a tool concurrently with other read-only calls in the same step.

## Knowledge

A `Knowledge` store is the agent's long-term memory. It is written to disk, can be shared across multiple agents, and is curated by the agent through `KnowledgeTool`.

Each entry is stored as a markdown page on disk; a compact index of one-line summaries is injected into the system prompt so the agent can decide which pages to read.

```rust
use agentwerk::Knowledge;

// Open a store and share it across agents:
let store = Knowledge::open("./.agentwerk")?;
let alice = Agent::new().knowledge(&store);
let bob = Agent::new().knowledge(&store);

// Raise the rendered-index char budget (default 4000):
let store = Knowledge::open("./.agentwerk")?.index_char_limit(12_000);
let agent = Agent::new().knowledge(&store);
```

| Method | Description |
|--------|-------------|
| `knowledge(&store)` | Bind a shared knowledge store to the agent. |

## Schemas

A `Schema` constrains the result an agent must produce for a ticket. A violation triggers a retry until `max_schema_retries` is exhausted.

```rust
use agentwerk::schemas::Schema;
use agentwerk::Ticket;

let schema = Schema::parse(json!({
    "type": "object",
    "properties": { "title": { "type": "string" } },
    "required": ["title"]
}))?;

tickets.ticket(Ticket::new("Write a report.").schema(schema));
```

## Compaction

When an agent's conversation grows close to the model's context window, agentwerk summarizes the older portion of the history and continues with the compacted version. Compaction runs automatically; no configuration is required.

Two seams trigger it:

| Trigger | When |
|---------|------|
| Proactive | Before each request, when the estimated input tokens cross `context_window - 20K (output reserve) - 13K (headroom)`. |
| Reactive | After the provider rejects a request as too large or flags the response with context-window overflow. |

In both cases the agent keeps the leading context and task messages, drops the rest, and replaces them with one summary message generated by a no-tools provider call. Each attempt emits `CompactionStarted` and either `CompactionFinished` (success) or `CompactionFailed` (the summarization call returned a provider error), so observers can see the lifecycle.

If a request still exceeds the window after one compaction, the ticket fails with `RequestFailed`.

## Events

Events report everything that happens while your agents work and give you deep insights into behavior or failures.

```rust
use std::sync::Arc;
use agentwerk::event::{Event, EventKind};

let agent = Agent::new()
    .event_handler(Arc::new(|event: Event| {
        if let EventKind::TicketDone { key } = &event.kind {
            eprintln!("[{}] done {key}", event.agent_name);
        }
    }));
```

| | Kind | Description |
|-|------|-------------|
| **Ticket** | `TicketStarted` | An agent claimed a ticket. |
| | `TicketDone` | A ticket finished successfully. |
| | `TicketFailed` | A ticket failed. |
| **Provider** | `RequestStarted` | A provider request started. |
| | `RequestFinished` | A provider request finished and reported its token usage. |
| | `RequestFailed` | A provider request failed and stopped the ticket. |
| | `TextChunkReceived` | A streamed text chunk arrived. |
| **Tool** | `ToolCallStarted` | A tool invocation started. |
| | `ToolCallFinished` | A tool invocation finished. |
| | `ToolCallFailed` | A tool invocation failed but the ticket continues. |
| **Compaction** | `CompactionStarted` | Compaction is about to summarize the conversation tail. |
| | `CompactionFinished` | Compaction finished and replaced the tail with a summary. |
| | `CompactionFailed` | The summarization call failed; the ticket is about to fail. |
| **Run** | `PolicyViolated` | A policy limit was breached and execution stopped. |

## Stats

`Stats` contain metrics about the progress of your agents' work, allowing to optimize your agentic system and identify bottlenecks.

```rust
let s = tickets.stats();
let scan = s.stats_for_label("scan");
```

| | Method | Description |
|-|--------|-------------|
| **Run** | `run_duration()` | Return the run's elapsed duration, live while agents work and frozen once the loop finishes. `None` until the first ticket starts. |
| **Work** | `work_duration()` | Return the sum of every finished ticket's start-to-end span. |
| | `avg_work_duration()` | Return the mean of the same span, or `None` until a ticket finishes. |
| **Tickets** | `tickets_created()` | Return the count of tickets created. |
| | `tickets_done()` | Return the count of tickets that finished successfully. |
| | `tickets_failed()` | Return the count of tickets that failed. |
| | `tickets_success_rate()` | Return `done / (done + failed)`, or `None` until a ticket finishes. |
| | `ticket_duration()` | Return the sum of every finished ticket's creation-to-end span. |
| | `avg_ticket_duration()` | Return the mean of the same span, or `None` until a ticket finishes. |
| **Tokens** | `input_tokens()` | Return the total input tokens across all provider responses. |
| | `output_tokens()` | Return the total output tokens across all provider responses. |
| **Activity** | `steps()` | Return the count of times an agent picked up a ticket to process. |
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
make hooks          # install Claude Code hooks
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
