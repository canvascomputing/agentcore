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
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let results = Agent::new()
        .provider(provider_from_env()?)
        .model(&model_from_env()?)
        .role("Answer questions about this repository.")
        .tool(ReadFileTool)
        .task("What does Cargo.toml describe?")
        .run_dry()
        .await;

    let answer = results
        .last()
        .map(|r| r.result_string())
        .unwrap_or_default();
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

> Configure an LLM provider first (see [Environment](#environment)).

## API

- [Providers](#providers): LLM backends agents send requests to.
- [Agents](#agents): Workers that pick up tickets and produce results.
- [Prompting](#prompting): How role, context, and task shape the model's input.
- [Ticket Systems](#ticket-systems): Shared queues that route tickets to agents.
- [Tools](#tools): Capabilities agents call to take action.
- [Memory](#memory): Durable facts the model curates across tickets.
- [Schemas](#schemas): JSON schemas that validate ticket results.
- [Events](#events): Lifecycle signals the run emits.
- [Stats](#stats): Counters and timings the run records.

### Providers

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

// Pick a provider and model from environment variables.
let provider = provider_from_env()?;
let model = model_from_env()?;
```

See [Environment](#environment) for the variable names.

### Agents

An `Agent` is a single worker that turns tasks into results. It calls the provider in a loop, invokes tools as the model requests them, and writes a result back when the task is finished.

```rust
let agent = Agent::new()
    .name("worker_0")
    .provider(provider)
    .model(&model)
    .role("You are an arithmetic worker.")
    .label("worker")
    .task("Compute 2+2.");
```

#### Build

Configure an agent: identity, provider, prompt, tools, events, and memory.

| | Method | Description |
|-|--------|-------------|
| **Construct** | `Agent::new()` | Create a new agent builder. |
| **Identity** | `name(s)` | Set the identifier used for routing and events. |
| | `label(l)` / `labels([..])` | Restrict the agent to tickets carrying matching labels. |
| **Provider** | `provider(p)` | Set the LLM provider. |
| | `model(m)` | Set the model the provider runs. |
| **Prompt** | `role(text)` | Set the role prompt. |
| | `context(text)` | Replace the default context block with custom markdown. |
| **Tools** | `tool(t)` / `tools([..])` | Register a tool the agent may call. |
| | `working_dir(p)` | Set the directory tools resolve paths against. |
| **Events** | `event_handler(fn)` | Set a custom event observer. |
| | `silent()` | Drop every event instead of logging it. |
| **Memory** | `memory(&store)` | Create a `Memory` so facts persist across tickets and restarts. |

#### Run

Start an agent with `run`, queue tasks while it's working, finish with `run_dry` and read the results.

```rust
let agent = Agent::new()
    .provider(provider)
    .model(&model)
    .run();

agent.task("Compute 2+2.");
agent.task_labeled("Compute 3+3.", "math");

let results = agent.run_dry().await;
```

`run()` starts the agent in the background. While it is running, `task` and `task_labeled` queue more work. `run_dry` waits for the queue to drain, stops the agent, and returns every finished ticket's `TicketResult`. `stop` and `join` are also available for abrupt cancellation.

Methods called after the agent is built:

| | Method | Description |
|-|--------|-------------|
| **Tasks** | `task(value)` | Create a task. |
| | `task_labeled(value, label)` | Create a task tagged with `label` for label-scoped routing. |
| | `task_schema(value, schema)` | Create a task whose result must validate against `schema`. |
| | `task_schema_labeled(value, schema, label)` | Create a labelled task whose result must validate against `schema`. |
| | `create(ticket)` | Add a caller-built `Ticket` to the queue. |
| **Run** | `run()` | Start a background run and return a `Running` handle. |
| | `run_dry().await` | Run until every queued task finishes and return every `TicketResult`. |
| **Inspect** | `get_name()` | Return the configured name. |
| | `get_labels()` | Return the configured label scope. |
| | `handles(labels)` | Return `true` when the label scope overlaps. |


### Prompting

A prompt has three main parts: `role`, `context`, and `task`, see the [prompting framework](https://github.com/canvascomputing/prompting).

#### Role

```rust
let agent = Agent::new().role("You are an arithmetic worker. Show your work.");
```

The role is the agent's identity and operating rules. It is set once at build time and reused on every ticket the agent handles.

#### Context

```rust
let agent = Agent::new().context("- Repo: example/widgets\n- Branch: main");
```

The context is the first user message of every ticket. When `context(text)` is not set, agentwerk generates a default block:

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

Override when the agent needs runtime facts the default block does not carry, such as a target file or a session identifier.

#### Task

```rust
agent.task("Compute 2+2.");
agent.task(serde_json::json!({ "file": "Cargo.toml", "find": "version" }));
```

The task is the per-ticket request. `value` may be a string or any serde-serializable type; structured tasks are pretty-printed as JSON. Use `task_labeled` for label routing, or `task_schema` / `task_schema_labeled` to attach a `Schema` the result must validate against.

### Ticket Systems

A `TicketSystem` lets multiple agents work through a shared backlog of tasks. Tasks become tickets in one queue, and agents claim them either by matching labels or by direct assignment to a named agent.

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
| **Construct** | `TicketSystem::new()` | Create a new ticket system that agents can share. |
| | `add(agent)` | Register an agent with the system. |
| | `interrupt_signal(signal)` | Override the cancel signal shared across agents. |
| | `workspace(dir)` | Set the workspace directory under which `results.jsonl` and `tickets.jsonl` are written. |
| **Tasks** | `task(value)` | Create a task. |
| | `task_labeled(value, label)` | Create a task tagged with `label` for label-scoped routing. |
| | `task_schema(value, schema)` | Create a task whose result must validate against `schema`. |
| | `task_schema_labeled(value, schema, label)` | Create a labelled task whose result must validate against `schema`. |
| | `create(ticket)` | Add a caller-built `Ticket` to the queue. |
| **Run** | `run()` | Start a background run and return a `Running` handle. |
| | `run_dry().await` | Run until every queued task finishes and return every `TicketResult`. |
| **Inspect** | `get(key)` / `tickets()` / `first()` | Look up finished tickets. |
| | `search(query)` | Return tickets whose task body matches `query`, case-insensitively. |
| | `filter(predicate)` | Return tickets matching `predicate`, in creation order. |
| | `find(predicate)` | Return the first ticket matching `predicate`. |
| | `count(predicate)` | Return the count of tickets matching `predicate`. |
| | `stats()` | Return the run counters and timings. |
| **Status** | `update_status(key, status)` | Transition a ticket through the state machine. |
| | `force_status(key, status)` | Force a ticket to `status`, bypassing the state machine. |

#### Workspace

When `workspace(dir)` is set, the system also appends one observational JSON line to `<dir>/tickets.jsonl` per lifecycle event:

- `{"event":"created","ts":<ms>,"key":"TICKET-N","reporter":...,"labels":[...],"assignee":...|null,"task":<value>}`
- `{"event":"started","ts":<ms>,"key":"TICKET-N","assignee":...}`
- `{"event":"done"|"failed","ts":<ms>,"key":"TICKET-N","duration_ms":<u64>,"work_ms":<u64>}`

The actual result payload still lives in `results.jsonl`; the `done` line is a transition marker. Without a workspace, the log is skipped.

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
| `max_steps(n)` | Cap the total number of steps. |
| `max_input_tokens(n)` | Cap the total input tokens. |
| `max_output_tokens(n)` | Cap the total output tokens. |
| `max_request_tokens(n)` | Cap the input tokens per request. |
| `max_schema_retries(n)` | Cap the schema-validation retry attempts. |
| `max_request_retries(n)` | Cap the retry attempts on recoverable provider errors. |
| `request_retry_delay(d)` | Set the base delay between request retries. |
| `max_time(d)` | Cap the run's elapsed duration. |

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
| **Shell** | `BashTool` | Runs shell commands matching an allowed pattern. |
| **Web** | `WebFetchTool` | Fetches a URL and returns its body. |
| **Tickets** | `WriteResultTool` | Writes the agent's result for the current ticket and marks it done. |
| | `ManageTicketsTool` | Reads the ticket queue and creates or edits tickets. |
| | `ReadTicketsTool` / `WriteTicketsTool` | Provide read-only and write-only access to the ticket queue. |
| **Memory** | `MemoryTool` | Adds, replaces, or removes entries in the agent's memory. |
| **Discovery** | `ToolSearchTool` | Discovers tools registered with `Tool::defer(true)`. |

`BashTool::unrestricted()` allows any command. `WriteResultTool` validates against the ticket's `schema`, appends an NDJSON line to `<workspace>/results.jsonl`, and attaches the `TicketResult` to the ticket. `MemoryTool` is auto-registered when `Agent::memory(&store)` is set.

### Memory

A `Memory` lets the model carry facts from one ticket to the next, even across process restarts. It is a file-backed store the model curates through `MemoryTool`. Current entries are rendered into the system prompt at the top of every ticket. Updates during a ticket appear in the prompt at the start of the next one. Multiple agents can share one store.

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

Both agents read and write the same `memory.jsonl` (one entry per line: `{"content": "...", "added_at": <ms>}`). Bind two agents to two different stores for independent memory. Pointing `Memory::open` at the same directory as `TicketSystem::workspace` co-locates `memory.jsonl`, `results.jsonl`, and `tickets.jsonl`.

Methods on `Memory`:

| | Method | Description |
|-|--------|-------------|
| **Open** | `Memory::open(dir)` | Open or create a store at `dir`. |
| **Read** | `entries()` | Return a clone of the current entries, in insertion order. |
| **Mutate** | `add(content)` | Append a new entry. |
| | `replace(old_text, content)` | Swap the unique entry containing `old_text`. |
| | `remove(old_text)` | Drop the unique entry containing `old_text`. |
| | `rewrite(entries)` | Replace every entry in a single call. |

`add` rejects empty content, duplicates, and content that would push the rendered prompt section past the size cap.

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

Via a fully-built `Ticket`. Chain `.schema(...)` with `.label(...)` for label routing, or with `.assign_to(...)` to pin to a named agent:

```rust
use agentwerk::Ticket;

// Label routing: any agent labelled "report" picks it up.
tickets.create(Ticket::new(body.clone()).schema(schema.clone()).label("report"));

// Direct assignment: pinned to a named agent.
tickets.create(Ticket::new(body).schema(schema).assign_to("report_writer"));
```

### Events

Agents emit lifecycle events that the caller observes:

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
| **Run** | `PolicyViolated` | A policy limit was breached and the run stopped. |

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

`Stats` collects counters and timings across the run. Read them after `run()` or `run_dry()`, or at any time during a run. `stats_for_label(label)` returns a slice with the same accessors, scoped to tickets carrying that label. `run_duration()` is `None` on a slice because the elapsed run duration stays global.

| | Method | Description |
|-|--------|-------------|
| **Run** | `run_duration()` | Return the elapsed time from first ticket start to run end, or `None` while the run is active. |
| | `elapsed()` | Return the live elapsed time since the run started, or `None` until the first ticket starts. |
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

> Configure an LLM provider first (see [Environment](#environment)).

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

GitHub Actions handles the crates.io publish via trusted publishing once the new tag is pushed (`git push --tags`).

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
| `LITELLM_PROVIDER` | LLM provider (`anthropic`, `mistral`, `openai`, `litellm`): explicit selection that overrides API-key auto-detection. |
