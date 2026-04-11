
# 🤖 `agent` - Minimal Agentic Framework

```
                          __   
 .---.-.-----.-----.-----|  |_ 
 |  _  |  _  |  -__|     |   _|
 |___._|___  |_____|__|__|____|
       |_____|                 
                    
  A minimal Rust framework for
  building agentic applications
```

- **Providers:** Anthropic, OpenAI-compatible (LiteLLM)
- **Tools:** read, write, edit, glob, grep, list, bash, tool search, custom
- **Output:** structured JSON Schema enforcement
- **Orchestration:** multi-agent spawning
- **Persistence:** session transcripts, task store
- **Tracking:** per-model cost breakdowns

## Quick Start

```rust
let provider = Arc::new(AnthropicProvider::new(api_key, transport));

let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are a helpful coding assistant.")
    .tool(ReadFileTool)
    .tool(GrepTool)
    .build()?;

let output = agent.run(InvocationContext {
    input: "Find all TODO comments".into(),
    state: HashMap::new(),
    working_directory: std::env::current_dir()?,
    provider,
    cost_tracker: CostTracker::new(),
    on_event: Arc::new(|_| {}),
    cancelled: Arc::new(AtomicBool::new(false)),
    session_store: None,
    command_queue: None,
    agent_id: generate_agent_id("assistant"),
}).await?;

println!("{}", output.content);
```

## API

An `LlmProvider` sends requests to an LLM. An `AgentBuilder` configures an agent with a model, system prompt, and tools. Calling `agent.run(ctx)` starts the agent loop: it calls the LLM, executes tool calls, and repeats until the LLM stops. The `InvocationContext` carries runtime state — provider, cost tracker, event callback, cancellation. `AgentOutput` contains the final text and optional structured data.

### AgentBuilder

```rust
let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are helpful.")
    .tool(ReadFileTool)
    .build()?;
```

| Method | Description |
|--------|-------------|
| `.name(impl Into<String>)` | Agent name (required) |
| `.description(impl Into<String>)` | Agent description |
| `.model(impl Into<String>)` | LLM model (required) |
| `.system_prompt(impl Into<String>)` | System prompt, supports `{key}` interpolation from `state` |
| `.tool(impl Tool)` | Register a tool (repeatable) |
| `.max_tokens(u32)` | Max response tokens (default: 4096) |
| `.max_turns(u32)` | Max loop iterations |
| `.max_budget(f64)` | USD cost limit |
| `.output_schema(Value)` | Enforce structured JSON output |
| `.sub_agent(Arc<dyn Agent>)` | Register sub-agent for `SpawnAgentTool` (repeatable) |
| `.prompt_builder(PromptBuilder)` | Inject memory, instructions, environment context |
| `.build()` | `Result<Arc<dyn Agent>>` |

### InvocationContext

```rust
let output = agent.run(InvocationContext {
    input: "Analyze this repo".into(),
    provider,
    cost_tracker: CostTracker::new(),
    on_event: Arc::new(|e| { /* handle events */ }),
    cancelled: Arc::new(AtomicBool::new(false)),
    ..
}).await?;
```

| Field | Type | Description |
|-------|------|-------------|
| `input` | `String` | User prompt |
| `state` | `HashMap<String, Value>` | Key-value pairs for `{key}` interpolation |
| `working_directory` | `PathBuf` | Base path for tool file operations |
| `provider` | `Arc<dyn LlmProvider>` | LLM provider |
| `cost_tracker` | `CostTracker` | Shared cost tracker (thread-safe, cloneable) |
| `on_event` | `Arc<dyn Fn(Event)>` | Event callback for streaming |
| `cancelled` | `Arc<AtomicBool>` | Cancellation flag, checked each turn |
| `session_store` | `Option<Arc<Mutex<SessionStore>>>` | JSONL transcript persistence |
| `command_queue` | `Option<Arc<CommandQueue>>` | Inter-agent notifications |
| `agent_id` | `String` | Unique ID (use `generate_agent_id()`) |

### AgentOutput

```rust
output.content              // final text response
output.usage                // accumulated token usage
output.structured_output    // Some(Value) if output_schema was set
```

### Event

```rust
on_event: Arc::new(|event| match &event {
    Event::Text { text, .. } => print!("{text}"),
    Event::ToolStart { tool, input, .. } => eprintln!("[{tool}]"),
    _ => {}
})
```

| Variant | Fields |
|---------|--------|
| `TurnStart` | `agent`, `turn` |
| `Text` | `agent`, `text` |
| `ToolStart` | `agent`, `tool`, `id`, `input` |
| `ToolEnd` | `agent`, `tool`, `id`, `result`, `is_error` |
| `Usage` | `agent`, `model`, `usage` |
| `AgentStart` | `agent` |
| `AgentEnd` | `agent`, `turns` |
| `Error` | `agent`, `error` |

### Tool

```rust
let tool = ToolBuilder::new("greet", "Say hello")
    .schema(json!({"type": "object", "properties": {"name": {"type": "string"}}}))
    .read_only(true)
    .handler(|input, _ctx| Box::pin(async move {
        Ok(ToolResult { content: format!("Hello, {}!", input["name"].as_str().unwrap()), is_error: false })
    }))
    .build();
```

| Method | Description |
|--------|-------------|
| `name()` | Tool name |
| `description()` | Description shown to the LLM |
| `input_schema()` | JSON Schema for input validation |
| `is_read_only()` | If true, runs concurrently with other read-only tools (default: false) |
| `should_defer()` | Exclude full schema until discovered via `ToolSearchTool` (default: false) |
| `call(input, ctx)` | Execute the tool, receives `ToolContext` with `working_directory` |

Built-in: `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `GlobTool`, `GrepTool`, `ListDirectoryTool`, `BashTool`, `ToolSearchTool`, `SpawnAgentTool`

### LlmProvider

```rust
let provider = AnthropicProvider::new(api_key, transport);
let provider = LiteLlmProvider::new(api_key, transport).base_url("http://localhost:4000".into());
```

`HttpTransport` wraps any HTTP client as `Box<dyn Fn(url, headers, body) -> Future<Result<Value>>>`.

### CostTracker

```rust
let tracker = CostTracker::new();   // pre-loaded with Claude pricing
tracker.record_usage("claude-sonnet-4-20250514", &response.usage);
println!("{}", tracker.summary());
// Total cost:            $0.0123
// claude-sonnet-4:  2.5k input, 800 output ($0.0123)
```

| Method | Description |
|--------|-------------|
| `::new()` | Pre-loaded with Claude pricing |
| `.model_pricing(model, costs)` | Add custom model pricing |
| `.record_usage(model, &usage)` | Record token usage |
| `.total_cost_usd()` | Total spend |
| `.total_requests()` | Total API calls |
| `.total_tool_calls()` | Total tool invocations |
| `.summary()` | Formatted multi-line breakdown |

## Development

```bash
make          # build
make test     # test
make fmt      # format
make example  # list and run examples
make litellm  # start LiteLLM proxy
```

### Examples

Examples auto-detect the provider: `ANTHROPIC_API_KEY`, `LITELLM_API_URL`, or a running proxy at `localhost:4000`.

```bash
make example name=llm_provider_call           # direct API call
make example name=agent_with_tools            # agent with custom tool
make example name=multi_agent_spawn           # multi-agent orchestration
make example name=task_and_session_store      # persistence
make example name=code_review                 # code review CLI
```

### LiteLLM

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Use Anthropic directly |
| `ANTHROPIC_BASE_URL` | API URL (default: `https://api.anthropic.com`) |
| `ANTHROPIC_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
| `LITELLM_API_KEY` | Auth key (optional) |
| `LITELLM_API_URL` | Use LiteLLM proxy (default: `http://localhost:4000`) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |

```bash
make litellm                     # default: anthropic
make litellm provider=openai     # uses OPENAI_API_KEY
```