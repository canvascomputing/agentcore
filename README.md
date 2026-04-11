
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

- **Providers:** Anthropic, Mistral, OpenAI-compatible (LiteLLM)
- **Tools:** read, write, edit, glob, grep, list, bash, tool search, custom
- **Output:** structured JSON Schema enforcement
- **Orchestration:** multi-agent spawning
- **Persistence:** session transcripts, task store
- **Tracking:** per-model cost breakdowns

## Quick Start

```rust
let provider = Arc::new(AnthropicProvider::from_api_key(api_key));

let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are a helpful coding assistant.")
    .tool(ReadFileTool)
    .tool(GrepTool)
    .build()?;

let mut ctx = InvocationContext::new(provider);
ctx.prompt = "Find all TODO comments".into();

let output = agent.run(ctx).await?;
println!("{}", output.response_raw);
```

## API

An `LlmProvider` sends requests to an LLM. An `AgentBuilder` configures an agent with a model, system prompt, and tools. Calling `agent.run(ctx)` starts the agent loop: it calls the LLM, executes tool calls, and repeats until the LLM stops. The `InvocationContext` carries runtime state — provider, cost tracker, event callback, cancellation. `AgentOutput` contains the raw text response, optional structured data, and token usage.

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
let mut ctx = InvocationContext::new(provider);
ctx.prompt = "Analyze this repo".into();
ctx.event_handler = Arc::new(|e| { /* handle events */ });
```

`InvocationContext::new(provider)` sets sensible defaults. Override what you need:

| Field | Type | Default |
|-------|------|---------|
| `agent_id` | `String` | `generate_agent_id("agent")` |
| `event_handler` | `Arc<dyn Fn(Event)>` | no-op |
| `cancel_signal` | `Arc<AtomicBool>` | `false` |
| `prompt` | `String` | empty |
| `template_variables` | `HashMap<String, Value>` | empty |
| `working_directory` | `PathBuf` | current dir |
| `provider` | `Arc<dyn LlmProvider>` | *required* |
| `cost_tracker` | `CostTracker` | new (pre-loaded with Claude pricing) |
| `session_store` | `Option<Arc<Mutex<SessionStore>>>` | `None` |
| `command_queue` | `Option<Arc<CommandQueue>>` | `None` |

### AgentOutput

```rust
let output = agent.run(ctx).await?;
```

| Field | Type | Description |
|-------|------|-------------|
| `response` | `Option<Value>` | Validated JSON if `output_schema` was set and the agent produced compliant data |
| `response_raw` | `String` | Free-form text from the LLM (always present, may be empty) |
| `token_usage` | `TokenUsage` | Accumulated token counts across all turns |

When you configure `.output_schema(schema)` on the builder, the agent automatically registers a `StructuredOutput` tool. The LLM calls this tool with JSON matching your schema, which is validated and captured in `response`.

### Event

```rust
on_event: Arc::new(|event| match &event {
    Event::TextChunk { content, .. } => print!("{content}"),
    Event::ToolCallStart { tool_name, .. } => eprintln!("[{tool_name}]"),
    _ => {}
})
```

| Variant | Fields |
|---------|--------|
| `AgentStart` | `agent_name` |
| `AgentEnd` | `agent_name`, `turns` |
| `AgentError` | `agent_name`, `message` |
| `TurnStart` | `agent_name`, `turn` |
| `TurnEnd` | `agent_name`, `turn` |
| `ToolCallStart` | `agent_name`, `tool_name`, `call_id`, `input` |
| `ToolCallEnd` | `agent_name`, `tool_name`, `call_id`, `output`, `is_error` |
| `TokenUsage` | `agent_name`, `model`, `usage` |
| `TextChunk` | `agent_name`, `content` |

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
let provider = AnthropicProvider::from_api_key(api_key);
let provider = MistralProvider::from_api_key(api_key);
let provider = LiteLlmProvider::from_api_key(api_key);
```

For custom HTTP transport, use `::new(api_key, transport)` instead.

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

Examples detect the provider based on environment variables:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Use Anthropic directly |
| `ANTHROPIC_BASE_URL` | API URL (default: `https://api.anthropic.com`) |
| `ANTHROPIC_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
| `MISTRAL_API_KEY` | Use Mistral directly |
| `MISTRAL_BASE_URL` | API URL (default: `https://api.mistral.ai`) |
| `MISTRAL_MODEL` | Model (default: `mistral-medium-2508`) |
| `LITELLM_API_KEY` | Auth key (optional) |
| `LITELLM_API_URL` | Use LiteLLM proxy (default: `http://localhost:4000`) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |

```bash
make example name=llm_provider_call           # direct API call
make example name=agent_with_tools            # agent with custom tool
make example name=multi_agent_spawn           # multi-agent orchestration
make example name=task_and_session_store      # persistence
make example name=code_review                 # code review CLI
```

### LiteLLM

Start a LiteLLM Docker container:

```bash
make litellm                      # default: anthropic
make litellm provider=anthropic   # uses ANTHROPIC_API_KEY
make litellm provider=mistral     # uses MISTRAL_API_KEY
make litellm provider=openai      # uses OPENAI_API_KEY
```