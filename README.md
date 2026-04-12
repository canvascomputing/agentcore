
# 🤖 `agent` - Minimal Agentic Framework

```
                             __   
    .---.-.-----.-----.-----|  |_ 
    |  _  |  _  |  -__|     |   _|
    |___._|___  |_____|__|__|____|
          |_____|                 

  A minimal Rust crate that gives any
  application agentic capabilities.
```

Build agents that reason, use tools, and manage long-running tasks autonomously.

- **Agentic loop** with automatic tool execution
- **Integration** with Anthropic, Mistral, and OpenAI-compatible APIs
- **Cost tracking** per model with budget limits
- **Agent spawning** in foreground and background
- **Basic tools** for file, search, and shell operations
- **Memory Layers** for session transcripts and task persistence
- **Output schemas** for type-safe model responses

## Use Cases

Ready-to-use CLI tools built with this project.

> Consider setting your LLM provider's environment variables for key, model or base URL.

### Project Scanner

Scans a directory and outputs a JSON summary with project description and languages used.

```bash
make use-case name=project-scanner              # scans current directory
make use-case name=project-scanner -- ./other    # scans a specific folder
cat project.json
```

Output:
```json
{
  "summary": "A minimal Rust framework for building agentic LLM applications with tool use",
  "languages": ["Rust"]
}
```

## API

Create an `LlmProvider`, define your `Tool`s, wire them into an `AgentBuilder`, call `agent.run()` with an `InvocationContext`, stream `Event`s during execution, and get back an `AgentOutput`.

### LlmProvider

Connect to any LLM. Uses reqwest by default, or bring your own transport.

```rust
use agent::{AnthropicProvider, MistralProvider, LiteLlmProvider};

let provider = AnthropicProvider::from_api_key(key);
let provider = MistralProvider::from_api_key(key);
let provider = LiteLlmProvider::from_api_key(key);
let provider = AnthropicProvider::new(key, custom_transport);
```

### Tool

Define what the agent can do. Read-only tools run concurrently.

```rust
use agent::{ToolBuilder, ToolResult};

let tool = ToolBuilder::new("greet", "Say hello")
    .schema(json!({...}))
    .read_only(true)
    .handler(|input, ctx| Box::pin(async move {
        Ok(ToolResult { content: "Hello!".into(), is_error: false })
    }))
    .build();
```

Built-in tools:

- `ReadFileTool`: read a file with line numbers, offset, and limit
- `WriteFileTool`: create or overwrite a file
- `EditFileTool`: find-and-replace string in a file
- `GlobTool`: find files by pattern (e.g. `**/*.rs`)
- `GrepTool`: search file contents by substring
- `ListDirectoryTool`: list directory entries with type and size
- `BashTool`: execute a shell command
- `ToolSearchTool`: discover available tools by keyword
- `SpawnAgentTool`: delegate work to a sub-agent

### AgentBuilder

Configure an agent with a model, prompt, tools, and guardrails.

```rust
use agent::AgentBuilder;

let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Help with {topic}")   // {key} replaced from template_variables
    .tool(ReadFileTool)
    .tool(GrepTool)
    .max_tokens(4096)
    .max_turns(10)
    .max_budget(1.0)                      // USD spend limit
    .output_schema(json!({...}))          // force structured JSON response
    .sub_agent(researcher)                // available to SpawnAgentTool
    .build()?;
```

### InvocationContext

Runtime state for a single agent run. Only `provider` is required.

```rust
use agent::InvocationContext;

let mut ctx = InvocationContext::new(provider);
ctx.prompt = "Find all TODOs".into();
```

Optional overrides:

```rust
// values for {key} placeholders in system_prompt
ctx.template_variables.insert("topic".into(), json!("rust"));

// base path for file tools
ctx.working_directory = PathBuf::from("./src");

// stream events (text chunks, tool calls, errors)
ctx.event_handler = Arc::new(|e| { ... });

// gracefully stop the agent loop
ctx.cancel_signal = Arc::new(AtomicBool::new(false));

// persist conversation transcripts to disk
ctx.session_store = Some(Arc::new(Mutex::new(store)));

// receive notifications from background sub-agents
ctx.command_queue = Some(Arc::new(CommandQueue::new()));
```

### Event

Lifecycle and progress notifications emitted during each agent run.

```rust
use agent::Event;

ctx.event_handler = Arc::new(|event| match &event {
    Event::TextChunk { content, .. } => print!("{content}"),
    Event::ToolCallStart { tool_name, input, .. } => eprintln!("[{tool_name}] {input}"),
    Event::ToolCallEnd { tool_name, is_error, .. } if *is_error => eprintln!("[error] {tool_name}"),
    Event::AgentStart { agent_name } => eprintln!("{agent_name} started"),
    Event::AgentEnd { agent_name, turns } => eprintln!("{agent_name} done in {turns} turns"),
    Event::AgentError { message, .. } => eprintln!("error: {message}"),
    Event::TokenUsage { model, usage, .. } => eprintln!("{model}: {} tokens", usage.input_tokens),
    _ => {}
});
```

### AgentOutput

The result of `agent.run()` — text, structured data, and token usage.

```rust
use agent::AgentOutput;

let output: AgentOutput = agent.run(ctx).await?;
output.response_raw
output.response
output.token_usage.input_tokens
output.token_usage.output_tokens
```

With `.output_schema()`, the agent returns validated JSON in `response`:

```rust
AgentBuilder::new()
    .output_schema(json!({
        "type": "object",
        "properties": { "category": { "type": "string" } },
        "required": ["category"]
    }))
    ...

output.response.unwrap()["category"]  // "billing"
```

### CostTracker

Track spend across all agents. Pre-loaded with Claude pricing.

```rust
use agent::CostTracker;

let tracker = CostTracker::new();
tracker.record_usage("claude-sonnet-4-20250514", &usage);
tracker.total_cost_usd()
tracker.total_requests()
tracker.total_tool_calls()
tracker.summary()
```

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
make example name=llm_provider_call
make example name=agent_with_tools
make example name=multi_agent_spawn
make example name=task_and_session_store
make example name=code_review
```

### LiteLLM

Start a LiteLLM Docker container:

```bash
make litellm # default: anthropic
make litellm provider=anthropic
make litellm provider=mistral
make litellm provider=openai
```