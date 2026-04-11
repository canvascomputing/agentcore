# Agent Loop and Command Queue

## Overview

This sub-plan implements the single-agent execution engine: the command queue for task notification routing, the `Agent` trait, `AgentBuilder`, `InvocationContext`, `LlmAgent` with its core loop, and the `Event` system for observability. After this, a single agent can call an LLM, execute tools, and track costs — but structured output enforcement is covered in [`output.md`](./output.md).

The `PromptBuilder` from the persistence increment is optional — if `prompt_builder` is `None` on the agent, no context message is injected. This means Agent can be implemented in parallel with Persistence.

## Dependencies

- [Core types](../1-base/types.md): `LlmProvider`, `CompletionRequest`, `ModelResponse`, `ContentBlock`, `StopReason`, `Usage`, `CostTracker`
- [Tool system](../3-tools/traits.md): `Tool`, `ToolResult`, `ToolRegistry`, `ToolSearchResult`, `execute_tool_calls`

**Parallelizable with** [Persistence](../5-persistence/).

## Files

```
crates/agent-core/src/agent.rs
```

---

## Specification

### 6. Task Queuing and Message Queue

A unified command queue routes user input, task notifications, and system events to the agent loop.

#### 6.1 Command Queue

```rust
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueuePriority {
    Now = 0,    // interrupt: abort current tool, send immediately
    Next = 1,   // mid-turn: process between tool result and next API call
    Later = 2,  // end-of-turn: wait for turn completion, then send as new query
}

#[derive(Debug, Clone)]
pub struct QueuedCommand {
    pub content: String,
    pub priority: QueuePriority,
    pub source: CommandSource,
    pub agent_id: Option<String>,   // routes to specific sub-agent (None = main)
}

#[derive(Debug, Clone)]
pub enum CommandSource {
    UserInput,
    TaskNotification { task_id: String },
    System,
}

/// Thread-safe priority queue for commands.
/// Processes: Now > Next > Later. Within same priority: FIFO.
pub struct CommandQueue {
    inner: Arc<Mutex<VecDeque<QueuedCommand>>>,
    notify: Arc<tokio::sync::Notify>,   // wake blocked consumers
}

impl CommandQueue {
    pub fn new() -> Self;

    /// Enqueue a command (default priority: Next).
    pub fn enqueue(&self, command: QueuedCommand);

    /// Enqueue a task notification (default priority: Later).
    /// Prevents task notifications from starving user input.
    pub fn enqueue_notification(&self, task_id: &str, summary: &str);

    /// Dequeue the highest-priority command, optionally filtered by agent_id.
    pub fn dequeue(&self, agent_id: Option<&str>) -> Option<QueuedCommand>;

    /// Block until a command is available, then dequeue.
    pub async fn wait_and_dequeue(&self, agent_id: Option<&str>) -> QueuedCommand;
}
```

**Example: Routing commands through the queue**

```rust
let queue = CommandQueue::new();

// User types a message while an agent is working → enqueue at Next priority
queue.enqueue(QueuedCommand {
    content: "Also check the error handling".into(),
    priority: QueuePriority::Next,
    source: CommandSource::UserInput,
    agent_id: None,
});

// Background agent finishes → notification at Later priority (won't starve user input)
queue.enqueue_notification("agent_42", "Research completed: found 3 relevant papers");

// Dequeue processes highest priority first: Now > Next > Later
let cmd = queue.dequeue(None);  // → "Also check the error handling" (Next)
let cmd = queue.dequeue(None);  // → notification about agent_42 (Later)
```

#### 6.2 How Task Notifications Flow

When a sub-agent completes (or fails), a notification is enqueued at `Later` priority:

1. The sub-agent completes its work.
2. `task_registry.complete(task_id, result)` is called to mark the task as done.
3. `command_queue.enqueue_notification(task_id, summary)` places a notification in the queue at `Later` priority.
4. The agent loop, after its current turn ends, calls `command_queue.dequeue(agent_id=None)` and receives the `QueuedCommand` with `source: TaskNotification`.
5. The notification is injected as the next user message: `"Task {id} completed: {summary}"`.
6. The agent processes the notification and decides the next action.

#### 6.3 Agent Loop Integration with Queue

The agent loop drains the command queue internally. After processing tool results and before the next LLM API call, the agent checks for queued commands addressed to it via `queue.dequeue(Some(&ctx.agent_id))`. `Now` and `Next` commands are injected as user messages; `Later` commands are re-enqueued for processing after the turn ends.

Applications only need to enqueue user input and call `agent.run()`:

```rust
let queue = Arc::new(CommandQueue::new());
let session_store = Arc::new(Mutex::new(SessionStore::new(base_dir, &session_id)));

loop {
    let input = read_user_input().await;
    let ctx = InvocationContext {
        input,
        session_store: Some(session_store.clone()),
        command_queue: Some(queue.clone()),
        agent_id: generate_agent_id("main"),
        // ... other fields ...
    };
    let output = agent.run(ctx).await?;
    println!("{}", output.content);
}
```

---

### 9. Agent System (`agent.rs`)

#### 9.1 Design Principles

1. **One trait** — Every agent (LLM, custom) implements `Agent`. There is no "SubAgent" type. A sub-agent is just an agent that another agent runs.
2. **Build-time vs runtime** — Static properties (model, tools, prompt template) go in the agent struct. Dynamic properties (input, state) flow through `InvocationContext` at runtime.
3. **Agents own their lifecycle** — Agents record their own transcripts via `session_store`, poll the `command_queue` between turns, and manage background sub-agent tracking. These shared resources are passed into `InvocationContext` as `Arc`-wrapped values, just like `provider` and `cost_tracker`.

#### 9.2 Agent Trait

```rust
/// Output returned by every agent type.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    /// The final text output.
    pub content: String,
    /// Aggregated token usage across all LLM calls (including children).
    pub usage: Usage,
    /// Structured output extracted from the StructuredOutput tool call.
    /// Present only when the agent was configured with an output_schema (see output.md).
    pub structured_output: Option<serde_json::Value>,
}

impl AgentOutput {
    pub fn empty(usage: Usage) -> Self {
        Self { content: String::new(), usage, structured_output: None }
    }
}

/// The single agent interface. Implemented by AgentBuilder-built agents
/// and any user-defined agent.
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run(&self, ctx: InvocationContext)
        -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>>;
}
```

#### 9.3 InvocationContext (Runtime Data)

```rust
/// Runtime context passed to Agent::run(). Cloned for child agents.
/// Contains only runtime data — agent configuration lives in the agent struct.
#[derive(Clone)]
pub struct InvocationContext {
    /// The user's prompt or workflow input for this agent.
    pub input: String,

    /// Key-value state for prompt interpolation. Agents read from it to
    /// resolve `{key}` placeholders in system prompts.
    pub state: HashMap<String, serde_json::Value>,

    /// Working directory for tool path resolution.
    pub working_directory: PathBuf,

    // --- Shared resources (same Arc across all agents in the tree) ---
    pub provider: Arc<dyn LlmProvider>,
    pub cost_tracker: CostTracker,
    pub on_event: Arc<dyn Fn(Event) + Send + Sync>,
    pub cancelled: Arc<AtomicBool>,

    /// Session store for transcript recording. Optional — when None,
    /// no transcript is recorded (useful for tests and ephemeral agents).
    pub session_store: Option<Arc<Mutex<SessionStore>>>,

    /// Command queue for receiving inter-agent notifications and user input.
    /// Optional — when None, no queue polling occurs.
    pub command_queue: Option<Arc<CommandQueue>>,

    /// Unique identifier for this agent invocation.
    pub agent_id: String,
}

impl InvocationContext {
    /// Create a child context for a sub-agent (clone with shared resources).
    pub fn child(&self, agent_name: &str) -> Self {
        let mut child = self.clone();
        child.agent_id = generate_agent_id(agent_name);
        child
    }

    /// Create a child context with a new input.
    pub fn with_input(&self, input: impl Into<String>) -> Self {
        let mut child = self.clone();
        child.input = input.into();
        child
    }
}
```

#### 9.4 Events

```rust
#[derive(Debug, Clone)]
pub enum Event {
    TurnStart { agent: String, turn: u32 },
    Text { agent: String, text: String },
    ToolStart { agent: String, tool: String, id: String },
    ToolEnd { agent: String, tool: String, id: String, result: String, is_error: bool },
    Usage { agent: String, model: String, usage: Usage },
    AgentStart { agent: String },
    AgentEnd { agent: String, turns: u32 },
    Error { agent: String, error: String },
}
```

**Example: Streaming text to stdout and logging tool calls**

```rust
let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| {
    match event {
        Event::Text { text, .. } => print!("{text}"),
        Event::ToolStart { agent, tool, .. } => eprintln!("[{agent}] Running tool: {tool}"),
        Event::ToolEnd { tool, is_error, result, .. } => {
            if is_error { eprintln!("[error] {tool}: {result}"); }
        }
        Event::AgentStart { agent } => eprintln!("[{agent}] Started"),
        Event::AgentEnd { agent, turns } => eprintln!("[{agent}] Completed in {turns} turns"),
        _ => {}
    }
});
```

#### 9.5 LLM Agent (internal, built via `AgentBuilder`)

```rust
/// An LLM-powered agent. Calls an LLM in a loop, executing tools until done.
/// Not public — users create these via `AgentBuilder::build()` which returns `Arc<dyn Agent>`.
struct LlmAgent {
    name: String,
    description: String,
    model: String,
    system_prompt: String,          // can contain {key} placeholders for state interpolation
    max_tokens: u32,
    max_turns: Option<u32>,
    max_budget: Option<f64>,
    output_schema: Option<OutputSchema>,  // enforce structured response (see output.md)
    max_schema_retries: u32,              // default: 3
    prompt_builder: Option<PromptBuilder>,
    tools: ToolRegistry,
}

impl Agent for LlmAgent {
    fn name(&self) -> &str { &self.name }
    fn description(&self) -> &str { &self.description }
    fn run(&self, ctx: InvocationContext)
        -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>>
    {
        Box::pin(async move { self.run_loop(ctx).await })
    }
}

impl LlmAgent {
    async fn run_loop(&self, ctx: InvocationContext) -> Result<AgentOutput> {
        let mut messages: Vec<Message> = Vec::new();
        let mut total_usage = Usage::default();
        let mut structured_output: Option<serde_json::Value> = None;
        let mut schema_retries: u32 = 0;
        let mut discovered_tools: HashSet<String> = HashSet::new();

        // 1. Interpolate system prompt
        let mut system_prompt = interpolate(&self.system_prompt, &ctx.state);

        // 1b. Append structured output instruction if output_schema is set
        if self.output_schema.is_some() {
            system_prompt.push_str("\n\n\
                IMPORTANT: You must provide your final response using the StructuredOutput tool \
                with the required structured format. After using any other tools needed to complete \
                the task, always call StructuredOutput with your final answer in the specified schema.");
        }

        // 2. Inject context message (memory, instruction files, env info)
        if let Some(ref pb) = self.prompt_builder {
            if let Some(context_msg) = pb.build_context_message() {
                messages.push(context_msg);
            }
        }

        // 3. Add user message
        messages.push(Message::User {
            content: vec![ContentBlock::Text { text: ctx.input.clone() }],
        });

        // 3b. Record user message in transcript
        if let Some(ref store) = ctx.session_store {
            store.lock().unwrap().record(TranscriptEntry {
                recorded_at: now_millis(),
                entry_type: EntryType::UserMessage,
                message: messages.last().unwrap().clone(),
                usage: None, model: None,
            }).ok();
        }

        // 4. Prepare structured output tool if output_schema is set
        let (tools, tool_choice) = if let Some(ref schema) = self.output_schema {
            let mut tools = self.tools.clone();
            tools.register(StructuredOutputTool::new(schema.clone()));
            let choice = if self.tools.is_empty() {
                Some(ToolChoice::Specific { name: STRUCTURED_OUTPUT_TOOL_NAME.into() })
            } else {
                None
            };
            (tools, choice)
        } else {
            (self.tools.clone(), None)
        };

        (ctx.on_event)(Event::AgentStart { agent: self.name.clone() });
        let mut turn: u32 = 0;

        loop {
            // === GUARDS ===
            if ctx.cancelled.load(Ordering::Relaxed) {
                return Err(AgenticError::Aborted);
            }
            turn += 1;
            if let Some(max) = self.max_turns {
                if turn > max { return Err(AgenticError::MaxTurnsExceeded(max)); }
            }
            if let Some(limit) = self.max_budget {
                if ctx.cost_tracker.total_cost_usd() >= limit {
                    return Err(AgenticError::BudgetExceeded {
                        spent: ctx.cost_tracker.total_cost_usd(), limit,
                    });
                }
            }

            (ctx.on_event)(Event::TurnStart { agent: self.name.clone(), turn });

            // === LLM CALL ===
            let response = ctx.provider.complete(CompletionRequest {
                model: self.model.clone(),
                system_prompt: system_prompt.clone(),
                messages: messages.clone(),
                tools: if tools.has_deferred_tools() {
                    tools.definitions_filtered(&discovered_tools)
                } else {
                    tools.definitions()
                },
                max_tokens: self.max_tokens,
                tool_choice: tool_choice.clone(),
            }).await?;

            // === RECORD USAGE ===
            total_usage.add(&response.usage);
            ctx.cost_tracker.record_usage(&response.model, &response.usage);
            (ctx.on_event)(Event::Usage {
                agent: self.name.clone(),
                model: response.model.clone(),
                usage: response.usage.clone(),
            });

            // === PARSE RESPONSE ===
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for block in &response.content {
                match block {
                    ContentBlock::Text { text: t } => {
                        text.push_str(t);
                        (ctx.on_event)(Event::Text { agent: self.name.clone(), text: t.clone() });
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(ToolCall {
                            id: id.clone(), name: name.clone(), input: input.clone(),
                        });
                    }
                    _ => {}
                }
            }
            messages.push(Message::Assistant { content: response.content.clone() });

            // Record assistant message in transcript
            if let Some(ref store) = ctx.session_store {
                store.lock().unwrap().record(TranscriptEntry {
                    recorded_at: now_millis(),
                    entry_type: EntryType::AssistantMessage,
                    message: Message::Assistant { content: response.content.clone() },
                    usage: Some(response.usage.clone()),
                    model: Some(response.model.clone()),
                }).ok();
            }

            // === STOP CHECK ===
            if response.stop_reason != StopReason::ToolUse || tool_calls.is_empty() {
                // Structured output retry enforcement
                if self.output_schema.is_some() && structured_output.is_none() {
                    schema_retries += 1;
                    if schema_retries > self.max_schema_retries {
                        return Err(AgenticError::SchemaRetryExhausted {
                            retries: self.max_schema_retries,
                        });
                    }
                    messages.push(Message::User {
                        content: vec![ContentBlock::Text {
                            text: "You MUST call the StructuredOutput tool to complete \
                                   this request. Call this tool now with the required schema."
                                .to_string(),
                        }],
                    });
                    continue;
                }

                (ctx.on_event)(Event::AgentEnd { agent: self.name.clone(), turns: turn });
                return Ok(AgentOutput { content: text, usage: total_usage, structured_output });
            }

            // === EXECUTE TOOLS ===
            let tools_arc = Arc::new(tools.clone());
            let tool_ctx = ToolContext {
                working_directory: ctx.working_directory.clone(),
                tool_registry: Some(tools_arc),
            };
            let tool_results = execute_tool_calls(&tool_calls, &tools, &tool_ctx).await;

            // Extract discovered tool names from tool_search results
            for call in &tool_calls {
                if call.name == "tool_search" {
                    for block in &tool_results {
                        if let ContentBlock::ToolResult { tool_use_id, content, is_error: false, .. } = block {
                            if *tool_use_id == call.id {
                                extract_discovered_tool_names(content, &mut discovered_tools);
                            }
                        }
                    }
                }
            }

            // Extract structured output from StructuredOutput tool results
            for call in &tool_calls {
                if call.name == STRUCTURED_OUTPUT_TOOL_NAME {
                    structured_output = Some(call.input.clone());
                }
            }

            messages.push(Message::User { content: tool_results });

            // Record tool results in transcript
            if let Some(ref store) = ctx.session_store {
                store.lock().unwrap().record(TranscriptEntry {
                    recorded_at: now_millis(),
                    entry_type: EntryType::ToolResult,
                    message: messages.last().unwrap().clone(),
                    usage: None, model: None,
                }).ok();
            }

            // === DRAIN COMMAND QUEUE ===
            if let Some(ref queue) = ctx.command_queue {
                while let Some(cmd) = queue.dequeue(Some(&ctx.agent_id)) {
                    match cmd.priority {
                        QueuePriority::Now | QueuePriority::Next => {
                            messages.push(Message::User {
                                content: vec![ContentBlock::Text { text: cmd.content }],
                            });
                        }
                        QueuePriority::Later => {
                            queue.enqueue(cmd);
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Replace {key} placeholders in a template with values from state.
fn interpolate(template: &str, state: &HashMap<String, serde_json::Value>) -> String {
    let mut result = template.to_string();
    for (key, value) in state {
        let replacement = match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        result = result.replace(&format!("{{{key}}}"), &replacement);
    }
    result
}

/// Extract tool names from tool_search result content. Parses `## tool_name`
/// headers emitted by ToolSearchTool and inserts them into the discovered set.
fn extract_discovered_tool_names(content: &str, discovered: &mut HashSet<String>) {
    for line in content.lines() {
        if let Some(name) = line.strip_prefix("## ") {
            let name = name.trim();
            if !name.is_empty() {
                discovered.insert(name.to_string());
            }
        }
    }
}
```

#### 9.6 AgentBuilder

```rust
pub struct AgentBuilder { /* ... */ }

impl AgentBuilder {
    pub fn new() -> Self;
    pub fn name(self, name: impl Into<String>) -> Self;
    pub fn description(self, desc: impl Into<String>) -> Self;
    pub fn model(self, model: impl Into<String>) -> Self;
    pub fn system_prompt(self, prompt: impl Into<String>) -> Self;
    pub fn max_tokens(self, max: u32) -> Self;
    pub fn max_turns(self, max: u32) -> Self;
    pub fn max_budget(self, budget: f64) -> Self;
    pub fn tool(self, tool: impl Tool + 'static) -> Self;
    pub fn output_schema(self, schema: serde_json::Value) -> Self;
    pub fn prompt_builder(self, pb: PromptBuilder) -> Self;
    pub fn sub_agent(self, agent: Arc<dyn Agent>) -> Self;
    pub fn build(self) -> Result<Arc<dyn Agent>>;
}
```

#### 9.7 Simple Usage Example

```rust
let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are a helpful coding assistant.")
    .tool(ReadFileTool)
    .tool(GlobTool)
    .build()?;

let ctx = InvocationContext {
    input: "What does this function do?".into(),
    state: HashMap::new(),
    working_directory: std::env::current_dir().unwrap(),
    provider: Arc::new(my_provider),
    cost_tracker: CostTracker::new(),
    on_event: Arc::new(|event| println!("{:?}", event)),
    cancelled: Arc::new(AtomicBool::new(false)),
    session_store: Some(Arc::new(Mutex::new(SessionStore::new(base_dir, &session_id)))),
    command_queue: Some(Arc::new(CommandQueue::new())),
    agent_id: generate_agent_id("assistant"),
};

let output = agent.run(ctx).await?;
println!("{}", output.content);
```

### Test Helpers (in `testutil.rs`)

```rust
/// Build a minimal InvocationContext for testing.
pub fn test_context(provider: Arc<dyn LlmProvider>) -> InvocationContext {
    InvocationContext {
        input: String::new(),
        state: HashMap::new(),
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        provider,
        cost_tracker: CostTracker::new(),
        on_event: Arc::new(|_| {}),
        cancelled: Arc::new(AtomicBool::new(false)),
        session_store: None,
        command_queue: None,
        agent_id: "test".into(),
    }
}

/// Build a test context that collects events into a Vec for assertions.
pub fn test_context_with_events(provider: Arc<dyn LlmProvider>) -> (InvocationContext, Arc<Mutex<Vec<Event>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let ctx = InvocationContext {
        input: String::new(),
        state: HashMap::new(),
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        provider,
        cost_tracker: CostTracker::new(),
        on_event: Arc::new(move |e| events_clone.lock().unwrap().push(e)),
        cancelled: Arc::new(AtomicBool::new(false)),
        session_store: None,
        command_queue: None,
        agent_id: "test".into(),
    };
    (ctx, events)
}
```

### EventCollector

```rust
/// Collects events emitted during agent execution for test assertions.
pub struct EventCollector {
    events: Arc<Mutex<Vec<Event>>>,
}

impl EventCollector {
    pub fn new() -> Self;
    pub fn callback(&self) -> Arc<dyn Fn(Event) + Send + Sync>;
    pub fn all(&self) -> Vec<Event>;
    pub fn texts(&self) -> Vec<String>;
    pub fn tool_starts(&self) -> Vec<String>;
    pub fn tool_ends(&self) -> Vec<(String, bool)>;
    pub fn agent_starts(&self) -> Vec<String>;
    pub fn agent_ends(&self) -> Vec<(String, u32)>;
    pub fn errors(&self) -> Vec<String>;
    pub fn count(&self) -> usize;
}
```

### TestHarness

```rust
/// High-level test harness combining MockProvider, EventCollector, and context
/// construction. Reduces agent test boilerplate to a single setup call.
pub struct TestHarness {
    provider: Arc<MockProvider>,
    events: EventCollector,
    cost_tracker: CostTracker,
    state: HashMap<String, serde_json::Value>,
    working_directory: PathBuf,
    cancelled: Arc<AtomicBool>,
}

impl TestHarness {
    pub fn new(provider: MockProvider) -> Self;
    pub fn with_state(mut self, key: &str, value: serde_json::Value) -> Self;
    pub fn with_working_dir(mut self, path: PathBuf) -> Self;
    pub fn build_context(&self, input: &str) -> InvocationContext;
    pub async fn run_agent(&self, agent: &dyn Agent, input: &str) -> Result<AgentOutput>;
    pub fn events(&self) -> &EventCollector;
    pub fn provider(&self) -> &MockProvider;
    pub fn cancel(&self);
}
```

## Work Items

1. **`agent.rs`** — Spec Sections 6, 9.1–9.7
   - `CommandQueue`, `QueuePriority`, `QueuedCommand`, `CommandSource`
   - `Agent` trait: `name()`, `description()`, `run()`
   - `AgentOutput`: content, usage, structured_output
   - `InvocationContext`: all fields + `child()`, `with_input()`
   - `Event` enum (all variants)
   - `LlmAgent` (internal):
     - `run_loop()` — core agent loop with `discovered_tools` tracking
     - Guards: cancellation, turn limit, budget limit
     - `interpolate()` — `{key}` placeholder replacement
     - `extract_discovered_tool_names()` — parses `## name` headers from tool_search results
     - Uses `definitions_filtered()` when deferred tools are present
     - Populates `ToolContext.tool_registry` for tool_search access
   - `AgentBuilder` — fluent builder returning `Arc<dyn Agent>`

2. **`testutil.rs`** (additions) — test_context, test_context_with_events, EventCollector, TestHarness
   - `test_context()`, `test_context_with_events()` — minimal `InvocationContext` builders
   - `EventCollector` — typed query methods (`texts`, `tool_starts`, `tool_ends`, `agent_starts`, `agent_ends`, `errors`, `count`)
   - `TestHarness` — combines MockProvider + EventCollector + context construction; builder pattern with `with_state()` and `with_working_dir()`

## Tests

### `agent.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `agent_loop_text_response` | Direct | Single LLM call returns text; output.content matches; structured_output=None; request_count=1 |
| `agent_loop_with_tool_execution` | Direct | Tool-use response triggers MockTool execution, then text response ends loop; request_count=2 |
| `agent_guards_table` | Table-driven (3 cases) | max_turns→MaxTurnsExceeded; max_budget→BudgetExceeded; cancel→Aborted |
| `state_interpolation_in_system_prompt` | Direct | `{topic}` placeholder replaced with state value "rust" |
| `events_emitted_during_agent_run` | Direct | EventCollector captures AgentStart, ToolStart, Text, AgentEnd in correct order |
| `agent_records_transcript` | Direct | Agent with `session_store: Some(...)` records user, assistant, and tool result entries |
| `agent_drains_command_queue` | Direct | Agent with `command_queue: Some(...)` picks up `Next` commands between turns |
| `agent_requeues_later_commands` | Direct | `Later` priority commands are re-enqueued, not processed mid-turn |
| `agent_sends_filtered_definitions_when_deferred` | Direct | Deferred tool has empty schema in first request |
| `agent_discovers_tools_via_search` | Direct | After `tool_search` returns a name, next request includes full definition |
| `extract_discovered_tool_names_parses_headers` | Direct | `"## read_file\n..."` adds `"read_file"` to set |
| `agent_no_filtering_without_deferred` | Direct | No deferred tools → `definitions()` used unchanged |

## Done Criteria

- `cargo test` passes all tests above
- An agent can: receive a prompt → call an LLM → execute tools → loop until done → return output
- Guards (turn limit, budget, cancellation) stop the agent loop correctly
- State interpolation replaces `{key}` placeholders in system prompts
- Events are emitted for all agent lifecycle stages
- `EventCollector` captures and queries all event types
- `TestHarness` reduces test boilerplate to a single `new()` + `run_agent()` call
