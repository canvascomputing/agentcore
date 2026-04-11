use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::cost::CostTracker;
use crate::error::{AgenticError, Result};
use crate::message::{ContentBlock, Message, StopReason, Usage};
use crate::prompt::PromptBuilder;
use crate::provider::{CompletionRequest, LlmProvider, ToolChoice};
use crate::tool::{
    Tool, ToolCall, ToolContext, ToolRegistry, ToolResult, execute_tool_calls,
};

// ---------------------------------------------------------------------------
// Command Queue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueuePriority {
    Now = 0,
    Next = 1,
    Later = 2,
}

#[derive(Debug, Clone)]
pub struct QueuedCommand {
    pub content: String,
    pub priority: QueuePriority,
    pub source: CommandSource,
    pub agent_id: Option<String>,
}

#[derive(Debug, Clone)]
pub enum CommandSource {
    UserInput,
    TaskNotification { task_id: String },
    System,
}

/// Thread-safe priority queue for commands.
pub struct CommandQueue {
    inner: Arc<Mutex<VecDeque<QueuedCommand>>>,
    notify: Arc<tokio::sync::Notify>,
}

impl CommandQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub fn enqueue(&self, command: QueuedCommand) {
        self.inner.lock().unwrap().push_back(command);
        self.notify.notify_one();
    }

    pub fn enqueue_notification(&self, task_id: &str, summary: &str) {
        self.enqueue(QueuedCommand {
            content: format!("Task {task_id} completed: {summary}"),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification {
                task_id: task_id.to_string(),
            },
            agent_id: None,
        });
    }

    pub fn dequeue(&self, agent_id: Option<&str>) -> Option<QueuedCommand> {
        let mut queue = self.inner.lock().unwrap();
        // Find highest priority (lowest ordinal) matching command
        let mut best_idx = None;
        let mut best_priority = None;

        for (i, cmd) in queue.iter().enumerate() {
            let matches = match (&cmd.agent_id, agent_id) {
                (None, _) => true,
                (Some(cmd_id), Some(filter_id)) => cmd_id == filter_id,
                (Some(_), None) => false,
            };
            if matches {
                if best_priority.is_none() || cmd.priority < *best_priority.as_ref().unwrap() {
                    best_idx = Some(i);
                    best_priority = Some(cmd.priority.clone());
                    if cmd.priority == QueuePriority::Now {
                        break; // Can't do better
                    }
                }
            }
        }

        best_idx.and_then(|i| queue.remove(i))
    }

    pub async fn wait_and_dequeue(&self, agent_id: Option<&str>) -> QueuedCommand {
        loop {
            if let Some(cmd) = self.dequeue(agent_id) {
                return cmd;
            }
            self.notify.notified().await;
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Agent trait and output
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub content: String,
    pub usage: Usage,
    pub structured_output: Option<Value>,
}

impl AgentOutput {
    pub fn empty(usage: Usage) -> Self {
        Self {
            content: String::new(),
            usage,
            structured_output: None,
        }
    }
}

pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>>;
}

// ---------------------------------------------------------------------------
// Session store placeholder
// ---------------------------------------------------------------------------

/// Placeholder for SessionStore (implemented in persistence increment).
pub struct SessionStore {
    _private: (),
}

impl SessionStore {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn record(&mut self, _entry: TranscriptEntry) -> Result<()> {
        Ok(())
    }
}

pub struct TranscriptEntry {
    pub recorded_at: u64,
    pub entry_type: EntryType,
    pub message: Message,
    pub usage: Option<Usage>,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub enum EntryType {
    UserMessage,
    AssistantMessage,
    ToolResult,
}

// ---------------------------------------------------------------------------
// InvocationContext
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct InvocationContext {
    pub input: String,
    pub state: HashMap<String, Value>,
    pub working_directory: PathBuf,
    pub provider: Arc<dyn LlmProvider>,
    pub cost_tracker: CostTracker,
    pub on_event: Arc<dyn Fn(Event) + Send + Sync>,
    pub cancelled: Arc<AtomicBool>,
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub command_queue: Option<Arc<CommandQueue>>,
    pub agent_id: String,
}

impl InvocationContext {
    pub fn child(&self, agent_name: &str) -> Self {
        let mut child = self.clone();
        child.agent_id = generate_agent_id(agent_name);
        child
    }

    pub fn with_input(&self, input: impl Into<String>) -> Self {
        let mut child = self.clone();
        child.input = input.into();
        child
    }
}

// ---------------------------------------------------------------------------
// Structured Output
// ---------------------------------------------------------------------------

/// A validated JSON Schema for structured output.
#[derive(Debug, Clone)]
pub struct OutputSchema {
    pub schema: Value,
}

impl OutputSchema {
    pub fn new(schema: Value) -> Result<Self> {
        if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
            return Err(AgenticError::SchemaValidation {
                path: String::new(),
                message: "output schema must have \"type\": \"object\"".into(),
            });
        }
        if schema.get("properties").is_none() {
            return Err(AgenticError::SchemaValidation {
                path: String::new(),
                message: "output schema must have \"properties\"".into(),
            });
        }
        Ok(Self { schema })
    }
}

const STRUCTURED_OUTPUT_TOOL_NAME: &str = "StructuredOutput";

struct StructuredOutputTool {
    schema: OutputSchema,
}

impl StructuredOutputTool {
    fn new(schema: OutputSchema) -> Self {
        Self { schema }
    }
}

impl Tool for StructuredOutputTool {
    fn name(&self) -> &str {
        STRUCTURED_OUTPUT_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Return your final response using the required output schema. \
         Call this tool exactly once at the end to provide the structured result."
    }

    fn input_schema(&self) -> Value {
        self.schema.schema.clone()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn call<'a>(
        &'a self,
        input: Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            validate_value(&input, &self.schema.schema)?;
            Ok(ToolResult {
                content: "Structured output accepted.".into(),
                is_error: false,
            })
        })
    }
}

/// Validate a JSON value against a JSON Schema object.
pub fn validate_value(value: &Value, schema: &Value) -> Result<()> {
    let schema_type = schema.get("type").and_then(|t| t.as_str()).unwrap_or("object");

    match schema_type {
        "object" => {
            let obj = value.as_object().ok_or_else(|| AgenticError::SchemaValidation {
                path: String::new(),
                message: "expected object".into(),
            })?;
            if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                for key in required {
                    if let Some(key_str) = key.as_str() {
                        if !obj.contains_key(key_str) {
                            return Err(AgenticError::SchemaValidation {
                                path: key_str.into(),
                                message: "missing required field".into(),
                            });
                        }
                    }
                }
            }
            if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
                for (key, prop_schema) in properties {
                    if let Some(prop_value) = obj.get(key) {
                        validate_value(prop_value, prop_schema).map_err(|e| match e {
                            AgenticError::SchemaValidation { path, message } => {
                                AgenticError::SchemaValidation {
                                    path: if path.is_empty() {
                                        key.clone()
                                    } else {
                                        format!("{key}.{path}")
                                    },
                                    message,
                                }
                            }
                            other => other,
                        })?;
                    }
                }
            }
            Ok(())
        }
        "array" => {
            let arr = value.as_array().ok_or_else(|| AgenticError::SchemaValidation {
                path: String::new(),
                message: "expected array".into(),
            })?;
            if let Some(items_schema) = schema.get("items") {
                for (i, item) in arr.iter().enumerate() {
                    validate_value(item, items_schema).map_err(|e| match e {
                        AgenticError::SchemaValidation { path, message } => {
                            AgenticError::SchemaValidation {
                                path: format!("[{i}].{path}"),
                                message,
                            }
                        }
                        other => other,
                    })?;
                }
            }
            Ok(())
        }
        "string" => {
            if value.is_string() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected string".into(),
                })
            }
        }
        "number" => {
            if value.is_number() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected number".into(),
                })
            }
        }
        "integer" => {
            if value.is_i64() || value.is_u64() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected integer".into(),
                })
            }
        }
        "boolean" => {
            if value.is_boolean() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected boolean".into(),
                })
            }
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// LlmAgent (internal)
// ---------------------------------------------------------------------------

struct LlmAgent {
    name: String,
    description: String,
    model: String,
    system_prompt: String,
    max_tokens: u32,
    max_turns: Option<u32>,
    max_budget: Option<f64>,
    output_schema: Option<OutputSchema>,
    max_schema_retries: u32,
    prompt_builder: Option<PromptBuilder>,
    tools: ToolRegistry,
    #[allow(dead_code)]
    sub_agents: Vec<Arc<dyn Agent>>,
}

impl Agent for LlmAgent {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>> {
        Box::pin(async move { self.run_loop(ctx).await })
    }
}

impl LlmAgent {
    async fn run_loop(&self, ctx: InvocationContext) -> Result<AgentOutput> {
        let mut messages: Vec<Message> = Vec::new();
        let mut total_usage = Usage::default();
        let mut structured_output: Option<Value> = None;
        let mut schema_retries: u32 = 0;
        let mut discovered_tools: HashSet<String> = HashSet::new();

        // 1. Interpolate system prompt
        let mut system_prompt = interpolate(&self.system_prompt, &ctx.state);

        // 1b. Append structured output instruction if output_schema is set
        if self.output_schema.is_some() {
            system_prompt.push_str(
                "\n\nIMPORTANT: You must provide your final response using the StructuredOutput tool \
                 with the required structured format. After using any other tools needed to complete \
                 the task, always call StructuredOutput with your final answer in the specified schema.",
            );
        }

        // 2. Inject context message
        if let Some(ref pb) = self.prompt_builder {
            if let Some(context_msg) = pb.build_context_message() {
                messages.push(context_msg);
            }
        }

        // 3. Add user message
        messages.push(Message::User {
            content: vec![ContentBlock::Text {
                text: ctx.input.clone(),
            }],
        });

        // 3b. Record user message in transcript
        if let Some(ref store) = ctx.session_store {
            store
                .lock()
                .unwrap()
                .record(TranscriptEntry {
                    recorded_at: now_millis(),
                    entry_type: EntryType::UserMessage,
                    message: messages.last().unwrap().clone(),
                    usage: None,
                    model: None,
                })
                .ok();
        }

        // 4. Prepare tools (with structured output tool if needed)
        let (tools, tool_choice) = if let Some(ref schema) = self.output_schema {
            let mut tools = self.tools.clone();
            tools.register(StructuredOutputTool::new(schema.clone()));
            let choice = if self.tools.is_empty() {
                Some(ToolChoice::Specific {
                    name: STRUCTURED_OUTPUT_TOOL_NAME.into(),
                })
            } else {
                None
            };
            (tools, choice)
        } else {
            (self.tools.clone(), None)
        };

        (ctx.on_event)(Event::AgentStart {
            agent: self.name.clone(),
        });
        let mut turn: u32 = 0;

        loop {
            // === GUARDS ===
            if ctx.cancelled.load(Ordering::Relaxed) {
                return Err(AgenticError::Aborted);
            }
            turn += 1;
            if let Some(max) = self.max_turns {
                if turn > max {
                    return Err(AgenticError::MaxTurnsExceeded(max));
                }
            }
            if let Some(limit) = self.max_budget {
                if ctx.cost_tracker.total_cost_usd() >= limit {
                    return Err(AgenticError::BudgetExceeded {
                        spent: ctx.cost_tracker.total_cost_usd(),
                        limit,
                    });
                }
            }

            (ctx.on_event)(Event::TurnStart {
                agent: self.name.clone(),
                turn,
            });

            // === LLM CALL ===
            let response = ctx
                .provider
                .complete(CompletionRequest {
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
                })
                .await?;

            // === RECORD USAGE ===
            total_usage.add(&response.usage);
            ctx.cost_tracker
                .record_usage(&response.model, &response.usage);
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
                        (ctx.on_event)(Event::Text {
                            agent: self.name.clone(),
                            text: t.clone(),
                        });
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                    }
                    _ => {}
                }
            }
            messages.push(Message::Assistant {
                content: response.content.clone(),
            });

            // Record assistant message in transcript
            if let Some(ref store) = ctx.session_store {
                store
                    .lock()
                    .unwrap()
                    .record(TranscriptEntry {
                        recorded_at: now_millis(),
                        entry_type: EntryType::AssistantMessage,
                        message: Message::Assistant {
                            content: response.content.clone(),
                        },
                        usage: Some(response.usage.clone()),
                        model: Some(response.model.clone()),
                    })
                    .ok();
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

                (ctx.on_event)(Event::AgentEnd {
                    agent: self.name.clone(),
                    turns: turn,
                });
                return Ok(AgentOutput {
                    content: text,
                    usage: total_usage,
                    structured_output,
                });
            }

            // === EXECUTE TOOLS ===
            // Emit tool start events
            for call in &tool_calls {
                (ctx.on_event)(Event::ToolStart {
                    agent: self.name.clone(),
                    tool: call.name.clone(),
                    id: call.id.clone(),
                });
                ctx.cost_tracker.record_tool_calls(1);
            }

            let mut tool_ctx = ToolContext::new(ctx.working_directory.clone())
                .with_registry(Arc::new(tools.clone()));
            tool_ctx.set_extension(ctx.clone());
            let tool_results = execute_tool_calls(&tool_calls, &tools, &tool_ctx).await;

            // Emit tool end events
            for block in &tool_results {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } = block
                {
                    let tool_name = tool_calls
                        .iter()
                        .find(|c| c.id == *tool_use_id)
                        .map(|c| c.name.clone())
                        .unwrap_or_default();
                    (ctx.on_event)(Event::ToolEnd {
                        agent: self.name.clone(),
                        tool: tool_name,
                        id: tool_use_id.clone(),
                        result: content.clone(),
                        is_error: *is_error,
                    });
                }
            }

            // Extract discovered tool names from tool_search results
            for call in &tool_calls {
                if call.name == "tool_search" {
                    for block in &tool_results {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error: false,
                        } = block
                        {
                            if *tool_use_id == call.id {
                                extract_discovered_tool_names(content, &mut discovered_tools);
                            }
                        }
                    }
                }
            }

            // Extract structured output from StructuredOutput tool
            for call in &tool_calls {
                if call.name == STRUCTURED_OUTPUT_TOOL_NAME {
                    structured_output = Some(call.input.clone());
                }
            }

            messages.push(Message::User {
                content: tool_results,
            });

            // Record tool results in transcript
            if let Some(ref store) = ctx.session_store {
                store
                    .lock()
                    .unwrap()
                    .record(TranscriptEntry {
                        recorded_at: now_millis(),
                        entry_type: EntryType::ToolResult,
                        message: messages.last().unwrap().clone(),
                        usage: None,
                        model: None,
                    })
                    .ok();
            }

            // === DRAIN COMMAND QUEUE ===
            if let Some(ref queue) = ctx.command_queue {
                while let Some(cmd) = queue.dequeue(Some(&ctx.agent_id)) {
                    match cmd.priority {
                        QueuePriority::Now | QueuePriority::Next => {
                            messages.push(Message::User {
                                content: vec![ContentBlock::Text {
                                    text: cmd.content,
                                }],
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

// ---------------------------------------------------------------------------
// AgentBuilder
// ---------------------------------------------------------------------------

pub struct AgentBuilder {
    name: Option<String>,
    description: String,
    model: Option<String>,
    system_prompt: String,
    max_tokens: u32,
    max_turns: Option<u32>,
    max_budget: Option<f64>,
    output_schema: Option<OutputSchema>,
    max_schema_retries: u32,
    prompt_builder: Option<PromptBuilder>,
    tools: ToolRegistry,
    sub_agents: Vec<Arc<dyn Agent>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            name: None,
            description: String::new(),
            model: None,
            system_prompt: String::new(),
            max_tokens: 4096,
            max_turns: None,
            max_budget: None,
            output_schema: None,
            max_schema_retries: 3,
            prompt_builder: None,
            tools: ToolRegistry::new(),
            sub_agents: Vec::new(),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    pub fn max_turns(mut self, max: u32) -> Self {
        self.max_turns = Some(max);
        self
    }

    pub fn max_budget(mut self, budget: f64) -> Self {
        self.max_budget = Some(budget);
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(OutputSchema::new(schema).expect("invalid output schema"));
        self
    }

    pub fn prompt_builder(mut self, pb: PromptBuilder) -> Self {
        self.prompt_builder = Some(pb);
        self
    }

    pub fn sub_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.sub_agents.push(agent);
        self
    }

    pub fn build(self) -> Result<Arc<dyn Agent>> {
        let name = self.name.ok_or_else(|| AgenticError::Other("AgentBuilder requires a name".into()))?;
        let model = self.model.ok_or_else(|| AgenticError::Other("AgentBuilder requires a model".into()))?;

        Ok(Arc::new(LlmAgent {
            name,
            description: self.description,
            model,
            system_prompt: self.system_prompt,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            max_budget: self.max_budget,
            output_schema: self.output_schema,
            max_schema_retries: self.max_schema_retries,
            prompt_builder: self.prompt_builder,
            tools: self.tools,
            sub_agents: self.sub_agents,
        }))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Replace {key} placeholders in a template with values from state.
fn interpolate(template: &str, state: &HashMap<String, Value>) -> String {
    let mut result = template.to_string();
    for (key, value) in state {
        let replacement = match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        result = result.replace(&format!("{{{key}}}"), &replacement);
    }
    result
}

/// Extract tool names from tool_search result content.
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

pub fn generate_agent_id(name: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{name}_{nanos}")
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::*;
    use std::sync::Arc;

    fn build_simple_agent() -> Arc<dyn Agent> {
        AgentBuilder::new()
            .name("test-agent")
            .model("mock-model")
            .system_prompt("You are a test assistant.")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn agent_loop_text_response() {
        let harness = TestHarness::new(MockProvider::text("Hello, world!"));
        let agent = build_simple_agent();

        let output = harness.run_agent(agent.as_ref(), "Hi").await.unwrap();
        assert_eq!(output.content, "Hello, world!");
        assert!(output.structured_output.is_none());
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn agent_loop_with_tool_execution() {
        let provider = MockProvider::tool_then_text(
            "echo_tool",
            serde_json::json!({"text": "ping"}),
            "Done!",
        );
        let agent = AgentBuilder::new()
            .name("test-agent")
            .model("mock-model")
            .system_prompt("You are helpful.")
            .tool(MockTool::new("echo_tool", false, "pong"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "Echo test").await.unwrap();
        assert_eq!(output.content, "Done!");
        assert_eq!(harness.provider().request_count(), 2);
    }

    #[tokio::test]
    async fn agent_guards_table() {
        // max_turns exceeded
        {
            let provider = MockProvider::new(vec![
                tool_response("t", "c1", serde_json::json!({})),
                tool_response("t", "c2", serde_json::json!({})),
                tool_response("t", "c3", serde_json::json!({})),
            ]);
            let agent = AgentBuilder::new()
                .name("test")
                .model("mock")
                .system_prompt("")
                .max_turns(2)
                .tool(MockTool::new("t", false, "ok"))
                .build()
                .unwrap();
            let harness = TestHarness::new(provider);
            let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
            assert!(matches!(err, AgenticError::MaxTurnsExceeded(2)));
        }

        // budget exceeded
        {
            let provider = MockProvider::new(vec![
                tool_response("t", "c1", serde_json::json!({})),
                text_response("done"),
            ]);
            let agent = AgentBuilder::new()
                .name("test")
                .model("mock")
                .system_prompt("")
                .max_budget(0.0) // any usage exceeds $0
                .tool(MockTool::new("t", false, "ok"))
                .build()
                .unwrap();
            let harness = TestHarness::new(provider);
            // First turn succeeds (budget checked before LLM call), second turn budget check fires
            let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
            assert!(matches!(err, AgenticError::BudgetExceeded { .. }));
        }

        // cancellation
        {
            let provider = MockProvider::new(vec![
                tool_response("t", "c1", serde_json::json!({})),
                text_response("done"),
            ]);
            let agent = AgentBuilder::new()
                .name("test")
                .model("mock")
                .system_prompt("")
                .tool(MockTool::new("t", false, "ok"))
                .build()
                .unwrap();
            let harness = TestHarness::new(provider);
            harness.cancel();
            let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
            assert!(matches!(err, AgenticError::Aborted));
        }
    }

    #[tokio::test]
    async fn state_interpolation_in_system_prompt() {
        let provider = MockProvider::text("Answer about rust");
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("You are an expert on {topic}.")
            .build()
            .unwrap();

        let harness = TestHarness::new(provider).with_state("topic", serde_json::json!("rust"));
        harness.run_agent(agent.as_ref(), "Tell me").await.unwrap();

        let prompts = harness.provider().system_prompts();
        assert!(prompts[0].contains("expert on rust"));
    }

    #[tokio::test]
    async fn events_emitted_during_agent_run() {
        let provider = MockProvider::tool_then_text(
            "read",
            serde_json::json!({}),
            "Done",
        );
        let agent = AgentBuilder::new()
            .name("assistant")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("read", true, "file contents"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "read it").await.unwrap();

        let events = harness.events();
        assert_eq!(events.agent_starts(), vec!["assistant"]);
        assert!(!events.tool_starts().is_empty());
        assert!(events.texts().contains(&"Done".to_string()));
        assert_eq!(events.agent_ends().len(), 1);
    }

    #[tokio::test]
    async fn agent_drains_command_queue() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build()
            .unwrap();

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "extra instruction".into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_id: Some("test".into()),
        });

        let harness = TestHarness::new(provider);
        let mut ctx = harness.build_context("start");
        ctx.command_queue = Some(queue);
        ctx.agent_id = "test".into();

        let output = agent.run(ctx).await.unwrap();
        assert_eq!(output.content, "final");
        // The second request should contain the extra instruction as a user message
        let requests = harness.provider().requests.lock().unwrap();
        let second_req = &requests[1];
        let has_extra = second_req.messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains("extra instruction"),
                _ => false,
            }),
            _ => false,
        });
        assert!(has_extra, "Extra instruction should be in second request");
    }

    #[tokio::test]
    async fn agent_requeues_later_commands() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build()
            .unwrap();

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "later task".into(),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification { task_id: "42".into() },
            agent_id: Some("test".into()),
        });

        let harness = TestHarness::new(provider);
        let mut ctx = harness.build_context("start");
        ctx.command_queue = Some(queue.clone());
        ctx.agent_id = "test".into();

        agent.run(ctx).await.unwrap();

        // Later command should still be in the queue
        let cmd = queue.dequeue(Some("test"));
        assert!(cmd.is_some());
        assert_eq!(cmd.unwrap().content, "later task");
    }

    #[tokio::test]
    async fn agent_sends_filtered_definitions_when_deferred() {
        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("always", true, "ok"))
            .tool(DeferredMockTool::new("deferred"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let deferred_def = req.tools.iter().find(|t| t.name == "deferred").unwrap();
        assert!(deferred_def.description.is_empty(), "Deferred tool should have empty description");
    }

    #[tokio::test]
    async fn agent_no_filtering_without_deferred() {
        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("read", true, "ok"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let def = req.tools.iter().find(|t| t.name == "read").unwrap();
        assert!(!def.description.is_empty());
    }

    #[test]
    fn extract_discovered_tool_names_parses_headers() {
        let mut discovered = HashSet::new();
        let content = "## read_file\nReads a file.\n\n## grep\nSearches content.";
        extract_discovered_tool_names(content, &mut discovered);
        assert!(discovered.contains("read_file"));
        assert!(discovered.contains("grep"));
        assert_eq!(discovered.len(), 2);
    }

    // --- Structured output tests ---

    #[tokio::test]
    async fn structured_output_extracted() {
        let schema_input = serde_json::json!({"category": "billing", "priority": "high"});
        let provider = MockProvider::new(vec![
            tool_response(STRUCTURED_OUTPUT_TOOL_NAME, "so1", schema_input.clone()),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("classifier")
            .model("mock")
            .system_prompt("Classify.")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "priority": {"type": "string"}
                },
                "required": ["category", "priority"]
            }))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "ticket").await.unwrap();
        assert!(output.structured_output.is_some());
        let so = output.structured_output.unwrap();
        assert_eq!(so["category"], "billing");
        assert_eq!(so["priority"], "high");
    }

    #[tokio::test]
    async fn structured_output_retry_on_noncompliance() {
        // First: text-only (no tool call) → retry
        // Second: text-only → retry
        // Third: calls StructuredOutput → success
        let provider = MockProvider::new(vec![
            text_response("thinking..."),
            text_response("still thinking..."),
            tool_response(
                STRUCTURED_OUTPUT_TOOL_NAME,
                "so1",
                serde_json::json!({"answer": "yes"}),
            ),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "question").await.unwrap();
        assert!(output.structured_output.is_some());
        // 1st text + 2nd text + tool call + done = at least 3 requests
        assert!(harness.provider().request_count() >= 3);
    }

    #[tokio::test]
    async fn structured_output_retry_exhausted() {
        // All text responses, never calls StructuredOutput
        let provider = MockProvider::new(vec![
            text_response("nope"),
            text_response("still nope"),
            text_response("nope again"),
            text_response("last nope"), // 4th text = exceeds max_schema_retries (3)
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
        assert!(matches!(err, AgenticError::SchemaRetryExhausted { retries: 3 }));
    }

    #[test]
    fn validate_value_table() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "age"]
        });

        // Valid complete
        assert!(validate_value(
            &serde_json::json!({"name": "Alice", "age": 30, "score": 9.5, "active": true, "tags": ["a", "b"]}),
            &schema
        ).is_ok());

        // Valid minimal (only required)
        assert!(validate_value(
            &serde_json::json!({"name": "Bob", "age": 25}),
            &schema
        ).is_ok());

        // Missing required field
        assert!(validate_value(
            &serde_json::json!({"name": "Carol"}),
            &schema
        ).is_err());

        // Wrong type for string field
        assert!(validate_value(
            &serde_json::json!({"name": 123, "age": 25}),
            &schema
        ).is_err());

        // Wrong type for integer field
        assert!(validate_value(
            &serde_json::json!({"name": "Dave", "age": "old"}),
            &schema
        ).is_err());

        // Wrong type for boolean field
        assert!(validate_value(
            &serde_json::json!({"name": "Eve", "age": 20, "active": "yes"}),
            &schema
        ).is_err());

        // Wrong array item type
        assert!(validate_value(
            &serde_json::json!({"name": "Frank", "age": 40, "tags": [1, 2]}),
            &schema
        ).is_err());

        // Non-object input
        assert!(validate_value(
            &serde_json::json!("not an object"),
            &schema
        ).is_err());
    }
}
