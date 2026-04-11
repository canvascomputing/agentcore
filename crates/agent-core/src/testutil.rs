use std::collections::HashMap;
use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::agent::{Agent, AgentOutput, Event, InvocationContext};
use crate::cost::CostTracker;
use crate::error::{AgenticError, Result};
use crate::message::{ContentBlock, ModelResponse, StopReason, Usage};
use crate::provider::{CompletionRequest, LlmProvider};
use crate::tool::{Tool, ToolContext, ToolRegistry, ToolResult};

/// A mock LLM provider that returns pre-configured responses in order.
pub struct MockProvider {
    responses: Mutex<VecDeque<ModelResponse>>,
    pub requests: Mutex<Vec<CompletionRequest>>,
    error_message: Option<String>,
}

impl MockProvider {
    /// Create with a queue of responses returned in FIFO order.
    pub fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Mutex::new(Vec::new()),
            error_message: None,
        }
    }

    /// Convenience: single text response with end_turn.
    pub fn text(text: &str) -> Self {
        Self::new(vec![text_response(text)])
    }

    /// Convenience: tool_use response followed by end_turn response.
    pub fn tool_then_text(tool_name: &str, input: serde_json::Value, final_text: &str) -> Self {
        Self::new(vec![
            tool_response(tool_name, "tool_call_1", input),
            text_response(final_text),
        ])
    }

    /// Zero responses — `.complete()` returns error immediately.
    pub fn empty() -> Self {
        Self::new(vec![])
    }

    /// Always returns the given error.
    pub fn error(err: AgenticError) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            error_message: Some(format!("{err}")),
        }
    }

    /// Returns a StructuredOutput tool_use response, then a text response.
    pub fn structured_output(input: serde_json::Value, final_text: &str) -> Self {
        Self::new(vec![
            tool_response("structured_output", "so_call_1", input),
            text_response(final_text),
        ])
    }

    /// Number of requests received.
    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    /// The most recent request, if any.
    pub fn last_request(&self) -> Option<CompletionRequest> {
        self.requests.lock().unwrap().last().cloned()
    }

    /// Extract system prompts from all recorded requests.
    pub fn system_prompts(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.system_prompt.clone())
            .collect()
    }
}

impl LlmProvider for MockProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        self.requests.lock().unwrap().push(request);

        Box::pin(async move {
            if let Some(ref msg) = self.error_message {
                return Err(AgenticError::Other(msg.clone()));
            }
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| AgenticError::Other("no more mock responses".into()))
        })
    }
}

/// Build a simple text-only ModelResponse.
pub fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
        model: "mock".to_string(),
    }
}

/// Build a tool_use ModelResponse.
pub fn tool_response(tool_name: &str, id: &str, input: serde_json::Value) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::ToolUse {
            id: id.to_string(),
            name: tool_name.to_string(),
            input,
        }],
        stop_reason: StopReason::ToolUse,
        usage: Usage::default(),
        model: "mock".to_string(),
    }
}

// ---------------------------------------------------------------------------
// MockTool
// ---------------------------------------------------------------------------

/// A mock tool for testing agent tool execution.
pub struct MockTool {
    pub name: String,
    pub read_only: bool,
    pub result: String,
    pub is_error: bool,
    pub calls: Mutex<Vec<serde_json::Value>>,
}

impl MockTool {
    pub fn new(name: &str, read_only: bool, result: &str) -> Self {
        Self {
            name: name.to_string(),
            read_only,
            result: result.to_string(),
            is_error: false,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn failing(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            read_only: false,
            result: message.to_string(),
            is_error: true,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    pub fn last_input(&self) -> Option<serde_json::Value> {
        self.calls.lock().unwrap().last().cloned()
    }
}

impl Tool for MockTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "A mock tool for testing"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    fn is_read_only(&self) -> bool {
        self.read_only
    }
    fn call<'a>(
        &'a self,
        input: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        self.calls.lock().unwrap().push(input);
        let result = ToolResult {
            content: self.result.clone(),
            is_error: self.is_error,
        };
        Box::pin(async move { Ok(result) })
    }
}

/// A mock tool that reports should_defer() = true.
pub struct DeferredMockTool {
    name: String,
}

impl DeferredMockTool {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl Tool for DeferredMockTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "A deferred mock tool"
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}})
    }
    fn should_defer(&self) -> bool {
        true
    }
    fn call<'a>(
        &'a self,
        _input: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async {
            Ok(ToolResult {
                content: "deferred result".into(),
                is_error: false,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Tool context helpers
// ---------------------------------------------------------------------------

pub fn test_tool_context() -> ToolContext {
    ToolContext::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

pub fn test_tool_context_with_registry(registry: Arc<ToolRegistry>) -> ToolContext {
    ToolContext::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .with_registry(registry)
}

// ---------------------------------------------------------------------------
// Agent test helpers
// ---------------------------------------------------------------------------

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
pub fn test_context_with_events(
    provider: Arc<dyn LlmProvider>,
) -> (InvocationContext, Arc<Mutex<Vec<Event>>>) {
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

// ---------------------------------------------------------------------------
// EventCollector
// ---------------------------------------------------------------------------

/// Collects events emitted during agent execution for test assertions.
pub struct EventCollector {
    events: Arc<Mutex<Vec<Event>>>,
}

impl EventCollector {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn callback(&self) -> Arc<dyn Fn(Event) + Send + Sync> {
        let events = self.events.clone();
        Arc::new(move |e| events.lock().unwrap().push(e))
    }

    pub fn all(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    pub fn texts(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn tool_starts(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::ToolStart { tool, .. } => Some(tool.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn tool_ends(&self) -> Vec<(String, bool)> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::ToolEnd {
                    tool, is_error, ..
                } => Some((tool.clone(), *is_error)),
                _ => None,
            })
            .collect()
    }

    pub fn agent_starts(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::AgentStart { agent } => Some(agent.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn agent_ends(&self) -> Vec<(String, u32)> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::AgentEnd { agent, turns } => Some((agent.clone(), *turns)),
                _ => None,
            })
            .collect()
    }

    pub fn errors(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::Error { error, .. } => Some(error.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------
// TestHarness
// ---------------------------------------------------------------------------

/// High-level test harness combining MockProvider, EventCollector, and context.
pub struct TestHarness {
    provider: Arc<MockProvider>,
    events: EventCollector,
    cost_tracker: CostTracker,
    state: HashMap<String, serde_json::Value>,
    working_directory: PathBuf,
    cancelled: Arc<AtomicBool>,
}

impl TestHarness {
    pub fn new(provider: MockProvider) -> Self {
        Self {
            provider: Arc::new(provider),
            events: EventCollector::new(),
            cost_tracker: CostTracker::new(),
            state: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_state(mut self, key: &str, value: serde_json::Value) -> Self {
        self.state.insert(key.to_string(), value);
        self
    }

    pub fn with_working_dir(mut self, path: PathBuf) -> Self {
        self.working_directory = path;
        self
    }

    pub fn build_context(&self, input: &str) -> InvocationContext {
        InvocationContext {
            input: input.to_string(),
            state: self.state.clone(),
            working_directory: self.working_directory.clone(),
            provider: self.provider.clone(),
            cost_tracker: self.cost_tracker.clone(),
            on_event: self.events.callback(),
            cancelled: self.cancelled.clone(),
            session_store: None,
            command_queue: None,
            agent_id: "test".into(),
        }
    }

    pub async fn run_agent(&self, agent: &dyn Agent, input: &str) -> Result<AgentOutput> {
        let ctx = self.build_context(input);
        agent.run(ctx).await
    }

    pub fn events(&self) -> &EventCollector {
        &self.events
    }

    pub fn provider(&self) -> &MockProvider {
        &self.provider
    }

    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}
