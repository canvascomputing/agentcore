use std::collections::HashMap;
use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::agent::{Agent, AgentOutput, Event, InvocationContext};
use crate::error::{AgenticError, Result};
use crate::provider::types::{ContentBlock, ModelResponse, StopReason, TokenUsage};
use crate::provider::{CompletionRequest, LlmProvider};
use crate::tools::{Tool, ToolContext, ToolResult};

/// A mock LLM provider that returns pre-configured responses in order.
pub struct MockProvider {
    responses: Mutex<VecDeque<ModelResponse>>,
    pub requests: Mutex<Vec<CompletionRequest>>,
    error_message: Option<String>,
}

impl MockProvider {
    pub fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Mutex::new(Vec::new()),
            error_message: None,
        }
    }

    pub fn text(text: &str) -> Self {
        Self::new(vec![text_response(text)])
    }

    pub fn tool_then_text(tool_name: &str, input: serde_json::Value, final_text: &str) -> Self {
        Self::new(vec![
            tool_response(tool_name, "tool_call_1", input),
            text_response(final_text),
        ])
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    pub fn last_request(&self) -> Option<CompletionRequest> {
        self.requests.lock().unwrap().last().cloned()
    }

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

pub fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: TokenUsage::default(),
        model: "mock".to_string(),
    }
}

pub fn tool_response(tool_name: &str, id: &str, input: serde_json::Value) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::ToolUse {
            id: id.to_string(),
            name: tool_name.to_string(),
            input,
        }],
        stop_reason: StopReason::ToolUse,
        usage: TokenUsage::default(),
        model: "mock".to_string(),
    }
}

// ---------------------------------------------------------------------------
// MockTool
// ---------------------------------------------------------------------------

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
        let result = if self.is_error {
            ToolResult::error(self.result.clone())
        } else {
            ToolResult::success(self.result.clone())
        };
        Box::pin(async move { Ok(result) })
    }
}

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
        Box::pin(async { Ok(ToolResult::success("deferred result")) })
    }
}

// ---------------------------------------------------------------------------
// Tool context helpers
// ---------------------------------------------------------------------------

pub fn test_tool_context() -> ToolContext {
    ToolContext::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

// ---------------------------------------------------------------------------
// EventCollector
// ---------------------------------------------------------------------------

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

    pub fn texts(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Event::ResponseTextChunk { content, .. } => Some(content.clone()),
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
                Event::ToolCallStart { tool_name, .. } => Some(tool_name.clone()),
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
                Event::AgentStart { agent_name } => Some(agent_name.clone()),
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
                Event::AgentEnd { agent_name, turns } => Some((agent_name.clone(), *turns)),
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TestHarness
// ---------------------------------------------------------------------------

pub struct TestHarness {
    provider: Arc<MockProvider>,
    events: EventCollector,
    template_variables: HashMap<String, serde_json::Value>,
    working_directory: PathBuf,
    cancel_signal: Arc<AtomicBool>,
}

impl TestHarness {
    pub fn new(provider: MockProvider) -> Self {
        Self {
            provider: Arc::new(provider),
            events: EventCollector::new(),
            template_variables: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            cancel_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_state(mut self, key: &str, value: serde_json::Value) -> Self {
        self.template_variables.insert(key.to_string(), value);
        self
    }

    pub fn build_context(&self, input: &str) -> InvocationContext {
        InvocationContext::new(self.provider.clone())
            .instruction_prompt(input)
            .template_variables(self.template_variables.clone())
            .working_directory(self.working_directory.clone())
            .event_handler(self.events.callback())
            .cancel_signal(self.cancel_signal.clone())
            .agent_name("test")
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
        self.cancel_signal
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}
