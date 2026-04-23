//! Mock provider, mock tool, and test harness that let unit tests drive the agent loop without reaching a live LLM.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::agent::queue::CommandQueue;
use crate::agent::{Agent, Event, EventKind, Output, Status};
use crate::error::Result;
use crate::provider::types::{
    CompletionResponse, ContentBlock, ResponseStatus, StreamEvent, TokenUsage,
};
use crate::provider::{CompletionRequest, Provider, ProviderError, ProviderResult};
use crate::tools::{ToolContext, ToolResult, Toolable};

/// A mock LLM provider that returns pre-configured responses in order.
///
/// Use `new()` for simple response sequences, or `with_results()` to interleave
/// errors and successes (useful for testing retry logic).
pub struct MockProvider {
    results: Mutex<VecDeque<ProviderResult<CompletionResponse>>>,
    pub requests: Mutex<Vec<CompletionRequest>>,
}

impl MockProvider {
    pub fn new(responses: Vec<CompletionResponse>) -> Self {
        Self {
            results: Mutex::new(responses.into_iter().map(Ok).collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn with_results(results: Vec<ProviderResult<CompletionResponse>>) -> Self {
        Self {
            results: Mutex::new(VecDeque::from(results)),
            requests: Mutex::new(Vec::new()),
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

impl Provider for MockProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<CompletionResponse>> + Send + '_>> {
        self.requests.lock().unwrap().push(request);

        Box::pin(async move {
            self.results.lock().unwrap().pop_front().unwrap_or_else(|| {
                Err(ProviderError::ResponseMalformed {
                    message: "no more mock responses".into(),
                })
            })
        })
    }

    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<CompletionResponse>> + Send + '_>> {
        Box::pin(async move {
            let response = self.complete(request).await?;
            for block in &response.content {
                if let ContentBlock::Text { text } = block {
                    on_event(StreamEvent::TextDelta {
                        index: 0,
                        text: text.clone(),
                    });
                }
            }
            on_event(StreamEvent::MessageDone);
            Ok(response)
        })
    }
}

pub fn text_response(text: &str) -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        status: ResponseStatus::EndTurn,
        usage: TokenUsage::default(),
        model: "mock".to_string(),
    }
}

pub fn truncated_response(text: &str) -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        status: ResponseStatus::OutputTruncated,
        usage: TokenUsage::default(),
        model: "mock".to_string(),
    }
}

pub fn tool_response(tool_name: &str, id: &str, input: serde_json::Value) -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            id: id.to_string(),
            name: tool_name.to_string(),
            input,
        }],
        status: ResponseStatus::ToolUse,
        usage: TokenUsage::default(),
        model: "mock".to_string(),
    }
}

pub struct MockTool {
    pub name: String,
    pub read_only: bool,
    pub outcome: ToolResult,
    pub calls: Mutex<Vec<serde_json::Value>>,
}

impl MockTool {
    pub fn new(name: &str, read_only: bool, result: &str) -> Self {
        Self {
            name: name.to_string(),
            read_only,
            outcome: ToolResult::success(result),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn error(name: &str, read_only: bool, error: &str) -> Self {
        Self {
            name: name.to_string(),
            read_only,
            outcome: ToolResult::error(error),
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl Toolable for MockTool {
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
        let outcome = self.outcome.clone();
        Box::pin(async move { Ok(outcome) })
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

impl Toolable for DeferredMockTool {
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

pub fn test_tool_context() -> ToolContext {
    ToolContext::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

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
            .filter_map(|e| match &e.kind {
                EventKind::TextChunkReceived { content } => Some(content.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn tool_starts(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::ToolCallStarted { tool_name, .. } => Some(tool_name.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn agent_starts(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::AgentStarted { .. } => Some(e.agent_name.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn all(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    pub fn agent_ends(&self) -> Vec<(String, u32, Status)> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::AgentFinished { turns, status } => {
                    Some((e.agent_name.clone(), *turns, status.clone()))
                }
                _ => None,
            })
            .collect()
    }
}

pub struct TestHarness {
    provider: Arc<MockProvider>,
    events: EventCollector,
    template_variables: HashMap<String, serde_json::Value>,
    working_directory: PathBuf,
    cancel_signal: Arc<AtomicBool>,
    // Only read from the cfg(test) branch in `run_agent` — in non-test builds
    // the field is always None and the reader is compiled out.
    #[allow(dead_code)]
    command_queue: Option<Arc<CommandQueue>>,
}

impl TestHarness {
    pub fn new(provider: MockProvider) -> Self {
        Self::with_provider(Arc::new(provider))
    }

    pub fn with_provider(provider: Arc<MockProvider>) -> Self {
        Self {
            provider,
            events: EventCollector::new(),
            template_variables: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            command_queue: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_provider_and_queue(
        provider: Arc<MockProvider>,
        queue: Arc<CommandQueue>,
    ) -> Self {
        let mut h = Self::with_provider(provider);
        h.command_queue = Some(queue);
        h
    }

    pub fn with_state(mut self, key: &str, value: serde_json::Value) -> Self {
        self.template_variables.insert(key.to_string(), value);
        self
    }

    /// Fully configure the given `Agent` with this harness's runtime bits
    /// (provider / event handler / wd / cancel / template vars), then run it
    /// against the supplied prompt.
    pub async fn run_agent(&self, agent: &Agent, input: &str) -> Result<Output> {
        let mut prepared = agent
            .clone()
            .provider(self.provider.clone())
            .instruction_prompt(input)
            .working_directory(self.working_directory.clone())
            .event_handler(self.events.callback())
            .cancel_signal(self.cancel_signal.clone());
        for (k, v) in &self.template_variables {
            prepared = prepared.template_variable(k.clone(), v.clone());
        }
        // If the harness carries a pre-built command queue, install it on the
        // agent so `Agent::compile` wires it onto the LoopRuntime.
        #[cfg(test)]
        if let Some(queue) = &self.command_queue {
            prepared = prepared.command_queue(queue.clone());
        }
        prepared.run().await
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

    /// Clone of the cancel signal — useful when a test needs to flip it from
    /// a spawned task while the harness is blocked on `run_agent`.
    #[cfg(test)]
    pub(crate) fn cancel_signal_for_test(&self) -> Arc<AtomicBool> {
        self.cancel_signal.clone()
    }
}
