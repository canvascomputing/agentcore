//! Mock provider, mock tool, and test harness that let unit tests drive the agent loop without reaching a live provider.
//!
//! These types are also intended for doctests across the crate: prefer
//! [`MockProvider`] over `no_run` whenever an example can run hermetically.
//! Use `no_run` only when the example must show real provider selection
//! (e.g. [`crate::Agent::provider_from_env`]).

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::agent::work::IncomingWork;
use crate::agent::Agent;
use crate::error::Result;
use crate::event::{Event, EventKind};
use crate::output::{Outcome, Output};
use crate::provider::types::{
    ContentBlock, ModelResponse, ResponseStatus, StreamEvent, TokenUsage,
};
use crate::provider::{ModelRequest, Provider, ProviderError, ProviderResult};
use crate::tools::{ToolContext, ToolLike, ToolResult};

/// A mock provider that returns pre-configured responses in order.
///
/// Use [`MockProvider::new`] for simple response sequences, or
/// [`MockProvider::with_results`] to interleave errors and successes (useful
/// for testing retry logic).
pub struct MockProvider {
    results: Mutex<VecDeque<ProviderResult<ModelResponse>>>,
    pub requests: Mutex<Vec<ModelRequest>>,
}

impl MockProvider {
    pub fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            results: Mutex::new(responses.into_iter().map(Ok).collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn with_results(results: Vec<ProviderResult<ModelResponse>>) -> Self {
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

    pub fn requests(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    pub fn last_request(&self) -> Option<ModelRequest> {
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
    fn respond(
        &self,
        request: ModelRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        self.requests.lock().unwrap().push(request);

        Box::pin(async move {
            let response = self
                .results
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    Err(ProviderError::ResponseMalformed {
                        message: "no more mock responses".into(),
                    })
                })?;
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

pub fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        status: ResponseStatus::EndTurn,
        usage: TokenUsage::default(),
        model: "mock".to_string(),
    }
}

pub fn truncated_response(text: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        status: ResponseStatus::OutputTruncated,
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

impl ToolLike for MockTool {
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

impl ToolLike for DeferredMockTool {
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

    pub fn agent_ends(&self) -> Vec<(String, u32, Outcome)> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::AgentFinished { turns, outcome } => {
                    Some((e.agent_name.clone(), *turns, outcome.clone()))
                }
                _ => None,
            })
            .collect()
    }
}

pub struct TestHarness {
    provider: Arc<MockProvider>,
    events: EventCollector,
    templates: HashMap<String, serde_json::Value>,
    working_dir: PathBuf,
    interrupt_signal: Arc<AtomicBool>,
    // Only read from the cfg(test) branch in `run_agent` — in non-test builds
    // the field is always None and the reader is compiled out.
    #[allow(dead_code)]
    incoming_work: Option<Arc<IncomingWork>>,
}

impl TestHarness {
    pub fn new(provider: MockProvider) -> Self {
        Self::with_provider(Arc::new(provider))
    }

    pub fn with_provider(provider: Arc<MockProvider>) -> Self {
        Self {
            provider,
            events: EventCollector::new(),
            templates: HashMap::new(),
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            interrupt_signal: Arc::new(AtomicBool::new(false)),
            incoming_work: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_provider_and_queue(
        provider: Arc<MockProvider>,
        queue: Arc<IncomingWork>,
    ) -> Self {
        let mut h = Self::with_provider(provider);
        h.incoming_work = Some(queue);
        h
    }

    pub fn with_state(mut self, key: &str, value: serde_json::Value) -> Self {
        self.templates.insert(key.to_string(), value);
        self
    }

    /// Fully configure the given `Agent` with this harness's runtime bits
    /// (provider / event handler / wd / cancel / template vars), then run it
    /// against the supplied prompt.
    pub async fn run_agent(&self, agent: &Agent, input: &str) -> Result<Output> {
        let mut prepared = agent
            .clone()
            .provider(self.provider.clone())
            .task(input)
            .working_dir(self.working_dir.clone())
            .event_handler(self.events.callback())
            .interrupt_signal(self.interrupt_signal.clone());
        for (k, v) in &self.templates {
            prepared = prepared.template(k.clone(), v.clone());
        }
        // If the harness carries a pre-built command queue, install it on the
        // agent so `Agent::compile` wires it onto the LoopRuntime.
        #[cfg(test)]
        if let Some(queue) = &self.incoming_work {
            prepared = prepared.incoming_work(queue.clone());
        }
        prepared.work().await
    }

    pub fn events(&self) -> &EventCollector {
        &self.events
    }

    pub fn provider(&self) -> &MockProvider {
        &self.provider
    }

    pub fn cancel(&self) {
        self.interrupt_signal
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Clone of the cancel signal — useful when a test needs to flip it from
    /// a spawned task while the harness is blocked on `run_agent`.
    #[cfg(test)]
    pub(crate) fn interrupt_signal_for_test(&self) -> Arc<AtomicBool> {
        self.interrupt_signal.clone()
    }
}
