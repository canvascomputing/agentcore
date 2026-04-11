use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use crate::error::{AgenticError, Result};
use crate::message::{ContentBlock, ModelResponse, StopReason, Usage};
use crate::provider::{CompletionRequest, LlmProvider};

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
