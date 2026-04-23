//! Anthropic Messages API provider, with SSE streaming and cache-aware token accounting.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;

use super::error::{ProviderError, ProviderResult};
use super::r#trait::{
    build_client, ModelRequest, Provider, ToolChoice, DEFAULT_REQUEST_TIMEOUT,
};
use super::types::{
    ModelResponse, ContentBlock, Message, ResponseStatus, StreamEvent, TokenUsage,
};

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    timeout: Duration,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".into(),
            client: build_client(DEFAULT_REQUEST_TIMEOUT),
            timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    pub(crate) fn from_env() -> Result<Self> {
        use super::environment::{env_or, env_required};
        Ok(Self::new(env_required("ANTHROPIC_API_KEY")?)
            .base_url(env_or("ANTHROPIC_BASE_URL", "https://api.anthropic.com")))
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self.client = build_client(self.timeout);
        self
    }

    #[cfg(test)]
    pub(crate) fn request_timeout(&self) -> Duration {
        self.timeout
    }

    fn api_headers(&self) -> Vec<(String, String)> {
        vec![
            ("x-api-key".into(), self.api_key.clone()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("content-type".into(), "application/json".into()),
        ]
    }

    fn serialize_request(&self, request: &ModelRequest) -> Value {
        let mut body = serde_json::json!({
            "model": request.model,
            "system": request.system_prompt,
            "messages": serialize_messages(&request.messages),
        });
        if let Some(n) = request.max_request_tokens {
            body["max_tokens"] = Value::from(n);
        }
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(serialize_tool_definition)
                .collect();
            body["tools"] = Value::Array(tools);
        }
        if let Some(ref choice) = request.tool_choice {
            body["tool_choice"] = serialize_tool_choice(choice);
        }
        body
    }

    fn parse_response(&self, json: Value) -> ModelResponse {
        ModelResponse {
            content: parse_content(&json),
            status: parse_status(&json),
            usage: parse_usage(&json),
            model: json["model"].as_str().unwrap_or("unknown").to_string(),
        }
    }

    async fn send_request(&self, body: Value) -> ProviderResult<reqwest::Response> {
        let url = format!("{}/v1/messages", self.base_url);
        let mut req = self.client.post(&url).json(&body);
        for (k, v) in self.api_headers() {
            req = req.header(k, v);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ProviderError::ConnectionFailed {
                message: e.to_string(),
            })?;

        super::map_http_errors(resp, classify_error).await
    }
}

/// Map Anthropic-specific error signatures to typed [`ProviderError`]
/// variants. Any status/body combination the match doesn't recognise
/// returns `None`, causing `map_http_errors` to fall through to
/// [`ProviderError::StatusUnclassified`] (or [`ProviderError::RateLimited`]
/// for 429/529).
fn classify_error(status: u16, body: &str) -> Option<ProviderError> {
    match status {
        401 => Some(ProviderError::AuthenticationFailed {
            message: body.into(),
        }),
        403 => Some(ProviderError::PermissionDenied {
            message: body.into(),
        }),
        404 => Some(ProviderError::ModelNotFound {
            message: body.into(),
        }),
        400 => classify_400(body),
        _ => None,
    }
}

fn classify_400(body: &str) -> Option<ProviderError> {
    let json: Value = serde_json::from_str(body).ok()?;
    let err = &json["error"];
    let type_ = err["type"].as_str().unwrap_or("");
    let message = err["message"].as_str().unwrap_or("").to_string();
    match type_ {
        "invalid_request_error" if message.contains("prompt is too long") => {
            Some(ProviderError::ContextWindowExceeded { message })
        }
        "not_found_error" => Some(ProviderError::ModelNotFound { message }),
        _ => None,
    }
}

impl AnthropicProvider {
    pub(crate) fn lookup_context_window_size(id: &str) -> Option<u64> {
        let m = id.to_ascii_lowercase();
        if m.contains("[1m]") {
            return Some(1_000_000);
        }
        if m.contains("claude-opus-4")
            || m.contains("claude-sonnet-4")
            || m.contains("claude-haiku-4")
            || m.contains("claude-3-7-sonnet")
            || m.contains("claude-3-5-sonnet")
            || m.contains("claude-3-5-haiku")
            || m.contains("claude-3-opus")
            || m.contains("claude-3-sonnet")
            || m.contains("claude-3-haiku")
        {
            return Some(200_000);
        }
        None
    }
}

impl Provider for AnthropicProvider {
    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async { super::r#trait::prewarm_with(&self.client, &self.base_url).await })
    }

    fn respond(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        let body = self.serialize_request(&request);

        Box::pin(async move {
            let resp = self.send_request(body).await?;
            let json: Value = resp
                .json()
                .await
                .map_err(|e| ProviderError::ResponseMalformed {
                    message: e.to_string(),
                })?;
            Ok(self.parse_response(json))
        })
    }

    fn respond_streaming(
        &self,
        request: ModelRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        let mut body = self.serialize_request(&request);
        body["stream"] = Value::Bool(true);

        Box::pin(async move {
            let resp = self.send_request(body).await?;
            stream_response(resp, &on_event).await
        })
    }
}

async fn stream_response(
    response: reqwest::Response,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
) -> ProviderResult<ModelResponse> {
    use super::stream::{SseEvent, StreamParser};
    use futures_util::StreamExt;

    let mut state = StreamState::default();
    let mut parser = StreamParser::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| ProviderError::StreamInterrupted {
            message: e.to_string(),
        })?;
        for event in parser.push(&chunk) {
            match event {
                SseEvent::Done => on_event(StreamEvent::MessageDone),
                SseEvent::Data(json) => handle_stream_event(&json, &mut state, on_event),
            }
        }
    }

    Ok(state.into_response())
}

/// Accumulator for a single Anthropic content block. Replaces four parallel
/// `Vec<Option<String>>` lookups with one typed slot per index.
enum BlockAcc {
    Text(String),
    Tool {
        id: String,
        name: String,
        input: String,
    },
}

#[derive(Default)]
struct StreamState {
    model: String,
    usage: TokenUsage,
    status: ResponseStatus,
    content_blocks: Vec<ContentBlock>,
    blocks: Vec<Option<BlockAcc>>,
}

impl StreamState {
    fn slot(&mut self, idx: usize) -> &mut Option<BlockAcc> {
        if self.blocks.len() <= idx {
            self.blocks.resize_with(idx + 1, || None);
        }
        &mut self.blocks[idx]
    }

    fn into_response(self) -> ModelResponse {
        ModelResponse {
            content: self.content_blocks,
            status: self.status,
            usage: self.usage,
            model: if self.model.is_empty() {
                "unknown".into()
            } else {
                self.model
            },
        }
    }
}

fn handle_stream_event(
    json: &Value,
    state: &mut StreamState,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
) {
    match json["type"].as_str().unwrap_or("") {
        "message_start" => handle_message_start(json, state),
        "content_block_start" => handle_block_start(json, state),
        "content_block_delta" => handle_block_delta(json, state, on_event),
        "content_block_stop" => handle_block_stop(json, state, on_event),
        "message_delta" => handle_message_delta(json, state, on_event),
        _ => {}
    }
}

fn handle_message_start(json: &Value, state: &mut StreamState) {
    let message = &json["message"];
    state.model = message["model"].as_str().unwrap_or("unknown").to_string();
    let u = &message["usage"];
    state.usage.input_tokens = u["input_tokens"].as_u64().unwrap_or(0);
    state.usage.cache_read_input_tokens = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
    state.usage.cache_creation_input_tokens =
        u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
}

fn handle_block_start(json: &Value, state: &mut StreamState) {
    let idx = json["index"].as_u64().unwrap_or(0) as usize;
    let block = &json["content_block"];
    let acc = match block["type"].as_str().unwrap_or("") {
        "tool_use" => BlockAcc::Tool {
            id: block["id"].as_str().unwrap_or("").to_string(),
            name: block["name"].as_str().unwrap_or("").to_string(),
            input: String::new(),
        },
        _ => BlockAcc::Text(String::new()),
    };
    *state.slot(idx) = Some(acc);
}

fn handle_block_delta(
    json: &Value,
    state: &mut StreamState,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
) {
    let idx = json["index"].as_u64().unwrap_or(0) as usize;
    let delta = &json["delta"];
    match delta["type"].as_str().unwrap_or("") {
        "text_delta" => {
            let text = delta["text"].as_str().unwrap_or("").to_string();
            if let Some(Some(BlockAcc::Text(buf))) = state.blocks.get_mut(idx) {
                buf.push_str(&text);
            }
            on_event(StreamEvent::TextDelta { index: idx, text });
        }
        "input_json_delta" => {
            let partial = delta["partial_json"].as_str().unwrap_or("");
            if let Some(Some(BlockAcc::Tool { input, .. })) = state.blocks.get_mut(idx) {
                input.push_str(partial);
            }
            on_event(StreamEvent::InputJsonDelta {
                index: idx,
                partial_json: partial.to_string(),
            });
        }
        _ => {}
    }
}

fn handle_block_stop(
    json: &Value,
    state: &mut StreamState,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
) {
    let idx = json["index"].as_u64().unwrap_or(0) as usize;
    match state.blocks.get_mut(idx).and_then(Option::take) {
        Some(BlockAcc::Text(text)) => state.content_blocks.push(ContentBlock::Text { text }),
        Some(BlockAcc::Tool { id, name, input }) => {
            let input = serde_json::from_str(&input).unwrap_or(Value::Object(Default::default()));
            state
                .content_blocks
                .push(ContentBlock::ToolUse { id, name, input });
        }
        None => {}
    }
    on_event(StreamEvent::ContentBlockStop { index: idx });
}

fn handle_message_delta(
    json: &Value,
    state: &mut StreamState,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
) {
    state.status = parse_status_str(json["delta"]["stop_reason"].as_str().unwrap_or("end_turn"));
    state.usage.output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0);
    on_event(StreamEvent::MessageDelta {
        status: state.status.clone(),
        usage: state.usage.clone(),
    });
}

fn serialize_messages(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .filter_map(|msg| {
            let (role, content) = match msg {
                Message::System { .. } => return None,
                Message::User { content } => ("user", content),
                Message::Assistant { content } => ("assistant", content),
            };
            Some(serde_json::json!({
                "role": role,
                "content": serialize_content_blocks(content),
            }))
        })
        .collect()
}

fn serialize_content_blocks(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks.iter().map(serialize_content_block).collect()
}

fn serialize_content_block(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text { text } => {
            serde_json::json!({"type": "text", "text": text})
        }
        ContentBlock::ToolUse { id, name, input } => {
            serde_json::json!({"type": "tool_use", "id": id, "name": name, "input": input})
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            serde_json::json!({"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error})
        }
    }
}

fn serialize_tool_definition(tool: &crate::tools::ToolDefinition) -> Value {
    serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    })
}

fn serialize_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Specific { name } => serde_json::json!({"type": "tool", "name": name}),
    }
}

fn parse_content(json: &Value) -> Vec<ContentBlock> {
    let Some(blocks) = json["content"].as_array() else {
        return Vec::new();
    };
    blocks
        .iter()
        .filter_map(|block| match block["type"].as_str()? {
            "text" => Some(ContentBlock::Text {
                text: block["text"].as_str().unwrap_or("").to_string(),
            }),
            "tool_use" => Some(ContentBlock::ToolUse {
                id: block["id"].as_str().unwrap_or("").to_string(),
                name: block["name"].as_str().unwrap_or("").to_string(),
                input: block["input"].clone(),
            }),
            _ => None,
        })
        .collect()
}

fn parse_status(json: &Value) -> ResponseStatus {
    parse_status_str(json["stop_reason"].as_str().unwrap_or("end_turn"))
}

fn parse_status_str(raw: &str) -> ResponseStatus {
    match raw {
        "end_turn" => ResponseStatus::EndTurn,
        "stop_sequence" => ResponseStatus::StopSequence,
        "tool_use" => ResponseStatus::ToolUse,
        "max_tokens" => ResponseStatus::OutputTruncated,
        "model_context_window_exceeded" => ResponseStatus::ContextWindowExceeded,
        "refusal" => ResponseStatus::Refused,
        "pause_turn" => ResponseStatus::PauseTurn,
        _ => ResponseStatus::EndTurn,
    }
}

fn parse_usage(json: &Value) -> TokenUsage {
    let usage = &json["usage"];
    TokenUsage {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
        cache_read_input_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        cache_creation_input_tokens: usage["cache_creation_input_tokens"].as_u64().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_request() -> ModelRequest {
        ModelRequest {
            model: "test-model".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text { text: "Hi".into() }],
            }],
            tools: vec![],
            max_request_tokens: Some(1024),
            tool_choice: None,
        }
    }

    fn provider() -> AnthropicProvider {
        AnthropicProvider::new("test-key")
    }

    #[test]
    fn timeout_builder_stores_duration() {
        let p = provider();
        assert_eq!(p.request_timeout(), DEFAULT_REQUEST_TIMEOUT);
        let p = p.timeout(Duration::from_secs(42));
        assert_eq!(p.request_timeout(), Duration::from_secs(42));
    }

    #[test]
    fn serialize_request_sets_model_and_system() {
        let body = provider().serialize_request(&simple_request());
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn serialize_request_omits_max_tokens_when_none() {
        let mut req = simple_request();
        req.max_request_tokens = None;
        let body = provider().serialize_request(&req);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn serialize_request_excludes_system_from_messages() {
        let body = provider().serialize_request(&simple_request());
        let messages = body["messages"].as_array().unwrap();
        for msg in messages {
            assert_ne!(msg["role"], "system");
        }
    }

    #[test]
    fn serialize_request_includes_tool_choice() {
        let mut req = simple_request();
        req.tool_choice = Some(ToolChoice::Specific {
            name: "read_file".into(),
        });
        let body = provider().serialize_request(&req);
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "read_file");
    }

    #[test]
    fn parse_response_extracts_text() {
        let json = serde_json::json!({
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-sonnet-4-20250514"
        });
        let resp = provider().parse_response(json);
        assert_eq!(resp.content.len(), 1);
        assert!(matches!(&resp.content[0], ContentBlock::Text { text } if text == "Hello!"));
        assert_eq!(resp.status, ResponseStatus::EndTurn);
        assert_eq!(resp.usage.input_tokens, 10);
    }

    #[test]
    fn parse_response_extracts_tool_use() {
        let json = serde_json::json!({
            "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "/tmp"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": "mock"
        });
        let resp = provider().parse_response(json);
        assert_eq!(resp.status, ResponseStatus::ToolUse);
        match &resp.content[0] {
            ContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "read");
                assert_eq!(input["path"], "/tmp");
            }
            other => panic!("Expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_empty_content() {
        let json = serde_json::json!({
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": "mock"
        });
        let resp = provider().parse_response(json);
        assert!(resp.content.is_empty());
    }

    #[test]
    fn parse_response_maps_status() {
        for (reason, expected) in [
            ("end_turn", ResponseStatus::EndTurn),
            ("stop_sequence", ResponseStatus::StopSequence),
            ("tool_use", ResponseStatus::ToolUse),
            ("max_tokens", ResponseStatus::OutputTruncated),
            (
                "model_context_window_exceeded",
                ResponseStatus::ContextWindowExceeded,
            ),
            ("refusal", ResponseStatus::Refused),
            ("pause_turn", ResponseStatus::PauseTurn),
        ] {
            let json = serde_json::json!({
                "content": [], "stop_reason": reason,
                "usage": {"input_tokens": 0, "output_tokens": 0}, "model": "m"
            });
            assert_eq!(provider().parse_response(json).status, expected);
        }
    }

    fn invalid_request(message: &str) -> String {
        serde_json::json!({
            "error": { "type": "invalid_request_error", "message": message }
        })
        .to_string()
    }

    #[test]
    fn context_window_exceeded_when_prompt_is_too_long() {
        let body = invalid_request("prompt is too long: 205000 > 200000");
        assert!(matches!(
            classify_error(400, &body),
            Some(ProviderError::ContextWindowExceeded { .. })
        ));
    }

    #[test]
    fn maps_400_not_found_error_to_model_not_found() {
        let body = serde_json::json!({
            "error": { "type": "not_found_error", "message": "model opus-9 not found" }
        })
        .to_string();
        assert!(matches!(
            classify_error(400, &body),
            Some(ProviderError::ModelNotFound { .. })
        ));
    }

    #[test]
    fn maps_http_401_to_authentication_failed() {
        assert!(matches!(
            classify_error(401, "boom"),
            Some(ProviderError::AuthenticationFailed { .. })
        ));
    }

    #[test]
    fn maps_http_403_to_permission_denied() {
        assert!(matches!(
            classify_error(403, "boom"),
            Some(ProviderError::PermissionDenied { .. })
        ));
    }

    #[test]
    fn maps_http_404_to_model_not_found() {
        assert!(matches!(
            classify_error(404, "boom"),
            Some(ProviderError::ModelNotFound { .. })
        ));
    }

    #[test]
    fn unrelated_400_falls_through() {
        let body = invalid_request("max_tokens must be a positive integer");
        assert!(classify_error(400, &body).is_none());
    }

    #[test]
    fn preserves_provider_message() {
        let Some(ProviderError::AuthenticationFailed { message }) =
            classify_error(401, "your api key is revoked")
        else {
            panic!("expected AuthenticationFailed");
        };
        assert_eq!(message, "your api key is revoked");
    }

    #[test]
    fn lookup_claude_4_family_returns_200k() {
        let lookup = AnthropicProvider::lookup_context_window_size;
        assert_eq!(lookup("claude-sonnet-4-20250514"), Some(200_000));
        assert_eq!(lookup("claude-opus-4-20250101"), Some(200_000));
        assert_eq!(lookup("claude-haiku-4-5-20251001"), Some(200_000));
    }

    #[test]
    fn lookup_claude_3_family_returns_200k() {
        let lookup = AnthropicProvider::lookup_context_window_size;
        assert_eq!(lookup("claude-3-5-sonnet-20241022"), Some(200_000));
        assert_eq!(lookup("claude-3-opus-20240229"), Some(200_000));
    }

    #[test]
    fn lookup_one_million_suffix_overrides_base_family() {
        let lookup = AnthropicProvider::lookup_context_window_size;
        assert_eq!(
            lookup("claude-opus-4-7[1m]"),
            Some(1_000_000),
            "explicit [1m] opt-in promotes to 1M"
        );
        assert_eq!(lookup("claude-sonnet-4-20250514[1m]"), Some(1_000_000));
    }

    #[test]
    fn lookup_unknown_models_return_none() {
        let lookup = AnthropicProvider::lookup_context_window_size;
        assert_eq!(lookup("gpt-4"), None);
        assert_eq!(lookup("mistral-large-2411"), None);
        assert_eq!(lookup("some-future-model"), None);
    }
}
