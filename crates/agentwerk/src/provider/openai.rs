//! OpenAI's Chat Completions API. The sibling `mistral` and `litellm` providers reuse this wire format against different base URLs.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;

use super::error::{ProviderError, ProviderResult};
use super::r#trait::{build_client, ModelRequest, Provider, ToolChoice, DEFAULT_REQUEST_TIMEOUT};
use super::stream::{SseEvent, StreamParser};
use super::types::{ContentBlock, Message, ModelResponse, ResponseStatus, StreamEvent, TokenUsage};

/// OpenAI-compatible LLM provider.
pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    cache_tokens: bool,
    timeout: Duration,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::raw(api_key, "https://api.openai.com", false)
    }

    /// Raw constructor used by sibling provider modules (`mistral`,
    /// `litellm`) to build an `OpenAiProvider` pointed at their own
    /// endpoint. Not part of the public API.
    pub(crate) fn raw(api_key: impl Into<String>, base_url: &str, cache_tokens: bool) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: build_client(DEFAULT_REQUEST_TIMEOUT),
            cache_tokens,
            timeout: DEFAULT_REQUEST_TIMEOUT,
        }
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

    pub(crate) fn from_env() -> Result<Self> {
        use super::environment::{env_or, env_required};
        Ok(Self::new(env_required("OPENAI_API_KEY")?)
            .base_url(env_or("OPENAI_BASE_URL", "https://api.openai.com")))
    }
}

impl OpenAiProvider {
    pub(crate) fn lookup_context_window_size(id: &str) -> Option<u64> {
        let m = id.to_ascii_lowercase();
        // Newest first so "gpt-4" doesn't shadow "gpt-4.1".
        if m.contains("gpt-4.1") {
            return Some(1_000_000);
        }
        if m.contains("gpt-5") {
            // 400K covers mini/nano; full/pro reach ~1M but we pick the family floor.
            return Some(400_000);
        }
        if m.starts_with("o3") || m.starts_with("o1") {
            return Some(200_000);
        }
        if m.contains("gpt-4o") || m.contains("gpt-4-turbo") {
            return Some(128_000);
        }
        if m.contains("gpt-4-32k") {
            return Some(32_768);
        }
        if m.contains("gpt-4") {
            return Some(8_192);
        }
        if m.contains("gpt-3.5-turbo") {
            return Some(16_385);
        }
        // Qwen (Alibaba) is routinely served via OpenAI-compatible endpoints
        // (vLLM, Ollama, LiteLLM), so its lookup lives here. Values are the
        // published *native* windows; deployments that disable YaRN or
        // configure a shorter `max_model_len` should override via
        // `Agent::model(Model::from_name(name).context_window_size(n))`.
        // Newest first so "qwen3" doesn't shadow "qwen3.5" / "qwen3.6".
        if m.contains("qwen3.6") || m.contains("qwen3.5") {
            return Some(262_144);
        }
        if m.contains("qwen3-coder") || m.contains("qwen3-next") {
            return Some(262_144);
        }
        if m.contains("-2507") && m.contains("qwen3") {
            return Some(262_144);
        }
        if m.contains("qwen3") {
            return Some(131_072);
        }
        if m.contains("qwen2.5") {
            // Qwen2.5 "1M" variants (e.g. Qwen2.5-7B-Instruct-1M) are a
            // separate release with a 1M-token native window.
            if m.contains("-1m") {
                return Some(1_000_000);
            }
            return Some(32_768);
        }
        None
    }
}

impl Provider for OpenAiProvider {
    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async { super::r#trait::prewarm_with(&self.client, &self.base_url).await })
    }

    fn respond(
        &self,
        request: ModelRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        let mut body = serialize_request(&request);
        body["stream"] = Value::Bool(true);
        body["stream_options"] = serde_json::json!({"include_usage": true});
        let url = format!("{}/v1/chat/completions", self.base_url);

        Box::pin(async move {
            let resp = self.send_raw(&url, body).await?;
            stream_response(resp, &on_event, self.cache_tokens).await
        })
    }
}

impl OpenAiProvider {
    async fn send_raw(&self, url: &str, body: Value) -> ProviderResult<reqwest::Response> {
        let resp = self
            .client
            .post(url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::ConnectionFailed {
                message: e.to_string(),
            })?;

        super::map_http_errors(resp, classify_error).await
    }
}

/// Map OpenAI-compatible error signatures to typed [`ProviderError`]
/// variants. Covers the OpenAI shape and the subset of it that LiteLLM /
/// Mistral reproduce (where `error.code` may be absent, so the message
/// text is the fallback signal).
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
    let code = err["code"].as_str().unwrap_or("");
    let message = err["message"].as_str().unwrap_or("").to_string();

    let is_context_window = code == "context_length_exceeded"
        || message.contains("maximum context length")
        || message.contains("context_length_exceeded");
    if is_context_window {
        return Some(ProviderError::ContextWindowExceeded { message });
    }
    match code {
        "model_not_found" => Some(ProviderError::ModelNotFound { message }),
        "content_filter" => Some(ProviderError::SafetyFilterTriggered { message }),
        _ => None,
    }
}

async fn stream_response(
    response: reqwest::Response,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
    cache_tokens: bool,
) -> ProviderResult<ModelResponse> {
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
                SseEvent::Data(json) => ingest_chunk(&json, &mut state, on_event, cache_tokens),
            }
        }
    }

    on_event(StreamEvent::MessageDelta {
        status: state.status.clone(),
        usage: state.usage.clone(),
    });
    Ok(state.into_response())
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct StreamState {
    text: String,
    tool_calls: HashMap<usize, ToolCallAccumulator>,
    status: ResponseStatus,
    usage: TokenUsage,
    model: String,
}

impl StreamState {
    fn into_response(self) -> ModelResponse {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        let mut sorted: Vec<_> = self.tool_calls.into_iter().collect();
        sorted.sort_by_key(|(idx, _)| *idx);
        for (_, acc) in sorted {
            let input =
                serde_json::from_str(&acc.arguments).unwrap_or(Value::Object(Default::default()));
            content.push(ContentBlock::ToolUse {
                id: acc.id,
                name: acc.name,
                input,
            });
        }
        ModelResponse {
            content,
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

fn ingest_chunk(
    json: &Value,
    state: &mut StreamState,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
    cache_tokens: bool,
) {
    if let Some(m) = json["model"].as_str() {
        state.model = m.to_string();
    }
    let choice = &json["choices"][0];
    update_text(&choice["delta"], state, on_event);
    update_tool_calls(&choice["delta"], state, on_event);
    if let Some(reason) = choice["finish_reason"].as_str() {
        state.status = parse_status_str(reason);
    }
    if let Some(u) = json.get("usage").filter(|u| !u.is_null()) {
        parse_streaming_usage(u, cache_tokens, &mut state.usage);
    }
}

fn update_text(
    delta: &Value,
    state: &mut StreamState,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
) {
    let Some(content) = delta["content"].as_str().filter(|s| !s.is_empty()) else {
        return;
    };
    state.text.push_str(content);
    on_event(StreamEvent::TextDelta {
        index: 0,
        text: content.to_string(),
    });
}

fn update_tool_calls(
    delta: &Value,
    state: &mut StreamState,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
) {
    let Some(calls) = delta["tool_calls"].as_array() else {
        return;
    };
    for call in calls {
        let idx = call["index"].as_u64().unwrap_or(0) as usize;
        let acc = state.tool_calls.entry(idx).or_default();
        if let Some(id) = call["id"].as_str() {
            acc.id = id.to_string();
        }
        if let Some(name) = call["function"]["name"].as_str() {
            acc.name = name.to_string();
        }
        if let Some(args) = call["function"]["arguments"].as_str() {
            acc.arguments.push_str(args);
            on_event(StreamEvent::InputJsonDelta {
                index: idx,
                partial_json: args.to_string(),
            });
        }
    }
}

fn parse_streaming_usage(u: &Value, cache_tokens: bool, dst: &mut TokenUsage) {
    dst.input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0);
    dst.output_tokens = u["completion_tokens"].as_u64().unwrap_or(0);
    if cache_tokens {
        dst.cache_read_input_tokens = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
        dst.cache_creation_input_tokens = u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    }
}

fn serialize_request(request: &ModelRequest) -> Value {
    let mut body = serde_json::json!({
        "model": request.model,
        "messages": serialize_messages(request),
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

fn serialize_messages(request: &ModelRequest) -> Vec<Value> {
    let mut messages = Vec::new();

    if !request.system_prompt.is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": request.system_prompt,
        }));
    }

    for msg in &request.messages {
        match msg {
            Message::System { content } => {
                messages.push(serde_json::json!({"role": "system", "content": content}));
            }
            Message::User { content } => {
                serialize_user_blocks(content, &mut messages);
            }
            Message::Assistant { content } => {
                messages.push(serialize_assistant_message(content));
            }
        }
    }

    messages
}

fn serialize_user_blocks(blocks: &[ContentBlock], messages: &mut Vec<Value>) {
    let mut text_parts = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content,
                }));
            }
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        messages.push(serde_json::json!({
            "role": "user",
            "content": text_parts.join("\n"),
        }));
    }
}

fn serialize_assistant_message(blocks: &[ContentBlock]) -> Value {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()},
                }));
            }
            _ => {}
        }
    }

    let content_str = text_parts.join("\n");
    let mut msg = serde_json::json!({
        "role": "assistant",
        "content": if content_str.is_empty() { Value::Null } else { Value::String(content_str) },
    });
    if !tool_calls.is_empty() {
        msg["tool_calls"] = Value::Array(tool_calls);
    }
    msg
}

fn serialize_tool_definition(tool: &crate::tools::ToolDefinition) -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    })
}

fn serialize_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::Specific { name } => {
            serde_json::json!({"type": "function", "function": {"name": name}})
        }
    }
}

fn parse_status_str(raw: &str) -> ResponseStatus {
    match raw {
        "stop" => ResponseStatus::EndTurn,
        "tool_calls" => ResponseStatus::ToolUse,
        "length" => ResponseStatus::OutputTruncated,
        "content_filter" => ResponseStatus::Refused,
        _ => ResponseStatus::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolDefinition;

    fn dummy_request() -> ModelRequest {
        ModelRequest {
            model: "test-model".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text {
                    text: "Hello".into(),
                }],
            }],
            tools: vec![],
            max_request_tokens: Some(1024),
            tool_choice: None,
        }
    }

    #[test]
    fn timeout_builder_stores_duration() {
        let p = OpenAiProvider::new("test-key");
        assert_eq!(p.request_timeout(), DEFAULT_REQUEST_TIMEOUT);
        let p = p.timeout(Duration::from_secs(42));
        assert_eq!(p.request_timeout(), Duration::from_secs(42));
    }

    #[test]
    fn serialize_system_prompt_as_message() {
        let body = serialize_request(&dummy_request());
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
    }

    #[test]
    fn serialize_basic_structure() {
        let body = serialize_request(&dummy_request());
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn serialize_request_omits_max_tokens_when_none() {
        let mut req = dummy_request();
        req.max_request_tokens = None;
        let body = serialize_request(&req);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn serialize_tools() {
        let request = ModelRequest {
            model: "test".into(),
            system_prompt: String::new(),
            messages: vec![],
            tools: vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get weather".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            }],
            max_request_tokens: Some(1024),
            tool_choice: Some(ToolChoice::Auto),
        };
        let body = serialize_request(&request);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn serialize_tool_choice_specific() {
        let request = ModelRequest {
            model: "test".into(),
            system_prompt: String::new(),
            messages: vec![],
            tools: vec![],
            max_request_tokens: Some(1024),
            tool_choice: Some(ToolChoice::Specific {
                name: "read_file".into(),
            }),
        };
        let body = serialize_request(&request);
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "read_file");
    }

    fn body_400(code: &str, message: &str) -> String {
        serde_json::json!({
            "error": {
                "type": "invalid_request_error",
                "code": code,
                "message": message,
            }
        })
        .to_string()
    }

    #[test]
    fn context_window_exceeded_by_code() {
        let body = body_400(
            "context_length_exceeded",
            "This model's maximum context length is 128000 tokens.",
        );
        assert!(matches!(
            classify_error(400, &body),
            Some(ProviderError::ContextWindowExceeded { .. })
        ));
    }

    #[test]
    fn context_window_exceeded_by_message_fallback() {
        // Mistral / LiteLLM path — `code` absent, classifier falls back to
        // matching the message text.
        let body = serde_json::json!({
            "error": {
                "type": "invalid_request_error",
                "message":
                    "This model's maximum context length is 32000 tokens, however you requested 40000.",
            }
        })
        .to_string();
        assert!(matches!(
            classify_error(400, &body),
            Some(ProviderError::ContextWindowExceeded { .. })
        ));
    }

    #[test]
    fn maps_400_content_filter_to_safety_filter_triggered() {
        let body = body_400("content_filter", "request blocked by policy");
        assert!(matches!(
            classify_error(400, &body),
            Some(ProviderError::SafetyFilterTriggered { .. })
        ));
    }

    #[test]
    fn maps_400_model_not_found_to_model_not_found() {
        let body = body_400("model_not_found", "the model `gpt-x` does not exist");
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
        let body = body_400("invalid_api_key", "incorrect API key provided");
        assert!(classify_error(400, &body).is_none());
    }

    #[test]
    fn preserves_provider_message() {
        let body = body_400(
            "context_length_exceeded",
            "maximum context length is 8192 tokens; requested 12000",
        );
        let Some(ProviderError::ContextWindowExceeded { message }) = classify_error(400, &body)
        else {
            panic!("expected ContextWindowExceeded");
        };
        assert_eq!(
            message,
            "maximum context length is 8192 tokens; requested 12000"
        );
    }

    #[test]
    fn lookup_gpt_5_family_returns_400k() {
        let lookup = OpenAiProvider::lookup_context_window_size;
        assert_eq!(lookup("gpt-5"), Some(400_000));
        assert_eq!(lookup("gpt-5-mini"), Some(400_000));
    }

    #[test]
    fn lookup_gpt_4_1_returns_1m() {
        let lookup = OpenAiProvider::lookup_context_window_size;
        assert_eq!(lookup("gpt-4.1"), Some(1_000_000));
    }

    #[test]
    fn lookup_o_series_returns_200k() {
        let lookup = OpenAiProvider::lookup_context_window_size;
        assert_eq!(lookup("o3-mini"), Some(200_000));
        assert_eq!(lookup("o1-preview"), Some(200_000));
    }

    #[test]
    fn lookup_gpt_4o_and_turbo_return_128k() {
        let lookup = OpenAiProvider::lookup_context_window_size;
        assert_eq!(lookup("gpt-4o"), Some(128_000));
        assert_eq!(lookup("gpt-4o-mini"), Some(128_000));
        assert_eq!(lookup("gpt-4-turbo-2024-04-09"), Some(128_000));
    }

    #[test]
    fn lookup_legacy_gpt_4_returns_8k() {
        let lookup = OpenAiProvider::lookup_context_window_size;
        assert_eq!(lookup("gpt-4"), Some(8_192));
        assert_eq!(lookup("gpt-4-32k"), Some(32_768));
    }

    #[test]
    fn lookup_gpt_3_5_turbo_returns_16k() {
        let lookup = OpenAiProvider::lookup_context_window_size;
        assert_eq!(lookup("gpt-3.5-turbo"), Some(16_385));
        assert_eq!(lookup("gpt-3.5-turbo-16k"), Some(16_385));
    }

    #[test]
    fn lookup_unknown_models_return_none() {
        let lookup = OpenAiProvider::lookup_context_window_size;
        assert_eq!(lookup("claude-sonnet-4-20250514"), None);
        assert_eq!(lookup("mistral-large-2411"), None);
        assert_eq!(lookup("llama-3-70b"), None);
    }
}
