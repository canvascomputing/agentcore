//! OpenAI-compatible provider supporting LiteLLM, Mistral, and any
//! API that speaks the OpenAI chat completions format.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::error::{AgenticError, Result};

use super::r#trait::{CompletionRequest, LlmProvider, ToolChoice};
use super::stream::{SseEvent, StreamParser};
use super::types::{ContentBlock, Message, ModelResponse, StopReason, StreamEvent, TokenUsage};

/// OpenAI-compatible LLM provider.
pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    cache_tokens: bool,
}

impl OpenAiProvider {
    pub fn from_api_key(api_key: impl Into<String>) -> Self {
        Self::new_with(api_key, "https://api.openai.com", reqwest::Client::new(), false)
    }

    pub fn new(api_key: impl Into<String>, client: reqwest::Client) -> Self {
        Self::new_with(api_key, "https://api.openai.com", client, false)
    }

    fn new_with(api_key: impl Into<String>, base_url: &str, client: reqwest::Client, cache_tokens: bool) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client,
            cache_tokens,
        }
    }

    pub fn base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    pub fn from_env() -> Result<(Self, String)> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| AgenticError::Other("OPENAI_API_KEY environment variable not set".into()))?;
        let mut provider = Self::from_api_key(key);
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            if !url.is_empty() {
                provider = provider.base_url(url);
            }
        }
        let model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "gpt-4o".into());
        Ok((provider, model))
    }
}

/// Convenience constructors for LiteLLM proxy.
pub struct LiteLlmProvider;

impl LiteLlmProvider {
    pub fn from_api_key(api_key: impl Into<String>) -> OpenAiProvider {
        OpenAiProvider::new_with(api_key, "http://localhost:4000", reqwest::Client::new(), true)
    }

    pub fn new(api_key: impl Into<String>, client: reqwest::Client) -> OpenAiProvider {
        OpenAiProvider::new_with(api_key, "http://localhost:4000", client, true)
    }

    pub fn from_env() -> (OpenAiProvider, String) {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_default();
        let url = std::env::var("LITELLM_API_URL")
            .unwrap_or_else(|_| "http://localhost:4000".into());
        let model = std::env::var("LITELLM_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        let provider = Self::from_api_key(key).base_url(url);
        (provider, model)
    }
}

/// Convenience constructors for Mistral API.
pub struct MistralProvider;

impl MistralProvider {
    pub fn from_api_key(api_key: impl Into<String>) -> OpenAiProvider {
        OpenAiProvider::new_with(api_key, "https://api.mistral.ai", reqwest::Client::new(), false)
    }

    pub fn new(api_key: impl Into<String>, client: reqwest::Client) -> OpenAiProvider {
        OpenAiProvider::new_with(api_key, "https://api.mistral.ai", client, false)
    }

    pub fn from_env() -> Result<(OpenAiProvider, String)> {
        let key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| AgenticError::Other("MISTRAL_API_KEY environment variable not set".into()))?;
        let mut provider = Self::from_api_key(key);
        if let Ok(url) = std::env::var("MISTRAL_BASE_URL") {
            if !url.is_empty() {
                provider = provider.base_url(url);
            }
        }
        let model = std::env::var("MISTRAL_MODEL")
            .unwrap_or_else(|_| "mistral-medium-2508".into());
        Ok((provider, model))
    }
}

impl LlmProvider for OpenAiProvider {
    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async { super::r#trait::prewarm_connection(&self.client, &self.base_url).await })
    }

    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        let body = serialize_request(&request);
        let url = format!("{}/v1/chat/completions", self.base_url);

        Box::pin(async move {
            let json = self.send_json(&url, body).await?;
            Ok(parse_response(json, self.cache_tokens))
        })
    }

    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
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
    async fn send_json(&self, url: &str, body: Value) -> Result<Value> {
        let resp = self.send_raw(url, body).await?;
        resp.json().await.map_err(|e| AgenticError::Other(e.to_string()))
    }

    async fn send_raw(&self, url: &str, body: Value) -> Result<reqwest::Response> {
        let resp = self.client
            .post(url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgenticError::Api {
                message: e.to_string(),
                status: None,
                retryable: true,
                retry_after_ms: None,
            })?;

        super::check_http_error(resp).await
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

async fn stream_response(
    response: reqwest::Response,
    on_event: &Arc<dyn Fn(StreamEvent) + Send + Sync>,
    cache_tokens: bool,
) -> Result<ModelResponse> {
    use futures_util::StreamExt;

    let mut parser = StreamParser::new();
    let mut text = String::new();
    let mut tool_calls: HashMap<usize, ToolCallAccumulator> = HashMap::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut usage = TokenUsage::default();
    let mut model = String::from("unknown");

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| AgenticError::Other(e.to_string()))?;

        for event in parser.push(&chunk) {
            match event {
                SseEvent::Done => {
                    on_event(StreamEvent::MessageDone);
                }
                SseEvent::Data(json) => {
                    if let Some(m) = json["model"].as_str() {
                        model = m.to_string();
                    }

                    let choice = &json["choices"][0];
                    let delta = &choice["delta"];

                    // Text content
                    if let Some(content) = delta["content"].as_str() {
                        if !content.is_empty() {
                            text.push_str(content);
                            on_event(StreamEvent::TextDelta {
                                index: 0,
                                text: content.to_string(),
                            });
                        }
                    }

                    // Tool calls (incremental)
                    if let Some(calls) = delta["tool_calls"].as_array() {
                        for call in calls {
                            let idx = call["index"].as_u64().unwrap_or(0) as usize;
                            let acc = tool_calls.entry(idx).or_insert_with(ToolCallAccumulator::new);

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

                    // Finish reason
                    if let Some(reason) = choice["finish_reason"].as_str() {
                        stop_reason = match reason {
                            "tool_calls" => StopReason::ToolUse,
                            "length" => StopReason::MaxTokens,
                            _ => StopReason::EndTurn,
                        };
                    }

                    // Usage (appears on final chunk when stream_options.include_usage is set)
                    if json.get("usage").is_some() && !json["usage"].is_null() {
                        let u = &json["usage"];
                        usage.input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0);
                        usage.output_tokens = u["completion_tokens"].as_u64().unwrap_or(0);
                        if cache_tokens {
                            usage.cache_read_input_tokens =
                                u["cache_read_input_tokens"].as_u64().unwrap_or(0);
                            usage.cache_creation_input_tokens =
                                u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
                        }
                    }
                }
            }
        }
    }

    // Assemble content blocks
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ContentBlock::Text { text });
    }
    let mut sorted_tools: Vec<_> = tool_calls.into_iter().collect();
    sorted_tools.sort_by_key(|(idx, _)| *idx);
    for (_, acc) in sorted_tools {
        let input = serde_json::from_str(&acc.arguments)
            .unwrap_or(Value::Object(Default::default()));
        content.push(ContentBlock::ToolUse {
            id: acc.id,
            name: acc.name,
            input,
        });
    }

    on_event(StreamEvent::MessageDelta {
        stop_reason: stop_reason.clone(),
        usage: usage.clone(),
    });

    Ok(ModelResponse {
        content,
        stop_reason,
        usage,
        model,
    })
}

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

fn serialize_request(request: &CompletionRequest) -> Value {
    let messages = serialize_messages(request);
    let tools: Vec<Value> = request.tools.iter().map(serialize_tool_definition).collect();

    let mut body = serde_json::json!({
        "model": request.model,
        "messages": messages,
    });

    if request.max_tokens != crate::UNLIMITED {
        body["max_tokens"] = Value::from(request.max_tokens);
    }

    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    if let Some(ref choice) = request.tool_choice {
        body["tool_choice"] = serialize_tool_choice(choice);
    }

    body
}

fn serialize_messages(request: &CompletionRequest) -> Vec<Value> {
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

fn serialize_tool_definition(tool: &crate::tools::tool::ToolDefinition) -> Value {
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

// ---------------------------------------------------------------------------
// Response parsing (non-streaming)
// ---------------------------------------------------------------------------

fn parse_response(json: Value, cache_tokens: bool) -> ModelResponse {
    let choice = &json["choices"][0];
    let message = &choice["message"];

    ModelResponse {
        content: parse_content(message),
        stop_reason: parse_stop_reason(choice),
        usage: parse_usage(&json, cache_tokens),
        model: json["model"].as_str().unwrap_or("unknown").to_string(),
    }
}

fn parse_content(message: &Value) -> Vec<ContentBlock> {
    let mut content = Vec::new();

    if let Some(text) = message["content"].as_str() {
        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }
    }

    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for call in tool_calls {
            let arguments_str = call["function"]["arguments"].as_str().unwrap_or("{}");
            content.push(ContentBlock::ToolUse {
                id: call["id"].as_str().unwrap_or("").to_string(),
                name: call["function"]["name"].as_str().unwrap_or("").to_string(),
                input: serde_json::from_str(arguments_str)
                    .unwrap_or(Value::Object(Default::default())),
            });
        }
    }

    content
}

fn parse_stop_reason(choice: &Value) -> StopReason {
    match choice["finish_reason"].as_str().unwrap_or("stop") {
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

fn parse_usage(json: &Value, cache_tokens: bool) -> TokenUsage {
    let usage = &json["usage"];
    TokenUsage {
        input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
        cache_read_input_tokens: if cache_tokens {
            usage["cache_read_input_tokens"].as_u64().unwrap_or(0)
        } else {
            0
        },
        cache_creation_input_tokens: if cache_tokens {
            usage["cache_creation_input_tokens"].as_u64().unwrap_or(0)
        } else {
            0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tool::ToolDefinition;

    fn dummy_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text { text: "Hello".into() }],
            }],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        }
    }

    // --- Serialization ---

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
    fn serialize_tools() {
        let request = CompletionRequest {
            model: "test".into(),
            system_prompt: String::new(),
            messages: vec![],
            tools: vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get weather".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            }],
            max_tokens: 1024,
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
        let request = CompletionRequest {
            model: "test".into(),
            system_prompt: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: Some(ToolChoice::Specific { name: "read_file".into() }),
        };
        let body = serialize_request(&request);
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "read_file");
    }

    // --- Response parsing ---

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4"
        });
        let resp = parse_response(json, false);
        assert_eq!(resp.content.len(), 1);
        assert!(matches!(&resp.content[0], ContentBlock::Text { text } if text == "Hello!"));
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.model, "gpt-4");
    }

    #[test]
    fn parse_tool_call_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4"
        });
        let resp = parse_response(json, true);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        match &resp.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_abc");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "/tmp");
            }
            other => panic!("Expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn parse_finish_reason_mapping() {
        for (reason, expected) in [
            ("stop", StopReason::EndTurn),
            ("tool_calls", StopReason::ToolUse),
            ("length", StopReason::MaxTokens),
        ] {
            let json = serde_json::json!({
                "choices": [{"message": {"content": "x"}, "finish_reason": reason}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "model": "m"
            });
            assert_eq!(parse_response(json, false).stop_reason, expected, "Failed for {reason}");
        }
    }

    #[test]
    fn parse_null_content() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": null}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "model": "m"
        });
        assert!(parse_response(json, false).content.is_empty());
    }

    #[test]
    fn parse_usage_with_cache_tokens() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 50,
                "cache_read_input_tokens": 20, "cache_creation_input_tokens": 10
            },
            "model": "m"
        });
        let resp = parse_response(json, true);
        assert_eq!(resp.usage.cache_read_input_tokens, 20);
        assert_eq!(resp.usage.cache_creation_input_tokens, 10);
    }

    #[test]
    fn parse_usage_without_cache_tokens() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 50,
                "cache_read_input_tokens": 20, "cache_creation_input_tokens": 10
            },
            "model": "m"
        });
        let resp = parse_response(json, false);
        assert_eq!(resp.usage.cache_read_input_tokens, 0);
        assert_eq!(resp.usage.cache_creation_input_tokens, 0);
    }
}
