use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::message::{ContentBlock, Message, ModelResponse, StopReason, Usage};
use crate::tool::ToolDefinition;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Specific { name: String },
}

/// Core LLM provider trait. Object-safe via boxed futures.
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>>;
}

/// Injectable HTTP transport: async fn(url, headers, body) -> response_json
pub type HttpTransport = Box<
    dyn Fn(&str, Vec<(&str, String)>, Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// Anthropic Provider
// ---------------------------------------------------------------------------

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    transport: HttpTransport,
}

impl AnthropicProvider {
    pub fn new(api_key: String, transport: HttpTransport) -> Self {
        Self {
            api_key,
            base_url: "https://api.anthropic.com".into(),
            transport,
        }
    }

    pub fn base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    fn serialize_request(&self, request: &CompletionRequest) -> Value {
        let messages = serialize_messages_anthropic(&request.messages);

        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": request.model,
            "system": request.system_prompt,
            "messages": messages,
            "max_tokens": request.max_tokens,
        });

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }

        if let Some(ref tc) = request.tool_choice {
            body["tool_choice"] = match tc {
                ToolChoice::Auto => serde_json::json!({"type": "auto"}),
                ToolChoice::Specific { name } => {
                    serde_json::json!({"type": "tool", "name": name})
                }
            };
        }

        body
    }

    fn parse_response(&self, json: Value) -> Result<ModelResponse> {
        let content = parse_anthropic_content(&json);

        let stop_reason_str = json["stop_reason"].as_str().unwrap_or("end_turn");
        let stop_reason = match stop_reason_str {
            "end_turn" => StopReason::EndTurn,
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let usage = Usage {
            input_tokens: json["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: json["usage"]["output_tokens"].as_u64().unwrap_or(0),
            cache_read_input_tokens: json["usage"]["cache_read_input_tokens"]
                .as_u64()
                .unwrap_or(0),
            cache_creation_input_tokens: json["usage"]["cache_creation_input_tokens"]
                .as_u64()
                .unwrap_or(0),
        };

        let model = json["model"].as_str().unwrap_or("unknown").to_string();

        Ok(ModelResponse {
            content,
            stop_reason,
            usage,
            model,
        })
    }
}

impl LlmProvider for AnthropicProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        let body = self.serialize_request(&request);
        let url = format!("{}/v1/messages", self.base_url);

        Box::pin(async move {
            let headers = vec![
                ("x-api-key", self.api_key.clone()),
                ("anthropic-version", "2023-06-01".into()),
                ("content-type", "application/json".into()),
            ];
            let response_json = (self.transport)(&url, headers, body).await?;
            self.parse_response(response_json)
        })
    }
}

fn serialize_messages_anthropic(messages: &[Message]) -> Vec<Value> {
    let mut result = Vec::new();
    for msg in messages {
        match msg {
            Message::System { .. } => {} // system prompt handled at top level
            Message::User { content } => {
                result.push(serde_json::json!({
                    "role": "user",
                    "content": serialize_content_blocks(content),
                }));
            }
            Message::Assistant { content } => {
                result.push(serde_json::json!({
                    "role": "assistant",
                    "content": serialize_content_blocks(content),
                }));
            }
        }
    }
    result
}

fn serialize_content_blocks(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => serde_json::json!({"type": "text", "text": text}),
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
        })
        .collect()
}

fn parse_anthropic_content(json: &Value) -> Vec<ContentBlock> {
    let Some(content_arr) = json["content"].as_array() else {
        return Vec::new();
    };
    content_arr
        .iter()
        .filter_map(|block| {
            let block_type = block["type"].as_str()?;
            match block_type {
                "text" => Some(ContentBlock::Text {
                    text: block["text"].as_str().unwrap_or("").to_string(),
                }),
                "tool_use" => Some(ContentBlock::ToolUse {
                    id: block["id"].as_str().unwrap_or("").to_string(),
                    name: block["name"].as_str().unwrap_or("").to_string(),
                    input: block["input"].clone(),
                }),
                _ => None,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LiteLLM / OpenAI-Compatible Provider
// ---------------------------------------------------------------------------

pub struct LiteLlmProvider {
    api_key: String,
    base_url: String,
    transport: HttpTransport,
}

impl LiteLlmProvider {
    pub fn new(api_key: String, transport: HttpTransport) -> Self {
        Self {
            api_key,
            base_url: "http://localhost:4000".into(),
            transport,
        }
    }

    pub fn base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    fn serialize_request(&self, request: &CompletionRequest) -> Value {
        let mut messages = Vec::new();

        // System prompt as first message
        if !request.system_prompt.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": request.system_prompt,
            }));
        }

        // Convert messages
        for msg in &request.messages {
            match msg {
                Message::System { content } => {
                    messages.push(serde_json::json!({"role": "system", "content": content}));
                }
                Message::User { content } => {
                    // Collect tool results as separate messages, text as content
                    let mut text_parts = Vec::new();
                    for block in content {
                        match block {
                            ContentBlock::Text { text } => {
                                text_parts.push(text.clone());
                            }
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
                Message::Assistant { content } => {
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();
                    for block in content {
                        match block {
                            ContentBlock::Text { text } => {
                                text_parts.push(text.clone());
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                tool_calls.push(serde_json::json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": input.to_string(),
                                    }
                                }));
                            }
                            _ => {}
                        }
                    }
                    let mut msg = serde_json::json!({"role": "assistant"});
                    let content_str = text_parts.join("\n");
                    if !content_str.is_empty() {
                        msg["content"] = Value::String(content_str);
                    } else {
                        msg["content"] = Value::Null;
                    }
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = Value::Array(tool_calls);
                    }
                    messages.push(msg);
                }
            }
        }

        // Tool definitions
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        });

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }

        if let Some(ref tc) = request.tool_choice {
            body["tool_choice"] = match tc {
                ToolChoice::Auto => serde_json::json!("auto"),
                ToolChoice::Specific { name } => {
                    serde_json::json!({"type": "function", "function": {"name": name}})
                }
            };
        }

        body
    }

    fn parse_response(&self, json: Value) -> Result<ModelResponse> {
        let choice = &json["choices"][0];
        let message = &choice["message"];

        let mut content = Vec::new();

        // Parse text content
        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                content.push(ContentBlock::Text {
                    text: text.to_string(),
                });
            }
        }

        // Parse tool calls
        if let Some(tool_calls) = message["tool_calls"].as_array() {
            for tc in tool_calls {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let arguments_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let input: Value =
                    serde_json::from_str(arguments_str).unwrap_or(Value::Object(Default::default()));
                content.push(ContentBlock::ToolUse { id, name, input });
            }
        }

        let finish_reason = choice["finish_reason"].as_str().unwrap_or("stop");
        let stop_reason = match finish_reason {
            "stop" => StopReason::EndTurn,
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let usage = Usage {
            input_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0),
            cache_read_input_tokens: json["usage"]["cache_read_input_tokens"]
                .as_u64()
                .unwrap_or(0),
            cache_creation_input_tokens: json["usage"]["cache_creation_input_tokens"]
                .as_u64()
                .unwrap_or(0),
        };

        let model = json["model"].as_str().unwrap_or("unknown").to_string();

        Ok(ModelResponse {
            content,
            stop_reason,
            usage,
            model,
        })
    }
}

impl LlmProvider for LiteLlmProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        let body = self.serialize_request(&request);
        let url = format!("{}/v1/chat/completions", self.base_url);

        Box::pin(async move {
            let headers = vec![
                ("authorization", format!("Bearer {}", self.api_key)),
                ("content-type", "application/json".into()),
            ];
            let response_json = (self.transport)(&url, headers, body).await?;
            self.parse_response(response_json)
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AgenticError;
    use std::sync::{Arc, Mutex};

    fn capture_transport(response: Value) -> (HttpTransport, Arc<Mutex<Vec<Value>>>) {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        let response = Arc::new(response);
        let transport: HttpTransport = Box::new(move |_url, _headers, body| {
            captured_clone.lock().unwrap().push(body.clone());
            let resp = (*response).clone();
            Box::pin(async move { Ok(resp) })
        });
        (transport, captured)
    }

    fn anthropic_response() -> Value {
        serde_json::json!({
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-sonnet-4-20250514"
        })
    }

    fn litellm_response(finish_reason: &str) -> Value {
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": null
                },
                "finish_reason": finish_reason
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4"
        })
    }

    fn simple_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text {
                    text: "Hi".into(),
                }],
            }],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        }
    }

    #[tokio::test]
    async fn anthropic_serializes_system_prompt_as_top_level() {
        let (transport, captured) = capture_transport(anthropic_response());
        let provider = AnthropicProvider::new("test-key".into(), transport);
        provider.complete(simple_request()).await.unwrap();

        let body = &captured.lock().unwrap()[0];
        assert_eq!(body["system"], "You are helpful.");
        // System prompt should NOT appear in messages
        let messages = body["messages"].as_array().unwrap();
        for msg in messages {
            assert_ne!(msg["role"], "system");
        }
    }

    #[tokio::test]
    async fn litellm_serializes_system_prompt_as_message() {
        let (transport, captured) = capture_transport(litellm_response("stop"));
        let provider = LiteLlmProvider::new("test-key".into(), transport);
        provider.complete(simple_request()).await.unwrap();

        let body = &captured.lock().unwrap()[0];
        assert!(body.get("system").is_none() || body["system"].is_null());
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
    }

    #[tokio::test]
    async fn litellm_parses_tool_calls_from_response() {
        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"/tmp/test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4"
        });
        let (transport, _) = capture_transport(response);
        let provider = LiteLlmProvider::new("key".into(), transport);
        let resp = provider.complete(simple_request()).await.unwrap();

        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_abc");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "/tmp/test.txt");
            }
            other => panic!("Expected ToolUse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn litellm_finish_reason_mapping() {
        let cases = vec![
            ("stop", StopReason::EndTurn),
            ("tool_calls", StopReason::ToolUse),
            ("length", StopReason::MaxTokens),
        ];
        for (reason, expected) in cases {
            let (transport, _) = capture_transport(litellm_response(reason));
            let provider = LiteLlmProvider::new("key".into(), transport);
            let resp = provider.complete(simple_request()).await.unwrap();
            assert_eq!(resp.stop_reason, expected, "Failed for finish_reason={reason}");
        }
    }

    #[tokio::test]
    async fn transport_error_propagated() {
        let transport: HttpTransport = Box::new(|_url, _headers, _body| {
            Box::pin(async { Err(AgenticError::Other("connection refused".into())) })
        });
        let provider = AnthropicProvider::new("key".into(), transport);
        let err = provider.complete(simple_request()).await.unwrap_err();
        assert!(format!("{err}").contains("connection refused"));
    }

    #[tokio::test]
    async fn anthropic_empty_content_array() {
        let response = serde_json::json!({
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": "claude-sonnet-4-20250514"
        });
        let (transport, _) = capture_transport(response);
        let provider = AnthropicProvider::new("key".into(), transport);
        let resp = provider.complete(simple_request()).await.unwrap();
        assert!(resp.content.is_empty());
    }

    #[tokio::test]
    async fn litellm_null_content_field() {
        let response = serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": null},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "model": "gpt-4"
        });
        let (transport, _) = capture_transport(response);
        let provider = LiteLlmProvider::new("key".into(), transport);
        let resp = provider.complete(simple_request()).await.unwrap();
        // null content → no text blocks
        assert!(resp.content.is_empty());
    }

    #[tokio::test]
    async fn anthropic_tool_choice_serialization() {
        let (transport, captured) = capture_transport(anthropic_response());
        let provider = AnthropicProvider::new("key".into(), transport);

        let mut req = simple_request();
        req.tool_choice = Some(ToolChoice::Specific {
            name: "read_file".into(),
        });
        provider.complete(req).await.unwrap();

        let body = &captured.lock().unwrap()[0];
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "read_file");
    }

    #[tokio::test]
    async fn litellm_tool_choice_serialization() {
        let (transport, captured) = capture_transport(litellm_response("stop"));
        let provider = LiteLlmProvider::new("key".into(), transport);

        let mut req = simple_request();
        req.tool_choice = Some(ToolChoice::Specific {
            name: "read_file".into(),
        });
        provider.complete(req).await.unwrap();

        let body = &captured.lock().unwrap()[0];
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "read_file");
    }
}
