use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::error::{AgenticError, Result};

use super::types::{ContentBlock, Message, ModelResponse, StopReason, StreamEvent, TokenUsage};
use super::r#trait::{CompletionRequest, LlmProvider, ToolChoice};

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn from_api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<(Self, String)> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| AgenticError::Other("ANTHROPIC_API_KEY environment variable not set".into()))?;
        let mut provider = Self::from_api_key(key);
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            if !url.is_empty() {
                provider = provider.base_url(url);
            }
        }
        let model = std::env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        Ok((provider, model))
    }

    pub fn new(api_key: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".into(),
            client,
        }
    }

    pub fn base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    fn api_headers(&self) -> Vec<(String, String)> {
        vec![
            ("x-api-key".into(), self.api_key.clone()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("content-type".into(), "application/json".into()),
        ]
    }

    pub(crate) fn serialize_request(&self, request: &CompletionRequest) -> Value {
        let messages = serialize_messages(&request.messages);
        let tools: Vec<Value> = request.tools.iter().map(serialize_tool_definition).collect();

        let mut body = serde_json::json!({
            "model": request.model,
            "system": request.system_prompt,
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

    pub(crate) fn parse_response(&self, json: Value) -> Result<ModelResponse> {
        Ok(ModelResponse {
            content: parse_content(&json),
            stop_reason: parse_stop_reason(&json),
            usage: parse_usage(&json),
            model: json["model"].as_str().unwrap_or("unknown").to_string(),
        })
    }

    async fn send_request(&self, body: Value) -> Result<reqwest::Response> {
        let url = format!("{}/v1/messages", self.base_url);
        let mut req = self.client.post(&url).json(&body);
        for (k, v) in self.api_headers() {
            req = req.header(k, v);
        }
        let resp = req.send().await.map_err(|e| AgenticError::Api {
            message: e.to_string(),
            status: None,
            retryable: true,
            retry_after_ms: None,
        })?;

        super::check_http_error(resp).await
    }
}

impl LlmProvider for AnthropicProvider {
    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async { super::r#trait::prewarm_connection(&self.client, &self.base_url).await })
    }

    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        let body = self.serialize_request(&request);

        Box::pin(async move {
            let resp = self.send_request(body).await?;
            let json: Value = resp.json().await.map_err(|e| AgenticError::Other(e.to_string()))?;
            self.parse_response(json)
        })
    }

    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
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
) -> Result<ModelResponse> {
    use futures_util::StreamExt;
    use super::stream::{SseEvent, StreamParser};

    let mut parser = StreamParser::new();
    let mut model = String::from("unknown");
    let mut usage = TokenUsage::default();
    let mut stop_reason = StopReason::EndTurn;
    let mut content_blocks: Vec<ContentBlock> = Vec::new();

    // Per-block accumulators, keyed by content block index
    let mut texts: Vec<Option<String>> = Vec::new();
    let mut tool_ids: Vec<Option<String>> = Vec::new();
    let mut tool_names: Vec<Option<String>> = Vec::new();
    let mut tool_inputs: Vec<Option<String>> = Vec::new();

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| AgenticError::Other(e.to_string()))?;

        for event in parser.push(&chunk) {
            match event {
                SseEvent::Done => {
                    on_event(StreamEvent::MessageDone);
                    continue;
                }
                SseEvent::Data(json) => {

            let event_type = json["type"].as_str().unwrap_or("");
            match event_type {
                "message_start" => {
                    model = json["message"]["model"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string();
                    usage.input_tokens =
                        json["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0);
                    usage.cache_read_input_tokens = json["message"]["usage"]
                        ["cache_read_input_tokens"]
                        .as_u64()
                        .unwrap_or(0);
                    usage.cache_creation_input_tokens = json["message"]["usage"]
                        ["cache_creation_input_tokens"]
                        .as_u64()
                        .unwrap_or(0);
                }
                "content_block_start" => {
                    let idx = json["index"].as_u64().unwrap_or(0) as usize;
                    let block = &json["content_block"];

                    // Grow accumulators to fit this index
                    while texts.len() <= idx {
                        texts.push(None);
                        tool_ids.push(None);
                        tool_names.push(None);
                        tool_inputs.push(None);
                    }

                    match block["type"].as_str().unwrap_or("") {
                        "tool_use" => {
                            tool_ids[idx] =
                                Some(block["id"].as_str().unwrap_or("").to_string());
                            tool_names[idx] =
                                Some(block["name"].as_str().unwrap_or("").to_string());
                            tool_inputs[idx] = Some(String::new());
                        }
                        _ => {
                            texts[idx] = Some(String::new());
                        }
                    }
                }
                "content_block_delta" => {
                    let idx = json["index"].as_u64().unwrap_or(0) as usize;
                    let delta = &json["delta"];

                    match delta["type"].as_str().unwrap_or("") {
                        "text_delta" => {
                            let text = delta["text"].as_str().unwrap_or("").to_string();
                            if let Some(Some(ref mut buf)) = texts.get_mut(idx) {
                                buf.push_str(&text);
                            }
                            on_event(StreamEvent::TextDelta { index: idx, text });
                        }
                        "input_json_delta" => {
                            let partial = delta["partial_json"].as_str().unwrap_or("");
                            if let Some(Some(ref mut buf)) = tool_inputs.get_mut(idx) {
                                buf.push_str(partial);
                            }
                            on_event(StreamEvent::InputJsonDelta {
                                index: idx,
                                partial_json: partial.to_string(),
                            });
                        }
                        _ => {}
                    }
                }
                "content_block_stop" => {
                    let idx = json["index"].as_u64().unwrap_or(0) as usize;

                    if let Some(Some(text)) = texts.get_mut(idx).map(|t| t.take()) {
                        content_blocks.push(ContentBlock::Text { text });
                    } else if let Some(Some(json_input)) =
                        tool_inputs.get_mut(idx).map(|t| t.take())
                    {
                        let input = serde_json::from_str(&json_input)
                            .unwrap_or(Value::Object(Default::default()));
                        content_blocks.push(ContentBlock::ToolUse {
                            id: tool_ids
                                .get_mut(idx)
                                .and_then(|t| t.take())
                                .unwrap_or_default(),
                            name: tool_names
                                .get_mut(idx)
                                .and_then(|t| t.take())
                                .unwrap_or_default(),
                            input,
                        });
                    }

                    on_event(StreamEvent::ContentBlockStop { index: idx });
                }
                "message_delta" => {
                    stop_reason = match json["delta"]["stop_reason"].as_str() {
                        Some("tool_use") => StopReason::ToolUse,
                        Some("max_tokens") => StopReason::MaxTokens,
                        _ => StopReason::EndTurn,
                    };
                    usage.output_tokens =
                        json["usage"]["output_tokens"].as_u64().unwrap_or(0);
                    on_event(StreamEvent::MessageDelta {
                        stop_reason: stop_reason.clone(),
                        usage: usage.clone(),
                    });
                }
                _ => {}
            }

                } // SseEvent::Data
            } // match event
        } // for event
    } // while chunk

    Ok(ModelResponse {
        content: content_blocks,
        stop_reason,
        usage,
        model,
    })
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

fn serialize_tool_definition(tool: &crate::tools::tool::ToolDefinition) -> Value {
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

fn parse_stop_reason(json: &Value) -> StopReason {
    match json["stop_reason"].as_str().unwrap_or("end_turn") {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

fn parse_usage(json: &Value) -> TokenUsage {
    let usage = &json["usage"];
    TokenUsage {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
        cache_read_input_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        cache_creation_input_tokens: usage["cache_creation_input_tokens"]
            .as_u64()
            .unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text { text: "Hi".into() }],
            }],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        }
    }

    fn provider() -> AnthropicProvider {
        AnthropicProvider::from_api_key("test-key")
    }

    #[test]
    fn serialize_request_sets_model_and_system() {
        let body = provider().serialize_request(&simple_request());
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["max_tokens"], 1024);
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
        req.tool_choice = Some(ToolChoice::Specific { name: "read_file".into() });
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
        let resp = provider().parse_response(json).unwrap();
        assert_eq!(resp.content.len(), 1);
        assert!(matches!(&resp.content[0], ContentBlock::Text { text } if text == "Hello!"));
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
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
        let resp = provider().parse_response(json).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
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
        let resp = provider().parse_response(json).unwrap();
        assert!(resp.content.is_empty());
    }

    #[test]
    fn parse_response_maps_stop_reasons() {
        for (reason, expected) in [
            ("end_turn", StopReason::EndTurn),
            ("tool_use", StopReason::ToolUse),
            ("max_tokens", StopReason::MaxTokens),
        ] {
            let json = serde_json::json!({
                "content": [], "stop_reason": reason,
                "usage": {"input_tokens": 0, "output_tokens": 0}, "model": "m"
            });
            assert_eq!(provider().parse_response(json).unwrap().stop_reason, expected);
        }
    }
}
