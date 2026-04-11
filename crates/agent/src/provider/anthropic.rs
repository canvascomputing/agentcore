use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;

use super::types::{ContentBlock, Message, ModelResponse, StopReason, TokenUsage};
use super::provider::{CompletionRequest, HttpTransport, LlmProvider, ToolChoice, default_transport};

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    transport: HttpTransport,
}

impl AnthropicProvider {
    pub fn from_api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".into(),
            transport: default_transport(),
        }
    }

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

        let usage = TokenUsage {
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
            Message::System { .. } => {}
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
