use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;

use super::types::{ContentBlock, Message, ModelResponse, StopReason, TokenUsage};
use super::provider::{CompletionRequest, HttpTransport, LlmProvider, ToolChoice, default_transport};

pub struct LiteLlmProvider {
    api_key: String,
    base_url: String,
    transport: HttpTransport,
}

impl LiteLlmProvider {
    pub fn from_api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "http://localhost:4000".into(),
            transport: default_transport(),
        }
    }

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

        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                content.push(ContentBlock::Text {
                    text: text.to_string(),
                });
            }
        }

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

        let usage = TokenUsage {
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
