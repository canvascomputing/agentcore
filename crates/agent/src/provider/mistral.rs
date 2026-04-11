use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;

use super::provider::{CompletionRequest, HttpTransport, LlmProvider, ToolChoice};
use super::types::{ContentBlock, Message, ModelResponse, StopReason, Usage};

pub struct MistralProvider {
    api_key: String,
    base_url: String,
    transport: HttpTransport,
}

impl MistralProvider {
    pub fn new(api_key: String, transport: HttpTransport) -> Self {
        Self {
            api_key,
            base_url: "https://api.mistral.ai".into(),
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

        let usage = Usage {
            input_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0),
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
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

impl LlmProvider for MistralProvider {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tool::ToolDefinition;

    fn dummy_provider() -> MistralProvider {
        let transport: HttpTransport = Box::new(|_, _, _| {
            Box::pin(async { Ok(serde_json::json!({})) })
        });
        MistralProvider::new("test-key".into(), transport)
    }

    #[test]
    fn serialize_basic() {
        let provider = dummy_provider();
        let request = CompletionRequest {
            model: "mistral-large-latest".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::User {
                content: vec![ContentBlock::Text {
                    text: "Hello".into(),
                }],
            }],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = provider.serialize_request(&request);
        assert_eq!(body["model"], "mistral-large-latest");
        assert_eq!(body["max_tokens"], 1024);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello");
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn serialize_tools() {
        let provider = dummy_provider();
        let request = CompletionRequest {
            model: "mistral-large-latest".into(),
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

        let body = provider.serialize_request(&request);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn parse_text_response() {
        let provider = dummy_provider();
        let json = serde_json::json!({
            "choices": [{
                "message": {"content": "Hello there!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "mistral-large-latest"
        });

        let response = provider.parse_response(json).unwrap();
        assert_eq!(response.content.len(), 1);
        assert!(matches!(&response.content[0], ContentBlock::Text { text } if text == "Hello there!"));
        assert!(matches!(response.stop_reason, StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.model, "mistral-large-latest");
    }

    #[test]
    fn parse_tool_call_response() {
        let provider = dummy_provider();
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "model": "mistral-large-latest"
        });

        let response = provider.parse_response(json).unwrap();
        assert_eq!(response.content.len(), 1);
        assert!(matches!(response.stop_reason, StopReason::ToolUse));
        if let ContentBlock::ToolUse { id, name, input } = &response.content[0] {
            assert_eq!(id, "call_123");
            assert_eq!(name, "get_weather");
            assert_eq!(input["city"], "Paris");
        } else {
            panic!("Expected ToolUse content block");
        }
    }

    #[test]
    fn parse_stop_reasons() {
        let provider = dummy_provider();
        let make_json = |reason: &str| {
            serde_json::json!({
                "choices": [{"message": {"content": "x"}, "finish_reason": reason}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": "m"
            })
        };

        assert!(matches!(
            provider.parse_response(make_json("stop")).unwrap().stop_reason,
            StopReason::EndTurn
        ));
        assert!(matches!(
            provider.parse_response(make_json("tool_calls")).unwrap().stop_reason,
            StopReason::ToolUse
        ));
        assert!(matches!(
            provider.parse_response(make_json("length")).unwrap().stop_reason,
            StopReason::MaxTokens
        ));
    }
}
