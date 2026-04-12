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
        let messages = serialize_messages(&request.messages);
        let tools: Vec<Value> = request.tools.iter().map(serialize_tool_definition).collect();

        let mut body = serde_json::json!({
            "model": request.model,
            "system": request.system_prompt,
            "messages": messages,
            "max_tokens": request.max_tokens,
        });

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        if let Some(ref choice) = request.tool_choice {
            body["tool_choice"] = serialize_tool_choice(choice);
        }

        body
    }

    fn parse_response(&self, json: Value) -> Result<ModelResponse> {
        Ok(ModelResponse {
            content: parse_content(&json),
            stop_reason: parse_stop_reason(&json),
            usage: parse_usage(&json),
            model: json["model"].as_str().unwrap_or("unknown").to_string(),
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
        .filter_map(|block| {
            match block["type"].as_str()? {
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
