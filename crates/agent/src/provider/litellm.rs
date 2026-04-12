use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;

use super::provider::{CompletionRequest, HttpTransport, LlmProvider, ToolChoice, default_transport};
use super::types::{ContentBlock, Message, ModelResponse, StopReason, TokenUsage};

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
}

impl LlmProvider for LiteLlmProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        let body = serialize_request(&request);
        let url = format!("{}/v1/chat/completions", self.base_url);

        Box::pin(async move {
            let headers = vec![
                ("authorization", format!("Bearer {}", self.api_key)),
                ("content-type", "application/json".into()),
            ];
            let json = (self.transport)(&url, headers, body).await?;
            Ok(parse_response(json, true))
        })
    }
}

fn serialize_request(request: &CompletionRequest) -> Value {
    let messages = serialize_messages(request);
    let tools: Vec<Value> = request.tools.iter().map(serialize_tool_definition).collect();

    let mut body = serde_json::json!({
        "model": request.model,
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

fn parse_response(json: Value, include_cache_tokens: bool) -> ModelResponse {
    let choice = &json["choices"][0];
    let message = &choice["message"];

    ModelResponse {
        content: parse_content(message),
        stop_reason: parse_stop_reason(choice),
        usage: parse_usage(&json, include_cache_tokens),
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

fn parse_usage(json: &Value, include_cache_tokens: bool) -> TokenUsage {
    let usage = &json["usage"];
    TokenUsage {
        input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
        cache_read_input_tokens: if include_cache_tokens {
            usage["cache_read_input_tokens"].as_u64().unwrap_or(0)
        } else {
            0
        },
        cache_creation_input_tokens: if include_cache_tokens {
            usage["cache_creation_input_tokens"].as_u64().unwrap_or(0)
        } else {
            0
        },
    }
}
