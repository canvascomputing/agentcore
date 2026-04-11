use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AgenticError, Result};

use super::types::{Message, ModelResponse};
use crate::tools::tool::ToolDefinition;

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

/// Default transport using reqwest.
pub fn default_transport() -> HttpTransport {
    Box::new(|url, headers, body| {
        let url = url.to_string();
        let headers: Vec<(String, String)> = headers
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Box::pin(async move {
            let client = reqwest::Client::new();
            let mut req = client.post(&url).json(&body);
            for (k, v) in &headers {
                req = req.header(k.as_str(), v.as_str());
            }
            let resp = req.send().await.map_err(|e| AgenticError::Other(e.to_string()))?;
            resp.json().await.map_err(|e| AgenticError::Other(e.to_string()))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AgenticError;
    use crate::provider::{AnthropicProvider, LiteLlmProvider};
    use super::super::types::{ContentBlock, StopReason};
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
