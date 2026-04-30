//! Provider-agnostic shape of a chat exchange — messages, content blocks, token usage, stop reasons, streaming events. The lingua franca between the loop and every provider.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: Vec<ContentBlock> },
    #[serde(rename = "assistant")]
    Assistant { content: Vec<ContentBlock> },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::System {
            content: text.into(),
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
}

/// Render this value as a user-role `Message`. Implemented by anything
/// that can sensibly become a single user turn — `Ticket` (the loop's
/// task seeding), and future sources like `Comment` or peer-message
/// notifications.
pub trait AsUserMessage {
    fn as_user_message(&self) -> Message;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// What the LLM API reported about why token generation ended.
///
/// Provider-agnostic superset of stop/finish reasons across LLM APIs.
///
/// References:
/// - <https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons>
/// - <https://github.com/BerriAI/litellm/issues/21348>
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Model finished generating naturally.
    /// Anthropic: `end_turn` | OpenAI: `stop` | Mistral: `stop`
    #[default]
    EndTurn,

    /// Model hit a caller-provided stop sequence.
    /// Anthropic: `stop_sequence` | OpenAI: `stop` (with stop param)
    StopSequence,

    /// Model emitted tool_use blocks and expects execution.
    /// Anthropic: `tool_use` | OpenAI: `tool_calls` | Mistral: `tool_calls`
    ToolUse,

    /// Output was truncated by the max_tokens limit.
    /// Anthropic: `max_tokens` | OpenAI: `length` | Mistral: `length`
    OutputTruncated,

    /// Input exceeded the model's context window.
    /// Anthropic: `model_context_window_exceeded`
    ContextWindowExceeded,

    /// Model refused to respond due to safety policy.
    /// Anthropic: `refusal` | OpenAI: `content_filter`
    Refused,

    /// Server-side tool loop hit its iteration limit.
    /// Anthropic: `pause_turn`
    PauseTurn,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
}

impl std::ops::AddAssign<&TokenUsage> for TokenUsage {
    fn add_assign(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: Vec<ContentBlock>,
    pub status: ResponseStatus,
    pub usage: TokenUsage,
    pub model: String,
}

/// Incremental event emitted during SSE streaming.
///
/// Events within a single response arrive in order and reference the content block they
/// belong to via `index`. A stream ends with `MessageDone`.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Appended text for the text block at `index`.
    TextDelta { index: usize, text: String },

    /// Partial JSON fragment for the tool-use block at `index`. Concatenate the fragments
    /// across deltas to reconstruct the full tool input; intermediate chunks are not valid JSON.
    InputJsonDelta { index: usize, partial_json: String },

    /// The content block at `index` is complete; no further deltas will target it.
    ContentBlockStop { index: usize },

    /// Final status and cumulative token usage for the response. Emitted after all content blocks.
    MessageDelta {
        status: ResponseStatus,
        usage: TokenUsage,
    },

    /// End of stream. No further events follow.
    MessageDone,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_serde_round_trip() {
        let msg = Message::User {
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        match deserialized {
            Message::User { content } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    ContentBlock::Text { text } => assert_eq!(text, "hello"),
                    other => panic!("Expected Text, got {other:?}"),
                }
            }
            other => panic!("Expected User, got {other:?}"),
        }
    }

    #[test]
    fn tool_use_block_serde() {
        let block = ContentBlock::ToolUse {
            id: "call_123".into(),
            name: "read_file".into(),
            input: serde_json::json!({"path": "/tmp/test.txt"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));

        let deserialized: ContentBlock = serde_json::from_str(&json).unwrap();
        match deserialized {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "/tmp/test.txt");
            }
            other => panic!("Expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn tool_result_is_error_defaults_false() {
        let json = r#"{"type":"tool_result","tool_use_id":"id1","content":"ok"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult { is_error, .. } => assert!(!is_error),
            other => panic!("Expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn usage_add_accumulates() {
        let mut usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 10,
            cache_creation_input_tokens: 5,
        };
        let other = TokenUsage {
            input_tokens: 200,
            output_tokens: 100,
            cache_read_input_tokens: 20,
            cache_creation_input_tokens: 10,
        };
        usage += &other;
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 150);
        assert_eq!(usage.cache_read_input_tokens, 30);
        assert_eq!(usage.cache_creation_input_tokens, 15);
    }
}
