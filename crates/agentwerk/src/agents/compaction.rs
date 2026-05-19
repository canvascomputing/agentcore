//! Context-window compaction: collapse an over-long transcript into
//! a single summary message before the next request would overflow.

use std::sync::Arc;

use crate::prompts::compaction_directive;
use crate::providers::types::StreamEvent;
use crate::providers::{
    ContentBlock, Message, ModelRequest, Provider, ProviderError, ProviderResult,
    ProviderToolDefinition, TokenUsage,
};

/// Tokens reserved for the model's response. The context window holds
/// input + output combined, so the input must leave at least this much
/// room for the next reply.
const RESERVED_RESPONSE_TOKENS: u64 = 20_000;

/// Headroom below the hard window limit so the warning fires with room
/// to spare. Also absorbs drift in the `bytes / 4` token estimate, which
/// tends to under-count code and JSON.
const COMPACTION_HEADROOM_TOKENS: u64 = 13_000;

/// Layer 2 (blocking) headroom: a much tighter line than the proactive
/// threshold. When the estimate crosses `window - BLOCKING_HEADROOM_TOKENS`
/// the loop synthesizes a `ContextWindowExceeded` before the provider
/// call goes out.
pub(crate) const BLOCKING_HEADROOM_TOKENS: u64 = 3_000;

/// Token count at which the proactive seam fires for a model with
/// context window `window`. `None` when the window is unknown.
pub(crate) fn threshold(window: Option<u64>) -> Option<u64> {
    window.map(|size| {
        size.saturating_sub(RESERVED_RESPONSE_TOKENS)
            .saturating_sub(COMPACTION_HEADROOM_TOKENS)
    })
}

/// Token count at which the Layer 2 blocking guard fires. Sits much
/// closer to the actual window than [`threshold`], so it only trips
/// when proactive compaction has already run (or could not).
pub(crate) fn blocking_threshold(window: Option<u64>) -> Option<u64> {
    window.map(|size| size.saturating_sub(BLOCKING_HEADROOM_TOKENS))
}

/// Estimate of the next request's input-token count: the last response's
/// reported input tokens plus a `bytes / 4` estimate over the full
/// request body the provider will see: every message in the current
/// vector, the system prompt, and every tool definition. Sums *all*
/// messages on purpose: this overcounts after the first iteration but
/// the resulting conservatism keeps the proactive seam ahead of the
/// real overflow.
pub(crate) fn estimate_next_request_tokens(
    last_usage: &TokenUsage,
    messages: &[Message],
    system_prompt: &str,
    tools: &[ProviderToolDefinition],
) -> u64 {
    let mut bytes: usize = messages.iter().map(message_bytes).sum();
    bytes += system_prompt.len();
    bytes += tools
        .iter()
        .map(|t| t.name.len() + t.description.len() + t.input_schema.to_string().len())
        .sum::<usize>();
    last_usage.input_tokens + (bytes / 4) as u64
}

fn message_bytes(message: &Message) -> usize {
    match message {
        Message::System { content } => content.len(),
        Message::User { content } | Message::Assistant { content } => {
            content.iter().map(block_bytes).sum()
        }
    }
}

fn block_bytes(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text { text } => text.len(),
        ContentBlock::ToolUse { name, input, .. } => name.len() + input.to_string().len(),
        ContentBlock::ToolResult { content, .. } => content.len(),
    }
}

/// `true` when the estimated next-request input crosses the
/// proactive compaction threshold. `false` when the window is unknown
/// or the estimate is still under it.
pub(crate) fn should_compact_proactively(
    window: Option<u64>,
    last_usage: &TokenUsage,
    messages: &[Message],
    system_prompt: &str,
    tools: &[ProviderToolDefinition],
) -> bool {
    let Some(threshold) = threshold(window) else {
        return false;
    };
    estimate_next_request_tokens(last_usage, messages, system_prompt, tools) >= threshold
}

/// Compact `messages` into a single summary line by sending them to
/// the provider with no tools and the compaction directive as the
/// system prompt. Returns `Ok(None)` when `messages` is too short
/// (zero or one entry) to be worth summarising; the caller treats
/// that as a no-op.
pub(crate) async fn compact(
    provider: &Arc<dyn Provider>,
    model: &str,
    messages: &[Message],
) -> ProviderResult<Option<String>> {
    if messages.len() <= 1 {
        return Ok(None);
    }
    let request = ModelRequest {
        model: model.to_string(),
        system_prompt: compaction_directive().to_string(),
        messages: messages.to_vec(),
        tools: Vec::new(),
        max_request_tokens: None,
        tool_choice: None,
    };
    let on_stream: Arc<dyn Fn(StreamEvent) + Send + Sync> = Arc::new(|_| {});
    let response = provider.respond(request, on_stream).await?;
    let summary = response
        .content
        .iter()
        .find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .ok_or_else(|| ProviderError::ResponseMalformed {
            message: "compaction reply contained no text".into(),
        })?;
    Ok(Some(summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_saturates_on_tiny_or_zero_window() {
        assert_eq!(threshold(Some(100)), Some(0));
        assert_eq!(threshold(Some(0)), Some(0));
    }

    #[test]
    fn threshold_is_none_for_unknown_window() {
        assert_eq!(threshold(None), None);
    }

    #[test]
    fn estimate_sums_last_input_tokens_and_byte_quarters() {
        // 400 bytes / 4 = 100; plus last response's 5_000 input tokens = 5_100.
        let usage = TokenUsage {
            input_tokens: 5_000,
            output_tokens: 200,
        };
        let messages = [Message::user("x".repeat(400))];
        assert_eq!(
            estimate_next_request_tokens(&usage, &messages, "", &[]),
            5_100,
        );
    }

    #[test]
    fn estimate_includes_system_prompt_and_tool_definitions() {
        // bytes = system_prompt + tool(name+description+schema) + message
        //       = 100 + (3 + 50 + "{}".len()) + 4 = 159
        // estimate = 0 + 159/4 = 39
        let usage = TokenUsage::default();
        let messages = [Message::user("hi!!")];
        let tools = vec![ProviderToolDefinition {
            name: "tot".into(),
            description: "x".repeat(50),
            input_schema: serde_json::json!({}),
        }];
        let system_prompt = "x".repeat(100);
        let got = estimate_next_request_tokens(&usage, &messages, &system_prompt, &tools);
        assert_eq!(got, 39);
    }

    #[test]
    fn should_compact_proactively_is_false_when_window_unknown() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        assert!(!should_compact_proactively(None, &usage, &messages, "", &[]));
    }

    #[test]
    fn should_compact_proactively_is_false_when_under_threshold() {
        let usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        assert!(!should_compact_proactively(
            Some(200_000),
            &usage,
            &messages,
            "",
            &[],
        ));
    }

    #[test]
    fn should_compact_proactively_is_true_when_estimate_crosses_threshold() {
        // Threshold = 200_000 - 20_000 - 13_000 = 167_000; estimate is
        // last_usage's 170_000 input tokens plus a trivial message.
        let usage = TokenUsage {
            input_tokens: 170_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        assert!(should_compact_proactively(
            Some(200_000),
            &usage,
            &messages,
            "",
            &[],
        ));
    }

    #[test]
    fn blocking_threshold_is_window_minus_3k() {
        assert_eq!(blocking_threshold(Some(200_000)), Some(197_000));
        assert_eq!(blocking_threshold(Some(2_000)), Some(0));
        assert_eq!(blocking_threshold(None), None);
    }

    // ---- compact ----

    use crate::providers::types::{ModelResponse, ResponseStatus};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex as StdMutex;

    /// Scripted provider: serves one canned result per `respond` call
    /// in FIFO order, and records the request it received so tests
    /// can assert on it.
    struct ScriptedProvider {
        results: StdMutex<Vec<ProviderResult<ModelResponse>>>,
        received: StdMutex<Vec<ModelRequest>>,
    }

    impl ScriptedProvider {
        fn new(results: Vec<ProviderResult<ModelResponse>>) -> Arc<Self> {
            Arc::new(Self {
                results: StdMutex::new(results),
                received: StdMutex::new(Vec::new()),
            })
        }

        fn last_request(&self) -> Option<ModelRequest> {
            self.received.lock().unwrap().last().cloned()
        }

        fn call_count(&self) -> usize {
            self.received.lock().unwrap().len()
        }
    }

    impl Provider for ScriptedProvider {
        fn respond(
            &self,
            request: ModelRequest,
            _on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
            self.received.lock().unwrap().push(request);
            let mut results = self.results.lock().unwrap();
            if results.is_empty() {
                panic!("ScriptedProvider out of scripted results");
            }
            let next = results.remove(0);
            Box::pin(async move { next })
        }
    }

    fn summary_response(text: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            status: ResponseStatus::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    #[tokio::test]
    async fn compact_returns_the_provider_summary() {
        let provider: Arc<dyn Provider> =
            ScriptedProvider::new(vec![Ok(summary_response("SUMMARY"))]);
        let messages = vec![
            Message::user("task"),
            Message::assistant("turn 0"),
            Message::user("turn 1 result"),
        ];

        let summary = compact(&provider, "mock", &messages)
            .await
            .expect("compact should succeed");

        assert_eq!(summary.as_deref(), Some("SUMMARY"));
    }

    #[tokio::test]
    async fn compact_is_a_noop_when_messages_are_too_short() {
        for len in [0, 1] {
            let provider = ScriptedProvider::new(Vec::new());
            let provider_handle: Arc<dyn Provider> = provider.clone();
            let messages: Vec<Message> = (0..len).map(|i| Message::user(format!("m{i}"))).collect();

            let summary = compact(&provider_handle, "mock", &messages)
                .await
                .expect("no-op should succeed");

            assert!(summary.is_none(), "len={len}: must short-circuit");
            assert_eq!(
                provider.call_count(),
                0,
                "len={len}: provider must not be called"
            );
        }
    }

    #[tokio::test]
    async fn compact_propagates_provider_error() {
        let provider: Arc<dyn Provider> =
            ScriptedProvider::new(vec![Err(ProviderError::ConnectionFailed {
                message: "dns".into(),
            })]);
        let messages = vec![Message::user("task"), Message::assistant("turn 0")];

        let err = compact(&provider, "mock", &messages)
            .await
            .expect_err("should propagate the connection failure");

        assert!(matches!(err, ProviderError::ConnectionFailed { .. }));
    }

    #[tokio::test]
    async fn compact_rejects_text_less_reply() {
        let no_text = ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: "x".into(),
                name: "irrelevant".into(),
                input: serde_json::json!({}),
            }],
            status: ResponseStatus::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
        };
        let provider: Arc<dyn Provider> = ScriptedProvider::new(vec![Ok(no_text)]);
        let messages = vec![Message::user("task"), Message::assistant("turn 0")];

        let err = compact(&provider, "mock", &messages)
            .await
            .expect_err("text-less reply must fail");

        assert!(matches!(err, ProviderError::ResponseMalformed { .. }));
    }

    #[tokio::test]
    async fn compact_builds_a_tool_less_request() {
        let provider = ScriptedProvider::new(vec![Ok(summary_response("SUMMARY"))]);
        let provider_handle: Arc<dyn Provider> = provider.clone();
        let messages = vec![
            Message::user("task"),
            Message::assistant("turn 0"),
            Message::user("turn 1 result"),
        ];

        compact(&provider_handle, "mock", &messages).await.unwrap();

        let req = provider.last_request().expect("provider was called");
        assert!(req.tools.is_empty(), "tools must be disabled");
        assert!(req.tool_choice.is_none(), "tool_choice must be unset");
        assert_eq!(req.messages.len(), messages.len());
        assert_eq!(req.system_prompt, compaction_directive());
    }
}
