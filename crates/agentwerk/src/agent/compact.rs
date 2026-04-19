//! Context-window compaction seam. `run()` is a stub;
//! `run_if_over_threshold()` is the live trigger wired into the agent loop.

use crate::agent::event::{Event, EventKind};
use crate::agent::werk::{AgentSpec, LoopState, Runtime};
use crate::error::{AgenticError, Result};
use crate::provider::types::{ContentBlock, Message};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactReason {
    Proactive,
    Reactive,
}

/// Tokens set aside for the model's response. The context window holds
/// input + output combined, so input must leave at least this much room
/// for the next reply. Treated as an upper bound on one response.
pub const RESERVED_RESPONSE_TOKENS: u64 = 20_000;

/// Headroom reserved below the hard window limit so compaction has room to
/// fire *and* finish before the real overflow. Also absorbs drift in the
/// `bytes / 4` token estimate, which usually under-counts code and JSON.
pub const COMPACTION_HEADROOM_TOKENS: u64 = 13_000;

/// Token count at which proactive compaction fires, given a context window size.
pub fn threshold_for_context_window_size(context_window_size: u64) -> u64 {
    context_window_size
        .saturating_sub(RESERVED_RESPONSE_TOKENS)
        .saturating_sub(COMPACTION_HEADROOM_TOKENS)
}

/// Estimate of the next request's input-token count: last API response's
/// reported input + cache tokens, plus a ~4-bytes-per-token estimate for
/// any messages appended since.
pub(crate) fn estimate_next_request_tokens(state: &LoopState) -> u64 {
    input_tokens_from_last_response(state) + estimate_tokens_from_message_bytes(&state.messages)
}

fn input_tokens_from_last_response(state: &LoopState) -> u64 {
    state.total_usage.input_tokens
        + state.total_usage.cache_read_input_tokens
        + state.total_usage.cache_creation_input_tokens
}

fn estimate_tokens_from_message_bytes(messages: &[Message]) -> u64 {
    (messages.iter().map(text_bytes_in_message).sum::<usize>() / 4) as u64
}

fn text_bytes_in_message(message: &Message) -> usize {
    match message {
        Message::System { content } => content.len(),
        Message::User { content } | Message::Assistant { content } => {
            content.iter().map(text_bytes_in_content_block).sum()
        }
    }
}

fn text_bytes_in_content_block(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text { text } => text.len(),
        ContentBlock::ToolUse { name, input, .. } => name.len() + input.to_string().len(),
        ContentBlock::ToolResult { content, .. } => content.len(),
    }
}

/// Invoke [`run`] when the estimated next-request size crosses the
/// threshold. No-op when the agent's model has no known context window size.
pub(crate) async fn run_if_over_threshold(
    runtime: &Runtime,
    spec: &AgentSpec,
    state: &mut LoopState,
) -> Result<()> {
    let Some(window_size) = spec.model.context_window_size else {
        return Ok(());
    };
    let threshold = threshold_for_context_window_size(window_size);
    let tokens = estimate_next_request_tokens(state);
    if tokens < threshold {
        return Ok(());
    }
    (runtime.event_handler)(Event::new(
        spec.name.clone(),
        EventKind::CompactTriggered {
            turn: state.turn,
            token_count: tokens,
            threshold,
            reason: CompactReason::Proactive,
        },
    ));
    run(runtime, spec, state, CompactReason::Proactive).await
}

/// Compact `state.messages` in place. Not yet implemented — returns
/// `AgenticError::NotImplemented` so callers see the trigger fired.
pub(crate) async fn run(
    _runtime: &Runtime,
    _spec: &AgentSpec,
    _state: &mut LoopState,
    _reason: CompactReason,
) -> Result<()> {
    Err(AgenticError::NotImplemented("context compaction"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_200k_model() {
        assert_eq!(threshold_for_context_window_size(200_000), 167_000);
    }

    #[test]
    fn threshold_saturates_on_tiny_window() {
        assert_eq!(threshold_for_context_window_size(100), 0);
        assert_eq!(threshold_for_context_window_size(0), 0);
    }

    #[test]
    fn estimate_scales_with_message_size() {
        let short = vec![Message::user("hi")];
        let long = vec![Message::user("x".repeat(400))];
        assert!(
            estimate_tokens_from_message_bytes(&long)
                > estimate_tokens_from_message_bytes(&short)
        );
        assert_eq!(estimate_tokens_from_message_bytes(&long), 100);
    }
}
