//! Context-window compaction seam. Decides when a conversation is too big for the model and hands off to the compaction strategy.

use crate::agent::r#loop::{LoopRuntime, LoopState};
use crate::agent::spec::AgentSpec;
use crate::error::Result;
use crate::event::{CompactReason, Event, EventKind};
use crate::provider::types::{ContentBlock, Message};
use crate::provider::Model;

/// Tokens set aside for the model's response. The context window holds
/// input + output combined, so input must leave at least this much room for
/// the next reply. Treated as an upper bound on one response.
pub(crate) const RESERVED_RESPONSE_TOKENS: u64 = 20_000;

/// Headroom reserved below the hard window limit so compaction has room to
/// fire *and* finish before the real overflow. Also absorbs drift in the
/// `bytes / 4` token estimate, which usually under-counts code and JSON.
pub(crate) const COMPACTION_HEADROOM_TOKENS: u64 = 13_000;

/// Token count at which proactive compaction fires for `model`. Returns
/// `None` when the model's context window size is unknown — callers treat
/// that as "no threshold; compaction is dormant".
pub(crate) fn compact_threshold(model: &Model) -> Option<u64> {
    model.context_window_size.map(|size| {
        size.saturating_sub(RESERVED_RESPONSE_TOKENS)
            .saturating_sub(COMPACTION_HEADROOM_TOKENS)
    })
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

/// Proactive seam: emit [`EventKind::ContextCompacted`] and invoke [`run`]
/// when the estimated next-request size crosses the threshold. No-op when
/// the agent's model has no known context window size.
pub(crate) async fn trigger_if_over_threshold(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
) -> Result<()> {
    let Some(threshold) = compact_threshold(spec.model()) else {
        return Ok(());
    };
    let tokens = estimate_next_request_tokens(state);
    if tokens < threshold {
        return Ok(());
    }
    (runtime.event_handler)(Event::new(
        spec.name.clone(),
        EventKind::ContextCompacted {
            turn: state.turn,
            token_count: tokens,
            threshold,
            reason: CompactReason::Proactive,
        },
    ));
    run(runtime, spec, state, CompactReason::Proactive).await
}

/// Reactive seam: emit [`EventKind::ContextCompacted`] (sentinel token
/// count / threshold of `0`) and invoke [`run`]. Fired when the provider
/// itself reports a context-window overflow — either pre-flight or
/// mid-generation.
pub(crate) async fn trigger_reactive(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    turn: u32,
) -> Result<()> {
    (runtime.event_handler)(Event::new(
        spec.name.clone(),
        EventKind::ContextCompacted {
            turn,
            token_count: 0,
            threshold: 0,
            reason: CompactReason::Reactive,
        },
    ));
    run(runtime, spec, state, CompactReason::Reactive).await
}

/// Compact `state.messages` in place. Currently a no-op: the
/// [`EventKind::ContextCompacted`] event fires so observers know the trigger
/// tripped, but the messages are left unchanged.
pub(crate) async fn run(
    _runtime: &LoopRuntime,
    _spec: &AgentSpec,
    _state: &mut LoopState,
    _reason: CompactReason,
) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_scales_with_message_size() {
        let short = vec![Message::user("hi")];
        let long = vec![Message::user("x".repeat(400))];
        assert!(
            estimate_tokens_from_message_bytes(&long) > estimate_tokens_from_message_bytes(&short)
        );
        assert_eq!(estimate_tokens_from_message_bytes(&long), 100);
    }

    #[test]
    fn compact_threshold_200k_model() {
        let m = Model::from_name("unknown").context_window_size(200_000);
        assert_eq!(compact_threshold(&m), Some(167_000));
    }

    #[test]
    fn compact_threshold_saturates_on_tiny_window() {
        let tiny = Model::from_name("unknown").context_window_size(100);
        let zero = Model::from_name("unknown").context_window_size(0);
        assert_eq!(compact_threshold(&tiny), Some(0));
        assert_eq!(compact_threshold(&zero), Some(0));
    }

    #[test]
    fn compact_threshold_none_for_unknown_window() {
        assert_eq!(compact_threshold(&Model::from_name("unknown")), None);
    }
}
