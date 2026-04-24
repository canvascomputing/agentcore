//! Context-window compaction seam. Emits `ContextCompacted` when a conversation grows too big for the model's window.

use crate::agent::r#loop::{LoopRuntime, LoopState};
use crate::agent::spec::AgentSpec;
use crate::error::Result;
use crate::event::{CompactReason, Event, EventKind};
use crate::provider::types::{ContentBlock, Message};
use crate::provider::Model;

/// Tokens set aside for the model's response. The context window holds
/// input + output combined, so input must leave at least this much room for
/// the next reply. Treated as an upper bound on one response.
const RESERVED_RESPONSE_TOKENS: u64 = 20_000;

/// Headroom reserved below the hard window limit so compaction has room to
/// fire *and* finish before the real overflow. Also absorbs drift in the
/// `bytes / 4` token estimate, which usually under-counts code and JSON.
const COMPACTION_HEADROOM_TOKENS: u64 = 13_000;

/// Token count at which proactive compaction fires for `model`. Returns
/// `None` when the model's context window size is unknown — callers treat
/// that as "no threshold; compaction is dormant".
fn compact_threshold(model: &Model) -> Option<u64> {
    model.context_window_size.map(|size| {
        size.saturating_sub(RESERVED_RESPONSE_TOKENS)
            .saturating_sub(COMPACTION_HEADROOM_TOKENS)
    })
}

/// Estimate of the next request's input-token count: last API response's
/// reported input + cache tokens, plus a ~4-bytes-per-token estimate for
/// any messages appended since.
fn estimate_next_request_tokens(state: &LoopState) -> u64 {
    let prior = state.usage.input_tokens
        + state.usage.cache_read_input_tokens
        + state.usage.cache_creation_input_tokens;
    let new_bytes: usize = state.messages.iter().map(message_bytes).sum();
    prior + (new_bytes / 4) as u64
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

/// Proactive seam: emit `ContextCompacted` when the estimated next-request
/// size crosses the threshold. No-op when the agent's model has no known
/// context window size.
pub(crate) async fn trigger_if_over_threshold(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &LoopState,
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
            turn: state.turns,
            tokens,
            threshold,
            reason: CompactReason::Proactive,
        },
    ));
    Ok(())
}

/// Reactive seam: emit `ContextCompacted` with sentinel token count and
/// threshold of `0`. Fired when the provider itself reports a context-window
/// overflow, either pre-flight or mid-generation.
pub(crate) async fn trigger_reactive(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    turn: u32,
) -> Result<()> {
    (runtime.event_handler)(Event::new(
        spec.name.clone(),
        EventKind::ContextCompacted {
            turn,
            tokens: 0,
            threshold: 0,
            reason: CompactReason::Reactive,
        },
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_bytes_scales_with_content_size() {
        let short = Message::user("hi");
        let long = Message::user("x".repeat(400));
        assert!(message_bytes(&long) > message_bytes(&short));
        assert_eq!(message_bytes(&long), 400);
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
