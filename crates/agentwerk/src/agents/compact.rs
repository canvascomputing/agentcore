//! Context-window observation seam. Emits `ContextCompacted` when a
//! conversation grows close to the model's context window (proactive)
//! or the provider reports it has overflowed (reactive). The seam never
//! mutates messages: the event is the contract.

use crate::event::{CompactReason, EventKind};
use crate::providers::{ContentBlock, Message, TokenUsage};

/// Tokens reserved for the model's response. The context window holds
/// input + output combined, so the input must leave at least this much
/// room for the next reply.
const RESERVED_RESPONSE_TOKENS: u64 = 20_000;

/// Headroom below the hard window limit so the warning fires with room
/// to spare. Also absorbs drift in the `bytes / 4` token estimate, which
/// tends to under-count code and JSON.
const COMPACTION_HEADROOM_TOKENS: u64 = 13_000;

/// Token count at which the proactive seam fires for a model with
/// context window `window`. `None` when the window is unknown.
pub(crate) fn threshold(window: Option<u64>) -> Option<u64> {
    window.map(|size| {
        size.saturating_sub(RESERVED_RESPONSE_TOKENS)
            .saturating_sub(COMPACTION_HEADROOM_TOKENS)
    })
}

/// Estimate of the next request's input-token count: the last response's
/// reported input tokens plus a `bytes / 4` estimate over any messages
/// appended since.
pub(crate) fn estimate_next_request_tokens(last_usage: &TokenUsage, messages: &[Message]) -> u64 {
    let new_bytes: usize = messages.iter().map(message_bytes).sum();
    last_usage.input_tokens + (new_bytes / 4) as u64
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

/// Proactive seam: the warning event when the estimated next-request
/// input crosses the threshold. `None` when the window is unknown or
/// the estimate is still under it.
pub(crate) fn proactive_event(
    window: Option<u64>,
    last_usage: &TokenUsage,
    messages: &[Message],
) -> Option<EventKind> {
    let threshold = threshold(window)?;
    let tokens = estimate_next_request_tokens(last_usage, messages);
    if tokens < threshold {
        return None;
    }
    Some(EventKind::ContextCompacted {
        tokens,
        threshold,
        reason: CompactReason::Proactive,
    })
}

/// Reactive seam: the warning event when the provider itself reports
/// context-window overflow. Carries sentinel `tokens = 0` and
/// `threshold = 0` since the authoritative numbers come from the
/// provider, not our estimator.
pub(crate) fn reactive_event() -> EventKind {
    EventKind::ContextCompacted {
        tokens: 0,
        threshold: 0,
        reason: CompactReason::Reactive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_subtracts_reserves_and_headroom() {
        assert_eq!(threshold(Some(200_000)), Some(167_000));
    }

    #[test]
    fn threshold_saturates_on_tiny_window() {
        assert_eq!(threshold(Some(100)), Some(0));
        assert_eq!(threshold(Some(0)), Some(0));
    }

    #[test]
    fn threshold_none_for_unknown_window() {
        assert_eq!(threshold(None), None);
    }

    #[test]
    fn estimate_scales_with_message_bytes() {
        let usage = TokenUsage::default();
        let short = [Message::user("hi")];
        let long = [Message::user("x".repeat(400))];
        assert!(
            estimate_next_request_tokens(&usage, &long)
                > estimate_next_request_tokens(&usage, &short)
        );
        assert_eq!(estimate_next_request_tokens(&usage, &long), 100);
    }

    #[test]
    fn estimate_adds_last_input_tokens() {
        let usage = TokenUsage {
            input_tokens: 5_000,
            output_tokens: 200,
        };
        let messages = [Message::user("x".repeat(400))];
        assert_eq!(estimate_next_request_tokens(&usage, &messages), 5_100);
    }

    #[test]
    fn proactive_event_none_when_window_unknown() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        assert!(proactive_event(None, &usage, &messages).is_none());
    }

    #[test]
    fn proactive_event_none_when_under_threshold() {
        let usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        assert!(proactive_event(Some(200_000), &usage, &messages).is_none());
    }

    #[test]
    fn proactive_event_some_when_over_threshold() {
        let usage = TokenUsage {
            input_tokens: 170_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        let event = proactive_event(Some(200_000), &usage, &messages)
            .expect("threshold should have tripped");
        match event {
            EventKind::ContextCompacted {
                tokens,
                threshold,
                reason,
            } => {
                assert_eq!(threshold, 167_000);
                assert!(tokens >= 170_000);
                assert_eq!(reason, CompactReason::Proactive);
            }
            other => panic!("expected ContextCompacted, got {other:?}"),
        }
    }

    #[test]
    fn reactive_event_carries_sentinel_zeros() {
        match reactive_event() {
            EventKind::ContextCompacted {
                tokens,
                threshold,
                reason,
            } => {
                assert_eq!(tokens, 0);
                assert_eq!(threshold, 0);
                assert_eq!(reason, CompactReason::Reactive);
            }
            other => panic!("expected ContextCompacted, got {other:?}"),
        }
    }
}
