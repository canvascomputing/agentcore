//! Context-window compaction seam. Decides when a conversation is too big for the model and hands off to the compaction strategy.

use crate::agent::event::{AgentEvent, AgentEventKind};
use crate::agent::r#loop::{LoopRuntime, LoopState};
use crate::agent::spec::AgentSpec;
use crate::error::{AgenticError, Result};
use crate::provider::types::{ContentBlock, Message};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactReason {
    Proactive,
    Reactive,
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

/// Proactive seam: emit [`AgentEventKind::CompactTriggered`] and invoke [`run`]
/// when the estimated next-request size crosses the threshold. No-op when
/// the agent's model has no known context window size.
pub(crate) async fn trigger_if_over_threshold(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
) -> Result<()> {
    let Some(threshold) = spec.model().compact_threshold() else {
        return Ok(());
    };
    let tokens = estimate_next_request_tokens(state);
    if tokens < threshold {
        return Ok(());
    }
    (runtime.event_handler)(AgentEvent::new(
        spec.name.clone(),
        AgentEventKind::CompactTriggered {
            turn: state.turn,
            token_count: tokens,
            threshold,
            reason: CompactReason::Proactive,
        },
    ));
    run(runtime, spec, state, CompactReason::Proactive).await
}

/// Reactive seam: emit [`AgentEventKind::CompactTriggered`] (sentinel token
/// count / threshold of `0`) and invoke [`run`]. Fired when the provider
/// itself reports a context-window overflow — either pre-flight or
/// mid-generation.
pub(crate) async fn trigger_reactive(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    turn: u32,
) -> Result<()> {
    (runtime.event_handler)(AgentEvent::new(
        spec.name.clone(),
        AgentEventKind::CompactTriggered {
            turn,
            token_count: 0,
            threshold: 0,
            reason: CompactReason::Reactive,
        },
    ));
    run(runtime, spec, state, CompactReason::Reactive).await
}

/// Compact `state.messages` in place. Not yet implemented — returns
/// `AgenticError::NotImplemented` so callers see the trigger fired.
pub(crate) async fn run(
    _runtime: &LoopRuntime,
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
    fn estimate_scales_with_message_size() {
        let short = vec![Message::user("hi")];
        let long = vec![Message::user("x".repeat(400))];
        assert!(
            estimate_tokens_from_message_bytes(&long) > estimate_tokens_from_message_bytes(&short)
        );
        assert_eq!(estimate_tokens_from_message_bytes(&long), 100);
    }
}
