//! Structured events the loop emits so callers can observe a run (turns, tool calls, compactions, completion) without wrapping the loop itself.

use std::sync::Arc;

use crate::agent::compact::CompactReason;
use crate::agent::output::Status;
use crate::provider::types::TokenUsage;

#[derive(Debug, Clone)]
pub struct Event {
    pub agent_name: String,
    pub kind: EventKind,
}

impl Event {
    pub(crate) fn new(agent_name: impl Into<String>, kind: EventKind) -> Self {
        Self {
            agent_name: agent_name.into(),
            kind,
        }
    }

    /// Default event handler: logs lifecycle and tool activity to stderr.
    ///
    /// Installed automatically when [`Agent`] is built without `.event_handler(...)`.
    /// Prints one line per notable event; chatty events (streamed text, token usage,
    /// turn/request boundaries, paused/resumed) are skipped. Call `.silent()` on the
    /// agent to opt out, or pass a custom handler for richer formatting.
    ///
    /// [`Agent`]: crate::agent::Agent
    pub fn default_logger() -> Arc<dyn Fn(Event) + Send + Sync> {
        Arc::new(|event: Event| {
            let agent = &event.agent_name;
            match &event.kind {
                EventKind::AgentStarted {
                    description: Some(d),
                } => {
                    eprintln!("[{agent}] start: {d}");
                }
                EventKind::AgentFinished { turns, status } => {
                    eprintln!("[{agent}] done ({turns} turns, {status:?})");
                }
                EventKind::ToolCallStarted {
                    tool_name, input, ..
                } => {
                    eprintln!("[{agent}] → {tool_name}({})", compact_input(input));
                }
                EventKind::ToolCallError {
                    tool_name, error, ..
                } => {
                    eprintln!("[{agent}] ✗ {tool_name}: {error}");
                }
                EventKind::ContextCompacted {
                    turn,
                    token_count,
                    threshold,
                    reason,
                } => {
                    eprintln!(
                        "[{agent}] compact turn={turn} {token_count}/{threshold} ({reason:?})"
                    );
                }
                EventKind::OutputTruncated { turn } => {
                    eprintln!("[{agent}] truncated turn={turn}");
                }
                EventKind::InputBudgetExhausted { usage, limit } => {
                    eprintln!("[{agent}] ✗ input budget exhausted ({usage}/{limit})");
                }
                EventKind::OutputBudgetExhausted { usage, limit } => {
                    eprintln!("[{agent}] ✗ output budget exhausted ({usage}/{limit})");
                }
                EventKind::RequestRetried {
                    attempt,
                    max_attempts,
                    error,
                } => {
                    eprintln!("[{agent}] ↻ retry {attempt}/{max_attempts} ({error})");
                }
                EventKind::RequestError { error } => {
                    eprintln!("[{agent}] ✗ request failed: {error}");
                }
                _ => {}
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum EventKind {
    AgentStarted {
        description: Option<String>,
    },
    AgentFinished {
        turns: u32,
        status: Status,
    },
    TurnStarted {
        turn: u32,
    },
    TurnFinished {
        turn: u32,
    },
    ToolCallStarted {
        tool_name: String,
        call_id: String,
        input: serde_json::Value,
    },
    ToolCallFinished {
        tool_name: String,
        call_id: String,
        output: String,
    },
    ToolCallError {
        tool_name: String,
        call_id: String,
        error: String,
    },
    TokensReported {
        model: String,
        usage: TokenUsage,
    },
    TextChunkReceived {
        content: String,
    },
    RequestStarted {
        model: String,
    },
    RequestFinished {
        model: String,
    },
    RequestRetried {
        attempt: u32,
        max_attempts: u32,
        error: String,
    },
    RequestError {
        error: String,
    },
    OutputTruncated {
        turn: u32,
    },
    ContextCompacted {
        turn: u32,
        token_count: u64,
        threshold: u64,
        reason: CompactReason,
    },
    InputBudgetExhausted {
        usage: u64,
        limit: u64,
    },
    OutputBudgetExhausted {
        usage: u64,
        limit: u64,
    },
    AgentPaused,
    AgentResumed,
}

/// Render a tool input as a single line, truncated to ~80 chars.
fn compact_input(input: &serde_json::Value) -> String {
    let one_line = input.to_string().replace('\n', " ");
    const MAX: usize = 80;
    if one_line.chars().count() <= MAX {
        one_line
    } else {
        let cut: String = one_line.chars().take(MAX).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::types::TokenUsage;

    /// Every variant must survive the default logger without panicking.
    /// Exhaustive match keeps this test honest when a new variant is added.
    #[test]
    fn default_logger_handles_every_variant() {
        let logger = Event::default_logger();
        let every: Vec<EventKind> = vec![
            EventKind::AgentStarted {
                description: Some("desc".into()),
            },
            EventKind::AgentStarted { description: None },
            EventKind::AgentFinished {
                turns: 3,
                status: Status::Completed,
            },
            EventKind::TurnStarted { turn: 1 },
            EventKind::TurnFinished { turn: 1 },
            EventKind::ToolCallStarted {
                tool_name: "glob".into(),
                call_id: "c1".into(),
                input: serde_json::json!({"pattern": "**/*.rs"}),
            },
            EventKind::ToolCallFinished {
                tool_name: "glob".into(),
                call_id: "c1".into(),
                output: "ok".into(),
            },
            EventKind::ToolCallError {
                tool_name: "glob".into(),
                call_id: "c1".into(),
                error: "boom".into(),
            },
            EventKind::TokensReported {
                model: "m".into(),
                usage: TokenUsage::default(),
            },
            EventKind::TextChunkReceived {
                content: "hi".into(),
            },
            EventKind::RequestStarted { model: "m".into() },
            EventKind::RequestFinished { model: "m".into() },
            EventKind::RequestRetried {
                attempt: 1,
                max_attempts: 5,
                error: "rate limited".into(),
            },
            EventKind::RequestError {
                error: "auth failed".into(),
            },
            EventKind::OutputTruncated { turn: 2 },
            EventKind::ContextCompacted {
                turn: 2,
                token_count: 9_000,
                threshold: 10_000,
                reason: CompactReason::Proactive,
            },
            EventKind::InputBudgetExhausted {
                usage: 4_200,
                limit: 4_000,
            },
            EventKind::OutputBudgetExhausted {
                usage: 5_200,
                limit: 5_000,
            },
            EventKind::AgentPaused,
            EventKind::AgentResumed,
        ];

        // If a new variant is added to EventKind, this match fails to
        // compile and the test list above must be extended.
        for kind in &every {
            match kind {
                EventKind::AgentStarted { .. }
                | EventKind::AgentFinished { .. }
                | EventKind::TurnStarted { .. }
                | EventKind::TurnFinished { .. }
                | EventKind::ToolCallStarted { .. }
                | EventKind::ToolCallFinished { .. }
                | EventKind::ToolCallError { .. }
                | EventKind::TokensReported { .. }
                | EventKind::TextChunkReceived { .. }
                | EventKind::RequestStarted { .. }
                | EventKind::RequestFinished { .. }
                | EventKind::RequestRetried { .. }
                | EventKind::RequestError { .. }
                | EventKind::OutputTruncated { .. }
                | EventKind::ContextCompacted { .. }
                | EventKind::InputBudgetExhausted { .. }
                | EventKind::OutputBudgetExhausted { .. }
                | EventKind::AgentPaused
                | EventKind::AgentResumed => {}
            }
        }

        for kind in every {
            logger(Event::new("test", kind));
        }
    }

    #[test]
    fn compact_input_truncates_long_json() {
        let long = serde_json::json!({ "text": "a".repeat(200) });
        let s = compact_input(&long);
        assert!(s.chars().count() <= 81); // 80 + ellipsis
        assert!(s.ends_with('…'));
    }

    #[test]
    fn compact_input_keeps_short_json_unchanged() {
        let short = serde_json::json!({ "p": "x" });
        assert_eq!(compact_input(&short), "{\"p\":\"x\"}");
    }
}
