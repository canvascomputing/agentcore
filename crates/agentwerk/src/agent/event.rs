//! Structured events the loop emits so callers can observe a run (turns, tool calls, compactions, completion) without wrapping the loop itself.

use std::sync::Arc;

use crate::agent::compact::CompactReason;
use crate::agent::output::AgentStatus;
use crate::provider::types::TokenUsage;

#[derive(Debug, Clone)]
pub struct AgentEvent {
    pub agent_name: String,
    pub kind: AgentEventKind,
}

impl AgentEvent {
    pub(crate) fn new(agent_name: impl Into<String>, kind: AgentEventKind) -> Self {
        Self {
            agent_name: agent_name.into(),
            kind,
        }
    }

    /// Default event handler: logs lifecycle and tool activity to stderr.
    ///
    /// Installed automatically when [`Agent`] is built without `.event_handler(...)`.
    /// Prints one line per notable event; chatty events (streamed text, token usage,
    /// turn/request boundaries, idle/resumed) are skipped. Call `.silent()` on the
    /// agent to opt out, or pass a custom handler for richer formatting.
    ///
    /// [`Agent`]: crate::agent::Agent
    pub fn default_logger() -> Arc<dyn Fn(AgentEvent) + Send + Sync> {
        Arc::new(|event: AgentEvent| {
            let agent = &event.agent_name;
            match &event.kind {
                AgentEventKind::AgentStart {
                    description: Some(d),
                } => {
                    eprintln!("[{agent}] start: {d}");
                }
                AgentEventKind::AgentEnd { turns, status } => {
                    eprintln!("[{agent}] done ({turns} turns, {status:?})");
                }
                AgentEventKind::ToolCallStart {
                    tool_name, input, ..
                } => {
                    eprintln!("[{agent}] → {tool_name}({})", compact_input(input));
                }
                AgentEventKind::ToolCallError {
                    tool_name, error, ..
                } => {
                    eprintln!("[{agent}] ✗ {tool_name}: {error}");
                }
                AgentEventKind::CompactTriggered {
                    turn,
                    token_count,
                    threshold,
                    reason,
                } => {
                    eprintln!(
                        "[{agent}] compact turn={turn} {token_count}/{threshold} ({reason:?})"
                    );
                }
                AgentEventKind::OutputTruncated { turn } => {
                    eprintln!("[{agent}] truncated turn={turn}");
                }
                _ => {}
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum AgentEventKind {
    AgentStart {
        description: Option<String>,
    },
    AgentEnd {
        turns: u32,
        status: AgentStatus,
    },
    TurnStart {
        turn: u32,
    },
    TurnEnd {
        turn: u32,
    },
    ToolCallStart {
        tool_name: String,
        call_id: String,
        input: serde_json::Value,
    },
    ToolCallEnd {
        tool_name: String,
        call_id: String,
        output: String,
    },
    ToolCallError {
        tool_name: String,
        call_id: String,
        error: String,
    },
    TokenUsage {
        model: String,
        usage: TokenUsage,
    },
    ResponseTextChunk {
        content: String,
    },
    RequestStart {
        model: String,
    },
    RequestEnd {
        model: String,
    },
    OutputTruncated {
        turn: u32,
    },
    CompactTriggered {
        turn: u32,
        token_count: u64,
        threshold: u64,
        reason: CompactReason,
    },
    AgentIdle,
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
        let logger = AgentEvent::default_logger();
        let every: Vec<AgentEventKind> = vec![
            AgentEventKind::AgentStart {
                description: Some("desc".into()),
            },
            AgentEventKind::AgentStart { description: None },
            AgentEventKind::AgentEnd {
                turns: 3,
                status: AgentStatus::Completed,
            },
            AgentEventKind::TurnStart { turn: 1 },
            AgentEventKind::TurnEnd { turn: 1 },
            AgentEventKind::ToolCallStart {
                tool_name: "glob".into(),
                call_id: "c1".into(),
                input: serde_json::json!({"pattern": "**/*.rs"}),
            },
            AgentEventKind::ToolCallEnd {
                tool_name: "glob".into(),
                call_id: "c1".into(),
                output: "ok".into(),
            },
            AgentEventKind::ToolCallError {
                tool_name: "glob".into(),
                call_id: "c1".into(),
                error: "boom".into(),
            },
            AgentEventKind::TokenUsage {
                model: "m".into(),
                usage: TokenUsage::default(),
            },
            AgentEventKind::ResponseTextChunk {
                content: "hi".into(),
            },
            AgentEventKind::RequestStart { model: "m".into() },
            AgentEventKind::RequestEnd { model: "m".into() },
            AgentEventKind::OutputTruncated { turn: 2 },
            AgentEventKind::CompactTriggered {
                turn: 2,
                token_count: 9_000,
                threshold: 10_000,
                reason: CompactReason::Proactive,
            },
            AgentEventKind::AgentIdle,
            AgentEventKind::AgentResumed,
        ];

        // If a new variant is added to AgentEventKind, this match fails to
        // compile and the test list above must be extended.
        for kind in &every {
            match kind {
                AgentEventKind::AgentStart { .. }
                | AgentEventKind::AgentEnd { .. }
                | AgentEventKind::TurnStart { .. }
                | AgentEventKind::TurnEnd { .. }
                | AgentEventKind::ToolCallStart { .. }
                | AgentEventKind::ToolCallEnd { .. }
                | AgentEventKind::ToolCallError { .. }
                | AgentEventKind::TokenUsage { .. }
                | AgentEventKind::ResponseTextChunk { .. }
                | AgentEventKind::RequestStart { .. }
                | AgentEventKind::RequestEnd { .. }
                | AgentEventKind::OutputTruncated { .. }
                | AgentEventKind::CompactTriggered { .. }
                | AgentEventKind::AgentIdle
                | AgentEventKind::AgentResumed => {}
            }
        }

        for kind in every {
            logger(AgentEvent::new("test", kind));
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
