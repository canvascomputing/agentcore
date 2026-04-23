//! Structured events the loop emits so callers can observe a run (turns, tool calls, compactions, completion) without wrapping the loop itself.

use std::sync::Arc;

use crate::output::Outcome;
use crate::provider::{RequestErrorKind, TokenUsage};

/// Why context compaction fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactReason {
    /// Estimated next-request tokens crossed the model's compact threshold
    /// before sending; compaction ran ahead of any failure.
    Proactive,
    /// A request failed mid-run with a context-overflow error from the
    /// provider; compaction ran in response.
    Reactive,
}

/// Which configured policy a [`EventKind::PolicyViolated`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyKind {
    /// `max_turns` — the agentic loop iteration cap.
    Turns,
    /// `max_input_tokens` — cumulative request-side token cap.
    InputTokens,
    /// `max_output_tokens` — cumulative reply-side token cap.
    OutputTokens,
    /// `max_schema_retries` — structured-output validation retry cap.
    SchemaRetries,
}

/// Classification of a [`EventKind::ToolCallFailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFailureKind {
    /// The tool returned `Ok(ToolResult::Error(...))`, or the registry could
    /// not find a tool by that name. The error message is visible to the
    /// model, which may recover by adjusting its arguments or picking a
    /// different tool.
    InBand,
    /// The tool's `call` raised `Err(...)` — a structural / harness-level
    /// failure the model cannot recover from by retrying.
    Infrastructure,
}

/// Observation emitted during an agent run.
///
/// Events are an out-of-band observation channel, distinct from the
/// [`Output`](crate::Output) returned by `.run()`. Observers (loggers, UIs,
/// tracers) receive the event stream; the caller of `.run()` receives the
/// final result.
///
/// Terminal-path invariants (verified in the loop):
/// - `AgentFinished` fires on every normal termination path — `Ok(Output)`
///   with `Outcome::Completed`, `Cancelled`, or `Failed`. Its `outcome` field
///   matches `output.outcome`. This holds even after `RequestFailed`: the loop
///   folds the failure into `Outcome::Failed` and emits `AgentFinished` last.
/// - `.run()` returns `Err(...)` only for pre-flight failures (missing
///   provider, unreadable prompt file, model not set); those never emit
///   `AgentFinished` because the loop never started.
/// - `ToolCallFailed` is never terminal for the run: more events follow as
///   the agent continues. The tool's error string goes back to the model as
///   a tool-result message.
#[derive(Debug, Clone)]
pub struct Event {
    /// Name of the agent that produced this event.
    pub agent_name: String,
    /// What happened.
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
                EventKind::AgentFinished { turns, outcome } => {
                    eprintln!("[{agent}] done ({turns} turns, {outcome:?})");
                }
                EventKind::ToolCallStarted {
                    tool_name, input, ..
                } => {
                    eprintln!("[{agent}] → {tool_name}({})", compact_input(input));
                }
                EventKind::ToolCallFailed {
                    tool_name,
                    message,
                    kind,
                    ..
                } => {
                    eprintln!("[{agent}] ✗ {tool_name} ({kind:?}): {message}");
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
                EventKind::PolicyViolated { kind, usage, limit } => {
                    eprintln!("[{agent}] policy violated: {kind:?} {usage}/{limit}");
                }
                EventKind::SchemaRetried {
                    attempt,
                    max_attempts,
                    path,
                    message,
                } => {
                    eprintln!(
                        "[{agent}] ↻ schema retry {attempt}/{max_attempts} at {path}: {message}"
                    );
                }
                EventKind::RequestRetried {
                    attempt,
                    max_attempts,
                    message,
                    ..
                } => {
                    eprintln!("[{agent}] ↻ retry {attempt}/{max_attempts} ({message})");
                }
                EventKind::RequestFailed { message, .. } => {
                    eprintln!("[{agent}] ✗ request failed: {message}");
                }
                _ => {}
            }
        })
    }
}

/// What an [`Event`] reports. Variants are grouped by lifecycle (`Agent*`),
/// turn (`Turn*`, `Request*`), tool (`ToolCall*`), context (`OutputTruncated`,
/// `ContextCompacted`, `*PolicyViolated`), and pause/resume.
#[derive(Debug, Clone)]
pub enum EventKind {
    /// Agent run began. `description` is set for sub-agents (the parent's
    /// human-readable label for the spawn) and `None` for root runs.
    AgentStarted { description: Option<String> },
    /// Agent run finished on an `Ok` path. `turns` is the loop iteration
    /// count; `outcome` is the exit reason.
    AgentFinished { turns: u32, outcome: Outcome },
    /// Agentic loop turn began.
    TurnStarted { turn: u32 },
    /// Agentic loop turn finished.
    TurnFinished { turn: u32 },
    /// Tool invocation began.
    ToolCallStarted {
        tool_name: String,
        call_id: String,
        input: serde_json::Value,
    },
    /// Tool invocation succeeded.
    ToolCallFinished {
        tool_name: String,
        call_id: String,
        output: String,
    },
    /// Tool invocation failed. The error is sent back to the model as a
    /// tool-result message; the run continues. `kind` distinguishes in-band
    /// failures (model-fixable) from infrastructure failures (harness-level).
    ToolCallFailed {
        tool_name: String,
        call_id: String,
        message: String,
        kind: ToolFailureKind,
    },
    /// Provider reported token counts for the last request.
    TokensReported { model: String, usage: TokenUsage },
    /// A streamed text chunk arrived from the provider.
    TextChunkReceived { content: String },
    /// Provider request began.
    RequestStarted { model: String },
    /// Provider request finished.
    RequestFinished { model: String },
    /// A transient provider error triggered a retry. `attempt` is the upcoming
    /// attempt number out of `max_attempts`; `kind` classifies the failure.
    RequestRetried {
        attempt: u32,
        max_attempts: u32,
        kind: RequestErrorKind,
        message: String,
    },
    /// Provider request failed after exhausting retries. The run returns
    /// `Err(...)`; no `AgentFinished` follows.
    RequestFailed {
        kind: RequestErrorKind,
        message: String,
    },
    /// The model's response was cut off at the configured length cap.
    OutputTruncated { turn: u32 },
    /// Conversation history was compacted to stay within the model's window.
    ContextCompacted {
        turn: u32,
        token_count: u64,
        threshold: u64,
        reason: CompactReason,
    },
    /// A configured policy (`max_turns`, `max_input_tokens`, `max_output_tokens`,
    /// `max_schema_retries`) was exceeded; the run is about to terminate with
    /// `Outcome::Failed`.
    PolicyViolated {
        kind: PolicyKind,
        usage: u64,
        limit: u64,
    },
    /// The model's terminal reply failed output-schema validation and the loop
    /// is sending a corrective message. `attempt` is the upcoming attempt
    /// number out of `max_attempts`; `path` and `message` come from the
    /// validator.
    SchemaRetried {
        attempt: u32,
        max_attempts: u32,
        path: String,
        message: String,
    },
    /// A keep-alive agent finished its current instruction and is parked
    /// waiting for the next message.
    AgentPaused,
    /// A keep-alive agent received a new instruction and resumed.
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
                outcome: Outcome::Completed,
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
            EventKind::ToolCallFailed {
                tool_name: "glob".into(),
                call_id: "c1".into(),
                message: "boom".into(),
                kind: ToolFailureKind::InBand,
            },
            EventKind::ToolCallFailed {
                tool_name: "glob".into(),
                call_id: "c2".into(),
                message: "panic".into(),
                kind: ToolFailureKind::Infrastructure,
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
                kind: RequestErrorKind::RateLimited,
                message: "rate limited".into(),
            },
            EventKind::RequestFailed {
                kind: RequestErrorKind::AuthenticationFailed,
                message: "auth failed".into(),
            },
            EventKind::OutputTruncated { turn: 2 },
            EventKind::ContextCompacted {
                turn: 2,
                token_count: 9_000,
                threshold: 10_000,
                reason: CompactReason::Proactive,
            },
            EventKind::PolicyViolated {
                kind: PolicyKind::Turns,
                usage: 5,
                limit: 5,
            },
            EventKind::SchemaRetried {
                attempt: 1,
                max_attempts: 3,
                path: "root.answer".into(),
                message: "expected integer".into(),
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
                | EventKind::ToolCallFailed { .. }
                | EventKind::TokensReported { .. }
                | EventKind::TextChunkReceived { .. }
                | EventKind::RequestStarted { .. }
                | EventKind::RequestFinished { .. }
                | EventKind::RequestRetried { .. }
                | EventKind::RequestFailed { .. }
                | EventKind::OutputTruncated { .. }
                | EventKind::ContextCompacted { .. }
                | EventKind::PolicyViolated { .. }
                | EventKind::SchemaRetried { .. }
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
