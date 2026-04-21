//! The execution kernel. Runs a compiled `Agent` turn by turn until it yields an `AgentOutput` or hits a guard.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::persistence::session::{SessionStore, TranscriptEntry, TranscriptEntryType};
use crate::provider::retry::compute_delay;
use crate::provider::types::{
    CompletionResponse, ContentBlock, Message, ResponseStatus, StreamEvent, TokenUsage,
};
use crate::provider::{CompletionRequest, Provider, ProviderError};
use crate::tools::{ToolCall, ToolContext, ToolRegistry, ToolResult};
use crate::util::{format_current_date, now_millis};

use super::compact;
use super::event::{AgentEvent, AgentEventKind};
use super::output::{AgentOutput, AgentStatistics, AgentStatus, OutputSchema};
use super::prompts::{self as prompts};
use super::queue::{CommandQueue, QueuePriority};
use super::spec::AgentSpec;

/// Per-run externals and resolved runtime values. Shared as `Arc<LoopRuntime>`.
/// The fields under "externals" inherit tree-wide (a child sub-agent reuses the
/// parent's provider, handlers, queue, etc.); `tools` and `template_variables`
/// are per-agent resolutions produced by `Agent::compile`.
pub(crate) struct LoopRuntime {
    // Externals — inherited across sub-agents.
    pub provider: Arc<dyn Provider>,
    pub event_handler: Arc<dyn Fn(AgentEvent) + Send + Sync>,
    pub cancel_signal: Arc<AtomicBool>,
    pub working_directory: PathBuf,
    pub command_queue: Option<Arc<CommandQueue>>,
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub metadata: Option<String>,
    pub discovered_tools: Arc<Mutex<HashSet<String>>>,

    // Per-agent resolutions. `tools` has `SpawnAgentTool` auto-wired when the agent
    // declared sub_agents; `template_variables` is a snapshot taken at compile time.
    pub tools: Arc<ToolRegistry>,
    pub template_variables: HashMap<String, Value>,
}

impl LoopRuntime {
    /// Build the environment metadata block prepended to the first user message.
    pub(crate) fn environment(working_directory: &Path) -> String {
        let working_directory = working_directory.display();
        let platform = std::env::consts::OS;
        let os_version = std::process::Command::new("uname")
            .arg("-r")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let date = format_current_date();
        format!(
            "<environment>\nWorking directory: {working_directory}\nPlatform: {platform}\nOS version: {os_version}\nDate: {date}\n</environment>"
        )
    }
}

/// Everything the loop mutates. Created fresh for each agent run.
#[derive(Default)]
pub(crate) struct LoopState {
    pub messages: Vec<Message>,
    pub total_usage: TokenUsage,
    pub request_count: u64,
    pub tool_call_count: u64,
    pub turn: u32,
    pub schema_retries: u32,
    pub is_idle: bool,
}

impl LoopState {
    /// Build the initial state. Both `context_prompt` and `instruction` are precomputed by the
    /// caller because assembling them needs `Agent`'s per-run fields, which this module does not see.
    pub(crate) fn initial(context_prompt: Option<String>, instruction: String) -> Self {
        let mut messages = Vec::new();
        if let Some(cp) = context_prompt {
            messages.push(Message::user(cp));
        }
        messages.push(Message::user(instruction));
        Self {
            messages,
            ..Self::default()
        }
    }
}

/// The agent execution loop — one turn per iteration.
///
/// Each turn sends the conversation to the model and processes the reply.
/// The loop **continues** when more work is needed and **exits** when the
/// agent has nothing left to do.
///
/// Continues when:
/// - the model called tools (run them and feed results back)
/// - the reply was truncated (ask the model to keep going)
/// - a peer agent or background task queued a message
/// - the agent was parked idle and a message woke it
/// - a required output schema wasn't met (retry)
///
/// Exits when:
/// - the model stopped and any required schema validates — return the
///   final answer
/// - a guard fired (cancel, turn limit, input-token budget) — return a
///   partial output with the reason attached
///
/// Context-window overflow is handled transparently: compaction is
/// triggered proactively when the estimated next request would overflow,
/// and reactively when the provider reports overflow mid-turn.
pub(crate) fn run_loop(
    runtime: Arc<LoopRuntime>,
    spec: Arc<AgentSpec>,
    mut state: LoopState,
    description: Option<String>,
) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send>> {
    Box::pin(async move {
        runtime.provider.prewarm().await;
        record_transcript(
            &runtime,
            TranscriptEntryType::UserMessage,
            state.messages.last().unwrap(),
            None,
        );
        emit_agent_start(&runtime, &spec, description);

        loop {
            if let Some(status) = check_guards(&runtime, &spec, &state) {
                return Ok(finish_early(&runtime, &spec, &mut state, status));
            }

            state.turn += 1;
            let turn = state.turn;

            emit_turn_start(&runtime, &spec, turn);
            emit_request_start(&runtime, &spec);

            let response = call_provider_with_retry(&runtime, &spec, &mut state, turn).await?;

            emit_request_end(&runtime, &spec);
            record_usage(&runtime, &spec, &mut state, &response);

            let (text, tool_calls) = parse_response(&response);
            state.messages.push(Message::Assistant {
                content: response.content.clone(),
            });
            record_transcript(
                &runtime,
                TranscriptEntryType::AssistantMessage,
                state.messages.last().unwrap(),
                Some((&response.usage, &response.model)),
            );

            // Mid-generation overflow: same reactive seam as the pre-flight error.
            if response.status == ResponseStatus::ContextWindowExceeded
                && spec.model().context_window_size.is_some()
            {
                compact::trigger_reactive(&runtime, &spec, &mut state, turn).await?;
            }

            compact::trigger_if_over_threshold(&runtime, &spec, &mut state).await?;

            let tool_use_ready =
                response.status == ResponseStatus::ToolUse && !tool_calls.is_empty();
            let response_truncated =
                response.status == ResponseStatus::OutputTruncated && tool_calls.is_empty();

            if tool_use_ready {
                let results = execute_tools(&runtime, &spec, &mut state, &tool_calls).await;
                state.messages.push(Message::User { content: results });
                record_transcript(
                    &runtime,
                    TranscriptEntryType::ToolResult,
                    state.messages.last().unwrap(),
                    None,
                );
                drain_command_queue(&runtime, &spec, &mut state);
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            if response_truncated {
                emit_output_truncated(&runtime, &spec, turn);
                state
                    .messages
                    .push(Message::user(prompts::MAX_TOKENS_CONTINUATION));
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            let drain_found_messages = drain_pending_messages(&runtime, &spec, &mut state);
            if drain_found_messages {
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            // Short-circuit: idle_until_message only runs when keep_alive is
            // enabled, preserving its side effects (emit Idle/Resumed).
            let idle_found_message =
                spec.keep_alive && idle_until_message(&runtime, &spec, &mut state).await;
            if idle_found_message {
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            let output_validation = match &spec.output_schema {
                None => Ok(None),
                Some(schema) => schema.validate(&text).map(Some),
            };

            if let Err(detail) = output_validation.as_ref() {
                state.schema_retries += 1;

                let retry_limit_exceeded = spec
                    .max_schema_retries
                    .filter(|&limit| state.schema_retries > limit);
                if let Some(limit) = retry_limit_exceeded {
                    return Err(AgenticError::SchemaRetryExhausted { retries: limit });
                }

                let retry_prompt = OutputSchema::retry_message(detail);
                state.messages.push(Message::user(retry_prompt));
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            let validated = output_validation.expect("Err handled above");
            let agent_end = AgentEventKind::AgentEnd {
                turns: state.turn,
                status: AgentStatus::Completed,
            };

            emit(&runtime, &spec, agent_end);
            emit_turn_end(&runtime, &spec, turn);
            return Ok(build_output(
                &state,
                text,
                validated,
                AgentStatus::Completed,
            ));
        }
    })
}

fn finish_early(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    status: AgentStatus,
) -> AgentOutput {
    let text = last_assistant_text(&state.messages);
    emit(
        runtime,
        spec,
        AgentEventKind::AgentEnd {
            turns: state.turn,
            status: status.clone(),
        },
    );
    build_output(state, text, None, status)
}

fn check_guards(runtime: &LoopRuntime, spec: &AgentSpec, state: &LoopState) -> Option<AgentStatus> {
    if runtime.cancel_signal.load(Ordering::Relaxed) {
        return Some(AgentStatus::Cancelled);
    }
    if let Some(limit) = spec.max_turns {
        if state.turn >= limit {
            return Some(AgentStatus::TurnLimitReached { limit });
        }
    }
    if let Some(limit) = spec.max_input_tokens {
        if state.total_usage.input_tokens >= limit {
            return Some(AgentStatus::BudgetExhausted {
                usage: state.total_usage.input_tokens,
                limit,
            });
        }
    }
    None
}

async fn call_provider(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &LoopState,
) -> Result<CompletionResponse> {
    let tool_defs = runtime
        .tools
        .definitions(&runtime.discovered_tools.lock().unwrap());
    let request = CompletionRequest {
        model: spec.model().id.clone(),
        system_prompt: spec.system_prompt(&runtime.template_variables),
        messages: state.messages.clone(),
        tools: tool_defs,
        max_output_tokens: spec.max_output_tokens,
        tool_choice: None,
    };

    let event_handler = runtime.event_handler.clone();
    let agent_name = spec.name.clone();
    let on_event = Arc::new(move |event: StreamEvent| {
        if let StreamEvent::TextDelta { text, .. } = &event {
            event_handler(AgentEvent::new(
                agent_name.clone(),
                AgentEventKind::ResponseTextChunk {
                    content: text.clone(),
                },
            ));
        }
    });

    runtime
        .provider
        .complete_streaming(request, on_event)
        .await
        .map_err(AgenticError::from)
}

/// One turn's provider call with two built-in resilience seams:
/// - transient errors (429, 529, 5xx) retry up to `spec.max_request_retries`
/// - a provider-reported `ContextWindowExceeded` fires
///   [`compact::trigger_reactive`] before propagating the original error
async fn call_provider_with_retry(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    turn: u32,
) -> Result<CompletionResponse> {
    let mut last_err = None;
    for attempt in 0..=spec.max_request_retries {
        match call_provider(runtime, spec, state).await {
            Ok(response) => return Ok(response),
            Err(AgenticError::Provider(ProviderError::ContextWindowExceeded {
                provider_message,
            })) if spec.model().context_window_size.is_some() => {
                compact::trigger_reactive(runtime, spec, state, turn).await?;
                // compact::run returns NotImplemented today; once
                // implemented this branch will retry the turn instead of
                // surfacing the error.
                return Err(AgenticError::Provider(
                    ProviderError::ContextWindowExceeded { provider_message },
                ));
            }
            Err(e) if e.is_retryable() && attempt < spec.max_request_retries => {
                let delay_ms =
                    compute_delay(spec.request_retry_backoff_ms, attempt, e.retry_after_ms());
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| AgenticError::Other("retry loop ended unexpectedly".into())))
}

fn record_usage(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    response: &CompletionResponse,
) {
    state.total_usage += &response.usage;
    state.request_count += 1;
    emit(
        runtime,
        spec,
        AgentEventKind::TokenUsage {
            model: response.model.clone(),
            usage: response.usage.clone(),
        },
    );
}

fn parse_response(response: &CompletionResponse) -> (String, Vec<ToolCall>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in &response.content {
        match block {
            ContentBlock::Text { text: chunk } => text.push_str(chunk),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            }),
            _ => {}
        }
    }
    (text, tool_calls)
}

async fn execute_tools(
    runtime: &Arc<LoopRuntime>,
    spec: &Arc<AgentSpec>,
    state: &mut LoopState,
    calls: &[ToolCall],
) -> Vec<ContentBlock> {
    state.tool_call_count += calls.len() as u64;
    for call in calls {
        emit(
            runtime,
            spec,
            AgentEventKind::ToolCallStart {
                tool_name: call.name.clone(),
                call_id: call.id.clone(),
                input: call.input.clone(),
            },
        );
    }

    let tool_ctx = ToolContext::new(runtime.working_directory.clone())
        .registry(Arc::clone(&runtime.tools))
        .runtime(Arc::clone(runtime))
        .caller_spec(Arc::clone(spec));

    let raw = runtime.tools.execute(calls, &tool_ctx).await;
    let mut blocks = Vec::with_capacity(raw.len());
    for (block, result) in raw {
        if let ContentBlock::ToolResult { tool_use_id, .. } = &block {
            let tool_name = calls
                .iter()
                .find(|c| c.id == *tool_use_id)
                .map(|c| c.name.clone())
                .unwrap_or_default();
            let event = match result {
                ToolResult::Success(output) => AgentEventKind::ToolCallEnd {
                    tool_name,
                    call_id: tool_use_id.clone(),
                    output,
                },
                ToolResult::Failure(error) => AgentEventKind::ToolCallError {
                    tool_name,
                    call_id: tool_use_id.clone(),
                    error,
                },
            };
            emit(runtime, spec, event);
        }

        blocks.push(block);
    }
    blocks
}

fn drain_command_queue(runtime: &LoopRuntime, spec: &AgentSpec, state: &mut LoopState) {
    let Some(queue) = runtime.command_queue.as_ref() else {
        return;
    };
    while let Some(cmd) = queue.dequeue_if(Some(&spec.name), |c| c.priority != QueuePriority::Later)
    {
        state.messages.push(Message::user(cmd.as_user_message()));
    }
}

/// Drain the command queue into `state.messages` and report whether anything
/// arrived. Used by `run_loop` to decide whether to continue the loop without
/// a new LLM request when the model gave no actionable output.
fn drain_pending_messages(runtime: &LoopRuntime, spec: &AgentSpec, state: &mut LoopState) -> bool {
    let before = state.messages.len();
    drain_command_queue(runtime, spec, state);
    state.messages.len() > before
}

/// Park the agent as idle, poll for incoming messages, then emit the resume
/// event. Returns `true` if a message arrived (loop should continue),
/// `false` on timeout or cancel (loop should finalize).
async fn idle_until_message(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
) -> bool {
    state.is_idle = true;
    emit_agent_idle(runtime, spec);
    let woken = wait_for_message(runtime, spec, state).await;
    state.is_idle = false;
    emit_agent_resumed(runtime, spec);
    woken
}

/// Park the agent between turn chains, polling the command queue for messages
/// visible to `spec.name`. Returns `true` if a message arrived (agent should
/// resume), `false` on cancel (agent should finalize).
///
/// Only called via `idle_until_message` when `spec.keep_alive` is true.
async fn wait_for_message(runtime: &LoopRuntime, spec: &AgentSpec, state: &mut LoopState) -> bool {
    const POLL_INTERVAL: Duration = Duration::from_millis(100);
    loop {
        if runtime.cancel_signal.load(Ordering::Relaxed) {
            return false;
        }
        let before = state.messages.len();
        drain_command_queue(runtime, spec, state);
        if state.messages.len() > before {
            return true;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn emit(runtime: &LoopRuntime, spec: &AgentSpec, kind: AgentEventKind) {
    (runtime.event_handler)(AgentEvent::new(spec.name.clone(), kind));
}

fn emit_agent_start(runtime: &LoopRuntime, spec: &AgentSpec, description: Option<String>) {
    emit(runtime, spec, AgentEventKind::AgentStart { description });
}

fn emit_turn_start(runtime: &LoopRuntime, spec: &AgentSpec, turn: u32) {
    emit(runtime, spec, AgentEventKind::TurnStart { turn });
}

fn emit_turn_end(runtime: &LoopRuntime, spec: &AgentSpec, turn: u32) {
    emit(runtime, spec, AgentEventKind::TurnEnd { turn });
}

fn emit_request_start(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(
        runtime,
        spec,
        AgentEventKind::RequestStart {
            model: spec.model().id.clone(),
        },
    );
}

fn emit_request_end(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(
        runtime,
        spec,
        AgentEventKind::RequestEnd {
            model: spec.model().id.clone(),
        },
    );
}

fn emit_output_truncated(runtime: &LoopRuntime, spec: &AgentSpec, turn: u32) {
    emit(runtime, spec, AgentEventKind::OutputTruncated { turn });
}

fn emit_agent_idle(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(runtime, spec, AgentEventKind::AgentIdle);
}

fn emit_agent_resumed(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(runtime, spec, AgentEventKind::AgentResumed);
}

fn record_transcript(
    runtime: &LoopRuntime,
    entry_type: TranscriptEntryType,
    message: &Message,
    usage_and_model: Option<(&TokenUsage, &str)>,
) {
    let Some(ref store) = runtime.session_store else {
        return;
    };
    store
        .lock()
        .unwrap()
        .record(TranscriptEntry {
            recorded_at: now_millis(),
            entry_type,
            message: message.clone(),
            usage: usage_and_model.map(|(u, _)| u.clone()),
            model: usage_and_model.map(|(_, m)| m.to_string()),
        })
        .ok();
}

fn build_output(
    state: &LoopState,
    text: String,
    response: Option<Value>,
    status: AgentStatus,
) -> AgentOutput {
    AgentOutput {
        response,
        response_raw: text,
        statistics: AgentStatistics {
            input_tokens: state.total_usage.input_tokens,
            output_tokens: state.total_usage.output_tokens,
            requests: state.request_count,
            tool_calls: state.tool_call_count,
            turns: state.turn,
        },
        status,
    }
}

fn last_assistant_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::Assistant { content } => {
                let text: String = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                Some(text)
            }
            _ => None,
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::super::agent::Agent;
    use super::*;
    use crate::agent::queue::{CommandSource, QueuedCommand};
    use crate::error::AgenticError;
    use crate::provider::types::ContentBlock;
    use crate::testutil::*;

    fn simple_agent() -> Agent {
        Agent::new()
            .name("test-agent")
            .model("mock-model")
            .identity_prompt("You are a test assistant.")
    }

    fn assert_lifecycle_events(harness: &TestHarness, output: &AgentOutput) {
        let events = harness.events().all();

        let agent_end_status = events.iter().find_map(|e| match &e.kind {
            AgentEventKind::AgentEnd { status, .. } => Some(status.clone()),
            _ => None,
        });
        assert_eq!(
            agent_end_status.as_ref(),
            Some(&output.status),
            "AgentEnd status must match output.status"
        );

        let last_significant = events
            .iter()
            .rev()
            .find(|e| !matches!(e.kind, AgentEventKind::TurnEnd { .. }));
        assert!(
            matches!(
                last_significant.map(|e| &e.kind),
                Some(AgentEventKind::AgentEnd { .. })
            ),
            "AgentEnd must be the last significant event"
        );

        for (i, event) in events.iter().enumerate() {
            if matches!(event.kind, AgentEventKind::OutputTruncated { .. }) {
                let after_agent_end = events[..i]
                    .iter()
                    .any(|e| matches!(e.kind, AgentEventKind::AgentEnd { .. }));
                assert!(!after_agent_end, "OutputTruncated at {i} after AgentEnd");
            }
        }
    }

    #[tokio::test]
    async fn simple_text_response() {
        let harness = TestHarness::new(MockProvider::text("Hello, world!"));
        let output = harness.run_agent(&simple_agent(), "Hi").await.unwrap();
        assert_eq!(output.response_raw, "Hello, world!");
        assert!(output.response.is_none());
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn failing_tool_emits_tool_call_error() {
        let provider = MockProvider::tool_then_text("boom", serde_json::json!({}), "acknowledged");
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::error("boom", false, "disk full"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        let saw_error = events.iter().any(|e| {
            matches!(
                &e.kind,
                AgentEventKind::ToolCallError { tool_name, error, .. }
                    if tool_name == "boom" && error == "disk full"
            )
        });
        let saw_end = events
            .iter()
            .any(|e| matches!(e.kind, AgentEventKind::ToolCallEnd { .. }));
        assert!(saw_error, "a failing tool must emit ToolCallError");
        assert!(!saw_end, "a failing tool must not also emit ToolCallEnd");
    }

    #[tokio::test]
    async fn simple_tool_execution() {
        let provider =
            MockProvider::tool_then_text("echo_tool", serde_json::json!({"text": "ping"}), "Done!");
        let agent = Agent::new()
            .name("test-agent")
            .model("mock-model")
            .identity_prompt("You are helpful.")
            .tool(MockTool::new("echo_tool", false, "pong"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "Echo test").await.unwrap();
        assert_eq!(output.response_raw, "Done!");
        assert_eq!(harness.provider().request_count(), 2);
    }

    #[tokio::test]
    async fn guard_max_turns() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            tool_response("t", "c2", serde_json::json!({})),
            tool_response("t", "c3", serde_json::json!({})),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_turns(2)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.status, AgentStatus::TurnLimitReached { limit: 2 });
        assert_eq!(output.statistics.turns, 2);
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn guard_cancellation() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("done"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        harness.cancel();
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.status, AgentStatus::Cancelled);
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn template_variable_interpolates_in_system_prompt() {
        let provider = MockProvider::text("Answer about rust");
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("You are an expert on {topic}.");

        let harness = TestHarness::new(provider).with_state("topic", serde_json::json!("rust"));
        harness.run_agent(&agent, "Tell me").await.unwrap();

        let prompts = harness.provider().system_prompts();
        assert!(prompts[0].contains("expert on rust"));
    }

    #[tokio::test]
    async fn events_emitted() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "Done");
        let agent = Agent::new()
            .name("assistant")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("read", true, "file contents"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "read it").await.unwrap();

        let events = harness.events();
        assert_eq!(events.agent_starts(), vec!["assistant"]);
        assert!(!events.tool_starts().is_empty());
        assert!(events.texts().contains(&"Done".to_string()));
        assert_eq!(events.agent_ends().len(), 1);
    }

    #[tokio::test]
    async fn command_queue_drains_next_priority() {
        use std::sync::Arc;
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "extra instruction".into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: Some("test".into()),
        });

        let harness = TestHarness::with_provider_and_queue(Arc::new(provider), queue);
        let output = harness.run_agent(&agent, "start").await.unwrap();
        assert_eq!(output.response_raw, "final");

        let requests = harness.provider().requests.lock().unwrap();
        let has_extra = requests[1].messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains("extra instruction"),
                _ => false,
            }),
            _ => false,
        });
        assert!(has_extra);
    }

    #[tokio::test]
    async fn command_queue_requeues_later_priority() {
        use std::sync::Arc;
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "later task".into(),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification {
                task_id: "42".into(),
            },
            agent_name: Some("test".into()),
        });

        let harness = TestHarness::with_provider_and_queue(Arc::new(provider), queue.clone());
        harness.run_agent(&agent, "start").await.unwrap();

        let cmd = queue.dequeue_if(Some("test"), |_| true);
        assert!(cmd.is_some());
        assert_eq!(cmd.unwrap().content, "later task");
    }

    #[tokio::test]
    async fn deferred_tool_filtering() {
        let provider = MockProvider::text("ok");
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("always", true, "ok"))
            .tool(DeferredMockTool::new("deferred"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let deferred_def = req.tools.iter().find(|t| t.name == "deferred").unwrap();
        assert!(deferred_def.description.is_empty());
    }

    #[tokio::test]
    async fn structured_output_extracted() {
        let schema_input = serde_json::json!({"category": "billing", "priority": "high"});
        let provider = MockProvider::new(vec![text_response(&schema_input.to_string())]);
        let agent = Agent::new()
            .name("classifier")
            .model("mock")
            .identity_prompt("Classify.")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": { "category": {"type": "string"}, "priority": {"type": "string"} },
                "required": ["category", "priority"]
            }));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "ticket").await.unwrap();
        let so = output.response.unwrap();
        assert_eq!(so["category"], "billing");
        assert_eq!(so["priority"], "high");
    }

    #[tokio::test]
    async fn structured_output_retry_exhausted() {
        let provider = MockProvider::new(vec![
            text_response("nope"),
            text_response("still nope"),
            text_response("nope again"),
            text_response("last nope"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .max_schema_retries(3);

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(&agent, "go").await.unwrap_err();
        assert!(matches!(
            err,
            AgenticError::SchemaRetryExhausted { retries: 3 }
        ));
    }

    #[tokio::test]
    async fn sub_agents_auto_wire_spawn_tool() {
        let sub = Agent::new()
            .name("helper")
            .model("mock")
            .identity_prompt("I help.");

        let provider = MockProvider::text("ok");
        let agent = Agent::new()
            .name("parent")
            .model("mock")
            .identity_prompt("I coordinate.")
            .sub_agents([sub]);

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        assert!(
            req.tools.iter().any(|t| t.name == "spawn_agent"),
            ".sub_agents() should register spawn_agent automatically"
        );
    }

    #[tokio::test]
    async fn missing_provider_fails_run() {
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("x")
            .instruction_prompt("do");
        let err = agent.run().await.unwrap_err();
        match err {
            AgenticError::Other(msg) => assert!(msg.contains("provider"), "got: {msg}"),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[allow(dead_code)]
    fn runtime_with_metadata(meta: &str) -> LoopRuntime {
        LoopRuntime {
            provider: Arc::new(MockProvider::text("ok")),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            working_directory: PathBuf::from("/tmp"),
            command_queue: None,
            session_store: None,
            metadata: Some(meta.to_string()),
            discovered_tools: Arc::new(Mutex::new(HashSet::new())),
            tools: Arc::new(ToolRegistry::new()),
            template_variables: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn simple_text_response_status_completed() {
        let harness = TestHarness::new(MockProvider::text("Hello!"));
        let output = harness.run_agent(&simple_agent(), "Hi").await.unwrap();
        assert_eq!(output.status, AgentStatus::Completed);
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn max_tokens_auto_continuation() {
        let provider = MockProvider::new(vec![
            truncated_response("partial response..."),
            text_response("...completed response"),
        ]);
        let harness = TestHarness::new(provider);
        let output = harness
            .run_agent(&simple_agent(), "write a long essay")
            .await
            .unwrap();

        assert_eq!(output.status, AgentStatus::Completed);
        assert_eq!(output.response_raw, "...completed response");
        assert_eq!(harness.provider().request_count(), 2);
        assert_lifecycle_events(&harness, &output);

        let req = &harness.provider().requests.lock().unwrap()[1];
        let has_continuation = req.messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains("cut off"),
                _ => false,
            }),
            _ => false,
        });
        assert!(has_continuation);
    }

    #[tokio::test]
    async fn max_tokens_continuation_events() {
        let provider =
            MockProvider::new(vec![truncated_response("partial"), text_response("done")]);
        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&simple_agent(), "go").await.unwrap();
        assert_lifecycle_events(&harness, &output);

        let truncated: Vec<u32> = harness
            .events()
            .all()
            .iter()
            .filter_map(|e| match &e.kind {
                AgentEventKind::OutputTruncated { turn } => Some(*turn),
                _ => None,
            })
            .collect();
        assert_eq!(truncated, vec![1]);
    }

    #[tokio::test]
    async fn token_budget_guard() {
        let mut response = tool_response("t", "c1", serde_json::json!({}));
        response.usage = TokenUsage {
            input_tokens: 5000,
            output_tokens: 100,
            ..Default::default()
        };
        let provider = MockProvider::new(vec![response, text_response("done")]);

        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_input_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(
            output.status,
            AgentStatus::BudgetExhausted {
                usage: 5000,
                limit: 4000
            }
        );
        assert_lifecycle_events(&harness, &output);
    }

    // ──────────────────────────────────────────────────────────────────────
    // keep_alive / idle wait — matrix-driven test suite
    //
    // Wake sources (W1-W6):   does a queue item wake an idle listener?
    // Lifecycle (L1-L6):      do one-shot / timeout / cancel / events behave?
    // ──────────────────────────────────────────────────────────────────────

    use crate::agent::queue::CommandQueue;

    const AGENT_NAME: &str = "test-agent";

    fn peer_msg(target: Option<&str>, from: &str, content: &str) -> QueuedCommand {
        QueuedCommand {
            content: content.into(),
            priority: QueuePriority::Next,
            source: CommandSource::PeerMessage {
                from: from.into(),
                summary: None,
            },
            agent_name: target.map(|s| s.into()),
        }
    }

    fn task_notification(target: Option<&str>, content: &str) -> QueuedCommand {
        QueuedCommand {
            content: content.into(),
            priority: QueuePriority::Next,
            source: CommandSource::TaskNotification {
                task_id: "task-1".into(),
            },
            agent_name: target.map(|s| s.into()),
        }
    }

    fn user_input(target: Option<&str>, content: &str) -> QueuedCommand {
        QueuedCommand {
            content: content.into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: target.map(|s| s.into()),
        }
    }

    /// Build a harness whose agent name is `AGENT_NAME` and which shares a
    /// fresh queue we return to the test for direct manipulation.
    fn listener_harness(provider: Arc<MockProvider>) -> (TestHarness, Arc<CommandQueue>) {
        let queue = Arc::new(CommandQueue::new());
        let harness = TestHarness::with_provider_and_queue(provider, queue.clone());
        (harness, queue)
    }

    /// Enqueue `cmd` after `delay_ms`. Used to drive wake-during-wait tests.
    fn enqueue_after(queue: &Arc<CommandQueue>, delay_ms: u64, cmd: QueuedCommand) {
        let q = queue.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            q.enqueue(cmd);
        });
    }

    /// Flip `cancel` after `delay_ms`. Used to break the unlimited idle wait
    /// once a test's assertion-relevant turns have completed.
    fn cancel_after(cancel: Arc<AtomicBool>, delay_ms: u64) {
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            cancel.store(true, Ordering::Relaxed);
        });
    }

    fn two_text_responses() -> Arc<MockProvider> {
        Arc::new(MockProvider::new(vec![
            text_response("first"),
            text_response("second"),
        ]))
    }

    #[tokio::test]
    async fn wake_on_peer_message_targeted_at_me() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, peer_msg(Some(AGENT_NAME), "peer", "hi"));
        cancel_after(harness.cancel_signal_for_test(), 400);

        let agent = simple_agent().keep_alive();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "peer message should wake the listener"
        );
    }

    #[tokio::test]
    async fn wake_on_task_notification_broadcast() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, task_notification(None, "Task foo completed"));
        cancel_after(harness.cancel_signal_for_test(), 400);

        let agent = simple_agent().keep_alive();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "broadcast task notification should wake the listener"
        );
    }

    #[tokio::test]
    async fn wake_on_user_input_targeted_at_me() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, user_input(Some(AGENT_NAME), "hello"));
        cancel_after(harness.cancel_signal_for_test(), 400);

        let agent = simple_agent().keep_alive();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "user input (targeted) should wake the listener"
        );
    }

    #[tokio::test]
    async fn wake_on_user_input_broadcast() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, user_input(None, "anyone?"));
        cancel_after(harness.cancel_signal_for_test(), 400);

        let agent = simple_agent().keep_alive();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "user input (broadcast) should wake the listener"
        );
    }

    #[tokio::test]
    async fn one_shot_when_keep_alive_unset() {
        let harness = TestHarness::new(MockProvider::text("done"));
        let agent = simple_agent();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(output.status, AgentStatus::Completed);
        assert_eq!(output.statistics.turns, 1);
    }

    #[tokio::test]
    async fn cancel_interrupts_keep_alive() {
        let (harness, _queue) = listener_harness(Arc::new(MockProvider::text("done")));

        let cancel = harness.cancel_signal_for_test();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            cancel.store(true, Ordering::Relaxed);
        });

        let agent = simple_agent().keep_alive();
        let t0 = std::time::Instant::now();
        let _ = harness.run_agent(&agent, "hi").await.unwrap();
        let elapsed = t0.elapsed();

        assert!(
            elapsed >= std::time::Duration::from_millis(100),
            "should have actually waited, elapsed = {elapsed:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(1_500),
            "should have exited promptly on cancel, elapsed = {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn idle_and_resumed_events_fire_in_order() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, peer_msg(Some(AGENT_NAME), "peer", "hi"));
        cancel_after(harness.cancel_signal_for_test(), 400);

        let agent = simple_agent().keep_alive();
        let _ = harness.run_agent(&agent, "hi").await.unwrap();

        let kinds: Vec<&'static str> = harness
            .events()
            .all()
            .iter()
            .filter_map(|e| match &e.kind {
                AgentEventKind::AgentIdle => Some("idle"),
                AgentEventKind::AgentResumed => Some("resumed"),
                _ => None,
            })
            .collect();
        let first_idle = kinds.iter().position(|k| *k == "idle").expect("idle fired");
        let first_resumed = kinds
            .iter()
            .position(|k| *k == "resumed")
            .expect("resumed fired");
        assert!(
            first_idle < first_resumed,
            "idle must precede resumed: {kinds:?}"
        );
    }

    #[tokio::test]
    async fn drain_before_exit_picks_up_preloaded_message() {
        let (harness, queue) = listener_harness(two_text_responses());
        queue.enqueue(peer_msg(Some(AGENT_NAME), "peer", "pre-loaded"));

        // No keep_alive — the drain-before-exit safety net must still catch it.
        let agent = simple_agent();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "drain-before-exit must inject the preloaded message and force a second turn"
        );
    }

    #[tokio::test]
    async fn drains_batch_of_messages_into_one_turn() {
        let (harness, queue) = listener_harness(two_text_responses());
        // Preload two messages. Both must arrive in a single drained turn,
        // not in two separate turns.
        queue.enqueue(peer_msg(Some(AGENT_NAME), "alice", "first"));
        queue.enqueue(peer_msg(Some(AGENT_NAME), "bob", "second"));

        cancel_after(harness.cancel_signal_for_test(), 300);

        let agent = simple_agent().keep_alive();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "two pending messages should drain into ONE additional turn, not two"
        );
    }

    #[test]
    fn compile_uses_externally_supplied_queue_and_cancel() {
        let queue = Arc::new(CommandQueue::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let agent = Agent::new()
            .model("mock")
            .provider(Arc::new(MockProvider::text("x")))
            .instruction_prompt("")
            .cancel_signal(cancel.clone())
            .command_queue(queue.clone());

        let (_spec, rt) = agent.compile(None).unwrap();

        assert!(
            Arc::ptr_eq(rt.command_queue.as_ref().unwrap(), &queue),
            "LoopRuntime should reuse the externally supplied queue"
        );
        assert!(
            Arc::ptr_eq(&rt.cancel_signal, &cancel),
            "LoopRuntime should reuse the externally supplied cancel signal"
        );
    }

    #[test]
    fn compile_allocates_default_queue_when_none_supplied() {
        let agent = Agent::new()
            .model("mock")
            .provider(Arc::new(MockProvider::text("x")))
            .instruction_prompt("");

        let (_spec, rt) = agent.compile(None).unwrap();
        assert!(
            rt.command_queue.is_some(),
            "default queue must be allocated so peer messaging still works"
        );
    }
}

#[cfg(test)]
mod retry_and_events_tests {
    use super::super::agent::Agent;
    use super::*;
    use crate::error::AgenticError;
    use crate::provider::ProviderError;
    use crate::testutil::*;

    fn rate_limit_error() -> ProviderError {
        ProviderError::RateLimited {
            message: "rate limited".into(),
            status: 429,
            retry_after_ms: None,
        }
    }

    #[tokio::test]
    async fn retry_succeeds_after_rate_limit() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Ok(text_response("hello")),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_backoff_ms(10);

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.response_raw, "hello");
        assert_eq!(harness.provider().request_count(), 3);
    }

    #[tokio::test]
    async fn no_retry_on_auth_error() {
        let provider = MockProvider::with_results(vec![Err(ProviderError::AuthenticationFailed {
            provider_message: "unauthorized".into(),
        })]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_backoff_ms(10);

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(&agent, "go").await.unwrap_err();
        assert!(matches!(
            err,
            AgenticError::Provider(ProviderError::AuthenticationFailed { .. })
        ));
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn event_sequence_complete() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "done");
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("read", true, "file contents"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        let names: Vec<&str> = events.iter().map(event_name).collect();
        assert_eq!(
            names,
            vec![
                "AgentStart",
                "TurnStart",
                "RequestStart",
                "RequestEnd",
                "TokenUsage",
                "ToolCallStart",
                "ToolCallEnd",
                "TurnEnd",
                "TurnStart",
                "RequestStart",
                "ResponseTextChunk",
                "RequestEnd",
                "TokenUsage",
                "AgentEnd",
                "TurnEnd",
            ]
        );
    }

    fn event_name(event: &AgentEvent) -> &'static str {
        match &event.kind {
            AgentEventKind::AgentStart { .. } => "AgentStart",
            AgentEventKind::AgentEnd { .. } => "AgentEnd",
            AgentEventKind::TurnStart { .. } => "TurnStart",
            AgentEventKind::TurnEnd { .. } => "TurnEnd",
            AgentEventKind::RequestStart { .. } => "RequestStart",
            AgentEventKind::RequestEnd { .. } => "RequestEnd",
            AgentEventKind::ResponseTextChunk { .. } => "ResponseTextChunk",
            AgentEventKind::ToolCallStart { .. } => "ToolCallStart",
            AgentEventKind::ToolCallEnd { .. } => "ToolCallEnd",
            AgentEventKind::ToolCallError { .. } => "ToolCallError",
            AgentEventKind::TokenUsage { .. } => "TokenUsage",
            AgentEventKind::OutputTruncated { .. } => "OutputTruncated",
            AgentEventKind::CompactTriggered { .. } => "CompactTriggered",
            AgentEventKind::AgentIdle => "AgentIdle",
            AgentEventKind::AgentResumed => "AgentResumed",
        }
    }
}
