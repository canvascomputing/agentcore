//! The agent execution loop. Drives one compiled `Agent` turn by turn until it returns an `Output`.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::event::{Event, EventKind, PolicyKind};
use crate::output::{Outcome, Output, OutputSchema, SchemaViolation, Statistics};
use crate::persistence::session::{SessionStore, TranscriptEntry};
use crate::provider::types::{ContentBlock, Message, ResponseStatus, StreamEvent, TokenUsage};
use crate::provider::{ModelRequest, Provider, ProviderError, RequestErrorKind};
use crate::tools::{ToolCall, ToolContext, ToolRegistry};
use crate::util::{
    cancellable_sleep, format_current_date, now_millis, wait_for_cancel, ExponentialRetry, Retry,
    Retryable,
};

use super::compact;
use super::error::AgentError;
use super::prompts::{self as prompts};
use super::queue::{CommandQueue, QueuePriority};
use super::spec::AgentSpec;

/// Externals plus per-agent resolutions shared by the loop. Externals
/// (provider, handlers, queue, store) inherit tree-wide so a child sub-agent
/// reuses the parent's; `tools` and `template_variables` are per-agent
/// resolutions produced by `Agent::compile`.
pub(crate) struct LoopRuntime {
    pub provider: Arc<dyn Provider>,
    pub event_handler: Arc<dyn Fn(Event) + Send + Sync>,
    pub cancel_signal: Arc<AtomicBool>,
    pub working_dir: PathBuf,
    pub environment: Option<String>,
    pub command_queue: Option<Arc<CommandQueue>>,
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub tool_registry: Arc<ToolRegistry>,
    pub template_variables: HashMap<String, Value>,
}

impl LoopRuntime {
    /// Build the environment block prepended to the first user message.
    pub(crate) fn environment(working_dir: &Path) -> String {
        let working_dir = working_dir.display();
        let platform = std::env::consts::OS;
        let os_version = std::process::Command::new("uname")
            .arg("-r")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let date = format_current_date();
        format!(
            "<environment>\nWorking directory: {working_dir}\nPlatform: {platform}\nOS version: {os_version}\nDate: {date}\n</environment>"
        )
    }
}

/// Per-run mutable state of the loop. Created fresh for each agent run.
#[derive(Default)]
pub(crate) struct LoopState {
    pub messages: Vec<Message>,
    pub errors: Vec<Error>,
    pub usage: TokenUsage,
    pub requests: u64,
    pub tool_calls: u64,
    pub turns: u32,
    pub schema_retries: u32,
    pub is_idle: bool,
}

impl LoopState {
    /// Build the initial state from precomputed prompts. Assembling them needs
    /// `Agent`'s per-run fields, which this module cannot see.
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

/// Drive one compiled agent to completion: provider call, tool calls, queue
/// drain, schema check, repeat. The body below is the recipe.
///
/// Always resolves to `Ok(Output)`. The `outcome` is `Completed`, `Cancelled`,
/// or `Failed`; on `Failed` the cause is the last entry of `output.errors`.
///
/// Boxed because `SpawnAgentTool` re-enters this function for sub-agents.
pub(crate) fn run_loop(
    runtime: Arc<LoopRuntime>,
    spec: Arc<AgentSpec>,
    mut state: LoopState,
) -> Pin<Box<dyn Future<Output = Result<Output>> + Send>> {
    Box::pin(async move {
        let emit = |kind: EventKind| {
            (runtime.event_handler)(Event::new(spec.name.clone(), kind));
        };
        let transcribe = |message: &Message, usage_and_model: Option<(&TokenUsage, &str)>| {
            if let Some(store) = runtime.session_store.as_ref() {
                store
                    .lock()
                    .unwrap()
                    .record(TranscriptEntry {
                        recorded_at: now_millis(),
                        message: message.clone(),
                        usage: usage_and_model.map(|(u, _)| u.clone()),
                        model: usage_and_model.map(|(_, m)| m.to_string()),
                    })
                    .ok();
            }
        };

        runtime.provider.prewarm().await;
        transcribe(state.messages.last().unwrap(), None);
        emit(EventKind::AgentStarted);

        let outcome = 'run: loop {
            // Guards: cancel, turn limit, token budgets
            if runtime.cancel_signal.load(Ordering::Relaxed) {
                break 'run Outcome::Cancelled;
            }
            if let Some(limit) = spec.max_turns {
                if state.turns >= limit {
                    let limit = u64::from(limit);
                    let kind = PolicyKind::Turns;
                    state
                        .errors
                        .push(AgentError::PolicyViolated { kind, limit }.into());
                    emit(EventKind::PolicyViolated { kind, limit });
                    break 'run Outcome::Failed;
                }
            }
            if let Some(limit) = spec.max_input_tokens {
                if state.usage.input_tokens >= limit {
                    let kind = PolicyKind::InputTokens;
                    state
                        .errors
                        .push(AgentError::PolicyViolated { kind, limit }.into());
                    emit(EventKind::PolicyViolated { kind, limit });
                    break 'run Outcome::Failed;
                }
            }
            if let Some(limit) = spec.max_output_tokens {
                if state.usage.output_tokens >= limit {
                    let kind = PolicyKind::OutputTokens;
                    state
                        .errors
                        .push(AgentError::PolicyViolated { kind, limit }.into());
                    emit(EventKind::PolicyViolated { kind, limit });
                    break 'run Outcome::Failed;
                }
            }

            // New turn
            state.turns += 1;
            let turn = state.turns;
            emit(EventKind::TurnStarted { turn });
            emit(EventKind::RequestStarted {
                model: spec.model().name.clone(),
            });

            // Provider call, retrying transient failures
            let retry = ExponentialRetry {
                base_delay: spec.request_retry_delay,
                max_attempts: spec.max_request_retries,
            };
            let mut attempt = 0u32;
            let response = 'fetch: loop {
                let request = ModelRequest {
                    model: spec.model().name.clone(),
                    system_prompt: spec.system_prompt(&runtime.template_variables),
                    messages: state.messages.clone(),
                    tools: runtime.tool_registry.definitions(),
                    max_request_tokens: spec.max_request_tokens,
                    tool_choice: None,
                };
                let on_event: Arc<dyn Fn(StreamEvent) + Send + Sync> = {
                    let event_handler = runtime.event_handler.clone();
                    let agent_name = spec.name.clone();
                    Arc::new(move |event| {
                        if let StreamEvent::TextDelta { text, .. } = &event {
                            event_handler(Event::new(
                                agent_name.clone(),
                                EventKind::TextChunkReceived {
                                    content: text.clone(),
                                },
                            ));
                        }
                    })
                };

                let call = tokio::select! {
                    biased;
                    _ = wait_for_cancel(&runtime.cancel_signal) => None,
                    r = runtime.provider.respond(request, on_event) => Some(r.map_err(Error::from)),
                };

                match call {
                    None => break 'run Outcome::Cancelled,
                    Some(Ok(response)) => break 'fetch response,
                    Some(Err(Error::Provider(ProviderError::ContextWindowExceeded {
                        message,
                    }))) if spec.model().context_window_size.is_some() => {
                        if let Err(compact_err) =
                            compact::trigger_reactive(&runtime, &spec, turn).await
                        {
                            state.errors.push(compact_err);
                        }
                        state
                            .errors
                            .push(Error::Provider(ProviderError::ContextWindowExceeded {
                                message,
                            }));
                        break 'run Outcome::Failed;
                    }
                    Some(Err(e)) if e.is_retryable() && attempt < retry.max_attempts() => {
                        let delay = retry.delay(attempt, e.retry_delay());
                        if !cancellable_sleep(delay, &runtime.cancel_signal).await {
                            state.errors.push(e);
                            break 'run Outcome::Cancelled;
                        }
                        attempt += 1;
                        let kind = match &e {
                            Error::Provider(pe) => pe.kind(),
                            _ => RequestErrorKind::StatusUnclassified,
                        };
                        emit(EventKind::RequestRetried {
                            attempt,
                            max_attempts: retry.max_attempts(),
                            kind,
                            message: format!("{e}"),
                        });
                        state.errors.push(e);
                    }
                    Some(Err(e)) => {
                        // Cancellation mid-flight surfaces as `None` above, not an
                        // error here — the signal check guards the rare race
                        // where the request itself errors out as cancel propagates.
                        if runtime.cancel_signal.load(Ordering::Relaxed) {
                            break 'run Outcome::Cancelled;
                        }
                        let kind = match &e {
                            Error::Provider(pe) => pe.kind(),
                            _ => RequestErrorKind::StatusUnclassified,
                        };
                        emit(EventKind::RequestFailed {
                            kind,
                            message: format!("{e}"),
                        });
                        state.errors.push(e);
                        break 'run Outcome::Failed;
                    }
                }
            };

            emit(EventKind::RequestFinished {
                model: spec.model().name.clone(),
            });
            state.usage += &response.usage;
            state.requests += 1;
            emit(EventKind::TokensReported {
                model: response.model.clone(),
                usage: response.usage.clone(),
            });

            // Parse the reply, append the assistant message
            let mut text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
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
            state.messages.push(Message::Assistant {
                content: response.content.clone(),
            });
            transcribe(
                state.messages.last().unwrap(),
                Some((&response.usage, &response.model)),
            );

            // Compaction: reactive on mid-generation overflow, then proactive
            if response.status == ResponseStatus::ContextWindowExceeded
                && spec.model().context_window_size.is_some()
            {
                if let Err(e) = compact::trigger_reactive(&runtime, &spec, turn).await {
                    state.errors.push(e);
                    break 'run Outcome::Failed;
                }
            }
            if let Err(e) = compact::trigger_if_over_threshold(&runtime, &spec, &state).await {
                state.errors.push(e);
                break 'run Outcome::Failed;
            }

            // Tool calls: run them, append results, drain the queue
            if response.status == ResponseStatus::ToolUse && !tool_calls.is_empty() {
                state.tool_calls += tool_calls.len() as u64;
                for call in &tool_calls {
                    emit(EventKind::ToolCallStarted {
                        tool_name: call.name.clone(),
                        call_id: call.id.clone(),
                        input: call.input.clone(),
                    });
                }
                let tool_ctx = ToolContext::new(runtime.working_dir.clone())
                    .registry(Arc::clone(&runtime.tool_registry))
                    .runtime(Arc::clone(&runtime))
                    .caller_spec(Arc::clone(&spec));
                let raw = runtime.tool_registry.execute(&tool_calls, &tool_ctx).await;
                let mut blocks = Vec::with_capacity(raw.len());
                for (block, outcome) in raw {
                    if let ContentBlock::ToolResult { tool_use_id, .. } = &block {
                        let tool_name = tool_calls
                            .iter()
                            .find(|c| c.id == *tool_use_id)
                            .map(|c| c.name.clone())
                            .unwrap_or_default();
                        match outcome {
                            Ok(output) => emit(EventKind::ToolCallFinished {
                                tool_name,
                                call_id: tool_use_id.clone(),
                                output,
                            }),
                            Err(err) => emit(EventKind::ToolCallFailed {
                                tool_name,
                                call_id: tool_use_id.clone(),
                                message: err.message(),
                                kind: (&err).into(),
                            }),
                        }
                    }
                    blocks.push(block);
                }
                state.messages.push(Message::User { content: blocks });
                transcribe(state.messages.last().unwrap(), None);
                if let Some(queue) = runtime.command_queue.as_ref() {
                    while let Some(cmd) =
                        queue.dequeue_if(Some(&spec.name), |c| c.priority != QueuePriority::Later)
                    {
                        state.messages.push(Message::user(cmd.as_user_message()));
                    }
                }
                emit(EventKind::TurnFinished { turn });
                continue;
            }

            // Truncated output: ask the model to keep going
            if response.status == ResponseStatus::OutputTruncated && tool_calls.is_empty() {
                emit(EventKind::OutputTruncated { turn });
                state
                    .messages
                    .push(Message::user(prompts::MAX_TOKENS_CONTINUATION));
                emit(EventKind::TurnFinished { turn });
                continue;
            }

            // Drain queued messages even without tool use
            let before = state.messages.len();
            if let Some(queue) = runtime.command_queue.as_ref() {
                while let Some(cmd) =
                    queue.dequeue_if(Some(&spec.name), |c| c.priority != QueuePriority::Later)
                {
                    state.messages.push(Message::user(cmd.as_user_message()));
                }
            }
            if state.messages.len() > before {
                emit(EventKind::TurnFinished { turn });
                continue;
            }

            // Park as idle and poll the queue when keep_alive is set
            if spec.keep_alive {
                state.is_idle = true;
                emit(EventKind::AgentPaused);
                const POLL_INTERVAL: Duration = Duration::from_millis(100);
                let woken = loop {
                    if runtime.cancel_signal.load(Ordering::Relaxed) {
                        break false;
                    }
                    let before = state.messages.len();
                    if let Some(queue) = runtime.command_queue.as_ref() {
                        while let Some(cmd) = queue
                            .dequeue_if(Some(&spec.name), |c| c.priority != QueuePriority::Later)
                        {
                            state.messages.push(Message::user(cmd.as_user_message()));
                        }
                    }
                    if state.messages.len() > before {
                        break true;
                    }
                    tokio::time::sleep(POLL_INTERVAL).await;
                };
                state.is_idle = false;
                emit(EventKind::AgentResumed);
                if woken {
                    emit(EventKind::TurnFinished { turn });
                    continue;
                }
            }

            // Schema validation: retry on violation, succeed otherwise
            let validated = match spec.output_schema.as_ref().map(|s| s.validate(&text)) {
                None => None,
                Some(Ok(value)) => Some(value),
                Some(Err(detail)) => {
                    if let Some(limit) = spec.max_schema_retries {
                        if state.schema_retries >= limit {
                            let limit = u64::from(limit);
                            let kind = PolicyKind::SchemaRetries;
                            state
                                .errors
                                .push(AgentError::PolicyViolated { kind, limit }.into());
                            emit(EventKind::PolicyViolated { kind, limit });
                            emit(EventKind::TurnFinished { turn });
                            break 'run Outcome::Failed;
                        }
                    }
                    state.schema_retries += 1;
                    let SchemaViolation { path, message } = &detail;
                    emit(EventKind::SchemaRetried {
                        attempt: state.schema_retries,
                        max_attempts: spec.max_schema_retries.unwrap_or(u32::MAX),
                        path: path.clone(),
                        message: message.clone(),
                    });
                    state
                        .messages
                        .push(Message::user(OutputSchema::retry_message(&detail)));
                    emit(EventKind::TurnFinished { turn });
                    continue;
                }
            };

            // Done: model stopped and any schema validates
            emit(EventKind::AgentFinished {
                turns: state.turns,
                outcome: Outcome::Completed,
            });
            emit(EventKind::TurnFinished { turn });
            return Ok(Output {
                name: spec.name.clone(),
                response: validated,
                response_raw: text,
                statistics: Statistics {
                    input_tokens: state.usage.input_tokens,
                    output_tokens: state.usage.output_tokens,
                    requests: state.requests,
                    tool_calls: state.tool_calls,
                    turns: state.turns,
                },
                outcome: Outcome::Completed,
                errors: std::mem::take(&mut state.errors),
            });
        };

        // Common early-exit path: build output from the last assistant text
        let response_raw = state
            .messages
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::Assistant { content } => Some(
                    content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<String>(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        emit(EventKind::AgentFinished {
            turns: state.turns,
            outcome,
        });
        Ok(Output {
            name: spec.name.clone(),
            response: None,
            response_raw,
            statistics: Statistics {
                input_tokens: state.usage.input_tokens,
                output_tokens: state.usage.output_tokens,
                requests: state.requests,
                tool_calls: state.tool_calls,
                turns: state.turns,
            },
            outcome,
            errors: std::mem::take(&mut state.errors),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::super::agent::Agent;
    use super::*;
    use crate::agent::queue::{CommandSource, QueuedCommand};
    use crate::error::Error;
    use crate::provider::types::ContentBlock;
    use crate::testutil::*;

    fn simple_agent() -> Agent {
        Agent::new()
            .name("test-agent")
            .model_name("mock-model")
            .identity_prompt("You are a test assistant.")
    }

    fn assert_lifecycle_events(harness: &TestHarness, output: &Output) {
        let events = harness.events().all();

        let agent_end_outcome = events.iter().find_map(|e| match &e.kind {
            EventKind::AgentFinished { outcome, .. } => Some(outcome.clone()),
            _ => None,
        });
        assert_eq!(
            agent_end_outcome.as_ref(),
            Some(&output.outcome),
            "AgentFinished outcome must match output.outcome"
        );

        let last_significant = events
            .iter()
            .rev()
            .find(|e| !matches!(e.kind, EventKind::TurnFinished { .. }));
        assert!(
            matches!(
                last_significant.map(|e| &e.kind),
                Some(EventKind::AgentFinished { .. })
            ),
            "AgentFinished must be the last significant event"
        );

        for (i, event) in events.iter().enumerate() {
            if matches!(event.kind, EventKind::OutputTruncated { .. }) {
                let after_agent_end = events[..i]
                    .iter()
                    .any(|e| matches!(e.kind, EventKind::AgentFinished { .. }));
                assert!(
                    !after_agent_end,
                    "OutputTruncated at {i} after AgentFinished"
                );
            }
        }
    }

    #[tokio::test]
    async fn simple_text_response() {
        let harness = TestHarness::new(MockProvider::text("Hello, world!"));
        let output = harness.run_agent(&simple_agent(), "Hi").await.unwrap();
        assert_eq!(output.response_raw, "Hello, world!");
        assert!(output.response.is_none());
        assert_eq!(harness.provider().requests(), 1);
    }

    #[tokio::test]
    async fn failing_tool_emits_tool_call_error() {
        let provider = MockProvider::tool_then_text("boom", serde_json::json!({}), "acknowledged");
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .tool(MockTool::error("boom", false, "disk full"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        let saw_error = events.iter().any(|e| {
            matches!(
                &e.kind,
                EventKind::ToolCallFailed { tool_name, message, .. }
                    if tool_name == "boom" && message == "disk full"
            )
        });
        let saw_end = events
            .iter()
            .any(|e| matches!(e.kind, EventKind::ToolCallFinished { .. }));
        assert!(saw_error, "a failing tool must emit ToolCallFailed");
        assert!(!saw_end, "a failing tool must not also emit ToolCallEnd");
    }

    #[tokio::test]
    async fn simple_tool_execution() {
        let provider =
            MockProvider::tool_then_text("echo_tool", serde_json::json!({"text": "ping"}), "Done!");
        let agent = Agent::new()
            .name("test-agent")
            .model_name("mock-model")
            .identity_prompt("You are helpful.")
            .tool(MockTool::new("echo_tool", false, "pong"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "Echo test").await.unwrap();
        assert_eq!(output.response_raw, "Done!");
        assert_eq!(harness.provider().requests(), 2);
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
            .model_name("mock")
            .identity_prompt("")
            .max_turns(2)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert_eq!(output.statistics.turns, 2);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Agent(AgentError::PolicyViolated {
                kind: PolicyKind::Turns,
                limit: 2
            }))
        ));
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
            .model_name("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        harness.cancel();
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Cancelled);
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn template_variable_interpolates_in_system_prompt() {
        let provider = MockProvider::text("Answer about rust");
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
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
            .model_name("mock")
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
            .model_name("mock")
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
            .model_name("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "later task".into(),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification,
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
            .model_name("mock")
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
            .model_name("mock")
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
            .model_name("mock")
            .identity_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .max_schema_retries(3);

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Agent(AgentError::PolicyViolated {
                kind: PolicyKind::SchemaRetries,
                limit: 3
            }))
        ));
    }

    #[tokio::test]
    async fn sub_agents_auto_wire_spawn_tool() {
        let sub = Agent::new()
            .name("helper")
            .model_name("mock")
            .identity_prompt("I help.");

        let provider = MockProvider::text("ok");
        let agent = Agent::new()
            .name("parent")
            .model_name("mock")
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
    #[should_panic(expected = ".provider()")]
    async fn missing_provider_panics_on_run() {
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("x")
            .instruction_prompt("do");
        let _ = agent.run().await;
    }

    #[allow(dead_code)]
    fn runtime_with_environment(env: &str) -> LoopRuntime {
        LoopRuntime {
            provider: Arc::new(MockProvider::text("ok")),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            working_dir: PathBuf::from("/tmp"),
            command_queue: None,
            session_store: None,
            environment: Some(env.to_string()),
            tool_registry: Arc::new(ToolRegistry::new()),
            template_variables: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn simple_text_response_status_completed() {
        let harness = TestHarness::new(MockProvider::text("Hello!"));
        let output = harness.run_agent(&simple_agent(), "Hi").await.unwrap();
        assert_eq!(output.outcome, Outcome::Completed);
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

        assert_eq!(output.outcome, Outcome::Completed);
        assert_eq!(output.response_raw, "...completed response");
        assert_eq!(harness.provider().requests(), 2);
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
                EventKind::OutputTruncated { turn } => Some(*turn),
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
            .model_name("mock")
            .identity_prompt("")
            .max_input_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Agent(AgentError::PolicyViolated {
                kind: PolicyKind::InputTokens,
                limit: 4000
            }))
        ));
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn max_request_tokens_propagates_into_request() {
        let provider = MockProvider::text("done");
        let agent = simple_agent().max_request_tokens(512);
        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();
        let req = harness.provider().last_request().unwrap();
        assert_eq!(req.max_request_tokens, Some(512));
    }

    #[tokio::test]
    async fn input_budget_guard_fires_at_exact_limit() {
        let mut first = tool_response("t", "c1", serde_json::json!({}));
        first.usage = TokenUsage {
            input_tokens: 4000,
            output_tokens: 100,
            ..Default::default()
        };
        let provider = MockProvider::new(vec![first, text_response("unused")]);

        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_input_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Agent(AgentError::PolicyViolated {
                kind: PolicyKind::InputTokens,
                limit: 4000
            }))
        ));
    }

    #[tokio::test]
    async fn output_budget_guard_fires_at_exact_limit() {
        let mut first = tool_response("t", "c1", serde_json::json!({}));
        first.usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 4000,
            ..Default::default()
        };
        let provider = MockProvider::new(vec![first, text_response("unused")]);

        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_output_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Agent(AgentError::PolicyViolated {
                kind: PolicyKind::OutputTokens,
                limit: 4000
            }))
        ));
    }

    #[tokio::test]
    async fn no_budget_error_when_budget_unset() {
        let mut response = text_response("done");
        response.usage = TokenUsage {
            input_tokens: 9_999_999,
            output_tokens: 9_999_999,
            ..Default::default()
        };
        let provider = MockProvider::new(vec![response]);
        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&simple_agent(), "go").await.unwrap();

        assert_eq!(output.outcome, Outcome::Completed);
        let saw_budget = output.errors.iter().any(|e| {
            matches!(
                e,
                Error::Agent(AgentError::PolicyViolated {
                    kind: PolicyKind::InputTokens | PolicyKind::OutputTokens,
                    ..
                })
            )
        });
        assert!(
            !saw_budget,
            "budget errors must not fire when no budget is configured"
        );
    }

    #[tokio::test]
    async fn output_budget_trips_before_input_budget() {
        let mut response = tool_response("t", "c1", serde_json::json!({}));
        response.usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 5000,
            ..Default::default()
        };
        let provider = MockProvider::new(vec![response, text_response("unused")]);

        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_input_tokens(10_000)
            .max_output_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Agent(AgentError::PolicyViolated {
                kind: PolicyKind::OutputTokens,
                limit: 4000
            }))
        ));
    }

    #[tokio::test]
    async fn output_token_budget_guard() {
        let mut response = tool_response("t", "c1", serde_json::json!({}));
        response.usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 5000,
            ..Default::default()
        };
        let provider = MockProvider::new(vec![response, text_response("done")]);

        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_output_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Agent(AgentError::PolicyViolated {
                kind: PolicyKind::OutputTokens,
                limit: 4000
            }))
        ));
        assert_lifecycle_events(&harness, &output);
    }

    // keep_alive / idle wait — matrix-driven test suite
    //
    // Wake sources (W1-W6):   does a queue item wake an idle listener?
    // Lifecycle (L1-L6):      do one-shot / timeout / cancel / events behave?

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
            source: CommandSource::TaskNotification,
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

        assert_eq!(output.outcome, Outcome::Completed);
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
                EventKind::AgentPaused => Some("idle"),
                EventKind::AgentResumed => Some("resumed"),
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
            .model_name("mock")
            .provider(Arc::new(MockProvider::text("x")))
            .instruction_prompt("")
            .cancel_signal(cancel.clone())
            .command_queue(queue.clone());

        let (_spec, rt) = agent.compile(None);

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
            .model_name("mock")
            .provider(Arc::new(MockProvider::text("x")))
            .instruction_prompt("");

        let (_spec, rt) = agent.compile(None);
        assert!(
            rt.command_queue.is_some(),
            "default queue must be allocated so peer messaging still works"
        );
    }
}

#[cfg(test)]
mod retry_and_events_tests {
    use std::sync::Mutex as StdMutex;

    use super::super::agent::Agent;
    use super::*;
    use crate::error::Error;
    use crate::provider::{Model, ProviderError};
    use crate::testutil::*;

    fn rate_limit_error() -> ProviderError {
        ProviderError::RateLimited {
            message: "rate limited".into(),
            status: 429,
            retry_delay: None,
        }
    }

    fn retries_in(events: &[Event]) -> Vec<(u32, u32, String)> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::RequestRetried {
                    attempt,
                    max_attempts,
                    message,
                    ..
                } => Some((*attempt, *max_attempts, message.clone())),
                _ => None,
            })
            .collect()
    }

    fn failures_in(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::RequestFailed { message, .. } => Some(message.clone()),
                _ => None,
            })
            .collect()
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
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(10));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.response_raw, "hello");
        assert_eq!(harness.provider().requests(), 3);
    }

    #[tokio::test]
    async fn no_retry_on_auth_error() {
        let provider = MockProvider::with_results(vec![Err(ProviderError::AuthenticationFailed {
            message: "unauthorized".into(),
        })]);
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(10));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Provider(ProviderError::AuthenticationFailed { .. }))
        ));
        assert_eq!(harness.provider().requests(), 1);
    }

    #[tokio::test]
    async fn event_sequence_complete() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "done");
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .tool(MockTool::new("read", true, "file contents"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        let names: Vec<&str> = events.iter().map(event_name).collect();
        assert_eq!(
            names,
            vec![
                "AgentStarted",
                "TurnStarted",
                "RequestStarted",
                "RequestFinished",
                "TokensReported",
                "ToolCallStarted",
                "ToolCallFinished",
                "TurnFinished",
                "TurnStarted",
                "RequestStarted",
                "TextChunkReceived",
                "RequestFinished",
                "TokensReported",
                "AgentFinished",
                "TurnFinished",
            ]
        );
    }

    fn event_name(event: &Event) -> &'static str {
        match &event.kind {
            EventKind::AgentStarted { .. } => "AgentStarted",
            EventKind::AgentFinished { .. } => "AgentFinished",
            EventKind::TurnStarted { .. } => "TurnStarted",
            EventKind::TurnFinished { .. } => "TurnFinished",
            EventKind::RequestStarted { .. } => "RequestStarted",
            EventKind::RequestFinished { .. } => "RequestFinished",
            EventKind::RequestRetried { .. } => "RequestRetried",
            EventKind::RequestFailed { .. } => "RequestFailed",
            EventKind::TextChunkReceived { .. } => "TextChunkReceived",
            EventKind::ToolCallStarted { .. } => "ToolCallStarted",
            EventKind::ToolCallFinished { .. } => "ToolCallFinished",
            EventKind::ToolCallFailed { .. } => "ToolCallFailed",
            EventKind::TokensReported { .. } => "TokensReported",
            EventKind::OutputTruncated { .. } => "OutputTruncated",
            EventKind::ContextCompacted { .. } => "ContextCompacted",
            EventKind::PolicyViolated { .. } => "PolicyViolated",
            EventKind::SchemaRetried { .. } => "SchemaRetried",
            EventKind::AgentPaused => "AgentPaused",
            EventKind::AgentResumed => "AgentResumed",
        }
    }

    #[tokio::test]
    async fn retry_emits_request_retried_with_attempt_numbers() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Ok(text_response("hello")),
        ]);
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(4)
            .request_retry_delay(Duration::from_millis(1));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let retries: Vec<(u32, u32)> = harness
            .events()
            .all()
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::RequestRetried {
                    attempt,
                    max_attempts,
                    ..
                } => Some((*attempt, *max_attempts)),
                _ => None,
            })
            .collect();
        assert_eq!(retries, vec![(1, 4), (2, 4)]);
        let failures = harness
            .events()
            .all()
            .iter()
            .filter(|e| matches!(e.kind, EventKind::RequestFailed { .. }))
            .count();
        assert_eq!(failures, 0, "no terminal failure on eventual success");
    }

    #[tokio::test]
    async fn terminal_error_emits_request_failed_once() {
        let provider = MockProvider::with_results(vec![Err(ProviderError::AuthenticationFailed {
            message: "unauthorized".into(),
        })]);
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(1));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);
        assert!(matches!(
            output.errors.last(),
            Some(Error::Provider(ProviderError::AuthenticationFailed { .. }))
        ));

        let events = harness.events().all();
        let failed: Vec<&str> = events
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::RequestFailed { message, .. } => Some(message.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(failed.len(), 1);
        assert!(failed[0].contains("unauthorized"));
        assert!(!events
            .iter()
            .any(|e| matches!(e.kind, EventKind::RequestRetried { .. })));
    }

    #[tokio::test]
    async fn retries_exhausted_emits_single_request_failed() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Err(rate_limit_error()),
        ]);
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(2)
            .request_retry_delay(Duration::from_millis(1));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);

        let events = harness.events().all();
        let retries: Vec<(u32, u32)> = retries_in(&events)
            .into_iter()
            .map(|(a, m, _)| (a, m))
            .collect();
        assert_eq!(retries, vec![(1, 2), (2, 2)]);
        assert_eq!(failures_in(&events).len(), 1);
    }

    #[tokio::test]
    async fn happy_path_emits_no_request_failed() {
        let provider = MockProvider::text("done");
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("");

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        assert!(retries_in(&events).is_empty());
        assert!(failures_in(&events).is_empty());
    }

    #[tokio::test]
    async fn max_retries_on_event_matches_spec_max_request_retries() {
        for max_retries in [0u32, 1, 3, 5] {
            let results: Vec<_> = (0..=max_retries).map(|_| Err(rate_limit_error())).collect();
            let provider = MockProvider::with_results(results);
            let agent = Agent::new()
                .name("test")
                .model_name("mock")
                .identity_prompt("")
                .max_request_retries(max_retries)
                .request_retry_delay(Duration::from_millis(1));

            let harness = TestHarness::new(provider);
            let output = harness.run_agent(&agent, "go").await.unwrap();
            assert_eq!(output.outcome, Outcome::Failed);

            let events = harness.events().all();
            let retries = retries_in(&events);
            assert_eq!(
                retries.len() as u32,
                max_retries,
                "max_retries={max_retries}"
            );
            for (_, evt_max_retries, _) in &retries {
                assert_eq!(
                    *evt_max_retries, max_retries,
                    "event.max_retries must equal spec.max_request_retries (got {evt_max_retries} for {max_retries})"
                );
            }
        }
    }

    #[tokio::test]
    async fn max_request_retries_zero_goes_straight_to_request_failed() {
        let provider = MockProvider::with_results(vec![Err(rate_limit_error())]);
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);

        let events = harness.events().all();
        assert!(retries_in(&events).is_empty());
        assert_eq!(failures_in(&events).len(), 1);
    }

    #[tokio::test]
    async fn request_retried_carries_provider_error_display() {
        let provider = MockProvider::with_results(vec![
            Err(ProviderError::ConnectionFailed {
                message: "dns lookup failed: no such host".into(),
            }),
            Ok(text_response("ok")),
        ]);
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(1));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        let retries = retries_in(&events);
        assert_eq!(retries.len(), 1);
        assert!(
            retries[0].2.contains("dns lookup failed"),
            "retry error must surface provider message, got: {}",
            retries[0].2
        );
    }

    #[tokio::test]
    async fn request_failed_carries_terminal_error_display_for_each_non_retryable_variant() {
        let cases: Vec<(ProviderError, &'static str)> = vec![
            (
                ProviderError::AuthenticationFailed {
                    message: "bad key 401".into(),
                },
                "bad key 401",
            ),
            (
                ProviderError::PermissionDenied {
                    message: "no access 403".into(),
                },
                "no access 403",
            ),
            (
                ProviderError::ModelNotFound {
                    message: "unknown-model-xyz".into(),
                },
                "unknown-model-xyz",
            ),
            (
                ProviderError::SafetyFilterTriggered {
                    message: "blocked by safety-filter-7".into(),
                },
                "safety-filter-7",
            ),
            (
                ProviderError::ResponseMalformed {
                    message: "malformed-json-token".into(),
                },
                "malformed-json-token",
            ),
        ];

        for (err, needle) in cases {
            let provider = MockProvider::with_results(vec![Err(err)]);
            let agent = Agent::new()
                .name("test")
                .model_name("mock")
                .identity_prompt("")
                .max_request_retries(3)
                .request_retry_delay(Duration::from_millis(1));

            let harness = TestHarness::new(provider);
            let output = harness.run_agent(&agent, "go").await.unwrap();
            assert_eq!(output.outcome, Outcome::Failed);

            let events = harness.events().all();
            let failures = failures_in(&events);
            assert_eq!(failures.len(), 1, "{needle}");
            assert!(
                failures[0].contains(needle),
                "RequestFailed must carry error detail '{needle}', got: {}",
                failures[0]
            );
            assert!(retries_in(&events).is_empty(), "{needle}");
        }
    }

    #[tokio::test]
    async fn context_window_exceeded_with_known_window_does_not_emit_request_failed() {
        let provider =
            MockProvider::with_results(vec![Err(ProviderError::ContextWindowExceeded {
                message: "context overflow".into(),
            })]);
        let agent = Agent::new()
            .name("test")
            .model(Model::from_name("mock").context_window_size(100_000))
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(1));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);

        let events = harness.events().all();
        let compacts = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::ContextCompacted { .. }))
            .count();
        assert_eq!(compacts, 1);
        assert!(failures_in(&events).is_empty());
    }

    #[tokio::test]
    async fn context_window_exceeded_with_unknown_window_emits_request_failed() {
        let provider =
            MockProvider::with_results(vec![Err(ProviderError::ContextWindowExceeded {
                message: "context overflow 413".into(),
            })]);
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(1));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);

        let events = harness.events().all();
        let failures = failures_in(&events);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("context overflow 413"));
    }

    #[tokio::test(start_paused = true)]
    async fn request_retried_fires_after_backoff_sleep_not_before() {
        let provider = MockProvider::with_results(vec![
            Err(ProviderError::RateLimited {
                message: "rl".into(),
                status: 429,
                retry_delay: Some(Duration::from_millis(1_000)),
            }),
            Ok(text_response("ok")),
        ]);
        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = collected.clone();
            Arc::new(move |e| c.lock().unwrap().push(e))
        };
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .provider(Arc::new(provider))
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(1_000))
            .event_handler(handler)
            .instruction_prompt("go");

        let run_fut = agent.run();
        let check_fut = async {
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            let retries = || {
                collected
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|e| matches!(e.kind, EventKind::RequestRetried { .. }))
                    .count()
            };
            assert_eq!(retries(), 0, "no retry event before sleep");

            tokio::time::advance(std::time::Duration::from_millis(999)).await;
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            assert_eq!(retries(), 0, "no retry event at 999ms into backoff");

            tokio::time::advance(std::time::Duration::from_millis(2)).await;
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            assert_eq!(retries(), 1, "retry event fires once sleep completes");
        };

        let (run_result, _) = tokio::join!(run_fut, check_fut);
        run_result.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn backoff_between_consecutive_retries_grows_exponentially() {
        // backoff_ms = 2_000 with additive 0..25% jitter, capped at 32_000:
        //   attempt 0 → [2_000, 2_500]ms
        //   attempt 1 → [4_000, 5_000]ms
        //   attempt 2 → [8_000, 10_000]ms
        // Cumulative arrival windows for retry events (worst-case widths):
        //   retry 1 → [ 2_000,  2_500]
        //   retry 2 → [ 6_000,  7_500]
        //   retry 3 → [14_000, 17_500]
        let provider = MockProvider::with_results(vec![
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Err(rate_limit_error()),
        ]);
        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = collected.clone();
            Arc::new(move |e| c.lock().unwrap().push(e))
        };
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .provider(Arc::new(provider))
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(2_000))
            .event_handler(handler)
            .instruction_prompt("go");

        let drain = || async {
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
        };
        let retries = || {
            collected
                .lock()
                .unwrap()
                .iter()
                .filter(|e| matches!(e.kind, EventKind::RequestRetried { .. }))
                .count()
        };

        let run_fut = agent.run();
        let check_fut = async {
            drain().await;
            assert_eq!(retries(), 0, "T=0: no retries yet");

            // Upper bound of attempt 0's sleep (2_500) → retry 1 must have fired.
            tokio::time::advance(std::time::Duration::from_millis(2_500)).await;
            drain().await;
            assert_eq!(retries(), 1, "T=2.5s: retry 1 fired");

            // Below the minimum arrival of retry 2 (6_000).
            tokio::time::advance(std::time::Duration::from_millis(3_499)).await;
            drain().await;
            assert_eq!(retries(), 1, "T≈6s-1ms: retry 2 has not fired yet");

            // Past upper bound of retry 2's window (7_500).
            tokio::time::advance(std::time::Duration::from_millis(1_501)).await;
            drain().await;
            assert_eq!(retries(), 2, "T=7.5s: retry 2 fired");

            // Below minimum arrival of retry 3 (14_000).
            tokio::time::advance(std::time::Duration::from_millis(6_499)).await;
            drain().await;
            assert_eq!(retries(), 2, "T≈14s-1ms: retry 3 has not fired yet");

            // Past upper bound of retry 3's window (17_500).
            tokio::time::advance(std::time::Duration::from_millis(3_501)).await;
            drain().await;
            assert_eq!(retries(), 3, "T=17.5s: retry 3 fired");
        };

        let (run_result, _) = tokio::join!(run_fut, check_fut);
        let output = run_result.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);

        let failures = collected
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e.kind, EventKind::RequestFailed { .. }))
            .count();
        assert_eq!(failures, 1, "one terminal failure after retries exhaust");
    }

    #[tokio::test]
    async fn cancel_during_backoff_exits_quickly_with_cancelled_status() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Err(rate_limit_error()),
        ]);
        let cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .provider(Arc::new(provider))
            .identity_prompt("")
            .max_request_retries(4)
            .request_retry_delay(Duration::from_millis(30_000))
            .cancel_signal(cancel.clone())
            .instruction_prompt("go");

        let cancel_setter = {
            let c = cancel.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                c.store(true, Ordering::Relaxed);
            }
        };

        let start = std::time::Instant::now();
        let (run_result, _) = tokio::join!(agent.run(), cancel_setter);
        let elapsed = start.elapsed();

        let output = run_result.unwrap();
        assert_eq!(output.outcome, Outcome::Cancelled);
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "cancel during a 30s backoff must exit within 2s (took {:?})",
            elapsed
        );
    }

    #[tokio::test]
    async fn custom_event_handler_observes_retry_and_failure() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Err(ProviderError::AuthenticationFailed {
                message: "terminal".into(),
            }),
        ]);
        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = collected.clone();
            Arc::new(move |e| c.lock().unwrap().push(e))
        };
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .provider(Arc::new(provider))
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(1))
            .event_handler(handler)
            .instruction_prompt("go");

        let output = agent.run().await.unwrap();
        assert_eq!(output.outcome, Outcome::Failed);

        let events = collected.lock().unwrap().clone();
        assert_eq!(
            retries_in(&events).len(),
            2,
            "custom handler must receive both retry events"
        );
        assert_eq!(
            failures_in(&events).len(),
            1,
            "custom handler must receive the terminal failure"
        );
    }
}
