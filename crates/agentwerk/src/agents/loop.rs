//! Multi-agent loop driver. One tokio task per registered agent,
//! reading the shared `TicketSystem` through the upgraded
//! `Weak<TicketSystem>` stamped at `bind_agent`.

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::event::{CompactReason, Event, EventKind, ToolFailureKind};
use crate::providers::types::{ResponseStatus, StreamEvent, TokenUsage};
use crate::providers::{AsUserMessage, ContentBlock, Message, Model, ModelRequest, ProviderError};
use crate::tools::{ToolCall, ToolContext, ToolError};

use super::agent::Agent;
use super::compact;
use super::retry::{ExponentialRetry, ImmediateRetry, Retry};
use super::stats::LoopStats;
use super::tickets::{policy_violated_kind, Status};
use crate::prompts::retry_directive;
use crate::tools::{missing_finisher_detail, FINISHER_TOOL_NAMES};

const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Supervise the agent set: poll the system's agent list every
/// `IDLE_POLL_INTERVAL`, spawn one `handle_tickets` task per newly
/// appended agent, and join all of them on shutdown. Detects late adds
/// from `tickets.agent()` calls that land after `run()` was spawned.
/// Exits when the interrupt signal flips.
pub(super) async fn run_main_loop(system: &crate::agents::tickets::TicketSystem) {
    let signal = Arc::clone(&system.interrupt_signal.lock().unwrap());
    let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    let mut last_spawned: usize = 0;

    loop {
        if signal.load(Ordering::Relaxed) {
            break;
        }
        let agents = system.clone_agents();
        let total = agents.len();
        for agent in agents.into_iter().skip(last_spawned) {
            handles.push(tokio::spawn(handle_tickets(agent)));
        }
        last_spawned = total;
        tokio::time::sleep(IDLE_POLL_INTERVAL).await;
    }

    for h in handles {
        let _ = h.await;
    }
}

/// Resolves once `signal` flips. 50 ms poll cadence; same as
/// `ToolContext::wait_for_cancel`. Pair with `tokio::select!` so
/// dropping the losing branch aborts the in-flight work.
pub(super) async fn wait_for_signal(signal: &Arc<AtomicBool>) {
    const POLL: Duration = Duration::from_millis(50);
    loop {
        if signal.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Terminate the current ticket with a request failure: emit
/// `RequestFailed`, record the error in run-wide and per-label stats,
/// then emit `TicketFailed`. Called from the four sites inside
/// `process_ticket` that turn a `ProviderError` into a ticket failure
/// (transient retries exhausted, terminal provider error, summarize
/// failure, context still over budget after compaction).
fn fail_request<F: Fn(EventKind)>(
    emit: &F,
    stats: &super::stats::Stats,
    labels: &[String],
    key: &str,
    err: &ProviderError,
) {
    emit(EventKind::RequestFailed {
        kind: err.kind(),
        message: err.to_string(),
    });
    stats.record_error();
    for l in labels {
        stats.stats_for_label(l).record_error();
    }
    emit(EventKind::TicketFailed {
        key: key.to_string(),
    });
}

/// Run one compaction attempt around `summarize_and_replace`: emit
/// `CompactionStarted`, call the summarizer, emit `CompactionFinished`
/// on success or `CompactionFailed` + `fail_request` on error.
///
/// `Continue(())` — compaction succeeded, caller proceeds.
/// `Break(())`    — compaction failed and the ticket has already been
///                  marked failed via `fail_request`; caller must
///                  `return` from `process_ticket`.
#[allow(clippy::too_many_arguments)]
async fn try_compact<F: Fn(EventKind)>(
    reason: CompactReason,
    provider: &Arc<dyn crate::providers::Provider>,
    model: &str,
    messages: &mut Vec<Message>,
    head_len: usize,
    emit: &F,
    stats: &super::stats::Stats,
    labels: &[String],
    key: &str,
) -> ControlFlow<()> {
    emit(EventKind::CompactionStarted { reason });
    match compact::summarize_and_replace(provider, model, messages, head_len).await {
        Ok(()) => {
            emit(EventKind::CompactionFinished { reason });
            ControlFlow::Continue(())
        }
        Err(e) => {
            emit(EventKind::CompactionFailed {
                reason,
                message: e.to_string(),
            });
            fail_request(emit, stats, labels, key, &e);
            ControlFlow::Break(())
        }
    }
}

/// Consume one slot of `compaction_retry`, then run reactive
/// compaction. If the budget is exhausted, fail the ticket with the
/// synthesized "after compaction" message and signal `Break(())`.
///
/// Called from two sites:
/// 1. The Layer 2 blocking check, when the estimate crosses the
///    blocking threshold before the provider call goes out.
/// 2. The reactive arm of the request retry loop, when the provider
///    itself returns `ContextWindowExceeded`.
///
/// Both sites map the return value: `Continue` => `continue 'outer`,
/// `Break` => `return` from `process_ticket`.
#[allow(clippy::too_many_arguments)]
async fn handle_overflow<F: Fn(EventKind)>(
    compaction_retry: &mut ImmediateRetry,
    provider: &Arc<dyn crate::providers::Provider>,
    model: &str,
    messages: &mut Vec<Message>,
    head_len: usize,
    emit: &F,
    stats: &super::stats::Stats,
    labels: &[String],
    key: &str,
) -> ControlFlow<()> {
    if compaction_retry.try_consume().is_none() {
        fail_request(
            emit,
            stats,
            labels,
            key,
            &ProviderError::ContextWindowExceeded {
                message: "context still exceeds window after compaction".into(),
            },
        );
        return ControlFlow::Break(());
    }
    try_compact(
        CompactReason::Reactive,
        provider,
        model,
        messages,
        head_len,
        emit,
        stats,
        labels,
        key,
    )
    .await
}

/// Per-agent claim loop. Picks the next eligible ticket and runs it
/// through `process_ticket`. Idles on `IDLE_POLL_INTERVAL` when no
/// work is queued; exits on cancel or policy violation.
pub(super) async fn handle_tickets(agent: Agent) {
    let ticket_system = agent
        .ticket_system
        .upgrade()
        .expect("Agent's TicketSystem was dropped before run() finished");
    let signal = Arc::clone(&ticket_system.interrupt_signal.lock().unwrap());
    loop {
        if signal.load(Ordering::Relaxed) {
            return;
        }
        let policies = ticket_system.policies();
        if let Some((kind, limit)) = policy_violated_kind(&policies, &ticket_system.stats) {
            let handler = agent.resolve_event_handler();
            handler(Event::new(
                agent.get_name(),
                EventKind::PolicyViolated { kind, limit },
            ));
            return;
        }
        // Path A: in-progress ticket already labelled with this agent's name.
        let path_a = ticket_system
            .find(|t| t.status == Status::InProgress && t.has_label(agent.get_name()))
            .map(|t| t.key().to_string());
        // Path B: atomically claim an open Todo whose labels match.
        let path_b = || {
            ticket_system.claim(
                |t| t.status == Status::Todo && agent.handles_labels(&t.labels),
                agent.get_name(),
            )
        };
        let key = match path_a.or_else(path_b) {
            Some(key) => key,
            None => {
                tokio::time::sleep(IDLE_POLL_INTERVAL).await;
                continue;
            }
        };

        process_ticket(&agent, &ticket_system, &signal, &key).await;
    }
}

/// One ticket from claimed → done/failed. Owns the per-ticket message vector.
async fn process_ticket(
    agent: &Agent,
    ticket_system: &Arc<crate::agents::tickets::TicketSystem>,
    interrupt_signal: &Arc<std::sync::atomic::AtomicBool>,
    key: &str,
) {
    let handler = agent.resolve_event_handler();
    let emit = |kind: EventKind| handler(Event::new(agent.get_name(), kind));

    ticket_system.stats.record_step();

    let Some(ticket) = ticket_system.get(key) else {
        return;
    };
    let labels = ticket.labels.clone();
    let task_msg = ticket.as_user_message();
    for l in &labels {
        ticket_system.stats.stats_for_label(l).record_step();
    }

    // Read knowledge index once, at the top of the ticket: the system
    // prompt stays byte-stable across every turn of this ticket so the
    // provider's prefix cache survives mid-ticket knowledge writes.
    // Cross-ticket and cross-agent writes become visible at the top of
    // the next ticket.
    let knowledge_contents = agent.knowledge_or_default().index();

    let policies = ticket_system.policies();
    let window = Model::from_name(agent.model_str()).context_window_size;
    let mut messages: Vec<Message> = Vec::new();
    if let Some(ctx) = agent.context_message(&policies, &ticket_system.stats) {
        messages.push(Message::user(ctx));
    }
    messages.push(task_msg);
    // Compaction always preserves the leading context + task messages.
    let head_len = messages.len();
    emit(EventKind::TicketStarted {
        key: key.to_string(),
    });

    let max_request_tokens = policies.max_request_tokens;
    let max_schema_retries = policies.max_schema_retries.unwrap_or(u32::MAX);
    // Consecutive schema-validation failures since the last successful
    // schema-checked tool call. Bounded by `max_schema_retries`.
    let mut consecutive_schema_failures: u32 = 0;
    // Token usage from the previous response, fed to the proactive
    // compaction seam at the top of each iteration.
    let mut last_usage: Option<TokenUsage> = None;
    // Compaction circuit breaker: allow one retry after compacting; if
    // the next request still overflows, give up rather than spin. The
    // counter resets on every successful response.
    let mut compaction_retry = ImmediateRetry::new(1);

    let on_stream: Arc<dyn Fn(StreamEvent) + Send + Sync> = {
        let handler = agent.resolve_event_handler();
        let name = agent.get_name().to_string();
        Arc::new(move |ev| {
            if let StreamEvent::TextDelta { text, .. } = ev {
                handler(Event::new(
                    &name,
                    EventKind::TextChunkReceived { content: text },
                ));
            }
        })
    };

    'outer: loop {
        if interrupt_signal.load(Ordering::Relaxed) {
            return;
        }
        match ticket_system.get(key) {
            Some(t) if matches!(t.status, Status::Done | Status::Failed) => {
                emit(terminal_event(t.status, key));
                return;
            }
            Some(_) => {}
            None => return,
        }

        // System prompt and tool definitions are byte-stable inputs to
        // the proactive estimate, the Layer 2 blocking check, and the
        // request itself: compute once per iteration and feed all
        // three.
        let system_prompt_now = agent.system_prompt(Some(&knowledge_contents));
        let tools_now = agent.tool_definitions();

        // ---- proactive compaction: collapse the tail before sending ----
        // Once the previous response's input-token estimate plus the
        // bytes appended since crosses the threshold, summarize the
        // tail in place so the next request fits.
        if let Some(usage) = &last_usage {
            if compact::should_compact_proactively(
                window,
                usage,
                &messages,
                &system_prompt_now,
                &tools_now,
            ) && try_compact(
                CompactReason::Proactive,
                &agent.provider_handle(),
                agent.model_str(),
                &mut messages,
                head_len,
                &emit,
                &ticket_system.stats,
                &labels,
                key,
            )
            .await
            .is_break()
            {
                return;
            }
        }

        // ---- Layer 2 blocking limit ----
        // After proactive (which may have shrunk `messages`), check the
        // estimate against `window - BLOCKING_HEADROOM_TOKENS`. Crossing
        // that line short-circuits the provider call: synthesize an
        // overflow and route it through the reactive seam. Uses
        // `handle_overflow`, so the synthetic path consumes the same
        // `compaction_retry` budget a real overflow would.
        if let Some(threshold) = compact::blocking_threshold(window) {
            let default_usage = TokenUsage::default();
            let usage = last_usage.as_ref().unwrap_or(&default_usage);
            let estimate = compact::estimate_next_request_tokens(
                usage,
                &messages,
                &system_prompt_now,
                &tools_now,
            );
            if estimate >= threshold {
                emit(EventKind::BlockingLimitExceeded {
                    estimated_tokens: estimate,
                    threshold_tokens: threshold,
                });
                match handle_overflow(
                    &mut compaction_retry,
                    &agent.provider_handle(),
                    agent.model_str(),
                    &mut messages,
                    head_len,
                    &emit,
                    &ticket_system.stats,
                    &labels,
                    key,
                )
                .await
                {
                    ControlFlow::Continue(()) => continue 'outer,
                    ControlFlow::Break(()) => return,
                }
            }
        }

        emit(EventKind::RequestStarted {
            model: agent.model_str().to_string(),
        });
        let request = ModelRequest {
            model: agent.model_str().to_string(),
            system_prompt: system_prompt_now,
            messages: messages.clone(),
            tools: tools_now,
            max_request_tokens,
            tool_choice: None,
        };
        let mut retry =
            ExponentialRetry::new(policies.request_retry_delay, policies.max_request_retries);

        // ---- request retry: transient errors back off and replay; an
        // overflow error compacts the tail and continues the outer
        // loop; any other terminal error fails the ticket. ----
        let provider = agent.provider_handle();
        let response = loop {
            let outcome = tokio::select! {
                biased;
                _ = wait_for_signal(interrupt_signal) => return,
                r = provider.respond(request.clone(), Arc::clone(&on_stream)) => r,
            };
            match outcome {
                Ok(r) => break r,
                Err(ProviderError::ContextWindowExceeded { .. }) => {
                    match handle_overflow(
                        &mut compaction_retry,
                        &agent.provider_handle(),
                        agent.model_str(),
                        &mut messages,
                        head_len,
                        &emit,
                        &ticket_system.stats,
                        &labels,
                        key,
                    )
                    .await
                    {
                        ControlFlow::Continue(()) => continue 'outer,
                        ControlFlow::Break(()) => return,
                    }
                }
                Err(e) if e.is_retryable() => match retry.try_consume() {
                    Some(attempt) => {
                        let delay = retry.delay(e.retry_delay());
                        emit(EventKind::RequestRetried {
                            attempt,
                            max_attempts: retry.max_attempts(),
                            kind: e.kind(),
                            message: e.to_string(),
                        });
                        tokio::select! {
                            biased;
                            _ = wait_for_signal(interrupt_signal) => return,
                            _ = tokio::time::sleep(delay) => {}
                        }
                    }
                    None => {
                        fail_request(&emit, &ticket_system.stats, &labels, key, &e);
                        return;
                    }
                },
                Err(e) => {
                    fail_request(&emit, &ticket_system.stats, &labels, key, &e);
                    return;
                }
            }
        };

        emit(EventKind::RequestFinished {
            model: response.model.clone(),
            usage: response.usage.clone(),
        });

        last_usage = Some(response.usage.clone());

        ticket_system
            .stats
            .record_request(response.usage.input_tokens, response.usage.output_tokens);
        for l in &labels {
            ticket_system
                .stats
                .stats_for_label(l)
                .record_request(response.usage.input_tokens, response.usage.output_tokens);
        }
        messages.push(Message::Assistant {
            content: response.content.clone(),
        });

        // ---- Layer 3: status-borne overflow ----
        // Some providers (e.g. Anthropic) signal overflow as a status
        // on a successful reply rather than a `ProviderError`. Route
        // it through `handle_overflow` so the reactive seam catches
        // it the same way it catches the error variant. The reset
        // below is intentionally AFTER this branch: a status-overflow
        // reply must not refill the compaction_retry budget.
        if response.status == ResponseStatus::ContextWindowExceeded {
            match handle_overflow(
                &mut compaction_retry,
                &agent.provider_handle(),
                agent.model_str(),
                &mut messages,
                head_len,
                &emit,
                &ticket_system.stats,
                &labels,
                key,
            )
            .await
            {
                ControlFlow::Continue(()) => continue 'outer,
                ControlFlow::Break(()) => return,
            }
        }

        compaction_retry.reset();

        let calls: Vec<ToolCall> = response
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect();

        // ---- terminal reply: model produced no tool calls ----
        // Any ticket without an attached result means the model ended
        // its turn without calling a finisher tool — retry with a
        // corrective directive. With a result attached, settle Done
        // (or Failed if a schema is set and validation now fails,
        // which only happens in the defensive case that the result
        // bypassed `write_result`'s pre-validation).
        if response.status != ResponseStatus::ToolUse || calls.is_empty() {
            let ticket = match ticket_system.get(key) {
                None => return,
                Some(t) => t,
            };
            match (&ticket.schema, ticket.result()) {
                (_, None) => {
                    consecutive_schema_failures = consecutive_schema_failures.saturating_add(1);
                    let registered: Vec<&str> = agent
                        .tool_definitions()
                        .iter()
                        .filter_map(|d| FINISHER_TOOL_NAMES.iter().find(|n| **n == d.name).copied())
                        .collect();
                    let detail = missing_finisher_detail(&registered);
                    emit(EventKind::SchemaRetried {
                        attempt: consecutive_schema_failures,
                        max_attempts: max_schema_retries,
                        message: detail.clone(),
                    });
                    messages.push(Message::User {
                        content: vec![ContentBlock::Text {
                            text: retry_directive(&detail),
                        }],
                    });
                    if consecutive_schema_failures >= max_schema_retries {
                        fail_with_schema_retries(ticket_system, key, max_schema_retries, &emit);
                        return;
                    }
                    continue;
                }
                (Some(schema), Some(attached)) if schema.validate(attached).is_err() => {
                    let _ = ticket_system.set_failed(key);
                    emit(terminal_event(Status::Failed, key));
                    return;
                }
                (_, Some(_)) => {
                    let _ = ticket_system.set_done(key);
                    emit(terminal_event(Status::Done, key));
                    return;
                }
            }
        }

        for call in &calls {
            emit(EventKind::ToolCallStarted {
                tool_name: call.name.clone(),
                call_id: call.id.clone(),
                input: call.input.clone(),
            });
        }

        let ctx = ToolContext::new(agent.dir_or_default())
            .interrupt_signal(Arc::clone(interrupt_signal))
            .registry(Arc::new(agent.tool_registry().clone()))
            .ticket_system(Arc::clone(ticket_system))
            .agent_name(agent.get_name().to_string())
            .knowledge(agent.knowledge_or_default());
        let outcomes = agent.tool_registry().execute(&calls, &ctx).await;

        let mut schema_failure_message: Option<String> = None;
        for (block, verdict) in &outcomes {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                let call = calls.iter().find(|c| &c.id == tool_use_id);
                let tool_name = call.map(|c| c.name.clone()).unwrap_or_default();
                match verdict {
                    Ok(output) => {
                        if call.is_some_and(|c| FINISHER_TOOL_NAMES.contains(&c.name.as_str())) {
                            consecutive_schema_failures = 0;
                        }
                        emit(EventKind::ToolCallFinished {
                            tool_name,
                            call_id: tool_use_id.clone(),
                            output: output.clone(),
                        });
                    }
                    Err(err) => {
                        if matches!(err, ToolError::SchemaValidationFailed { .. }) {
                            consecutive_schema_failures =
                                consecutive_schema_failures.saturating_add(1);
                            if schema_failure_message.is_none() {
                                schema_failure_message = Some(err.message());
                            }
                        }
                        emit(EventKind::ToolCallFailed {
                            tool_name,
                            call_id: tool_use_id.clone(),
                            message: err.message(),
                            kind: match err {
                                ToolError::ToolNotFound { .. } => ToolFailureKind::ToolNotFound,
                                ToolError::ExecutionFailed { .. } => {
                                    ToolFailureKind::ExecutionFailed
                                }
                                ToolError::SchemaValidationFailed { .. } => {
                                    ToolFailureKind::SchemaValidationFailed
                                }
                            },
                        });
                    }
                }
            }
        }

        // ---- schema retry: tool's done-side validation failed → directive + replay ----
        // Emit even on the exhausting attempt so observers see the
        // sequence `SchemaRetried(N) → PolicyViolated`. The directive
        // rides in the same user message as the tool-result blocks.
        let mut blocks: Vec<ContentBlock> = outcomes.into_iter().map(|(b, _)| b).collect();
        if let Some(validator_message) = &schema_failure_message {
            let detail = format!(
                "Your output did not match the required schema. Reply with a \
                 single JSON value conforming to the schema, with no surrounding \
                 text and no code fences. Validator said: {validator_message}"
            );
            emit(EventKind::SchemaRetried {
                attempt: consecutive_schema_failures,
                max_attempts: max_schema_retries,
                message: detail.clone(),
            });
            blocks.push(ContentBlock::Text {
                text: retry_directive(&detail),
            });
        }
        messages.push(Message::User { content: blocks });

        for _ in 0..calls.len() {
            ticket_system.stats.record_tool_call();
            for l in &labels {
                ticket_system.stats.stats_for_label(l).record_tool_call();
            }
        }

        if consecutive_schema_failures >= max_schema_retries {
            fail_with_schema_retries(ticket_system, key, max_schema_retries, &emit);
            return;
        }
    }
}

fn fail_with_schema_retries<F: Fn(EventKind)>(
    ticket_system: &crate::agents::tickets::TicketSystem,
    key: &str,
    max_schema_retries: u32,
    emit: &F,
) {
    emit(EventKind::PolicyViolated {
        kind: crate::event::PolicyKind::MaxSchemaRetries,
        limit: u64::from(max_schema_retries),
    });
    let _ = ticket_system.set_failed(key);
    emit(EventKind::TicketFailed {
        key: key.to_string(),
    });
}

fn terminal_event(status: Status, key: &str) -> EventKind {
    match status {
        Status::Done => EventKind::TicketDone {
            key: key.to_string(),
        },
        Status::Failed => EventKind::TicketFailed {
            key: key.to_string(),
        },
        other => unreachable!("terminal_event called with non-terminal status {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    //! Loop-level tests for request retries, schema retries, the
    //! mark-done shortcut, and cancellation. Each test scripts a
    //! `MockProvider` sequence and asserts on the event stream and
    //! ticket status.
    use std::pin::Pin;
    use std::sync::Mutex as StdMutex;

    use crate::event::{CompactReason, PolicyKind};
    use crate::providers::types::{ModelResponse, TokenUsage};
    use crate::providers::{Provider, ProviderError, ProviderResult};
    use crate::schemas::Schema;
    use crate::tools::ManageTicketsTool;

    use super::super::tickets::{Ticket, TicketSystem};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // ---- mock provider ----

    /// Scripted provider. Pops one `ProviderResult` per `respond`
    /// call; falls back to a non-retryable error once exhausted. Also
    /// records each call's `request.messages` for tests that need to
    /// inspect what the loop fed in.
    struct MockProvider {
        results: StdMutex<Vec<ProviderResult<ModelResponse>>>,
        requests: AtomicUsize,
        received: StdMutex<Vec<Vec<Message>>>,
        received_system_prompts: StdMutex<Vec<String>>,
    }

    impl MockProvider {
        fn with_results(results: Vec<ProviderResult<ModelResponse>>) -> Arc<Self> {
            Arc::new(Self {
                results: StdMutex::new(results),
                requests: AtomicUsize::new(0),
                received: StdMutex::new(Vec::new()),
                received_system_prompts: StdMutex::new(Vec::new()),
            })
        }

        fn requests(&self) -> usize {
            self.requests.load(Ordering::Relaxed)
        }

        fn received(&self) -> Vec<Vec<Message>> {
            self.received.lock().unwrap().clone()
        }

        fn received_system_prompts(&self) -> Vec<String> {
            self.received_system_prompts.lock().unwrap().clone()
        }
    }

    impl Provider for MockProvider {
        fn respond(
            &self,
            request: ModelRequest,
            _on_event: Arc<dyn Fn(crate::providers::types::StreamEvent) + Send + Sync>,
        ) -> Pin<Box<dyn std::future::Future<Output = ProviderResult<ModelResponse>> + Send + '_>>
        {
            self.received.lock().unwrap().push(request.messages.clone());
            self.received_system_prompts
                .lock()
                .unwrap()
                .push(request.system_prompt.clone());
            self.requests.fetch_add(1, Ordering::Relaxed);
            // Non-retryable fallback once exhausted: a retryable
            // fallback would spin up retry chains in tests that don't
            // want them.
            let next = {
                let mut results = self.results.lock().unwrap();
                if results.is_empty() {
                    Err(ProviderError::AuthenticationFailed {
                        message: "MockProvider exhausted".into(),
                    })
                } else {
                    results.remove(0)
                }
            };
            // Yield once: the failure-then-Path-A-re-claim path has no
            // Pending await otherwise, hot-loops the agent task on the
            // current_thread runtime, and starves the run-dry watcher.
            Box::pin(async move {
                tokio::task::yield_now().await;
                next
            })
        }
    }

    // ---- response builders ----

    /// `write_result_tool` call carrying a string `result`. For
    /// no-schema tickets this settles the ticket Done; for schema-bound
    /// tickets it relies on the schema accepting strings.
    fn write_result_response(result: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "write_result_tool".into(),
                input: serde_json::json!({ "result": result }),
            }],
            status: ResponseStatus::ToolUse,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    /// `write_result_tool` call carrying a structured `result` value.
    /// Used by schema-bound ticket tests.
    fn write_result_value(result: serde_json::Value) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "write_result_tool".into(),
                input: serde_json::json!({ "result": result }),
            }],
            status: ResponseStatus::ToolUse,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    /// `knowledge_tool` `write` call: the model's first turn writes a page.
    /// The loop's tool dispatch will upsert it in the bound `Knowledge`.
    fn knowledge_write_response(slug: &str, summary: &str, content: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "knowledge_tool".into(),
                input: serde_json::json!({"action": "write", "slug": slug, "summary": summary, "content": content}),
            }],
            status: ResponseStatus::ToolUse,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    /// `knowledge_tool` `read` call.
    fn knowledge_read_response(slug: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-2".into(),
                name: "knowledge_tool".into(),
                input: serde_json::json!({"action": "read", "slug": slug}),
            }],
            status: ResponseStatus::ToolUse,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    fn text_response(text: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            status: ResponseStatus::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    fn rate_limit() -> ProviderError {
        ProviderError::RateLimited {
            message: "rate limited".into(),
            status: 429,
            retry_delay: None,
        }
    }

    fn connection_failed(message: &str) -> ProviderError {
        ProviderError::ConnectionFailed {
            message: message.into(),
        }
    }

    // ---- event filters ----

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

    fn schema_retries_in(events: &[Event]) -> Vec<(u32, u32, String)> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::SchemaRetried {
                    attempt,
                    max_attempts,
                    message,
                } => Some((*attempt, *max_attempts, message.clone())),
                _ => None,
            })
            .collect()
    }

    // ---- harness ----

    /// Run one ticket against `provider`; return collected events, the
    /// provider handle (for request-count assertions), and the settled
    /// ticket.
    async fn run_one(
        provider: Arc<MockProvider>,
        max_request_retries: u32,
        max_schema_retries: u32,
        schema: Option<Schema>,
    ) -> (Vec<Event>, Arc<MockProvider>, Option<Ticket>) {
        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = Arc::clone(&collected);
            Arc::new(move |e| c.lock().unwrap().push(e))
        };

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(max_request_retries)
            .request_retry_delay(Duration::from_millis(1))
            .max_schema_retries(max_schema_retries)
            // Short timeout: tests where the loop bails leave the ticket
            // InProgress, so Path A would re-claim forever without it.
            .max_time(Duration::from_millis(200));

        let agent = Agent::new()
            .name("tester")
            .provider(provider.clone() as Arc<dyn Provider>)
            .model("mock")
            .role("test")
            // ManageTicketsTool drives create/edit. WriteResultTool is
            // auto-registered on every Agent and is the only path to
            // settle a ticket.
            .tool(ManageTicketsTool)
            .event_handler(handler);
        tickets.agent(agent);

        if let Some(schema) = schema {
            tickets.ticket(Ticket::new("go").schema(schema));
        } else {
            tickets.task("go");
        }

        let _ = tickets.finish().await;
        let events = collected.lock().unwrap().clone();
        let settled = tickets.first();
        (events, provider, settled)
    }

    // =====================================================================
    // Bucket A — request retries
    // =====================================================================

    #[tokio::test]
    async fn retry_succeeds_after_rate_limit() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit()),
            Err(rate_limit()),
            Ok(write_result_response("ok")),
        ]);
        let (events, provider, settled) = run_one(provider, 3, 10, None).await;

        assert_eq!(provider.requests(), 3);
        assert_eq!(retries_in(&events).len(), 2);
        assert!(failures_in(&events).is_empty());
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn no_retry_on_auth_error() {
        let provider = MockProvider::with_results(vec![Err(ProviderError::AuthenticationFailed {
            message: "unauthorized".into(),
        })]);
        let (events, _, _) = run_one(provider, 3, 10, None).await;

        // Path A re-claims an unfailed ticket, so several
        // `RequestFailed`s land before the run-dry timeout. The first
        // one carries the scripted error; what matters is that no
        // retries fire.
        assert!(retries_in(&events).is_empty());
        let failures = failures_in(&events);
        assert!(!failures.is_empty());
        assert!(failures[0].contains("unauthorized"));
    }

    #[tokio::test]
    async fn retries_exhausted_emits_request_failed() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit()),
            Err(rate_limit()),
            Err(rate_limit()),
        ]);
        let (events, _, _) = run_one(provider, 2, 10, None).await;

        let retries: Vec<(u32, u32)> = retries_in(&events)
            .into_iter()
            .map(|(a, m, _)| (a, m))
            .collect();
        assert_eq!(retries, vec![(1, 2), (2, 2)]);
        // Path A re-claims an unfailed ticket, so the same scenario
        // can emit several `RequestFailed`s before the run-dry timeout
        // cuts the loop. The first one is the contract under test.
        let failures = failures_in(&events);
        assert!(!failures.is_empty());
        assert!(failures[0].contains("rate limited"));
    }

    #[tokio::test]
    async fn happy_path_emits_no_request_failed() {
        let provider = MockProvider::with_results(vec![Ok(write_result_response("ok"))]);
        let (events, _, settled) = run_one(provider, 3, 10, None).await;

        assert!(retries_in(&events).is_empty());
        assert!(failures_in(&events).is_empty());
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn max_retries_on_event_matches_policy() {
        for max_retries in [0u32, 1, 3, 5] {
            // Exactly `max_retries + 1` retryable errors so the first
            // process_ticket cycle exhausts them. Any Path A re-claim
            // afterwards hits the MockProvider's non-retryable
            // exhausted-fallback, which doesn't add extra retries.
            let results: Vec<_> = (0..=max_retries).map(|_| Err(rate_limit())).collect();
            let provider = MockProvider::with_results(results);
            let (events, _, _) = run_one(provider, max_retries, 10, None).await;

            let retries = retries_in(&events);
            assert_eq!(
                retries.len() as u32,
                max_retries,
                "max_retries={max_retries}",
            );
            for (_, evt_max, _) in &retries {
                assert_eq!(*evt_max, max_retries);
            }
        }
    }

    #[tokio::test]
    async fn max_request_retries_zero_goes_straight_to_request_failed() {
        let provider = MockProvider::with_results(vec![Err(rate_limit())]);
        let (events, _, _) = run_one(provider, 0, 10, None).await;

        // Same Path-A re-claim caveat as the other terminal-error
        // tests: assert structure (no retries, at least one failure),
        // not exact counts.
        assert!(retries_in(&events).is_empty());
        assert!(!failures_in(&events).is_empty());
    }

    #[tokio::test]
    async fn request_retried_attempt_numbers_are_one_based() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit()),
            Err(rate_limit()),
            Ok(write_result_response("ok")),
        ]);
        let (events, _, _) = run_one(provider, 4, 10, None).await;

        let attempts: Vec<u32> = retries_in(&events).into_iter().map(|(a, ..)| a).collect();
        assert_eq!(attempts, vec![1, 2]);
    }

    #[tokio::test]
    async fn request_retried_carries_provider_error_display() {
        let provider = MockProvider::with_results(vec![
            Err(connection_failed("dns lookup failed: no such host")),
            Ok(write_result_response("ok")),
        ]);
        let (events, _, _) = run_one(provider, 3, 10, None).await;

        let retries = retries_in(&events);
        assert_eq!(retries.len(), 1);
        assert!(retries[0].2.contains("dns lookup failed"));
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
            let (events, _, _) = run_one(provider, 3, 10, None).await;

            // Same Path-A re-claim caveat as the other terminal-error
            // tests: the first failure is the scripted one; later
            // entries come from re-claim cycles hitting the
            // exhausted-fallback.
            let failures = failures_in(&events);
            assert!(!failures.is_empty(), "{needle}");
            assert!(failures[0].contains(needle), "{needle}: {}", failures[0]);
            assert!(retries_in(&events).is_empty(), "{needle}");
        }
    }

    // =====================================================================
    // Bucket B — backoff timing
    // =====================================================================

    #[tokio::test(start_paused = true)]
    async fn request_retried_fires_after_backoff_sleep_not_before() {
        let provider = MockProvider::with_results(vec![
            Err(ProviderError::RateLimited {
                message: "rl".into(),
                status: 429,
                retry_delay: Some(Duration::from_millis(1_000)),
            }),
            Ok(write_result_response("ok")),
        ]);
        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = Arc::clone(&collected);
            Arc::new(move |e| c.lock().unwrap().push(e))
        };

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(3)
            .request_retry_delay(Duration::from_millis(1));
        let agent = Agent::new()
            .name("tester")
            .provider(provider as Arc<dyn Provider>)
            .model("mock")
            .role("test")
            .event_handler(handler);
        tickets.agent(agent);
        tickets.task("go");

        let run_fut = tickets.finish();
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
            assert_eq!(retries(), 1, "retry event fires immediately on Err");

            tokio::time::advance(Duration::from_millis(999)).await;
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            // sleep is still in progress; no second retry yet
            assert_eq!(retries(), 1);
            tokio::time::advance(Duration::from_millis(2)).await;
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            // sleep done; mark_done_response served on the next attempt
        };

        let (_, _) = tokio::join!(run_fut, check_fut);
    }

    // =====================================================================
    // Bucket E — text-only replies (no-tool branch terminates the ticket)
    // =====================================================================

    #[tokio::test]
    async fn text_reply_no_schema_retries_then_recovers() {
        // First reply is text-only with no result → retry directive
        // fires. Second reply calls write_result_tool successfully.
        let provider = MockProvider::with_results(vec![
            Ok(text_response("Hello!")),
            Ok(write_result_response("done")),
        ]);
        let (events, provider, settled) = run_one(provider, 3, 10, None).await;

        assert_eq!(provider.requests(), 2);
        let retries = schema_retries_in(&events);
        assert_eq!(retries.len(), 1);
        assert!(retries[0].2.contains("write_result_tool"));
        let done = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketDone { .. }))
            .count();
        let failed = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFailed { .. }))
            .count();
        assert_eq!(done, 1);
        assert_eq!(failed, 0);
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn text_reply_no_schema_exhausts_retries_and_fails() {
        // Three text-only replies with `max_schema_retries(2)` exhaust
        // the budget and fail the ticket with a MaxSchemaRetries
        // PolicyViolated event.
        let provider = MockProvider::with_results(vec![
            Ok(text_response("a")),
            Ok(text_response("b")),
            Ok(text_response("c")),
        ]);
        let (events, _, settled) = run_one(provider, 3, 2, None).await;

        let retries = schema_retries_in(&events);
        assert_eq!(retries.len(), 2);
        let policy_violated = events.iter().any(|e| {
            matches!(
                &e.kind,
                EventKind::PolicyViolated {
                    kind: PolicyKind::MaxSchemaRetries,
                    limit: 2,
                },
            )
        });
        assert!(policy_violated, "expected MaxSchemaRetries PolicyViolated");
        assert_eq!(settled.unwrap().status, Status::Failed);
    }

    #[tokio::test]
    async fn text_reply_with_schema_retries_then_recovers() {
        // Schema-bound ticket; first reply is text-only (no result
        // attached) → retry directive fires. Second reply attaches a
        // valid result via write_result_tool → Done.
        let provider = MockProvider::with_results(vec![
            Ok(text_response("Hello!")),
            Ok(write_result_value(serde_json::json!({"partial_sum": 1}))),
        ]);
        let (events, provider, settled) =
            run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;

        assert_eq!(provider.requests(), 2);
        let retries = schema_retries_in(&events);
        assert_eq!(retries.len(), 1);
        assert!(retries[0].2.contains("write_result_tool"));
        let done = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketDone { .. }))
            .count();
        let failed = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFailed { .. }))
            .count();
        assert_eq!(done, 1);
        assert_eq!(failed, 0);
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn write_result_settles_ticket_done_with_valid_json() {
        let provider = MockProvider::with_results(vec![Ok(write_result_value(
            serde_json::json!({"partial_sum": 42}),
        ))]);
        let (events, provider, settled) =
            run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;

        assert_eq!(provider.requests(), 1);
        let done = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketDone { .. }))
            .count();
        let failed = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFailed { .. }))
            .count();
        assert_eq!(done, 1);
        assert_eq!(failed, 0);
        let settled = settled.unwrap();
        assert_eq!(settled.status, Status::Done);
        assert_eq!(settled.result().unwrap()["partial_sum"], 42);
    }

    // =====================================================================
    // Bucket C — schema retries
    // =====================================================================

    fn schema_for_partial_sum() -> Schema {
        Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {
                "partial_sum": { "type": "integer" }
            },
            "required": ["partial_sum"]
        }))
        .expect("valid schema")
    }

    #[tokio::test]
    async fn schema_violation_emits_schema_retried_with_attempt_numbers() {
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("not json")),
            Ok(write_result_response("not json again")),
            Ok(write_result_value(serde_json::json!({"partial_sum": 42}))),
        ]);
        let (events, _, settled) = run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;

        let schema_retries = schema_retries_in(&events);
        let attempts: Vec<u32> = schema_retries.iter().map(|(a, ..)| *a).collect();
        assert_eq!(attempts, vec![1, 2]);
        for (_, max_attempts, _) in &schema_retries {
            assert_eq!(*max_attempts, 10);
        }
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn schema_retry_appends_directive_to_user_message() {
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("not json")),
            Ok(write_result_value(serde_json::json!({"partial_sum": 1}))),
        ]);
        let (events, _, _) = run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;
        // We can't peek at the second request directly without a richer
        // mock. Instead, assert the schema-retry event message carries
        // the validator detail (which is what the directive uses for
        // {detail} substitution).
        let schema_retries = schema_retries_in(&events);
        assert_eq!(schema_retries.len(), 1);
        assert!(
            !schema_retries[0].2.is_empty(),
            "schema-retry message must carry validator detail"
        );
    }

    #[tokio::test]
    async fn schema_retry_exhausted_emits_policy_violated_and_force_fails_ticket() {
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("nope")),
            Ok(write_result_response("still nope")),
            Ok(write_result_response("never")),
        ]);
        let (events, _, settled) = run_one(provider, 3, 2, Some(schema_for_partial_sum())).await;

        let policy_violated = events.iter().any(|e| {
            matches!(
                &e.kind,
                EventKind::PolicyViolated {
                    kind: PolicyKind::MaxSchemaRetries,
                    limit: 2,
                },
            )
        });
        assert!(policy_violated, "expected MaxSchemaRetries PolicyViolated");
        assert_eq!(settled.unwrap().status, Status::Failed);
    }

    // =====================================================================
    // Bucket D — cancellation interactions with retries
    // =====================================================================

    #[tokio::test(start_paused = true)]
    async fn cancel_during_backoff_sleep_aborts_immediately() {
        let provider = MockProvider::with_results(vec![Err(ProviderError::RateLimited {
            message: "rl".into(),
            status: 429,
            retry_delay: Some(Duration::from_secs(60)),
        })]);
        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = Arc::clone(&collected);
            Arc::new(move |e| c.lock().unwrap().push(e))
        };
        let tickets = TicketSystem::new();
        tickets
            .max_request_retries(3)
            .request_retry_delay(Duration::from_secs(60));
        let agent = Agent::new()
            .name("tester")
            .provider(provider as Arc<dyn Provider>)
            .model("mock")
            .role("test")
            .event_handler(handler);
        tickets.agent(agent);
        tickets.task("go");

        let run_fut = tickets.finish();
        let cancel_handle = Arc::clone(&tickets);
        let cancel_fut = async {
            // Let the loop hit the inter-attempt sleep.
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            cancel_handle.cancel();
            // wait_for_signal polls on a 50ms cadence; advance past it.
            tokio::time::advance(Duration::from_millis(100)).await;
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
        };

        let _ = tokio::join!(run_fut, cancel_fut);
        let events = collected.lock().unwrap().clone();
        // One RequestRetried fires (the initial Err triggers it);
        // cancel kicks in during the 60s backoff sleep so the loop
        // exits before any further provider request.
        assert_eq!(retries_in(&events).len(), 1);
        assert!(failures_in(&events).is_empty());
    }

    // =====================================================================
    // Bucket F — cross-ticket memory (body capture)
    // =====================================================================

    /// First user-side text in each `User` message, in order, with the
    /// auto-injected `## Context` block filtered out so cross-ticket
    /// tests can assert on task bodies without re-stating the environment
    /// prelude every time.
    fn user_texts(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .filter_map(|m| match m {
                Message::User { content } => content.iter().find_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .filter(|text| !text.starts_with("## Context\n\n"))
            .collect()
    }

    #[tokio::test]
    async fn messages_contain_only_the_current_tickets_task() {
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("ok")),
            Ok(write_result_response("ok")),
        ]);
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_schema_retries(10)
            .max_time(Duration::from_millis(500));
        tickets.agent(
            Agent::new()
                .name("tester")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .tool(ManageTicketsTool),
        );
        tickets.task("first");
        tickets.task("second");
        let _ = tickets.finish().await;

        let calls = provider.received();
        assert_eq!(calls.len(), 2);
        assert_eq!(user_texts(&calls[0]), vec!["first".to_string()]);
        assert_eq!(user_texts(&calls[1]), vec!["second".to_string()]);
    }

    #[tokio::test]
    async fn model_writes_in_ticket_n_become_visible_in_ticket_n_plus_one_system_prompt() {
        use crate::agents::Knowledge;

        // Ticket 1: model writes a knowledge page (turn 1) then finishes (turn 2).
        // Ticket 2: model finishes immediately (turn 3). The system prompt at
        // turn 3 must contain the index entry written at turn 1.
        let provider = MockProvider::with_results(vec![
            Ok(knowledge_write_response(
                "api-config",
                "API runs on port 3000",
                "# API Config\n\nPort 3000.",
            )),
            Ok(write_result_response("done 1")),
            Ok(write_result_response("done 2")),
        ]);
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let knowledge_dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(knowledge_dir.path()).unwrap();

        let tickets = TicketSystem::new();

        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_time(Duration::from_millis(500));
        tickets.agent(
            Agent::new()
                .name("tester")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .knowledge(&store),
        );
        tickets.task("first");
        tickets.task("second");
        let _ = tickets.finish().await;

        let prompts = provider.received_system_prompts();
        assert_eq!(prompts.len(), 3);
        assert!(
            !prompts[0].contains("api-config"),
            "ticket 1 turn 1 sees an empty knowledge store: {:?}",
            prompts[0]
        );
        assert!(
            prompts[2].contains("## Knowledge"),
            "ticket 2 should render the knowledge section: {:?}",
            prompts[2]
        );
        assert!(
            prompts[2].contains("API runs on port 3000"),
            "ticket 2 should see ticket 1's write: {:?}",
            prompts[2]
        );
    }

    #[tokio::test]
    async fn system_prompt_does_not_change_after_mid_ticket_knowledge_write() {
        use crate::agents::Knowledge;

        // One ticket, two turns: the model writes a knowledge page in turn 1,
        // then finishes in turn 2. The two turns must see byte-identical system
        // prompts so the provider's prefix cache survives the mid-ticket write.
        let provider = MockProvider::with_results(vec![
            Ok(knowledge_write_response(
                "mid-ticket",
                "Written mid-ticket",
                "# Mid\n\nContent.",
            )),
            Ok(write_result_response("ok")),
        ]);
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let knowledge_dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(knowledge_dir.path()).unwrap();

        let tickets = TicketSystem::new();

        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_time(Duration::from_millis(500));
        tickets.agent(
            Agent::new()
                .name("tester")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .knowledge(&store),
        );
        tickets.task("hi");
        let _ = tickets.finish().await;

        let prompts = provider.received_system_prompts();
        assert_eq!(prompts.len(), 2);
        assert_eq!(
            prompts[0], prompts[1],
            "mid-ticket knowledge write must not change the system prompt within the same ticket"
        );
        // Disk write was durable, so the next ticket would see it.
        assert!(store.index().contains("mid-ticket"));
    }

    #[tokio::test]
    async fn agent_a_writes_in_one_ticket_then_agent_b_sees_it_in_its_next_ticket() {
        use crate::agents::Knowledge;

        // Two agents share one Knowledge via the Arc passed to knowledge(&store).
        // Drive alice's ticket to completion first, then enqueue bob's so the
        // ordering is deterministic. Bob's ticket-1 system prompt must show
        // alice's write in the index.
        let p_a = MockProvider::with_results(vec![
            Ok(knowledge_write_response(
                "alice-note",
                "Note from Alice",
                "# Alice\n\nAlice's note.",
            )),
            Ok(write_result_response("alice done")),
        ]);
        let p_b = MockProvider::with_results(vec![Ok(write_result_response("bob done"))]);

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let knowledge_dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(knowledge_dir.path()).unwrap();

        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_time(Duration::from_millis(500));

        tickets.agent(
            Agent::new()
                .name("alice")
                .label("a")
                .provider(p_a.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .knowledge(&store),
        );
        tickets.agent(
            Agent::new()
                .name("bob")
                .label("b")
                .provider(p_b.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .knowledge(&store),
        );

        tickets.task_labeled("alice work", "a");
        let _ = tickets.finish().await;
        assert!(store.index().contains("alice-note"));

        // finish() resets the signal at entry, so no manual reset needed.
        tickets.task_labeled("bob work", "b");
        let _ = tickets.finish().await;

        let bob_prompts = p_b.received_system_prompts();
        assert_eq!(bob_prompts.len(), 1, "bob processed exactly one ticket");
        assert!(
            bob_prompts[0].contains("Note from Alice"),
            "bob should see alice's write: {:?}",
            bob_prompts[0]
        );
    }

    #[tokio::test]
    async fn knowledge_write_then_read_across_tickets() {
        use crate::agents::Knowledge;

        // Two tickets processed sequentially by one agent bound to a Knowledge store.
        //
        // Ticket 1 (3 turns):
        //   1. Model calls knowledge_tool write (api-config)
        //   2. Model calls knowledge_tool read (api-config)
        //   3. Model calls write_result_tool to finish
        //
        // Ticket 2 (1 turn):
        //   1. Model calls write_result_tool immediately

        let provider = MockProvider::with_results(vec![
            // Ticket 1, turn 1: write a page
            Ok(knowledge_write_response(
                "api-config",
                "API runs on port 3000",
                "# API Config\n\nThe API server listens on port 3000.\nRate limit: 100 req/min.\nSee also: [[error-codes]]",
            )),
            // Ticket 1, turn 2: read the page back
            Ok(knowledge_read_response("api-config")),
            // Ticket 1, turn 3: finish
            Ok(write_result_response("done 1")),
            // Ticket 2, turn 1: finish immediately
            Ok(write_result_response("done 2")),
        ]);

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let knowledge_dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(knowledge_dir.path()).unwrap();

        let tickets = TicketSystem::new();

        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_time(Duration::from_millis(500));
        tickets.agent(
            Agent::new()
                .name("tester")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .knowledge(&store),
        );
        tickets.task("first");
        tickets.task("second");
        let _ = tickets.finish().await;

        let prompts = provider.received_system_prompts();
        assert_eq!(prompts.len(), 4);

        // Ticket 1, turn 1: store is empty at start — no ## Knowledge
        assert!(
            !prompts[0].contains("## Knowledge"),
            "ticket 1 turn 1 should not have Knowledge section: {:?}",
            prompts[0]
        );

        // Mid-ticket writes do not change the prompt (prefix cache stability)
        assert_eq!(
            prompts[0], prompts[1],
            "ticket 1 turn 2 prompt must be byte-identical to turn 1"
        );
        assert_eq!(
            prompts[0], prompts[2],
            "ticket 1 turn 3 prompt must be byte-identical to turn 1"
        );

        // Ticket 2, turn 1: the index is visible
        assert!(
            prompts[3].contains("## Knowledge"),
            "ticket 2 should render the knowledge section: {:?}",
            prompts[3]
        );
        assert!(
            prompts[3].contains("api-config"),
            "ticket 2 should see the page slug: {:?}",
            prompts[3]
        );
        assert!(
            prompts[3].contains("API runs on port 3000"),
            "ticket 2 should see the index summary: {:?}",
            prompts[3]
        );
        // The full page body should NOT be in the prompt — only the index summary
        assert!(
            !prompts[3].contains("Rate limit: 100 req/min"),
            "ticket 2 should NOT contain full page body: {:?}",
            prompts[3]
        );

        // Disk state: page file exists with correct content
        let page_path = knowledge_dir.path().join("pages").join("api-config.md");
        assert!(page_path.exists(), "page file should exist on disk");
        let page_raw = std::fs::read_to_string(&page_path).unwrap();
        assert!(page_raw.contains("Rate limit: 100 req/min"));
        assert!(page_raw.contains("---")); // frontmatter present

        // Disk state: index.md exists with correct entry
        let index_path = knowledge_dir.path().join("index.md");
        assert!(index_path.exists(), "index.md should exist on disk");
        let index_raw = std::fs::read_to_string(&index_path).unwrap();
        assert!(index_raw.contains("- **api-config** — API runs on port 3000"));

        // The read action (turn 2) should have returned the body WITHOUT frontmatter.
        // We verify this by checking the messages the provider received: turn 3's
        // input includes the tool result from the read action (the last user
        // message before the assistant response at turn 3).
        let received = provider.received();
        let turn3_messages = &received[2];
        // Collect ALL tool results from the messages sent at turn 3.
        let all_tool_results: Vec<&String> = turn3_messages
            .iter()
            .filter_map(|m| match m {
                Message::User { content } => Some(
                    content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolResult { content, .. } => Some(content),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .flatten()
            .collect();
        // The read result is the one that contains the page body, not the
        // "page written" confirmation from the write action.
        let read_result = all_tool_results
            .iter()
            .find(|r| !r.starts_with("page written"))
            .expect("should have a non-write tool result (the read result)");
        assert!(
            !read_result.contains("---"),
            "read result should not contain frontmatter delimiters: {read_result}"
        );
        assert!(
            !read_result.contains("updated:"),
            "read result should not contain updated field: {read_result}"
        );
        assert!(
            read_result.contains("Rate limit: 100 req/min"),
            "read result should contain page body: {read_result}"
        );
    }

    // ---- late-add tests ----
    //
    // (No companion test for "supervisor does not re-spawn the same agent on
    //  every poll": with synchronous mock providers, observable side effects
    //  collapse to one provider call regardless of whether the agent task is
    //  spawned once or many times, because the only ticket transitions to
    //  Done atomically before any second poll could race. The index-tracker
    //  correctness is verified by inspection of `run_main_loop`.)

    #[tokio::test]
    async fn add_after_run_spawns_new_agent() {
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));

        let run_handle = tickets.start();

        // Let the first scan run (no agents registered yet).
        tokio::time::sleep(Duration::from_millis(150)).await;

        let provider = MockProvider::with_results(vec![Ok(write_result_response("ok"))]);
        tickets.agent(
            Agent::new()
                .name("late")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .tool(ManageTicketsTool),
        );
        tickets.ticket(Ticket::new("hello").label("late"));

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let done = tickets
                .tickets()
                .iter()
                .any(|t| t.status == Status::Done && t.task.as_str() == Some("hello"));
            if done {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                run_handle.stop().await;
                panic!("late-added agent did not finish ticket within 5s");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        run_handle.stop().await;

        assert_eq!(provider.requests(), 1);
    }

    #[tokio::test]
    async fn late_added_agent_joined_on_shutdown() {
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));

        let run_handle = tickets.start();

        tokio::time::sleep(Duration::from_millis(150)).await;

        let provider = MockProvider::with_results(vec![Ok(write_result_response("ok"))]);
        tickets.agent(
            Agent::new()
                .name("late")
                .provider(provider as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .tool(ManageTicketsTool),
        );
        tickets.ticket(Ticket::new("x").label("late"));

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let done = tickets
                .tickets()
                .iter()
                .any(|t| t.status == Status::Done && t.task.as_str() == Some("x"));
            if done {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                run_handle.stop().await;
                panic!("late-added agent did not finish ticket within 5s");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // The run must join the late-spawned task on shutdown rather than
        // orphan it. If it did orphan it, start() would still return on signal
        // flip, but the late task would dangle.
        tokio::time::timeout(Duration::from_secs(2), run_handle.stop())
            .await
            .expect("start() did not return within 2s of signal flip");
    }

    // ---- Run lifecycle tests ----

    #[tokio::test]
    async fn finish_drains_late_added_tickets() {
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("a-done")),
            Ok(write_result_response("b-done")),
        ]);
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));
        tickets.agent(
            Agent::new()
                .name("worker")
                .provider(provider as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .tool(ManageTicketsTool),
        );

        tickets.start();

        // Queue tickets after the run is in flight.
        tickets.task("a");
        tickets.task("b");

        let results = tokio::time::timeout(Duration::from_secs(5), tickets.finish())
            .await
            .expect("finish did not finish within 5s");

        assert_eq!(results.all_results().len(), 2);
        assert_eq!(results.last_result().as_deref(), Some("b-done"));
    }

    #[tokio::test]
    async fn cancel_stops_a_running_workshop() {
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));

        tickets.start();
        tickets.cancel();

        tokio::time::timeout(Duration::from_secs(2), tickets.stop())
            .await
            .expect("run did not exit within 2s of cancel()");
    }

    #[tokio::test]
    async fn stop_is_abrupt() {
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));

        tickets.start();

        tokio::time::timeout(Duration::from_secs(2), tickets.stop())
            .await
            .expect("run did not exit within 2s of stop()");
    }

    #[tokio::test]
    async fn finish_after_run_resets_signal() {
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("first")),
            Ok(write_result_response("second")),
        ]);
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));
        tickets.agent(
            Agent::new()
                .name("worker")
                .provider(provider as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .tool(ManageTicketsTool),
        );

        // First run: spawn, finish. Leaves the interrupt signal flipped.
        tickets.task("first");
        tickets.finish().await;
        assert_eq!(tickets.last_result().as_deref(), Some("first"));

        // Second run must reset the signal at entry; otherwise the run
        // exits before claiming the new ticket.
        tickets.task("second");
        tokio::time::timeout(Duration::from_secs(5), tickets.finish())
            .await
            .expect("second finish did not finish within 5s");
        assert_eq!(tickets.last_result().as_deref(), Some("second"));
    }

    #[tokio::test]
    async fn agent_finish_forwards_to_bound_system() {
        let results_dir = crate::test_util::TempDir::new().unwrap();
        let provider = MockProvider::with_results(vec![Ok(write_result_response("forwarded"))]);
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1));
        let agent = tickets.agent(
            Agent::new()
                .name("worker")
                .provider(provider as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .tool(ManageTicketsTool),
        );

        agent.task("hello");
        let sys = tokio::time::timeout(Duration::from_secs(5), agent.finish())
            .await
            .expect("agent.finish did not finish within 5s");
        assert_eq!(sys.last_result().as_deref(), Some("forwarded"));
    }

    // =====================================================================
    // Bucket G — context-window compaction warnings
    // =====================================================================

    #[tokio::test]
    async fn first_overflow_attempts_compaction_before_request_failed() {
        // Single scripted overflow with no tail to summarize (messages
        // is just [ctx, task] when overflow hits, so the summarizer
        // short-circuits and returns Ok). The contract: the loop tries
        // compaction before the ticket fails. The Started/Finished
        // pair fires from the attempt, then turn 2 surfaces the
        // RequestFailed via the exhausted-fallback.
        let provider =
            MockProvider::with_results(vec![Err(ProviderError::ContextWindowExceeded {
                message: "prompt is 250000 tokens, exceeds 200000".into(),
            })]);
        let (events, _, _) = run_one(provider, 0, 10, None).await;

        let started_idx = events
            .iter()
            .position(|e| matches!(&e.kind, EventKind::CompactionStarted { .. }))
            .expect("compaction must have started");
        let finished_idx = events
            .iter()
            .position(|e| matches!(&e.kind, EventKind::CompactionFinished { .. }))
            .expect("compaction must have finished (no-op when tail is empty)");
        let request_failed_idx = events
            .iter()
            .position(|e| matches!(&e.kind, EventKind::RequestFailed { .. }))
            .expect("the ticket must surface a request failure");
        assert!(started_idx < finished_idx);
        assert!(finished_idx < request_failed_idx);
    }

    /// Tool-less assistant reply with caller-chosen usage. Used to drive
    /// the summarizer call inside the compaction tests, and to seed
    /// `last_usage` for the proactive-threshold test.
    fn text_response_with_usage(text: &str, usage: TokenUsage) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            status: ResponseStatus::EndTurn,
            usage,
            model: "mock".into(),
        }
    }

    fn compaction_started_count(events: &[Event], expected: CompactReason) -> usize {
        events
            .iter()
            .filter(|e| match &e.kind {
                EventKind::CompactionStarted { reason } => *reason == expected,
                _ => false,
            })
            .count()
    }

    fn compaction_finished_count(events: &[Event], expected: CompactReason) -> usize {
        events
            .iter()
            .filter(|e| match &e.kind {
                EventKind::CompactionFinished { reason } => *reason == expected,
                _ => false,
            })
            .count()
    }

    #[tokio::test]
    async fn reactive_overflow_compacts_then_succeeds() {
        // Sequence (4 provider calls):
        //   1. text reply — pads the message tail
        //   2. ContextWindowExceeded — main request rejected, summarize fires
        //   3. "SUMMARY" — the summarize call's response
        //   4. write_result — settles the ticket
        //
        // The fourth request must carry the compacted history:
        // [task, SUMMARY] (no context message; run_one suppresses it).
        let provider = MockProvider::with_results(vec![
            Ok(text_response("turn 1")),
            Err(ProviderError::ContextWindowExceeded {
                message: "exceeded".into(),
            }),
            Ok(text_response_with_usage("SUMMARY", TokenUsage::default())),
            Ok(write_result_response("ok")),
        ]);
        let (events, provider, settled) = run_one(provider, 0, 10, None).await;

        assert_eq!(provider.requests(), 4);
        assert_eq!(
            compaction_started_count(&events, CompactReason::Reactive),
            1
        );
        assert_eq!(
            compaction_finished_count(&events, CompactReason::Reactive),
            1
        );
        assert!(failures_in(&events).is_empty());
        assert_eq!(settled.unwrap().status, Status::Done);

        // Prove compaction actually happened: the fourth (retried)
        // request's user-side texts are just the task and the summary,
        // not the original turn-1 trail.
        let fourth = &provider.received()[3];
        assert_eq!(
            user_texts(fourth),
            vec!["go".to_string(), "SUMMARY".to_string()]
        );
    }

    #[tokio::test]
    async fn reactive_overflow_twice_in_a_row_fails_the_ticket() {
        // Turn 1 pads the messages, turn 2 overflows (summarize runs),
        // turn 3 overflows again. Two consecutive overflows trip the
        // circuit breaker and the first RequestFailed carries the
        // synthesized "after compaction" message. The same Path-A
        // re-claim caveat as the other terminal-error tests applies,
        // so later failures come from the MockProvider's
        // exhausted-fallback.
        let provider = MockProvider::with_results(vec![
            Ok(text_response("turn 1")),
            Err(ProviderError::ContextWindowExceeded {
                message: "first overflow".into(),
            }),
            Ok(text_response_with_usage("SUMMARY", TokenUsage::default())),
            Err(ProviderError::ContextWindowExceeded {
                message: "second overflow".into(),
            }),
        ]);
        let (events, _, _) = run_one(provider, 0, 10, None).await;

        // First overflow: Started + Finished pair (compaction succeeded).
        // Second overflow: circuit breaker trips before Started can fire.
        assert_eq!(
            compaction_started_count(&events, CompactReason::Reactive),
            1
        );
        assert_eq!(
            compaction_finished_count(&events, CompactReason::Reactive),
            1
        );
        let failures = failures_in(&events);
        assert!(!failures.is_empty());
        assert!(
            failures[0].contains("after compaction"),
            "expected the synthesized circuit-breaker message, got {:?}",
            failures[0],
        );
    }

    /// Run one ticket against `provider` with a model whose context
    /// window is known (so proactive compaction can fire) and without
    /// the auto-injected context prelude (so `head_len` is exactly 1
    /// and the test can reason about request shapes precisely).
    async fn run_one_with_real_model(
        provider: Arc<MockProvider>,
    ) -> (Vec<Event>, Arc<MockProvider>, Option<Ticket>) {
        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = Arc::clone(&collected);
            Arc::new(move |e| c.lock().unwrap().push(e))
        };

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_schema_retries(10)
            .max_time(Duration::from_millis(200));

        let agent = Agent::new()
            .name("tester")
            .provider(provider.clone() as Arc<dyn Provider>)
            .model("claude-sonnet-4-20250514")
            .role("test")
            .context("static") // suppresses the dynamic context-message default
            .tool(ManageTicketsTool)
            .event_handler(handler);
        tickets.agent(agent);
        tickets.task("go");

        let _ = tickets.finish().await;
        let events = collected.lock().unwrap().clone();
        let settled = tickets.first();
        (events, provider, settled)
    }

    #[tokio::test]
    async fn proactive_threshold_triggers_compaction_before_next_request() {
        // Turn 1: text reply with input_tokens above the 167K threshold
        //         (200K window − 20K reserve − 13K headroom). No tool
        //         call, so the loop pushes a retry directive and loops.
        // Top of turn 2: proactive_event fires; summarizer collapses
        //                the messages tail to [ctx, task, "SUMMARY"].
        // Turn 2:  write_result_response finishes the ticket.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage(
                "thinking...",
                TokenUsage {
                    input_tokens: 170_000,
                    output_tokens: 0,
                },
            )),
            Ok(text_response_with_usage("SUMMARY", TokenUsage::default())),
            Ok(write_result_response("done")),
        ]);
        let (events, provider, settled) = run_one_with_real_model(provider).await;

        assert_eq!(provider.requests(), 3);
        assert_eq!(
            compaction_started_count(&events, CompactReason::Proactive),
            1
        );
        assert_eq!(
            compaction_finished_count(&events, CompactReason::Proactive),
            1
        );
        assert_eq!(settled.unwrap().status, Status::Done);

        // The third request — the retry after compaction — sees the
        // compacted message vector: context, task, summary.
        let third = &provider.received()[2];
        assert_eq!(third.len(), 3);
        match &third[2] {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => assert_eq!(text, "SUMMARY"),
                other => panic!("expected text summary, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }

        // Both compaction events must fall between turn 1's
        // RequestStarted and turn 2's RequestStarted: proactive
        // compaction runs after the first response and before the
        // retried request.
        let started_idx = events
            .iter()
            .position(|e| matches!(&e.kind, EventKind::CompactionStarted { .. }))
            .expect("compaction must start");
        let finished_idx = events
            .iter()
            .position(|e| matches!(&e.kind, EventKind::CompactionFinished { .. }))
            .expect("compaction must finish");
        let request_started: Vec<usize> = events
            .iter()
            .enumerate()
            .filter_map(|(i, e)| matches!(&e.kind, EventKind::RequestStarted { .. }).then_some(i))
            .collect();
        assert!(request_started.len() >= 2);
        assert!(started_idx > request_started[0] && started_idx < request_started[1]);
        assert!(finished_idx > started_idx && finished_idx < request_started[1]);
    }

    #[tokio::test]
    async fn summarize_rate_limited_kills_ticket_without_retry() {
        // Turn 1: text reply with 170K input_tokens primes the
        // proactive seam to fire at the top of turn 2 (170K + a
        // trivial bytes/4 > 167K threshold for the 200K Sonnet
        // window).
        // Turn 2 (summarize call): RateLimited. The variant is
        // retryable in the main request loop, but summarize_and_replace
        // has no retry policy: the error propagates immediately,
        // CompactionFailed{Proactive} fires, and the ticket dies.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage(
                "thinking...",
                TokenUsage {
                    input_tokens: 170_000,
                    output_tokens: 0,
                },
            )),
            Err(rate_limit()),
        ]);
        let (events, _, _) = run_one_with_real_model(provider).await;

        assert_eq!(
            compaction_started_count(&events, CompactReason::Proactive),
            1,
        );
        assert!(
            events.iter().any(|e| matches!(
                &e.kind,
                EventKind::CompactionFailed {
                    reason: CompactReason::Proactive,
                    message,
                } if message.contains("rate limited"),
            )),
            "rate-limit error must surface verbatim in CompactionFailed{{Proactive}}",
        );
        assert!(
            retries_in(&events).is_empty(),
            "summarize call has no retry policy; got {:?}",
            retries_in(&events),
        );
        let failures = failures_in(&events);
        assert!(!failures.is_empty(), "ticket must surface a request failure");
        assert!(
            failures[0].contains("rate limited"),
            "first failure must carry the rate-limited error; got {:?}",
            failures[0],
        );
    }

    #[tokio::test]
    async fn summary_empty_text_replaces_tail_with_empty_user_message() {
        // Turn 1: high-input-tokens text reply forces proactive on
        // turn 2.
        // Turn 2 (summarize call): Ok with empty text. summarize_and_replace
        // accepts it as a valid summary today, so the tail collapses
        // to a single user message whose content is "".
        // Turn 3 (retried main request): write_result_response settles
        // the ticket.
        //
        // Documents that summarize_and_replace does not reject empty
        // summaries: the loop will continue with effectively no
        // working context.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage(
                "thinking...",
                TokenUsage {
                    input_tokens: 170_000,
                    output_tokens: 0,
                },
            )),
            Ok(text_response_with_usage("", TokenUsage::default())),
            Ok(write_result_response("done")),
        ]);
        let (events, provider, settled) = run_one_with_real_model(provider).await;

        assert_eq!(
            compaction_started_count(&events, CompactReason::Proactive),
            1,
        );
        assert_eq!(
            compaction_finished_count(&events, CompactReason::Proactive),
            1,
            "empty text counts as a valid summary today",
        );
        assert_eq!(settled.unwrap().status, Status::Done);

        // The third request sees [ctx, task, user("")]: the empty
        // user message is the collapsed summary.
        let third = &provider.received()[2];
        assert_eq!(third.len(), 3);
        match &third[2] {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => assert_eq!(text, ""),
                other => panic!("expected empty text block, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn response_status_context_window_exceeded_triggers_reactive_compaction() {
        // A successful response carrying `status=ContextWindowExceeded`
        // is the same overflow signal as `ProviderError::ContextWindowExceeded`,
        // delivered on a 200 OK. Layer 3 routes it through
        // `handle_overflow`. The setup turn before the overflow
        // grows the tail past the summarizer's two-message no-op
        // floor; otherwise summarize_and_replace returns Ok(()) with
        // no provider call and the next response slot would shift.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage("thinking", TokenUsage::default())),
            Ok(ModelResponse {
                content: vec![ContentBlock::Text {
                    text: "oops".into(),
                }],
                status: ResponseStatus::ContextWindowExceeded,
                usage: TokenUsage::default(),
                model: "mock".into(),
            }),
            Ok(text_response_with_usage("SUMMARY", TokenUsage::default())),
            Ok(write_result_response("recovered")),
        ]);
        let (events, _, settled) = run_one_with_real_model(provider).await;

        assert_eq!(
            compaction_started_count(&events, CompactReason::Reactive),
            1,
            "ResponseStatus::ContextWindowExceeded must trigger reactive compaction",
        );
        assert_eq!(
            compaction_finished_count(&events, CompactReason::Reactive),
            1,
        );
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn response_status_context_window_exceeded_consumes_compaction_retry_budget() {
        // After the setup turn, two consecutive responses carry the
        // overflow status. The first consumes the ImmediateRetry
        // budget via handle_overflow; the second finds no budget
        // left (the reset below the status branch was skipped on the
        // overflow path) and the ticket fails with the synthesized
        // "after compaction" message.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage("thinking", TokenUsage::default())),
            Ok(ModelResponse {
                content: vec![ContentBlock::Text {
                    text: "first overflow".into(),
                }],
                status: ResponseStatus::ContextWindowExceeded,
                usage: TokenUsage::default(),
                model: "mock".into(),
            }),
            Ok(text_response_with_usage("SUMMARY", TokenUsage::default())),
            Ok(ModelResponse {
                content: vec![ContentBlock::Text {
                    text: "second overflow".into(),
                }],
                status: ResponseStatus::ContextWindowExceeded,
                usage: TokenUsage::default(),
                model: "mock".into(),
            }),
        ]);
        let (events, _, _) = run_one_with_real_model(provider).await;

        assert_eq!(
            compaction_started_count(&events, CompactReason::Reactive),
            1,
            "only the first overflow consumes the retry budget",
        );
        let failures = failures_in(&events);
        assert!(!failures.is_empty());
        assert!(
            failures[0].contains("after compaction"),
            "second overflow must surface the exhausted-budget message; got {:?}",
            failures[0],
        );
    }

    #[tokio::test]
    async fn huge_tool_result_is_persisted_as_knowledge_page_and_ticket_finishes_done() {
        use crate::agents::Knowledge;
        use crate::tools::{Tool, ToolResult};

        // Layer 1: an ~800 KB tool result is far above PER_TOOL_CAP
        // (50K). ToolRegistry::execute caps it to a stub before the
        // ContentBlock::ToolResult lands in messages, and persists the
        // full content as a Knowledge page named `tool-result-call-1`.
        // The model then finishes the ticket in one more turn. No
        // compaction fires; no failure surfaces.
        let provider = MockProvider::with_results(vec![
            Ok(ModelResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "dump".into(),
                    input: serde_json::json!({}),
                }],
                status: ResponseStatus::ToolUse,
                usage: TokenUsage::default(),
                model: "mock".into(),
            }),
            Ok(write_result_response("done")),
        ]);

        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = Arc::clone(&collected);
            Arc::new(move |e| c.lock().unwrap().push(e))
        };

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let knowledge_dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(knowledge_dir.path()).unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_schema_retries(10)
            .max_time(Duration::from_millis(500));

        let dump = Tool::new("dump", "Returns ~800 KB of text")
            .handler(|_input, _ctx| async move {
                Ok(ToolResult::success("x".repeat(800_000)))
            });

        let agent = Agent::new()
            .name("tester")
            .provider(provider.clone() as Arc<dyn Provider>)
            .model("claude-sonnet-4-20250514")
            .role("test")
            .context("static")
            .knowledge(&store)
            .tool(dump)
            .event_handler(handler);
        tickets.agent(agent);
        tickets.task("go");

        let _ = tickets.finish().await;
        let events = collected.lock().unwrap().clone();
        let settled = tickets.first();

        assert_eq!(provider.requests(), 2);
        assert_eq!(
            compaction_started_count(&events, CompactReason::Proactive),
            0,
            "Layer 1 prevents the messages from ever crossing the proactive threshold",
        );
        assert!(failures_in(&events).is_empty());
        assert_eq!(settled.unwrap().status, Status::Done);

        // The Knowledge store carries the full original content under
        // the expected slug, and the index summary names the tool.
        let body = store
            .read_page("tool-result-call-1")
            .expect("knowledge page must exist");
        assert!(body.contains(&"x".repeat(800_000)));
        assert!(store.index().contains("dump output"));

        // The second request (after the tool call) sees the stub in
        // place of the giant blob.
        let stub_visible = provider.received()[1].iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::ToolResult { content, .. } => {
                    content.contains("<persisted-output>")
                        && content.contains("tool-result-call-1")
                }
                _ => false,
            }),
            _ => false,
        });
        assert!(stub_visible, "stub must appear in the second request's messages");
    }

    #[tokio::test]
    async fn proactive_compact_does_not_consume_reactive_budget() {
        // Both seams must fire on the same ticket. To trip proactive
        // without relying on tool-result bytes (which Layer 1 now
        // stubs), prime `last_usage` via a high input_tokens reply.
        //
        // Sequence:
        //   1. text(170K input)  : primes proactive on step 2.
        //   2. text("SUMMARY-A") : proactive summarize on step 2.
        //   3. text(default usage): main of step 2; clears last_usage
        //                           so proactive is quiet on step 3.
        //   4. Err(ContextWindowExceeded): main of step 3 overflows;
        //                                  reactive consumes its budget.
        //   5. text("SUMMARY-B") : reactive summarize on step 3.
        //   6. write_result_response: settles the ticket on step 3 retry.
        //
        // Proactive must not consume compaction_retry: the reactive
        // seam in step 3 still has its full ImmediateRetry budget.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage(
                "thinking...",
                TokenUsage {
                    input_tokens: 170_000,
                    output_tokens: 0,
                },
            )),
            Ok(text_response_with_usage("SUMMARY-A", TokenUsage::default())),
            Ok(text_response_with_usage("thinking again", TokenUsage::default())),
            Err(ProviderError::ContextWindowExceeded {
                message: "main request overflow after proactive".into(),
            }),
            Ok(text_response_with_usage("SUMMARY-B", TokenUsage::default())),
            Ok(write_result_response("done")),
        ]);
        let (events, provider, settled) = run_one_with_real_model(provider).await;

        assert_eq!(provider.requests(), 6);
        assert_eq!(
            compaction_started_count(&events, CompactReason::Proactive),
            1,
        );
        assert_eq!(
            compaction_finished_count(&events, CompactReason::Proactive),
            1,
        );
        assert_eq!(
            compaction_started_count(&events, CompactReason::Reactive),
            1,
            "reactive must have a full budget after a successful proactive",
        );
        assert_eq!(
            compaction_finished_count(&events, CompactReason::Reactive),
            1,
        );
        assert!(failures_in(&events).is_empty());
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn parallel_moderate_results_aggregate_offloads_largest_first() {
        use crate::agents::Knowledge;
        use crate::tools::{Tool, ToolResult};

        // Five parallel calls to `size_tool`, each below PER_TOOL_CAP
        // (50K) but together over PER_STEP_CAP (200K). The aggregate
        // pass stubs the largest first; one offload brings the step
        // under budget, so only `c1` is replaced.
        let provider = MockProvider::with_results(vec![
            Ok(ModelResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "size_tool".into(),
                        input: serde_json::json!({"bytes": 48_000}),
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "size_tool".into(),
                        input: serde_json::json!({"bytes": 47_000}),
                    },
                    ContentBlock::ToolUse {
                        id: "c3".into(),
                        name: "size_tool".into(),
                        input: serde_json::json!({"bytes": 46_000}),
                    },
                    ContentBlock::ToolUse {
                        id: "c4".into(),
                        name: "size_tool".into(),
                        input: serde_json::json!({"bytes": 45_000}),
                    },
                    ContentBlock::ToolUse {
                        id: "c5".into(),
                        name: "size_tool".into(),
                        input: serde_json::json!({"bytes": 44_000}),
                    },
                ],
                status: ResponseStatus::ToolUse,
                usage: TokenUsage::default(),
                model: "mock".into(),
            }),
            Ok(write_result_response("done")),
        ]);

        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = Arc::clone(&collected);
            Arc::new(move |e| c.lock().unwrap().push(e))
        };

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let knowledge_dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(knowledge_dir.path()).unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_schema_retries(10)
            .max_time(Duration::from_millis(500));

        let size_tool = Tool::new("size_tool", "Returns N bytes of 'x'")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {"bytes": {"type": "integer"}},
                "required": ["bytes"],
            }))
            .read_only(true)
            .handler(|input, _ctx| async move {
                let bytes = input["bytes"].as_u64().unwrap_or(0) as usize;
                Ok(ToolResult::success("x".repeat(bytes)))
            });

        tickets.agent(
            Agent::new()
                .name("tester")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .knowledge(&store)
                .tool(size_tool)
                .event_handler(handler),
        );
        tickets.task("go");

        let _ = tickets.finish().await;
        let settled = tickets.first();
        assert_eq!(settled.unwrap().status, Status::Done);

        // The second request carries one stub (for c1) and four
        // unchanged tool results (c2..c5).
        let second = &provider.received()[1];
        let tool_results: Vec<&String> = second
            .iter()
            .flat_map(|m| match m {
                Message::User { content } => content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolResult { content, .. } => Some(content),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect();
        let stub_count = tool_results
            .iter()
            .filter(|c| c.starts_with("<persisted-output>"))
            .count();
        assert_eq!(stub_count, 1, "exactly one tool result gets offloaded");

        // The lone stub belongs to c1 (the largest input).
        let stub = tool_results
            .iter()
            .find(|c| c.starts_with("<persisted-output>"))
            .expect("stub must be present");
        assert!(stub.contains("tool-result-c1"));

        // The offloaded content survives in Knowledge at full size.
        let body = store.read_page("tool-result-c1").unwrap();
        assert!(body.starts_with(&"x".repeat(48_000)));
    }

    #[tokio::test]
    async fn next_ticket_sees_offloaded_results_in_knowledge_index() {
        use crate::agents::Knowledge;
        use crate::tools::{Tool, ToolResult};

        // Ticket 1 offloads a giant `dump` result; ticket 2 should
        // see the page under `## Knowledge` in its system prompt at
        // step 1 (knowledge index is captured at the top of each
        // ticket).
        let provider = MockProvider::with_results(vec![
            Ok(ModelResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "dump".into(),
                    input: serde_json::json!({}),
                }],
                status: ResponseStatus::ToolUse,
                usage: TokenUsage::default(),
                model: "mock".into(),
            }),
            Ok(write_result_response("first")),
            Ok(write_result_response("second")),
        ]);

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let knowledge_dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(knowledge_dir.path()).unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_time(Duration::from_millis(500));

        let dump = Tool::new("dump", "Returns ~800 KB of text")
            .handler(|_input, _ctx| async move {
                Ok(ToolResult::success("x".repeat(800_000)))
            });

        tickets.agent(
            Agent::new()
                .name("tester")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("mock")
                .role("test")
                .knowledge(&store)
                .tool(dump),
        );
        tickets.task("first");
        tickets.task("second");
        let _ = tickets.finish().await;

        let prompts = provider.received_system_prompts();
        assert_eq!(prompts.len(), 3);
        assert!(
            !prompts[0].contains("tool-result-call-1"),
            "ticket 1's first step has an empty knowledge index",
        );
        assert!(
            prompts[2].contains("## Knowledge"),
            "ticket 2 should render the knowledge section: {:?}",
            prompts[2],
        );
        assert!(
            prompts[2].contains("tool-result-call-1"),
            "ticket 2 should see the offloaded slug: {:?}",
            prompts[2],
        );
        assert!(
            prompts[2].contains("dump output"),
            "ticket 2's index entry names the tool: {:?}",
            prompts[2],
        );
    }

    // =====================================================================
    // Bucket H — Layer 2 blocking limit (pre-flight)
    // =====================================================================

    fn blocking_limit_count(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::BlockingLimitExceeded { .. }))
            .count()
    }

    #[tokio::test]
    async fn blocking_limit_exceeded_emits_event_and_skips_provider_call() {
        // Step 1 primes last_usage above the 197K blocking threshold
        // for the 200K Sonnet window. On step 2 both seams cross
        // their lines: proactive runs first, blocking runs after, and
        // handle_overflow routes the synthetic overflow through the
        // reactive seam.
        //
        // The contract: every BlockingLimitExceeded is immediately
        // followed by a CompactionStarted{Reactive} (or a
        // RequestFailed when the budget is exhausted), and never by a
        // RequestStarted. The synthetic path does not call the
        // provider.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage(
                "thinking",
                TokenUsage {
                    input_tokens: 198_000,
                    output_tokens: 0,
                },
            )),
            Ok(text_response_with_usage("SUMMARY-A", TokenUsage::default())),
            Ok(text_response_with_usage("SUMMARY-B", TokenUsage::default())),
            Ok(text_response_with_usage("SUMMARY-C", TokenUsage::default())),
        ]);
        let (events, _, _) = run_one_with_real_model(provider).await;

        assert!(
            blocking_limit_count(&events) >= 1,
            "blocking guard must trip when estimate >= window - 3K",
        );

        for window in events.windows(2) {
            if matches!(&window[0].kind, EventKind::BlockingLimitExceeded { .. }) {
                assert!(
                    matches!(
                        &window[1].kind,
                        EventKind::CompactionStarted { reason: CompactReason::Reactive }
                            | EventKind::RequestFailed { .. },
                    ),
                    "BlockingLimitExceeded must be followed by CompactionStarted{{Reactive}} \
                     or RequestFailed, never a provider call. Got: {:?}",
                    window[1].kind,
                );
            }
        }
    }

    #[tokio::test]
    async fn blocking_limit_uses_compaction_retry_budget() {
        // Two iterations trip the blocking guard. The first consumes
        // the ImmediateRetry budget via handle_overflow; the second
        // finds no budget left and fail_request surfaces the
        // synthesized "context still exceeds window after compaction"
        // message.
        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage("low", TokenUsage::default())),
            Ok(text_response_with_usage(
                "huge",
                TokenUsage {
                    input_tokens: 198_000,
                    output_tokens: 0,
                },
            )),
            Ok(text_response_with_usage("SUMMARY-A", TokenUsage::default())),
            Ok(text_response_with_usage("SUMMARY-B", TokenUsage::default())),
            Ok(text_response_with_usage("SUMMARY-C", TokenUsage::default())),
        ]);
        let (events, _, _) = run_one_with_real_model(provider).await;

        assert!(
            blocking_limit_count(&events) >= 2,
            "two consecutive iterations should trip the blocking guard",
        );

        let failures = failures_in(&events);
        assert!(!failures.is_empty());
        assert!(
            failures[0].contains("after compaction"),
            "first failure must carry the synthesized \"after compaction\" message; got {:?}",
            failures[0],
        );
    }

    #[tokio::test]
    async fn blocking_limit_does_not_fire_in_first_iteration() {
        // last_usage is None on step 1, so the estimate floor is 0 +
        // (tiny prompt/tools/messages) / 4. Far below the 197K
        // blocking threshold. No BlockingLimitExceeded event.
        let provider = MockProvider::with_results(vec![Ok(write_result_response("done"))]);
        let (events, _, settled) = run_one_with_real_model(provider).await;

        assert_eq!(blocking_limit_count(&events), 0);
        assert_eq!(settled.unwrap().status, Status::Done);
    }

    #[tokio::test]
    async fn blocking_limit_includes_system_prompt_and_tools_in_estimate() {
        // last_usage = 195_500 sits below the 197K blocking threshold
        // on its own. A tool with a hefty description adds ~1.5K
        // tokens (6K bytes / 4) to the estimate; the addition pushes
        // the total over the line and the blocking guard fires.
        // Without that tool contribution, blocking would not trip.
        use crate::tools::{Tool, ToolResult};

        let provider = MockProvider::with_results(vec![
            Ok(text_response_with_usage(
                "thinking",
                TokenUsage {
                    input_tokens: 195_500,
                    output_tokens: 0,
                },
            )),
            Ok(text_response_with_usage("SUMMARY-A", TokenUsage::default())),
            Ok(text_response_with_usage("SUMMARY-B", TokenUsage::default())),
            Ok(text_response_with_usage("SUMMARY-C", TokenUsage::default())),
        ]);

        let collected: Arc<StdMutex<Vec<Event>>> = Arc::new(StdMutex::new(Vec::new()));
        let handler: Arc<dyn Fn(Event) + Send + Sync> = {
            let c = Arc::clone(&collected);
            Arc::new(move |e| c.lock().unwrap().push(e))
        };

        let big_desc = "x".repeat(6_000);
        let big_tool = Tool::new("big_tool", big_desc)
            .handler(|_input, _ctx| async { Ok(ToolResult::success("ok")) });

        let results_dir = crate::test_util::TempDir::new().unwrap();
        let tickets = TicketSystem::new();
        tickets
            .dir(results_dir.path().to_path_buf())
            .max_request_retries(0)
            .request_retry_delay(Duration::from_millis(1))
            .max_schema_retries(10)
            .max_time(Duration::from_millis(500));

        tickets.agent(
            Agent::new()
                .name("tester")
                .provider(provider.clone() as Arc<dyn Provider>)
                .model("claude-sonnet-4-20250514")
                .role("test")
                .context("static")
                .tool(big_tool)
                .event_handler(handler),
        );
        tickets.task("go");

        let _ = tickets.finish().await;
        let events = collected.lock().unwrap().clone();

        assert!(
            blocking_limit_count(&events) >= 1,
            "tool-description bytes must contribute to the estimate and push it over threshold",
        );
    }
}
