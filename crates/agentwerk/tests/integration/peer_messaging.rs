//! End-to-end coverage for peer agent messaging via `SendMessageTool`.
//!
//! Uses a real LLM provider (`make test_integration`). The orchestrator
//! backgrounds a worker, sends it a message, and we verify the flow
//! executed: both agents ran, the orchestrator made at least two tool
//! calls (spawn + send), and the worker completed.

use super::common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use agentwerk::{Agent, Event, EventKind, SendMessageTool};

#[tokio::test]
async fn orchestrator_sends_message_to_backgrounded_worker(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    // Fresh secret per run — a hardcoded value can match by coincidence
    // even when peer messaging is broken.
    let secret: u32 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos()
        % 90_000
        + 10_000;

    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = events.clone();
    let event_handler = Arc::new(move |e: Event| {
        if let Some(line) = format_event(&e) {
            eprintln!("[{}] {line}", e.agent_name);
        }
        collected.lock().unwrap().push(e);
    });

    // Shared cancel signal — set on the orchestrator; the worker inherits it
    // via Agent::compile (child path). We flip it from the test body after the
    // worker has had time to process the peer message, to release its idle wait.
    let cancel = Arc::new(AtomicBool::new(false));

    let worker = Agent::new()
        .name("worker")
        .model_name(&model)
        .identity_prompt(
            "You are a worker waiting for a message. Your conversation will \
             include one message of the form '[message from orchestrator: ...] \
             the secret is N'. Respond with exactly the number N and end your turn.",
        )
        .keep_alive()
        .max_turns(3);

    let orchestrator_identity = format!(
        "You coordinate work. Do exactly these two steps in order:\n\
         1. Call spawn_agent with agent=\"worker\", background=true, \
            description=\"worker\", instruction=\"wait for a message\".\n\
         2. Call send_message with to=\"worker\", message=\"the secret is {secret}\".\n\
         Then end your turn with a short confirmation."
    );
    let orchestrator_instruction = format!("Start the worker and send it the secret {secret}.");

    let output = Agent::new()
        .provider(provider)
        .model_name(&model)
        .name("orchestrator")
        .identity_prompt(orchestrator_identity)
        .instruction_prompt(orchestrator_instruction)
        .sub_agents([worker])
        .tool(SendMessageTool)
        .cancel_signal(cancel.clone())
        .max_turns(6)
        .event_handler(event_handler)
        .run()
        .await?;

    common::print_result(&output);

    // Give the worker time to drain the peer message and produce its text
    // response, then cancel so its idle wait exits cleanly.
    tokio::time::sleep(tokio::time::Duration::from_millis(2_000)).await;
    cancel.store(true, Ordering::Relaxed);
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let all = events.lock().unwrap();
    let worker_started = all
        .iter()
        .any(|e| matches!(e.kind, EventKind::AgentStarted { .. }) && e.agent_name == "worker");
    let worker_ended = all
        .iter()
        .any(|e| matches!(e.kind, EventKind::AgentFinished { .. }) && e.agent_name == "worker");
    let send_ok = all.iter().any(|e| {
        e.agent_name == "orchestrator"
            && matches!(
                &e.kind,
                EventKind::ToolCallFinished { tool_name, .. }
                    if tool_name == "send_message"
            )
    });
    let worker_text: String = all
        .iter()
        .filter(|e| e.agent_name == "worker")
        .filter_map(|e| match &e.kind {
            EventKind::TextChunkReceived { content } => Some(content.as_str()),
            _ => None,
        })
        .collect();

    assert!(worker_started, "worker must start");
    assert!(send_ok, "orchestrator's send_message must succeed");
    assert!(worker_ended, "worker must complete");
    assert!(
        worker_text.contains(&secret.to_string()),
        "worker must echo the secret {secret} received via peer message; got {worker_text:?}"
    );
    assert!(
        output.statistics.tool_calls >= 2,
        "orchestrator should call at least spawn_agent + send_message, got {}",
        output.statistics.tool_calls
    );

    Ok(())
}

/// Render an event as a single crisp line. Returns `None` for the noisy
/// streaming/usage events that would flood the log.
fn format_event(e: &Event) -> Option<String> {
    match &e.kind {
        EventKind::AgentStarted { description } => Some(match description {
            Some(d) => format!("start  ({d})"),
            None => "start".into(),
        }),
        EventKind::AgentFinished { turns, status } => {
            Some(format!("end    ({turns} turns, {status:?})"))
        }
        EventKind::TurnStarted { turn } => Some(format!("turn   {turn}")),
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => Some(format!("tool   {tool_name}({})", one_line(input))),
        EventKind::ToolCallFinished {
            tool_name, output, ..
        } => Some(format!("tool   {tool_name} -> ok {}", truncate(output, 80))),
        EventKind::ToolCallFailed {
            tool_name, message, ..
        } => Some(format!(
            "tool   {tool_name} -> err {}",
            truncate(message, 80)
        )),
        EventKind::ContextCompacted {
            turn,
            token_count,
            threshold,
            reason,
        } => Some(format!(
            "compact turn={turn} {token_count}/{threshold} ({reason:?})"
        )),
        EventKind::OutputTruncated { turn } => Some(format!("truncated turn={turn}")),
        EventKind::AgentPaused => Some("idle".into()),
        EventKind::AgentResumed => Some("resumed".into()),
        // Quiet: TurnFinished, RequestStarted/Finished, TextChunkReceived, TokensReported.
        _ => None,
    }
}

fn one_line(v: &serde_json::Value) -> String {
    truncate(&v.to_string(), 80)
}

fn truncate(s: &str, n: usize) -> String {
    let one = s.replace('\n', " ");
    if one.chars().count() <= n {
        one
    } else {
        let cut: String = one.chars().take(n).collect();
        format!("{cut}…")
    }
}
