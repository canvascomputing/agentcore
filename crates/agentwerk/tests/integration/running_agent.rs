//! End-to-end coverage for `Agent::spawn()` / `AgentHandle`.
//!
//! Uses a real LLM provider (`make test_integration`). The test plays the
//! role of an external controller: it spawns an agent via `.spawn()`, feeds
//! it two instructions through `send` (one via the original handle, one via
//! a clone), cancels via a third clone on a spawned task, then awaits the
//! returned output future.

use super::common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use agentwerk::event::EventKind;
use agentwerk::{Agent, Event};

#[tokio::test]
async fn external_sender_delivers_two_instructions_and_clone_cancels(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let secret_a = five_digit_secret(0);
    let secret_b = five_digit_secret(1);
    eprintln!("[test] secrets generated: a={secret_a} b={secret_b}");

    let events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = events.clone();
    let text_buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let buf_for_handler = text_buf.clone();
    let event_handler = Arc::new(move |e: Event| {
        match &e.kind {
            EventKind::TextChunkReceived { content } => {
                buf_for_handler.lock().unwrap().push_str(content);
            }
            _ => {
                let mut buf = buf_for_handler.lock().unwrap();
                let trimmed = buf.trim();
                if !trimmed.is_empty() {
                    eprintln!("[{}] resp   {}", e.agent_name, truncate(trimmed, 120));
                    buf.clear();
                }
            }
        }
        if let Some(line) = format_event(&e) {
            eprintln!("[{}] {line}", e.agent_name);
        }
        collected.lock().unwrap().push(e);
    });

    let (agent, output) = Agent::new()
        .name("listener")
        .model_name(&model)
        .provider(provider)
        .identity_prompt(
            "You receive user messages. A secret-bearing message has the exact form \
             'the secret is N' where N is a number. \
             \n\nRules (follow strictly):\n\
             1. If the current user message is of the form 'the secret is N', reply with \
                exactly the number N on its own line and end your turn. Output nothing else.\n\
             2. Otherwise, reply with exactly the single word 'ready' and end your turn. \
                Do not invent numbers. Do not echo any example. Do not restate the rules.",
        )
        .instruction_prompt("wait")
        .max_turns(10)
        .event_handler(event_handler)
        .spawn();

    wait_for(&events, |all| {
        all.iter().any(|e| matches!(e.kind, EventKind::AgentPaused))
    })
    .await?;

    eprintln!("[test] sending secret_a={secret_a} via original handle");
    agent.send(format!("the secret is {secret_a}"));
    wait_for(&events, |all| {
        listener_text(all).contains(&secret_a.to_string())
    })
    .await?;

    let via_clone = agent.clone();
    eprintln!("[test] sending secret_b={secret_b} via cloned handle");
    via_clone.send(format!("the secret is {secret_b}"));
    wait_for(&events, |all| {
        listener_text(all).contains(&secret_b.to_string())
    })
    .await?;

    let canceler = agent.clone();
    tokio::spawn(async move {
        eprintln!("[test] cancelling via cloned handle from spawned task");
        canceler.cancel();
    });

    let output = output.await?;
    common::print_result(&output);

    assert!(
        output.statistics.turns <= 10,
        "ran past max_turns: {}",
        output.statistics.turns
    );
    let all = events.lock().unwrap();
    assert!(
        all.iter().any(|e| matches!(e.kind, EventKind::AgentPaused)),
        "agent should have idled at least once"
    );
    assert!(
        all.iter()
            .any(|e| matches!(e.kind, EventKind::AgentResumed)),
        "agent should have resumed at least once"
    );
    let text = listener_text(&all);
    assert!(
        text.contains(&secret_a.to_string()),
        "listener must echo secret_a {secret_a}; got {text:?}"
    );
    assert!(
        text.contains(&secret_b.to_string()),
        "listener must echo secret_b {secret_b}; got {text:?}"
    );
    Ok(())
}

fn five_digit_secret(salt: u32) -> u32 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    // Knuth multiplicative hash mixes the salt into the entropy so different
    // salts produce independent-looking numbers, not adjacent ones.
    let mixed = nanos.wrapping_add(salt.wrapping_mul(2_654_435_761));
    (mixed % 90_000) + 10_000
}

fn listener_text(events: &[Event]) -> String {
    events
        .iter()
        .filter(|e| e.agent_name == "listener")
        .filter_map(|e| match &e.kind {
            EventKind::TextChunkReceived { content } => Some(content.as_str()),
            _ => None,
        })
        .collect()
}

async fn wait_for<F>(events: &Arc<Mutex<Vec<Event>>>, pred: F) -> std::result::Result<(), String>
where
    F: Fn(&[Event]) -> bool,
{
    const TIMEOUT: Duration = Duration::from_secs(60);
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if pred(&events.lock().unwrap()) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out after {TIMEOUT:?} waiting for event condition"
            ));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Render an event as a single crisp line. Returns `None` for the noisy
/// streaming/usage events that would flood the log.
fn format_event(e: &Event) -> Option<String> {
    match &e.kind {
        EventKind::AgentStarted { description } => Some(match description {
            Some(d) => format!("start  ({d})"),
            None => "start".into(),
        }),
        EventKind::AgentFinished { turns, outcome } => {
            Some(format!("end    ({turns} turns, {outcome:?})"))
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
