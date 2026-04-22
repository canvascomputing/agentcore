//! High-level unit tests for the `AgentHandle` / `AgentOutputFuture` pair
//! returned by `Agent::spawn()`.
//!
//! The agent loop runs on a background tokio task. The foreground code
//! interacts with it through two values:
//!
//! - [`AgentHandle`] — cheap to clone. Public surface:
//!   - `send(instruction)` — enqueue a new instruction; picked up at the
//!     next turn boundary or immediately if the agent is parked idle.
//!   - `cancel()` — flip the shared cancel signal.
//!   - `is_cancelled()` — read that signal.
//!   - `is_stopped()` — read the terminal-state flag; `true` once the loop
//!     has emitted `AgentFinished`.
//!   - Dropping the last handle auto-cancels (RAII leak protection).
//! - [`AgentOutputFuture`] — resolves to the final `AgentOutput` when the
//!   loop exits. Polling it twice returns an error.
//!
//! These tests exercise each operation through its public surface only.
//! `MockProvider` avoids network calls; event observation synchronises the
//! foreground with the background loop's state.
//!
//! Run with `cargo test --test unit running_agent`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agentwerk::provider::types::CompletionResponse;
use agentwerk::testutil::{text_response, MockProvider};
use agentwerk::{
    Agent, AgentHandle, AgentOutputFuture, AgentStatus, AgenticError, CompletionRequest,
    ContentBlock, Event, EventKind, Message,
};

#[tokio::test]
async fn output_resolves_with_final_text_after_cancel() {
    let events = EventLog::new();
    let (handle, output) = Agent::new()
        .name("demo")
        .model("mock")
        .provider(Arc::new(MockProvider::text("hello world")))
        .identity_prompt("")
        .instruction_prompt("greet")
        .event_handler(events.handler())
        .spawn();

    // Wait until the loop has produced its terminal output and parked idle;
    // cancelling before that would abort turn 1 with `Cancelled` status.
    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    handle.cancel();
    let output = output.await.expect("run should succeed");
    assert_eq!(output.response_raw, "hello world");
    assert_eq!(output.status, AgentStatus::Completed);
}

#[tokio::test]
async fn awaiting_the_future_twice_returns_an_error() {
    let (handle, mut output) = Agent::new()
        .model("mock")
        .provider(Arc::new(MockProvider::text("done")))
        .instruction_prompt("x")
        .spawn();

    handle.cancel();
    let _first = (&mut output).await.expect("first await succeeds");
    let second = output.await;
    assert!(
        matches!(&second, Err(AgenticError::Other(msg)) if msg.contains("polled after completion")),
        "expected 'polled after completion' error, got {second:?}",
    );
}

#[tokio::test]
async fn send_injects_an_instruction_into_the_next_turn() {
    let events = EventLog::new();
    let (provider, handle, output) = spawn_agent(
        vec![text_response("first"), text_response("second")],
        &events,
    );

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    handle.send("follow-up");
    wait_until(|| provider.request_count() >= 2).await;

    let second = provider.last_request().expect("second request");
    let last_user = last_user_text(&second).expect("user message in second request");
    assert!(
        last_user.contains("follow-up"),
        "injected instruction must appear in turn 2's user message; got {last_user:?}",
    );

    handle.cancel();
    let _ = output.await;
}

#[tokio::test]
async fn is_cancelled_returns_true_after_cancel() {
    let (handle, output) = Agent::new()
        .model("mock")
        .provider(Arc::new(MockProvider::text("done")))
        .identity_prompt("")
        .instruction_prompt("x")
        .spawn();

    assert!(!handle.is_cancelled());
    handle.cancel();
    assert!(handle.is_cancelled());
    let _ = output.await;
}

#[tokio::test]
async fn cancel_breaks_an_idle_agent_out_of_its_wait() {
    let events = EventLog::new();
    let (_provider, handle, output) = spawn_agent(vec![text_response("first")], &events);

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    handle.cancel();
    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentResumed))
        .await;
    let out = output.await.expect("output");
    assert_eq!(out.status, AgentStatus::Completed);
}

#[tokio::test]
async fn is_stopped_stays_false_during_idle() {
    let events = EventLog::new();
    let (_provider, handle, output) = spawn_agent(vec![text_response("first")], &events);

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    assert!(!handle.is_stopped(), "idle is not stopped");

    handle.cancel();
    let _ = output.await;
}

#[tokio::test]
async fn is_stopped_becomes_true_after_cancel_during_idle() {
    let events = EventLog::new();
    let (_provider, handle, output) = spawn_agent(vec![text_response("first")], &events);

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    handle.cancel();
    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentResumed))
        .await;
    let _ = output.await.expect("output");
    assert!(handle.is_stopped());
}

#[tokio::test]
async fn send_and_cancel_on_a_clone_reach_the_original_task() {
    let events = EventLog::new();
    let (provider, original, output) = spawn_agent(
        vec![text_response("first"), text_response("second")],
        &events,
    );
    let sender = original.clone();
    let canceller = original.clone();

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    sender.send("via-clone");
    wait_until(|| provider.request_count() >= 2).await;

    let second = provider.last_request().expect("second request");
    assert!(last_user_text(&second).unwrap().contains("via-clone"));

    assert!(!original.is_cancelled());
    canceller.cancel();
    assert!(
        original.is_cancelled() && sender.is_cancelled(),
        "cancel from one clone must be visible from every clone",
    );

    let _ = output.await.expect("output");
}

#[tokio::test]
async fn dropping_the_last_handle_terminates_the_agent() {
    let events = EventLog::new();
    let (_provider, handle, output) = spawn_agent(vec![text_response("first")], &events);

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    drop(handle);
    let out = output.await.expect("output");
    assert_eq!(out.status, AgentStatus::Completed);
}

/// Spawn a fresh agent wired to a MockProvider and the given event log.
fn spawn_agent(
    responses: Vec<CompletionResponse>,
    events: &EventLog,
) -> (Arc<MockProvider>, AgentHandle, AgentOutputFuture) {
    let provider = Arc::new(MockProvider::new(responses));
    let (h, o) = Agent::new()
        .name("root")
        .model("mock")
        .provider(provider.clone())
        .identity_prompt("")
        .instruction_prompt("initial")
        .event_handler(events.handler())
        .spawn();
    (provider, h, o)
}

struct EventLog {
    events: Arc<Mutex<Vec<Event>>>,
}

impl EventLog {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn handler(&self) -> Arc<dyn Fn(Event) + Send + Sync> {
        let events = self.events.clone();
        Arc::new(move |e| events.lock().unwrap().push(e))
    }

    async fn wait_for<F: Fn(&Event) -> bool>(&self, pred: F) {
        for _ in 0..200 {
            if self.events.lock().unwrap().iter().any(&pred) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let seen: Vec<_> = self
            .events
            .lock()
            .unwrap()
            .iter()
            .map(|e| format!("{}:{:?}", e.agent_name, e.kind))
            .collect();
        panic!("timed out after 5s waiting for event; saw: {seen:#?}");
    }
}

async fn wait_until<F: FnMut() -> bool>(mut pred: F) {
    for _ in 0..200 {
        if pred() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out after 5s waiting for condition");
}

fn last_user_text(req: &CompletionRequest) -> Option<String> {
    req.messages.iter().rev().find_map(|m| match m {
        Message::User { content } => Some(
            content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    })
}
