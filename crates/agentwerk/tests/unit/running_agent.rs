//! High-level unit tests for the `AgentWorking` / `OutputFuture` pair
//! returned by `Agent::retain()`.
//!
//! The agent loop runs on a background tokio task. The foreground code
//! interacts with it through two values:
//!
//! - [`AgentWorking`] — cheap to clone. Public surface:
//!   - `send(instruction)` — enqueue a new instruction; picked up at the
//!     next turn boundary or immediately if the agent is parked idle.
//!   - `interrupt()` — flip the shared cancel signal.
//!   - `is_interrupted()` — read that signal.
//!   - Dropping the last handle auto-interrupts (RAII leak protection).
//! - [`OutputFuture`] — resolves to the final `Output` when the
//!   loop exits. Polling it twice returns an error.
//!
//! These tests exercise each operation through its public surface only.
//! `MockProvider` avoids network calls; event observation synchronises the
//! foreground with the background loop's state.
//!
//! Run with `cargo test --test unit running_agent`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agentwerk::agent::{AgentWorking, OutputFuture};
use agentwerk::event::EventKind;
use agentwerk::output::Outcome;
use agentwerk::provider::types::ModelResponse;
use agentwerk::provider::{ContentBlock, Message, ModelRequest};
use agentwerk::testutil::{text_response, MockProvider};
use agentwerk::{Agent, Event};

#[tokio::test]
async fn output_resolves_with_final_text_after_interrupt() {
    let events = EventLog::new();
    let (handle, output) = Agent::new()
        .name("demo")
        .model_name("mock")
        .provider(Arc::new(MockProvider::text("hello world")))
        .identity_prompt("")
        .instruction_prompt("greet")
        .event_handler(events.handler())
        .retain();

    // Wait until the loop has produced its terminal output and parked idle;
    // interrupting before that would abort turn 1 with `Cancelled` status.
    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    handle.interrupt();
    let output = output.await.expect("run should succeed");
    assert_eq!(output.response_raw, "hello world");
    assert_eq!(output.outcome, Outcome::Completed);
}

#[tokio::test]
async fn send_injects_an_instruction_into_the_next_turn() {
    let events = EventLog::new();
    let (provider, handle, output) = retain_agent(
        vec![text_response("first"), text_response("second")],
        &events,
    );

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    handle.send("follow-up");
    wait_until(|| provider.requests() >= 2).await;

    let second = provider.last_request().expect("second request");
    let last_user = last_user_text(&second).expect("user message in second request");
    assert!(
        last_user.contains("follow-up"),
        "injected instruction must appear in turn 2's user message; got {last_user:?}",
    );

    handle.interrupt();
    let _ = output.await;
}

#[tokio::test]
async fn is_interrupted_returns_true_after_interrupt() {
    let (handle, output) = Agent::new()
        .model_name("mock")
        .provider(Arc::new(MockProvider::text("done")))
        .identity_prompt("")
        .instruction_prompt("x")
        .retain();

    assert!(!handle.is_interrupted());
    handle.interrupt();
    assert!(handle.is_interrupted());
    let _ = output.await;
}

#[tokio::test]
async fn interrupt_breaks_an_idle_agent_out_of_its_wait() {
    let events = EventLog::new();
    let (_provider, handle, output) = retain_agent(vec![text_response("first")], &events);

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    handle.interrupt();
    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentResumed))
        .await;
    let out = output.await.expect("output");
    assert_eq!(out.outcome, Outcome::Completed);
}

#[tokio::test]
async fn send_and_interrupt_on_a_clone_reach_the_original_task() {
    let events = EventLog::new();
    let (provider, original, output) = retain_agent(
        vec![text_response("first"), text_response("second")],
        &events,
    );
    let sender = original.clone();
    let interrupter = original.clone();

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    sender.send("via-clone");
    wait_until(|| provider.requests() >= 2).await;

    let second = provider.last_request().expect("second request");
    assert!(last_user_text(&second).unwrap().contains("via-clone"));

    assert!(!original.is_interrupted());
    interrupter.interrupt();
    assert!(
        original.is_interrupted() && sender.is_interrupted(),
        "interrupt from one clone must be visible from every clone",
    );

    let _ = output.await.expect("output");
}

#[tokio::test]
async fn dropping_the_last_handle_terminates_the_agent() {
    let events = EventLog::new();
    let (_provider, handle, output) = retain_agent(vec![text_response("first")], &events);

    events
        .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
        .await;
    drop(handle);
    let out = output.await.expect("output");
    assert_eq!(out.outcome, Outcome::Completed);
}

/// Retain a fresh agent wired to a MockProvider and the given event log.
fn retain_agent(
    responses: Vec<ModelResponse>,
    events: &EventLog,
) -> (Arc<MockProvider>, AgentWorking, OutputFuture) {
    let provider = Arc::new(MockProvider::new(responses));
    let (h, o) = Agent::new()
        .name("root")
        .model_name("mock")
        .provider(provider.clone())
        .identity_prompt("")
        .instruction_prompt("initial")
        .event_handler(events.handler())
        .retain();
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

fn last_user_text(req: &ModelRequest) -> Option<String> {
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
