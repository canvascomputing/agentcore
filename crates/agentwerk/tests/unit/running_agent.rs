//! High-level unit tests for the `RunningAgent` interface.
//!
//! `RunningAgent` is the handle returned by `Agent::create()`. The agent
//! loop runs on a background tokio task; the handle is how foreground code
//! interacts with it. Five operations form the public surface:
//!
//! - `send(instruction)` — enqueue a new instruction. Picked up at the next
//!   turn boundary, or immediately if the agent is parked in keep-alive idle.
//! - `cancel()` — flip the shared cancel signal.
//! - `is_cancelled()` — read that signal.
//! - `run().await` — await the background task and receive `AgentOutput`.
//!   May be called at most once across all clones.
//! - `clone()` — produce another handle to the same task. `send` and `cancel`
//!   are visible from any clone; the JoinHandle is held only once.
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
    Agent, AgentEvent, AgentEventKind, AgentStatus, AgenticError, CompletionRequest, ContentBlock,
    Message, RunningAgent,
};

// ---------------------------------------------------------------------------
// `run()` — awaiting the background task
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_returns_final_output_for_a_one_shot_agent() {
    let running = Agent::new()
        .name("demo")
        .model("mock")
        .provider(Arc::new(MockProvider::text("hello world")))
        .identity_prompt("")
        .instruction_prompt("greet")
        .create();

    let output = running.run().await.expect("run should succeed");
    assert_eq!(output.response_raw, "hello world");
    assert_eq!(output.status, AgentStatus::Completed);
}

#[tokio::test]
async fn run_can_only_be_awaited_once() {
    let running = Agent::new()
        .model("mock")
        .provider(Arc::new(MockProvider::text("done")))
        .instruction_prompt("x")
        .create();

    let _first = running.run().await.expect("first run succeeds");
    let second = running.run().await;
    assert!(
        matches!(&second, Err(AgenticError::Other(msg)) if msg.contains("already called")),
        "expected 'already called' error, got {second:?}",
    );
}

// ---------------------------------------------------------------------------
// `send()` — instructions injected after the loop started
// ---------------------------------------------------------------------------

#[tokio::test]
async fn send_injects_an_instruction_into_the_next_turn() {
    // Keep-alive parks the loop in idle after turn 1. `send("follow-up")`
    // wakes it; turn 2's provider request must carry the injected text as a
    // user message.
    let events = EventLog::new();
    let (provider, running) = keep_alive_agent(
        vec![text_response("first"), text_response("second")],
        &events,
    );

    events
        .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
        .await;
    running.send("follow-up");
    wait_until(|| provider.request_count() >= 2).await;

    let second = provider.last_request().expect("second request");
    let last_user = last_user_text(&second).expect("user message in second request");
    assert!(
        last_user.contains("follow-up"),
        "injected instruction must appear in turn 2's user message; got {last_user:?}",
    );

    // cleanup: wake the keep-alive loop so the background task exits.
    running.cancel();
    let _ = running.run().await;
}

// ---------------------------------------------------------------------------
// `cancel()` / `is_cancelled()` — shared atomic signal
// ---------------------------------------------------------------------------

#[tokio::test]
async fn is_cancelled_returns_true_after_cancel() {
    let running = Agent::new()
        .model("mock")
        .provider(Arc::new(MockProvider::text("done")))
        .identity_prompt("")
        .instruction_prompt("x")
        .create();

    assert!(!running.is_cancelled());
    running.cancel();
    assert!(running.is_cancelled());
    let _ = running.run().await;
}

#[tokio::test]
async fn cancel_breaks_an_idle_agent_out_of_its_wait() {
    // Keep-alive: the loop parks in AgentIdle after turn 1. Without cancel,
    // `run().await` would hang forever. A concurrent `cancel()` wakes the
    // idle wait (AgentResumed). The agent had already produced its terminal
    // reply, so the status is `Completed` — cancel-during-idle closes out
    // the work, it doesn't abort an in-flight turn (which would yield
    // `Cancelled` via the turn-boundary guard).
    let events = EventLog::new();
    let (_provider, running) = keep_alive_agent(vec![text_response("first")], &events);

    events
        .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
        .await;
    running.cancel();
    events
        .wait_for(|e| matches!(e.kind, AgentEventKind::AgentResumed))
        .await;
    let out = running.run().await.expect("output");
    assert_eq!(out.status, AgentStatus::Completed);
}

// ---------------------------------------------------------------------------
// `clone()` — another handle to the same task
// ---------------------------------------------------------------------------

#[tokio::test]
async fn send_and_cancel_on_a_clone_reach_the_original_task() {
    // Three clones, three roles:
    //   - `sender`    calls send()   — instruction must reach the loop
    //   - `canceller` calls cancel() — signal must be visible to all clones
    //   - `awaiter`   calls run()    — still receives the final output
    let events = EventLog::new();
    let (provider, awaiter) = keep_alive_agent(
        vec![text_response("first"), text_response("second")],
        &events,
    );
    let sender = awaiter.clone();
    let canceller = awaiter.clone();

    events
        .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
        .await;
    sender.send("via-clone");
    wait_until(|| provider.request_count() >= 2).await;

    let second = provider.last_request().expect("second request");
    assert!(last_user_text(&second).unwrap().contains("via-clone"));

    assert!(!awaiter.is_cancelled());
    canceller.cancel();
    assert!(
        awaiter.is_cancelled() && sender.is_cancelled(),
        "cancel from one clone must be visible from every clone",
    );

    let _ = awaiter.run().await.expect("output");
}

// ---------------------------------------------------------------------------
// Helpers — pedagogical, not library API. Kept private to this test file.
// ---------------------------------------------------------------------------

/// Build a keep-alive agent wired to a fresh `MockProvider` and the given
/// event log. The provider `Arc` is returned so the test can inspect
/// captured requests.
fn keep_alive_agent(
    responses: Vec<CompletionResponse>,
    events: &EventLog,
) -> (Arc<MockProvider>, RunningAgent) {
    let provider = Arc::new(MockProvider::new(responses));
    let running = Agent::new()
        .name("root")
        .model("mock")
        .provider(provider.clone())
        .identity_prompt("")
        .instruction_prompt("initial")
        .keep_alive_unlimited()
        .event_handler(events.handler())
        .create();
    (provider, running)
}

/// Shared event buffer plus a polling helper. Mirrors the pattern used in
/// `running.rs` inline tests: event collection in a `Mutex`, polled every
/// 25ms up to 5s.
struct EventLog {
    events: Arc<Mutex<Vec<AgentEvent>>>,
}

impl EventLog {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn handler(&self) -> Arc<dyn Fn(AgentEvent) + Send + Sync> {
        let events = self.events.clone();
        Arc::new(move |e| events.lock().unwrap().push(e))
    }

    async fn wait_for<F: Fn(&AgentEvent) -> bool>(&self, pred: F) {
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

/// Join every `Text` block of the most recent user message into a single
/// string. Used to locate an injected instruction in a later turn.
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
