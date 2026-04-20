//! `RunningAgent` — a handle to an agent whose loop is running on a background
//! tokio task.
//!
//! Obtained via [`Agent::create`](crate::Agent::create). The loop is already
//! executing by the time `create()` returns; hold this handle to inject new
//! instructions, cancel, or await the final output.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::task::JoinHandle;

use crate::agent::output::AgentOutput;
use crate::agent::queue::{CommandQueue, CommandSource, QueuePriority, QueuedCommand};
use crate::error::{AgenticError, Result};

/// Handle to an agent whose loop is running on a background tokio task.
///
/// Cheap to clone (Arc-backed). `send` and `cancel` can be called from any
/// clone; `run` may be called at most once across all clones and yields the
/// final output.
#[derive(Clone)]
pub struct RunningAgent {
    inner: Arc<Inner>,
}

struct Inner {
    queue: Arc<CommandQueue>,
    cancel: Arc<AtomicBool>,
    join: Mutex<Option<JoinHandle<Result<AgentOutput>>>>,
}

impl RunningAgent {
    pub(crate) fn new(
        queue: Arc<CommandQueue>,
        cancel: Arc<AtomicBool>,
        join: JoinHandle<Result<AgentOutput>>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                queue,
                cancel,
                join: Mutex::new(Some(join)),
            }),
        }
    }

    /// Deliver a new instruction to the running agent. Broadcast across the
    /// run-tree; picked up at the next turn boundary by `drain_command_queue`
    /// (or immediately if the agent is parked in keep-alive idle).
    pub fn send(&self, instruction: impl Into<String>) {
        self.inner.queue.enqueue(QueuedCommand {
            content: instruction.into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: None,
        });
    }

    /// Signal the agent to stop. The loop observes this at the next turn
    /// boundary and exits with [`AgentStatus::Cancelled`](crate::agent::AgentStatus).
    pub fn cancel(&self) {
        self.inner.cancel.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if `cancel()` has been called.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancel.load(Ordering::Relaxed)
    }

    /// Await the agent's completion and return its final output. May only be
    /// called once across all clones; later calls return an error.
    pub async fn run(&self) -> Result<AgentOutput> {
        let handle = self
            .inner
            .join
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| AgenticError::Other("RunningAgent::run already called".into()))?;
        handle
            .await
            .map_err(|e| AgenticError::Other(format!("agent task failed: {e}")))?
    }

    #[cfg(test)]
    pub(crate) fn queue_for_test(&self) -> Arc<CommandQueue> {
        self.inner.queue.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    use crate::agent::{Agent, AgentEvent, AgentEventKind, AgentStatus};
    use crate::provider::types::{ContentBlock, Message};
    use crate::testutil::{text_response, MockProvider};

    fn new_agent() -> RunningAgent {
        Agent::new()
            .model("mock")
            .provider(Arc::new(MockProvider::text("done")))
            .instruction_prompt("x")
            .create()
    }

    #[tokio::test]
    async fn send_enqueues_user_input_command() {
        let running = new_agent();
        running.send("hi");
        let queue = running.queue_for_test();
        let cmd = queue
            .dequeue_if(Some("anyone"), |_| true)
            .expect("queued command");
        assert_eq!(cmd.content, "hi");
        assert!(matches!(cmd.priority, QueuePriority::Next));
        assert!(matches!(cmd.source, CommandSource::UserInput));
        assert!(cmd.agent_name.is_none());
        let _ = running.run().await;
    }

    #[tokio::test]
    async fn cancel_sets_signal() {
        let running = new_agent();
        assert!(!running.is_cancelled());
        running.cancel();
        assert!(running.is_cancelled());
        let _ = running.run().await;
    }

    #[tokio::test]
    async fn clone_shares_state() {
        let running = new_agent();
        let other = running.clone();
        other.send("relay");
        let cmd = running
            .queue_for_test()
            .dequeue_if(Some("anyone"), |_| true)
            .expect("queued command");
        assert_eq!(cmd.content, "relay");
        let _ = running.run().await;
    }

    #[tokio::test]
    async fn double_run_returns_error() {
        let running = new_agent();
        let _first = running.run().await;
        let second = running.run().await;
        assert!(matches!(second, Err(AgenticError::Other(_))));
    }

    #[tokio::test]
    async fn send_reaches_next_provider_request() {
        let provider = Arc::new(MockProvider::new(vec![
            text_response("first"),
            text_response("second"),
        ]));
        let events: Arc<StdMutex<Vec<AgentEvent>>> = Arc::new(StdMutex::new(Vec::new()));
        let events_for_handler = events.clone();

        let running = Agent::new()
            .name("root")
            .model("mock")
            .provider(provider.clone())
            .identity_prompt("")
            .instruction_prompt("initial")
            .keep_alive_unlimited()
            .event_handler(Arc::new(move |e: AgentEvent| {
                events_for_handler.lock().unwrap().push(e);
            }))
            .create();

        wait_for_event(&events, |e| matches!(e.kind, AgentEventKind::AgentIdle)).await;
        running.send("follow-up");
        wait_until(|| provider.request_count() >= 2).await;

        let second = provider.last_request().expect("second request");
        let last_user_text = second
            .messages
            .iter()
            .rev()
            .find_map(|m| match m {
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
            .expect("user message in second request");
        assert!(
            last_user_text.contains("follow-up"),
            "injected instruction must appear in the next provider request; got {last_user_text:?}"
        );

        running.cancel();
        let out = running.run().await.expect("output");
        assert!(matches!(
            out.status,
            AgentStatus::Completed | AgentStatus::Cancelled
        ));
    }

    #[tokio::test]
    async fn cancel_breaks_idle_wait() {
        let provider = Arc::new(MockProvider::text("first"));
        let events: Arc<StdMutex<Vec<AgentEvent>>> = Arc::new(StdMutex::new(Vec::new()));
        let events_for_handler = events.clone();

        let running = Agent::new()
            .name("root")
            .model("mock")
            .provider(provider)
            .identity_prompt("")
            .instruction_prompt("initial")
            .keep_alive_unlimited()
            .event_handler(Arc::new(move |e: AgentEvent| {
                events_for_handler.lock().unwrap().push(e);
            }))
            .create();

        wait_for_event(&events, |e| matches!(e.kind, AgentEventKind::AgentIdle)).await;
        running.cancel();
        wait_for_event(&events, |e| matches!(e.kind, AgentEventKind::AgentResumed)).await;
        let _ = running.run().await.expect("output");
    }

    async fn wait_for_event<F>(events: &Arc<StdMutex<Vec<AgentEvent>>>, pred: F)
    where
        F: Fn(&AgentEvent) -> bool,
    {
        for _ in 0..200 {
            if events.lock().unwrap().iter().any(&pred) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let seen: Vec<_> = events
            .lock()
            .unwrap()
            .iter()
            .map(|e| format!("{}:{:?}", e.agent_name, e.kind))
            .collect();
        panic!("timed out after 5s waiting for event; saw: {seen:#?}");
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
}
