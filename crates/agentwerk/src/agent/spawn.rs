//! Starts an agent on a background task and returns a handle + future pair, so the caller can send instructions, cancel, or await the result without blocking.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use tokio::task::JoinHandle;

use crate::agent::agent::Agent;
use crate::agent::output::AgentOutput;
use crate::agent::queue::{CommandQueue, CommandSource, QueuePriority, QueuedCommand};
use crate::error::{AgenticError, Result};

/// Shared atomic state between every [`AgentHandle`] clone, the
/// [`AgentOutputFuture`], and the running loop.
pub(crate) struct HandleState {
    pub(crate) queue: Arc<CommandQueue>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) stopped: Arc<AtomicBool>,
}

/// Drop-detection token: every handle and the future own a clone. When the
/// last one drops, `Drop::drop` sets the shared cancel flag, so the loop
/// exits on its next poll.
pub(crate) struct LifeToken {
    cancel: Arc<AtomicBool>,
}

impl LifeToken {
    pub(crate) fn new(cancel: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new(Self { cancel })
    }
}

impl Drop for LifeToken {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Cheap, clonable handle to an agent whose loop runs on a background tokio
/// task. Obtained from [`Agent::spawn`](crate::Agent::spawn).
///
/// While any clone of the handle is alive, the loop idles after producing
/// output; dropping the last clone (or calling [`cancel`](Self::cancel))
/// signals the loop to exit.
#[derive(Clone)]
pub struct AgentHandle {
    state: Arc<HandleState>,
    _life: Arc<LifeToken>,
}

impl AgentHandle {
    pub(crate) fn new(state: Arc<HandleState>, life: Arc<LifeToken>) -> Self {
        Self { state, _life: life }
    }

    /// Deliver a new instruction to the running agent. Picked up at the next
    /// turn boundary, or immediately if the agent is parked idle.
    pub fn send(&self, instruction: impl Into<String>) {
        self.state.queue.enqueue(QueuedCommand {
            content: instruction.into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: None,
        });
    }

    /// Signal the agent to stop. The loop observes this at the next turn
    /// boundary or idle-wait poll and exits.
    pub fn cancel(&self) {
        self.state.cancel.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if a cancel signal has been raised (explicitly via
    /// [`cancel`](Self::cancel) or implicitly via the last handle being
    /// dropped).
    pub fn is_cancelled(&self) -> bool {
        self.state.cancel.load(Ordering::Relaxed)
    }

    /// Returns `true` once the loop has terminated — i.e. it has emitted
    /// [`AgentEnd`](crate::agent::AgentEventKind::AgentEnd) and will not
    /// return to an idle wait. Reports *reality* (the loop is over) as
    /// opposed to [`is_cancelled`](Self::is_cancelled) which reports the
    /// *request*.
    pub fn is_stopped(&self) -> bool {
        self.state.stopped.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn queue_for_test(&self) -> Arc<CommandQueue> {
        self.state.queue.clone()
    }
}

/// Future that resolves to the agent's final
/// [`AgentOutput`](crate::agent::AgentOutput).
///
/// The future does not own a [`LifeToken`]: only [`AgentHandle`] clones do.
/// Dropping this future just abandons the result; whether the loop keeps
/// running is decided by whether any handles remain.
pub struct AgentOutputFuture {
    join: Mutex<Option<JoinHandle<Result<AgentOutput>>>>,
}

impl AgentOutputFuture {
    pub(crate) fn new(join: JoinHandle<Result<AgentOutput>>) -> Self {
        Self {
            join: Mutex::new(Some(join)),
        }
    }
}

impl Future for AgentOutputFuture {
    type Output = Result<AgentOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.join.lock().unwrap();
        let join = match guard.as_mut() {
            Some(j) => j,
            None => {
                return Poll::Ready(Err(AgenticError::Other(
                    "AgentOutputFuture polled after completion".into(),
                )))
            }
        };
        let pinned = Pin::new(join);
        match pinned.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(result)) => {
                *guard = None;
                Poll::Ready(result)
            }
            Poll::Ready(Err(e)) => {
                *guard = None;
                Poll::Ready(Err(AgenticError::Other(format!("agent task failed: {e}"))))
            }
        }
    }
}

impl Agent {
    /// Start the agent on a background tokio task and return a pair:
    ///
    /// - [`AgentHandle`] — cheap, clonable handle for injecting new
    ///   instructions, cancelling, or inspecting state.
    /// - [`AgentOutputFuture`] — resolves to the final
    ///   [`AgentOutput`](crate::agent::AgentOutput) once the loop exits.
    ///
    /// The loop idles after each terminal output as long as any handle is
    /// alive. Dropping the last handle calls [`AgentHandle::cancel`] for you
    /// (RAII safety); an explicit `.cancel()` does the same thing. For a
    /// pure one-shot run without a handle, use [`Agent::run`] instead — a
    /// `let (_, out) = agent.spawn(); out.await?` pattern will cancel
    /// before the first turn completes.
    ///
    /// Requires a running tokio runtime (`tokio::spawn` is invoked
    /// synchronously). Requires `.provider()` and `.instruction_prompt()`.
    pub fn spawn(self) -> (AgentHandle, AgentOutputFuture) {
        let queue = Arc::new(CommandQueue::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let life = LifeToken::new(cancel.clone());

        let prepared = self
            .cancel_signal(cancel.clone())
            .command_queue(queue.clone())
            .keep_alive();

        let stopped_for_task = stopped.clone();
        let join = tokio::spawn(async move {
            let result = prepared.run().await;
            stopped_for_task.store(true, Ordering::Relaxed);
            result
        });

        let state = Arc::new(HandleState {
            queue,
            cancel,
            stopped,
        });
        let handle = AgentHandle::new(state, life);
        let output = AgentOutputFuture::new(join);
        (handle, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    use crate::agent::{Agent, AgentEvent, AgentEventKind, AgentStatus};
    use crate::provider::types::{CompletionResponse, ContentBlock, Message};
    use crate::testutil::{text_response, MockProvider};
    use crate::CompletionRequest;

    #[tokio::test]
    async fn spawn_returns_handle_and_future() {
        let (handle, output) = one_shot_agent("hello");
        let clone = handle.clone();
        // AgentHandle is Clone; AgentOutputFuture is a Future. Cancel so the
        // keep-alive loop terminates.
        clone.cancel();
        let _: Result<AgentOutput> = output.await;
    }

    #[tokio::test]
    async fn spawn_starts_loop_immediately() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);
        // AgentStart is the first event emitted by run_loop — observable
        // before we await the future.
        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentStart { .. }))
            .await;
        handle.cancel();
        let _ = output.await;
    }

    #[tokio::test]
    async fn send_enqueues_user_input_command() {
        let (handle, output) = one_shot_agent("done");
        handle.send("hi");
        let queue = handle.queue_for_test();
        let cmd = queue
            .dequeue_if(Some("anyone"), |_| true)
            .expect("queued command");
        assert_eq!(cmd.content, "hi");
        assert!(matches!(cmd.priority, QueuePriority::Next));
        assert!(matches!(cmd.source, CommandSource::UserInput));
        assert!(cmd.agent_name.is_none());
        handle.cancel();
        let _ = output.await;
    }

    #[tokio::test]
    async fn send_reaches_next_provider_request() {
        let events = EventLog::new();
        let (provider, handle, output) = keep_alive_agent_with_provider(
            vec![text_response("first"), text_response("second")],
            &events,
        );

        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
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
        let out = output.await.expect("output");
        assert!(matches!(
            out.status,
            AgentStatus::Completed | AgentStatus::Cancelled
        ));
    }

    #[tokio::test]
    async fn clone_shares_queue() {
        let (handle, output) = one_shot_agent("done");
        let other = handle.clone();
        other.send("relay");
        let cmd = handle
            .queue_for_test()
            .dequeue_if(Some("anyone"), |_| true)
            .expect("queued command");
        assert_eq!(cmd.content, "relay");
        handle.cancel();
        let _ = output.await;
    }

    #[tokio::test]
    async fn clone_shares_cancel() {
        let (handle, output) = one_shot_agent("done");
        let other = handle.clone();
        assert!(!handle.is_cancelled());
        other.cancel();
        assert!(handle.is_cancelled() && other.is_cancelled());
        let _ = output.await;
    }

    #[tokio::test]
    async fn clone_shares_is_stopped() {
        let (handle, output) = one_shot_agent("done");
        let other = handle.clone();
        assert!(!handle.is_stopped() && !other.is_stopped());
        handle.cancel();
        let _ = output.await;
        assert!(handle.is_stopped() && other.is_stopped());
    }

    #[tokio::test]
    async fn cancel_during_idle_preserves_completed_status() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
            .await;
        handle.cancel();
        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentResumed))
            .await;
        let out = output.await.expect("output");
        assert_eq!(out.status, AgentStatus::Completed);
    }

    #[tokio::test]
    async fn cancel_from_spawned_task() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
            .await;
        let canceller = handle.clone();
        tokio::spawn(async move {
            canceller.cancel();
        });
        let _ = output.await.expect("output");
        assert!(handle.is_stopped());
    }

    #[tokio::test]
    async fn dropping_last_handle_triggers_cancel() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
            .await;
        drop(handle);
        let out = output.await.expect("output");
        assert_eq!(out.status, AgentStatus::Completed);
    }

    #[tokio::test]
    async fn dropping_one_of_two_handles_does_not_cancel() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        let survivor = handle.clone();
        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
            .await;
        drop(handle);
        // Cancel is NOT set while another handle is alive.
        assert!(!survivor.is_cancelled());
        // cleanup
        survivor.cancel();
        let _ = output.await;
    }

    #[tokio::test]
    async fn dropping_future_alone_does_not_cancel() {
        // The future holds no LifeToken, so dropping it doesn't cancel. The
        // loop keeps running — cleanup belongs to the handle.
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
            .await;
        drop(output);
        assert!(!handle.is_cancelled());
        handle.cancel();
        wait_until(|| handle.is_stopped()).await;
    }

    #[tokio::test]
    async fn keep_alive_idle_and_resumed_events_still_fire() {
        let events = EventLog::new();
        let (provider, handle, output) = keep_alive_agent_with_provider(
            vec![text_response("first"), text_response("second")],
            &events,
        );
        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentIdle))
            .await;
        handle.send("wake up");
        wait_until(|| provider.request_count() >= 2).await;
        events
            .wait_for(|e| matches!(e.kind, AgentEventKind::AgentResumed))
            .await;
        handle.cancel();
        let _ = output.await;
    }

    #[tokio::test]
    async fn awaiting_future_twice_returns_error() {
        // AgentOutputFuture consumes its inner JoinHandle on completion;
        // polling again surfaces an AgenticError::Other.
        let (handle, mut output) = one_shot_agent("done");
        handle.cancel();
        let _first = (&mut output).await;
        let second = output.await;
        assert!(matches!(second, Err(AgenticError::Other(_))));
    }

    fn one_shot_agent(text: &str) -> (AgentHandle, AgentOutputFuture) {
        Agent::new()
            .name("demo")
            .model("mock")
            .provider(Arc::new(MockProvider::text(text)))
            .identity_prompt("")
            .instruction_prompt("x")
            .spawn()
    }

    fn keep_alive_agent(
        responses: Vec<CompletionResponse>,
        events: &EventLog,
    ) -> (AgentHandle, AgentOutputFuture) {
        let (_, h, o) = keep_alive_agent_with_provider(responses, events);
        (h, o)
    }

    fn keep_alive_agent_with_provider(
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
        events: Arc<StdMutex<Vec<AgentEvent>>>,
    }

    impl EventLog {
        fn new() -> Self {
            Self {
                events: Arc::new(StdMutex::new(Vec::new())),
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
}
