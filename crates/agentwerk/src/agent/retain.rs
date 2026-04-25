//! Starts an agent on a background task and returns a handle + future pair, so the caller can send instructions, cancel, or await the result without blocking.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::agent::agent::Agent;
use crate::agent::error::AgentError;
use crate::agent::work::{Task, TaskSource, Work, WorkPriority};
use crate::error::Result;
use crate::output::Output;

/// RAII token: flips the shared cancel flag when its last clone drops, so
/// abandoning the handle without an explicit `.interrupt()` still unblocks
/// the loop.
struct CancelGuard {
    cancel: Arc<AtomicBool>,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Cheap, clonable handle to an agent whose loop runs on a background tokio
/// task. Obtained from [`Agent::retain`](crate::Agent::retain).
///
/// While any clone of the handle is alive, the loop idles after producing
/// output; dropping the last clone (or calling [`interrupt`](Self::interrupt))
/// signals the loop to exit.
#[derive(Clone)]
pub struct AgentWorking {
    work: Arc<Work>,
    cancel: Arc<AtomicBool>,
    #[allow(dead_code)]
    guard: Arc<CancelGuard>,
}

impl AgentWorking {
    /// Hand the running agent a new task. Picked up at the next turn
    /// boundary, or immediately if the agent is parked idle.
    pub fn task(&self, task: impl Into<String>) {
        self.work.add(Task {
            content: task.into(),
            priority: WorkPriority::Next,
            source: TaskSource::UserInput,
            agent_name: None,
        });
    }

    /// Signal the agent to stop. The loop observes this at the next turn
    /// boundary or idle-wait poll and exits.
    pub fn interrupt(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if an interrupt signal has been raised (explicitly via
    /// [`interrupt`](Self::interrupt) or implicitly via the last handle being
    /// dropped).
    pub fn is_interrupted(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Resolves to the agent's final [`Output`](crate::output::Output) once the
/// background loop exits.
///
/// Only [`AgentWorking`] clones keep the agent alive; dropping this
/// (without awaiting) just abandons the result. Whether the loop keeps
/// running is decided by whether any handles remain.
///
/// Implements [`IntoFuture`]: `.await` consumes the value by move, so a
/// double-await is a compile error rather than a runtime failure.
pub struct OutputFuture {
    join: JoinHandle<Result<Output>>,
}

impl IntoFuture for OutputFuture {
    type Output = Result<Output>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<Output>> + Send + 'static>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            match self.join.await {
                Ok(result) => result,
                Err(e) => Err(AgentError::AgentCrashed {
                    message: e.to_string(),
                }
                .into()),
            }
        })
    }
}

impl Agent {
    /// Start the agent on a background tokio task and return a pair:
    ///
    /// - [`AgentWorking`]: cheap, clonable handle for injecting new
    ///   instructions, cancelling, or inspecting state.
    /// - [`OutputFuture`]: resolves to the final
    ///   [`Output`](crate::output::Output) once the loop exits.
    ///
    /// The loop idles after each terminal output as long as any handle is
    /// alive. Dropping the last handle calls [`AgentWorking::interrupt`] for you
    /// (RAII safety); an explicit `.interrupt()` does the same thing. For a
    /// pure one-shot run without a handle, use [`Agent::task`] instead: a
    /// `let (_, out) = agent.retain(); out.await?` pattern will interrupt
    /// before the first turn completes.
    ///
    /// Requires a running tokio runtime (`tokio::spawn` is invoked
    /// synchronously). Requires `.provider()` and either an initial
    /// `.task(...)` set in the builder or follow-up
    /// [`AgentWorking::task`] calls on the returned handle.
    pub fn retain(self) -> (AgentWorking, OutputFuture) {
        let work = Arc::new(Work::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let guard = Arc::new(CancelGuard {
            cancel: cancel.clone(),
        });

        let prepared = self
            .interrupt_signal(cancel.clone())
            .incoming_work(work.clone())
            .keep_alive();

        let join = tokio::spawn(async move { prepared.work().await });

        (
            AgentWorking {
                work,
                cancel,
                guard,
            },
            OutputFuture { join },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    use crate::agent::Agent;
    use crate::event::{Event, EventKind};
    use crate::output::Outcome;
    use crate::provider::types::{ContentBlock, Message, ModelResponse};
    use crate::provider::ModelRequest;
    use crate::testutil::{text_response, MockProvider};

    #[tokio::test]
    async fn retain_returns_handle_and_future() {
        let (handle, output) = one_shot_agent("hello");
        let clone = handle.clone();
        // AgentWorking is Clone; OutputFuture is a Future. Interrupt so the
        // keep-alive loop terminates.
        clone.interrupt();
        let _: Result<Output> = output.await;
    }

    #[tokio::test]
    async fn retain_starts_loop_immediately() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);
        // AgentStarted is the first event emitted by run_loop: observable
        // before we await the future.
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentStarted { .. }))
            .await;
        handle.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn task_adds_user_input_work() {
        let (handle, output) = one_shot_agent("done");
        handle.task("hi");
        let task = handle
            .work
            .take_if(Some("anyone"), |_| true)
            .expect("pending task");
        assert_eq!(task.content, "hi");
        assert!(matches!(task.priority, WorkPriority::Next));
        assert!(matches!(task.source, TaskSource::UserInput));
        assert!(task.agent_name.is_none());
        handle.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn task_reaches_next_provider_request() {
        let events = EventLog::new();
        let (provider, handle, output) = keep_alive_agent_with_provider(
            vec![text_response("first"), text_response("second")],
            &events,
        );

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        handle.task("follow-up");
        wait_until(|| provider.requests() >= 2).await;

        let second = provider.last_request().expect("second request");
        let last_user = last_user_text(&second).expect("user message in second request");
        assert!(
            last_user.contains("follow-up"),
            "injected instruction must appear in turn 2's user message; got {last_user:?}",
        );

        handle.interrupt();
        let out = output.await.expect("output");
        assert!(matches!(
            out.outcome,
            Outcome::Completed | Outcome::Cancelled
        ));
    }

    #[tokio::test]
    async fn clone_shares_work() {
        let (handle, output) = one_shot_agent("done");
        let other = handle.clone();
        other.task("relay");
        let task = handle
            .work
            .take_if(Some("anyone"), |_| true)
            .expect("pending task");
        assert_eq!(task.content, "relay");
        handle.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn clone_shares_interrupt() {
        let (handle, output) = one_shot_agent("done");
        let other = handle.clone();
        assert!(!handle.is_interrupted());
        other.interrupt();
        assert!(handle.is_interrupted() && other.is_interrupted());
        let _ = output.await;
    }

    #[tokio::test]
    async fn interrupt_during_idle_preserves_completed_status() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

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
    async fn interrupt_from_spawned_task() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        let interrupter = handle.clone();
        tokio::spawn(async move {
            interrupter.interrupt();
        });
        let _ = output.await.expect("output");
    }

    #[tokio::test]
    async fn dropping_last_handle_triggers_interrupt() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        drop(handle);
        let out = output.await.expect("output");
        assert_eq!(out.outcome, Outcome::Completed);
    }

    #[tokio::test]
    async fn dropping_one_of_two_handles_does_not_interrupt() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        let survivor = handle.clone();
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        drop(handle);
        // Interrupt is NOT set while another handle is alive.
        assert!(!survivor.is_interrupted());
        // cleanup
        survivor.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn dropping_future_alone_does_not_interrupt() {
        // The future holds no CancelGuard, so dropping it doesn't interrupt. The
        // loop keeps running: cleanup belongs to the handle.
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        drop(output);
        assert!(!handle.is_interrupted());
        handle.interrupt();
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentFinished { .. }))
            .await;
    }

    #[tokio::test]
    async fn keep_alive_idle_and_resumed_events_still_fire() {
        let events = EventLog::new();
        let (provider, handle, output) = keep_alive_agent_with_provider(
            vec![text_response("first"), text_response("second")],
            &events,
        );
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        handle.task("wake up");
        wait_until(|| provider.requests() >= 2).await;
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentResumed))
            .await;
        handle.interrupt();
        let _ = output.await;
    }

    fn one_shot_agent(text: &str) -> (AgentWorking, OutputFuture) {
        Agent::new()
            .name("demo")
            .model("mock")
            .provider(Arc::new(MockProvider::text(text)))
            .role("")
            .task("x")
            .retain()
    }

    fn keep_alive_agent(
        responses: Vec<ModelResponse>,
        events: &EventLog,
    ) -> (AgentWorking, OutputFuture) {
        let (_, h, o) = keep_alive_agent_with_provider(responses, events);
        (h, o)
    }

    fn keep_alive_agent_with_provider(
        responses: Vec<ModelResponse>,
        events: &EventLog,
    ) -> (Arc<MockProvider>, AgentWorking, OutputFuture) {
        let provider = Arc::new(MockProvider::new(responses));
        let (h, o) = Agent::new()
            .name("root")
            .model("mock")
            .provider(provider.clone())
            .role("")
            .task("initial")
            .event_handler(events.handler())
            .retain();
        (provider, h, o)
    }

    struct EventLog {
        events: Arc<StdMutex<Vec<Event>>>,
    }

    impl EventLog {
        fn new() -> Self {
            Self {
                events: Arc::new(StdMutex::new(Vec::new())),
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
}
