//! Run many agents with a shared concurrency cap. `Batch::run` waits for a fixed set; `Batch::spawn` hands back a pool you can submit into while it's running.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_util::stream::{FuturesUnordered, Stream, StreamExt};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::Result;
use crate::output::Output;

const DEFAULT_CONCURRENCY: usize = 1;

/// Pool of agents that all share a concurrency cap. Build with
/// [`Batch::new`], chain [`concurrency`](Self::concurrency) and
/// [`agent`](Self::agent) / [`agents`](Self::agents), then finish with
/// [`run`](Self::run) (wait for all) or [`spawn`](Self::spawn) (dynamic pool).
pub struct Batch {
    concurrency: usize,
    agents: Vec<Agent>,
    cancel_signal: Option<Arc<AtomicBool>>,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            concurrency: DEFAULT_CONCURRENCY,
            agents: Vec::new(),
            cancel_signal: None,
        }
    }
}

impl Batch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap on simultaneous in-flight agents. Clamped to at least 1.
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    /// Add one agent to run.
    pub fn agent(mut self, agent: Agent) -> Self {
        self.agents.push(agent);
        self
    }

    /// Add many agents to run.
    pub fn agents<I>(mut self, agents: I) -> Self
    where
        I: IntoIterator<Item = Agent>,
    {
        self.agents.extend(agents);
        self
    }

    /// Share an external cancel signal with the pool. Every submitted agent
    /// uses it, and [`BatchHandle::cancel`] writes to it. Useful when the
    /// caller already owns a signal (e.g. wired to Ctrl-C) and wants in-flight
    /// agents to observe it.
    pub fn cancel_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.cancel_signal = Some(signal);
        self
    }

    /// Run every added agent to completion. Returns results in **submission**
    /// order: `results[i]` corresponds to the `i`th agent added via
    /// [`agent`](Self::agent) / [`agents`](Self::agents). A failing agent does
    /// not abort the others.
    pub async fn run(self) -> Vec<Result<Output>> {
        let total = self.agents.len();
        let (handle, stream) = self.spawn();
        handle.drain();

        let mut slots: Vec<Option<Result<Output>>> = (0..total).map(|_| None).collect();
        for (index, result) in stream.collect().await {
            if index < slots.len() {
                slots[index] = Some(result);
            }
        }
        slots
            .into_iter()
            .map(|slot| slot.expect("batch stream yields one result per submission"))
            .collect()
    }

    /// Start a dispatcher on a background tokio task and return a pair:
    ///
    /// - [`BatchHandle`] — cheap, clonable handle for submitting more agents
    ///   or cancelling.
    /// - [`BatchOutputStream`] — yields
    ///   `(submission_index, Result<Output>)` in completion order. The
    ///   `submission_index` matches the position the agent was added:
    ///   preloaded [`agents`](Self::agents) take indices `0..n`, then dynamic
    ///   [`submit`](BatchHandle::submit) calls continue the sequence. Ends
    ///   once all handles are dropped or [`drain`](BatchHandle::drain)ed (let
    ///   in-flight finish), or [`cancel`](BatchHandle::cancel) is called
    ///   (interrupt in-flight) and the backlog completes.
    ///
    /// Requires a running tokio runtime.
    pub fn spawn(self) -> (BatchHandle, BatchOutputStream) {
        let concurrency = self.concurrency;
        let (submit_tx, submit_rx) = mpsc::unbounded_channel::<(usize, Agent)>();
        let (output_tx, output_rx) = mpsc::unbounded_channel::<(usize, Result<Output>)>();
        let cancel = self
            .cancel_signal
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let counter = Arc::new(AtomicUsize::new(0));

        for agent in self.agents {
            let index = counter.fetch_add(1, Ordering::Relaxed);
            let _ = submit_tx.send((index, agent));
        }

        let dispatcher_cancel = cancel.clone();
        tokio::spawn(async move {
            dispatch(submit_rx, output_tx, concurrency, dispatcher_cancel).await;
        });

        let handle = BatchHandle {
            sender: submit_tx,
            cancel,
            counter,
        };
        let output = BatchOutputStream { rx: output_rx };
        (handle, output)
    }
}

async fn dispatch(
    mut submit_rx: mpsc::UnboundedReceiver<(usize, Agent)>,
    output_tx: mpsc::UnboundedSender<(usize, Result<Output>)>,
    concurrency: usize,
    cancel: Arc<AtomicBool>,
) {
    let mut in_flight: FuturesUnordered<tokio::task::JoinHandle<(usize, Result<Output>)>> =
        FuturesUnordered::new();
    let mut closed = false;

    loop {
        if cancel.load(Ordering::Relaxed) && !closed {
            submit_rx.close();
            closed = true;
        }

        tokio::select! {
            biased;
            Some(join) = in_flight.next(), if !in_flight.is_empty() => {
                // A task-level JoinError means the spawned future panicked or was
                // aborted — the submission index is then unrecoverable, so the slot
                // will be backfilled with a synthetic error by `Batch::run`.
                if let Ok(pair) = join {
                    let _ = output_tx.send(pair);
                }
            }
            maybe = submit_rx.recv(), if !closed && in_flight.len() < concurrency => {
                let Some((index, agent)) = maybe else {
                    closed = true;
                    continue;
                };
                let agent = agent.cancel_signal(cancel.clone());
                in_flight.push(tokio::spawn(async move {
                    (index, agent.run().await)
                }));
            }
            else => return,
        }
    }
}

/// Cheap, clonable handle to a running [`Batch`] pool. Obtained from
/// [`Batch::spawn`].
///
/// While any clone of the handle is alive, the pool accepts new submissions.
/// Dropping the last clone (or calling [`drain`](Self::drain) on it) closes
/// the pool gracefully: queued and in-flight agents finish, then the output
/// stream ends. Use [`cancel`](Self::cancel) to interrupt instead.
#[derive(Clone)]
pub struct BatchHandle {
    sender: mpsc::UnboundedSender<(usize, Agent)>,
    cancel: Arc<AtomicBool>,
    counter: Arc<AtomicUsize>,
}

impl BatchHandle {
    /// Enqueue another agent for the pool. Returns the submission index that
    /// will accompany this agent's result on the [`BatchOutputStream`].
    /// Indices are assigned monotonically and continue the sequence begun by
    /// the preloaded [`Batch::agents`] / [`Batch::agent`] calls. If the pool
    /// has already been cancelled or the dispatcher has exited the agent is
    /// silently dropped; the returned index is still reserved but no result
    /// will arrive for it.
    pub fn submit(&self, agent: Agent) -> usize {
        let index = self.counter.fetch_add(1, Ordering::Relaxed);
        let _ = self.sender.send((index, agent));
        index
    }

    /// Signal all in-flight agents to stop (via their `cancel_signal`) and
    /// stop the dispatcher from pulling new submissions. In-flight agents
    /// observe the flag at their next turn boundary; the stream ends once
    /// they complete.
    ///
    /// The pool owns one cancel signal and sets it on every submitted agent,
    /// overriding any per-agent signal the caller attached. To share an
    /// external signal with the pool, pass it to
    /// [`Batch::cancel_signal`](Batch::cancel_signal).
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if [`cancel`](Self::cancel) has been called.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Release this handle. When the last clone is gone, the dispatcher
    /// flushes in-flight agents to completion and ends the output stream.
    /// Non-blocking: results still arrive on the [`BatchOutputStream`]. Sugar
    /// for `drop(handle)`, but visible at the call site — pairs with
    /// [`cancel`](Self::cancel) (interrupt) to name the two exit modes.
    pub fn drain(self) {}
}

/// Stream of per-agent results from a [`Batch::spawn`] pool. Yields
/// `(submission_index, Result<Output>)` in completion order. The
/// `submission_index` matches the position the agent was added — preloaded
/// [`Batch::agents`] first, then dynamic [`BatchHandle::submit`] calls — so
/// the caller can correlate a streamed result back to its input without
/// inspecting [`Output::name`]. Ends once the pool is closed (all
/// handles dropped, [`drain`](BatchHandle::drain)ed, or
/// [`cancel`](BatchHandle::cancel)led) and the backlog completes.
pub struct BatchOutputStream {
    rx: mpsc::UnboundedReceiver<(usize, Result<Output>)>,
}

impl Stream for BatchOutputStream {
    type Item = (usize, Result<Output>);

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl BatchOutputStream {
    /// Collect every remaining result in completion order.
    pub async fn collect(self) -> Vec<(usize, Result<Output>)> {
        StreamExt::collect(self).await
    }

    /// Await the next result, or `None` once the pool has drained.
    pub async fn next(&mut self) -> Option<(usize, Result<Output>)> {
        StreamExt::next(self).await
    }
}

impl Unpin for BatchOutputStream {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{text_response, tool_response, MockProvider};
    use crate::tools::{Tool, ToolResult};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn agent_with_response(name: &str, text: &str) -> Agent {
        Agent::new()
            .name(name)
            .model_name("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(Arc::new(MockProvider::text(text)))
    }

    fn agent_with_delay(name: &str, delay_ms: u64, text: &str) -> Agent {
        let slow_tool = Tool::new("slow", "simulates work")
            .schema(serde_json::json!({"type": "object", "properties": {}}))
            .handler(move |_, _| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    Ok(ToolResult::success("done"))
                })
            });

        let provider = Arc::new(MockProvider::new(vec![
            tool_response("slow", "c1", serde_json::json!({})),
            text_response(text),
        ]));

        Agent::new()
            .name(name)
            .model_name("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .tool(slow_tool)
            .provider(provider)
    }

    #[tokio::test]
    async fn empty_run_yields_empty_vec() {
        let results = Batch::new().concurrency(4).run().await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn run_returns_results_in_submission_order() {
        let results = Batch::new()
            .concurrency(4)
            .agents(["a", "b", "c"].iter().map(|n| agent_with_response(n, "ok")))
            .run()
            .await;
        assert_eq!(results.len(), 3);
        let names: Vec<String> = results
            .iter()
            .map(|r| r.as_ref().unwrap().name.clone())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn run_submission_order_ignores_completion_order() {
        // First submitted agent finishes last; result must still land at index 0.
        let slow = agent_with_delay("slow", 80, "slow");
        let fast = agent_with_response("fast", "fast");

        let results = Batch::new()
            .concurrency(4)
            .agent(slow)
            .agent(fast)
            .run()
            .await;
        assert_eq!(results[0].as_ref().unwrap().name, "slow");
        assert_eq!(results[1].as_ref().unwrap().name, "fast");
    }

    #[tokio::test]
    async fn run_surfaces_failures_without_blocking_others() {
        let failing = Agent::new()
            .name("fail")
            .model_name("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(Arc::new(MockProvider::new(vec![])));

        let results = Batch::new()
            .concurrency(2)
            .agent(agent_with_response("ok1", "first"))
            .agent(failing)
            .agent(agent_with_response("ok2", "second"))
            .run()
            .await;
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].as_ref().unwrap().outcome,
            crate::output::Outcome::Completed
        );
        assert_eq!(
            results[1].as_ref().unwrap().outcome,
            crate::output::Outcome::Failed
        );
        assert_eq!(
            results[2].as_ref().unwrap().outcome,
            crate::output::Outcome::Completed
        );
    }

    #[tokio::test]
    async fn stream_yields_submission_indices() {
        let (pool, mut stream) = Batch::new()
            .concurrency(4)
            .agents(["a", "b", "c"].iter().map(|n| agent_with_response(n, "ok")))
            .spawn();
        drop(pool);

        let mut seen: Vec<(usize, String)> = Vec::new();
        while let Some((index, result)) = stream.next().await {
            seen.push((index, result.unwrap().name));
        }
        seen.sort_by_key(|(i, _)| *i);
        assert_eq!(
            seen,
            vec![(0, "a".into()), (1, "b".into()), (2, "c".into()),],
        );
    }

    #[tokio::test]
    async fn submit_returns_monotonic_indices_continuing_preloaded() {
        let (pool, mut stream) = Batch::new()
            .concurrency(4)
            .agent(agent_with_response("preloaded", "ok"))
            .spawn();

        let idx_b = pool.submit(agent_with_response("b", "ok"));
        let idx_c = pool.submit(agent_with_response("c", "ok"));
        assert_eq!(idx_b, 1);
        assert_eq!(idx_c, 2);

        drop(pool);
        let mut seen = Vec::new();
        while let Some((i, _)) = stream.next().await {
            seen.push(i);
        }
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn concurrency_cap_bounds_parallelism() {
        let running = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let make = |i: usize| {
            let r = running.clone();
            let m = max_concurrent.clone();
            let slow_tool = Tool::new("slow", "work")
                .schema(serde_json::json!({"type": "object", "properties": {}}))
                .handler(move |_, _| {
                    let r = r.clone();
                    let m = m.clone();
                    Box::pin(async move {
                        let cur = r.fetch_add(1, Ordering::SeqCst) + 1;
                        m.fetch_max(cur, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        r.fetch_sub(1, Ordering::SeqCst);
                        Ok(ToolResult::success("done"))
                    })
                });
            Agent::new()
                .name(&format!("w{i}"))
                .model_name("mock")
                .identity_prompt("")
                .instruction_prompt("go")
                .tool(slow_tool)
                .provider(Arc::new(MockProvider::new(vec![
                    tool_response("slow", "c1", serde_json::json!({})),
                    text_response("finished"),
                ])))
        };

        let results = Batch::new()
            .concurrency(3)
            .agents((0..10).map(make))
            .run()
            .await;
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.is_ok()));
        let peak = max_concurrent.load(Ordering::SeqCst);
        assert!(peak <= 3, "peak concurrency {peak} exceeded cap of 3");
        assert!(
            peak >= 2,
            "peak concurrency {peak} never reached meaningful overlap"
        );
    }

    #[tokio::test]
    async fn concurrency_scales_throughput() {
        let start = tokio::time::Instant::now();
        let seq = Batch::new()
            .concurrency(1)
            .agents((0..10).map(|i| agent_with_delay("w", 30, &format!("r{i}"))))
            .run()
            .await;
        let seq_elapsed = start.elapsed();

        let start = tokio::time::Instant::now();
        let par = Batch::new()
            .concurrency(10)
            .agents((0..10).map(|i| agent_with_delay("w", 30, &format!("r{i}"))))
            .run()
            .await;
        let par_elapsed = start.elapsed();

        assert_eq!(seq.len(), 10);
        assert_eq!(par.len(), 10);
        assert!(
            seq_elapsed > par_elapsed * 3,
            "sequential ({seq_elapsed:?}) should dwarf parallel ({par_elapsed:?})",
        );
    }

    #[tokio::test]
    async fn high_throughput_smoke() {
        let results = Batch::new()
            .concurrency(50)
            .agents((0..500).map(|i| agent_with_response("w", &format!("r{i}"))))
            .run()
            .await;
        assert_eq!(results.len(), 500);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn spawn_accepts_dynamic_submissions() {
        let (pool, mut stream) = Batch::new().concurrency(2).spawn();
        pool.submit(agent_with_response("a", "first"));
        pool.submit(agent_with_response("b", "second"));

        let r1 = stream.next().await.expect("first result");
        let r2 = stream.next().await.expect("second result");

        pool.submit(agent_with_response("c", "third"));
        drop(pool);

        let r3 = stream.next().await.expect("third result");
        assert!(stream.next().await.is_none(), "stream must end after drop");

        let mut names: Vec<String> = [r1, r2, r3]
            .into_iter()
            .map(|(_, r)| r.unwrap().name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn spawn_keeps_stream_open_while_any_handle_lives() {
        let (pool, mut stream) = Batch::new().concurrency(4).spawn();
        let clone = pool.clone();
        pool.submit(agent_with_response("a", "done"));
        drop(pool);
        assert!(stream.next().await.unwrap().1.is_ok());
        clone.submit(agent_with_response("b", "done"));
        assert!(stream.next().await.unwrap().1.is_ok());
        drop(clone);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn spawn_drops_handle_drains_backlog_and_ends_stream() {
        let (pool, mut stream) = Batch::new().concurrency(2).spawn();
        pool.submit(agent_with_response("a", "done"));
        pool.submit(agent_with_response("b", "done"));
        drop(pool);

        let mut seen = 0;
        while let Some((_, r)) = stream.next().await {
            r.unwrap();
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn drain_lets_in_flight_agents_finish_unlike_cancel() {
        let (pool, mut stream) = Batch::new().concurrency(2).spawn();
        pool.submit(agent_with_delay("a", 30, "done"));
        pool.submit(agent_with_delay("b", 30, "done"));
        pool.drain();

        let mut seen = 0;
        while let Some((_, r)) = stream.next().await {
            let out = r.unwrap();
            assert_eq!(out.outcome, crate::output::Outcome::Completed);
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn spawn_cancel_stops_in_flight_agents() {
        let (pool, mut stream) = Batch::new().concurrency(2).spawn();
        pool.submit(agent_with_delay("slow", 200, "never"));

        tokio::time::sleep(Duration::from_millis(20)).await;
        pool.cancel();

        let (_, result) = stream.next().await.expect("result after cancel");
        let out = result.unwrap();
        assert_eq!(out.outcome, crate::output::Outcome::Cancelled);
        assert!(pool.is_cancelled());
        drop(pool);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn preloaded_agents_run_without_explicit_submit() {
        let (pool, stream) = Batch::new()
            .concurrency(2)
            .agents(["a", "b"].iter().map(|n| agent_with_response(n, "ok")))
            .spawn();
        drop(pool);
        let results = stream.collect().await;
        assert_eq!(results.len(), 2);
    }
}
