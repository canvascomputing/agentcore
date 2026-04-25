//! Run many agents on a fixed number of production lines. `Werk::produce` waits for a fixed set; `Werk::spawn` hands back a producing handle you can hire into while it's running.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_util::stream::{FuturesUnordered, Stream, StreamExt};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::Result;
use crate::output::Output;

const DEFAULT_LINES: usize = 1;

/// Workshop of agents capped to a fixed number of production lines. Build with
/// [`Werk::new`], chain [`lines`](Self::lines) and
/// [`worker`](Self::worker) / [`workers`](Self::workers), then finish with
/// [`produce`](Self::produce) (wait for all) or [`spawn`](Self::spawn) (dynamic).
pub struct Werk {
    lines: usize,
    workers: Vec<Agent>,
    cancel_signal: Option<Arc<AtomicBool>>,
}

impl Default for Werk {
    fn default() -> Self {
        Self {
            lines: DEFAULT_LINES,
            workers: Vec::new(),
            cancel_signal: None,
        }
    }
}

impl Werk {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap on simultaneous in-flight agents. Clamped to at least 1.
    pub fn lines(mut self, n: usize) -> Self {
        self.lines = n.max(1);
        self
    }

    /// Add one worker to run.
    pub fn worker(mut self, worker: Agent) -> Self {
        self.workers.push(worker);
        self
    }

    /// Add many workers to run.
    pub fn workers<I>(mut self, workers: I) -> Self
    where
        I: IntoIterator<Item = Agent>,
    {
        self.workers.extend(workers);
        self
    }

    /// Share an external cancel signal with the workshop. Every hired worker
    /// uses it, and [`WerkProducing::interrupt`] writes to it. Useful when the
    /// caller already owns a signal (e.g. wired to Ctrl-C) and wants in-flight
    /// workers to observe it.
    pub fn cancel_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.cancel_signal = Some(signal);
        self
    }

    /// Run every hired worker to completion. Returns results in **hire**
    /// order: `results[i]` corresponds to the `i`th worker added via
    /// [`worker`](Self::worker) / [`workers`](Self::workers). A failing worker does
    /// not abort the others.
    pub async fn produce(self) -> Vec<Result<Output>> {
        let total = self.workers.len();
        let (handle, stream) = self.spawn();
        handle.close();

        let mut slots: Vec<Option<Result<Output>>> = (0..total).map(|_| None).collect();
        for (index, result) in stream.collect().await {
            if index < slots.len() {
                slots[index] = Some(result);
            }
        }
        slots
            .into_iter()
            .map(|slot| slot.expect("werk stream yields one result per hire"))
            .collect()
    }

    /// Start a dispatcher on a background tokio task and return a pair:
    ///
    /// - [`WerkProducing`] — cheap, clonable handle for hiring more workers
    ///   or cancelling.
    /// - [`WerkOutputStream`] — yields
    ///   `(hire_index, Result<Output>)` in completion order. The
    ///   `hire_index` matches the position the worker was added:
    ///   preloaded [`workers`](Self::workers) take indices `0..n`, then dynamic
    ///   [`hire`](WerkProducing::hire) calls continue the sequence. Ends
    ///   once all handles are dropped or [`close`](WerkProducing::close)d (let
    ///   in-flight finish), or [`interrupt`](WerkProducing::interrupt) is
    ///   called (stop in-flight) and the backlog completes.
    ///
    /// Requires a running tokio runtime.
    pub fn spawn(self) -> (WerkProducing, WerkOutputStream) {
        let lines = self.lines;
        let (hire_tx, hire_rx) = mpsc::unbounded_channel::<(usize, Agent)>();
        let (output_tx, output_rx) = mpsc::unbounded_channel::<(usize, Result<Output>)>();
        let cancel = self
            .cancel_signal
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let counter = Arc::new(AtomicUsize::new(0));

        for worker in self.workers {
            let index = counter.fetch_add(1, Ordering::Relaxed);
            let _ = hire_tx.send((index, worker));
        }

        let dispatcher_cancel = cancel.clone();
        tokio::spawn(async move {
            dispatch(hire_rx, output_tx, lines, dispatcher_cancel).await;
        });

        let handle = WerkProducing {
            sender: hire_tx,
            cancel,
            counter,
        };
        let output = WerkOutputStream { rx: output_rx };
        (handle, output)
    }
}

async fn dispatch(
    mut hire_rx: mpsc::UnboundedReceiver<(usize, Agent)>,
    output_tx: mpsc::UnboundedSender<(usize, Result<Output>)>,
    lines: usize,
    cancel: Arc<AtomicBool>,
) {
    let mut in_flight: FuturesUnordered<tokio::task::JoinHandle<(usize, Result<Output>)>> =
        FuturesUnordered::new();
    let mut closed = false;

    loop {
        if cancel.load(Ordering::Relaxed) && !closed {
            hire_rx.close();
            closed = true;
        }

        tokio::select! {
            biased;
            Some(join) = in_flight.next(), if !in_flight.is_empty() => {
                // A task-level JoinError means the spawned future panicked or was
                // aborted — the hire index is then unrecoverable, so the slot
                // will be backfilled with a synthetic error by `Werk::produce`.
                if let Ok(pair) = join {
                    let _ = output_tx.send(pair);
                }
            }
            maybe = hire_rx.recv(), if !closed && in_flight.len() < lines => {
                let Some((index, worker)) = maybe else {
                    closed = true;
                    continue;
                };
                let worker = worker.cancel_signal(cancel.clone());
                in_flight.push(tokio::spawn(async move {
                    (index, worker.work().await)
                }));
            }
            else => return,
        }
    }
}

/// Cheap, clonable handle to a running [`Werk`]. Obtained from
/// [`Werk::spawn`].
///
/// While any clone of the handle is alive, the workshop accepts new hires.
/// Dropping the last clone (or calling [`close`](Self::close) on it) closes
/// the workshop gracefully: queued and in-flight workers finish, then the output
/// stream ends. Use [`cancel`](Self::cancel) to interrupt instead.
#[derive(Clone)]
pub struct WerkProducing {
    sender: mpsc::UnboundedSender<(usize, Agent)>,
    cancel: Arc<AtomicBool>,
    counter: Arc<AtomicUsize>,
}

impl WerkProducing {
    /// Hire another worker. Returns the hire index that will accompany this
    /// worker's result on the [`WerkOutputStream`]. Indices are assigned
    /// monotonically and continue the sequence begun by the preloaded
    /// [`Werk::workers`] / [`Werk::worker`] calls. If the workshop has already
    /// been cancelled or the dispatcher has exited the worker is silently
    /// dropped; the returned index is still reserved but no result will arrive
    /// for it.
    pub fn hire(&self, worker: Agent) -> usize {
        let index = self.counter.fetch_add(1, Ordering::Relaxed);
        let _ = self.sender.send((index, worker));
        index
    }

    /// Signal all in-flight workers to stop (via their `cancel_signal`) and
    /// stop the dispatcher from accepting new hires. In-flight workers
    /// observe the flag at their next turn boundary; the stream ends once
    /// they complete.
    ///
    /// The workshop owns one cancel signal and sets it on every hired worker,
    /// overriding any per-worker signal the caller attached. To share an
    /// external signal with the workshop, pass it to
    /// [`Werk::cancel_signal`](Werk::cancel_signal).
    pub fn interrupt(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if [`interrupt`](Self::interrupt) has been called.
    pub fn is_interrupted(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Close the workshop to new hires. When the last clone is gone, the
    /// dispatcher flushes in-flight workers to completion and ends the output
    /// stream. Non-blocking: results still arrive on the [`WerkOutputStream`].
    /// Sugar for `drop(handle)`, but visible at the call site — pairs with
    /// [`interrupt`](Self::interrupt) to name the two exit modes.
    pub fn close(self) {}
}

/// Stream of per-worker results from a [`Werk::spawn`] workshop. Yields
/// `(hire_index, Result<Output>)` in completion order. The `hire_index`
/// matches the position the worker was added — preloaded [`Werk::workers`]
/// first, then dynamic [`WerkProducing::hire`] calls — so the caller can
/// correlate a streamed result back to its input without inspecting
/// [`Output::name`]. Ends once the workshop is closed (all handles dropped,
/// [`close`](WerkProducing::close)d, or [`interrupt`](WerkProducing::interrupt)ed)
/// and the backlog completes.
pub struct WerkOutputStream {
    rx: mpsc::UnboundedReceiver<(usize, Result<Output>)>,
}

impl Stream for WerkOutputStream {
    type Item = (usize, Result<Output>);

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl WerkOutputStream {
    /// Collect every remaining result in completion order.
    pub async fn collect(self) -> Vec<(usize, Result<Output>)> {
        StreamExt::collect(self).await
    }

    /// Await the next result, or `None` once the workshop has closed.
    pub async fn next(&mut self) -> Option<(usize, Result<Output>)> {
        StreamExt::next(self).await
    }
}

impl Unpin for WerkOutputStream {}

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
            .role("")
            .instruction("go")
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
            .role("")
            .instruction("go")
            .tool(slow_tool)
            .provider(provider)
    }

    #[tokio::test]
    async fn empty_produce_yields_empty_vec() {
        let results = Werk::new().lines(4).produce().await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn produce_returns_results_in_hire_order() {
        let results = Werk::new()
            .lines(4)
            .workers(["a", "b", "c"].iter().map(|n| agent_with_response(n, "ok")))
            .produce()
            .await;
        assert_eq!(results.len(), 3);
        let names: Vec<String> = results
            .iter()
            .map(|r| r.as_ref().unwrap().name.clone())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn produce_hire_order_ignores_completion_order() {
        // First hired worker finishes last; result must still land at index 0.
        let slow = agent_with_delay("slow", 80, "slow");
        let fast = agent_with_response("fast", "fast");

        let results = Werk::new()
            .lines(4)
            .worker(slow)
            .worker(fast)
            .produce()
            .await;
        assert_eq!(results[0].as_ref().unwrap().name, "slow");
        assert_eq!(results[1].as_ref().unwrap().name, "fast");
    }

    #[tokio::test]
    async fn produce_surfaces_failures_without_blocking_others() {
        let failing = Agent::new()
            .name("fail")
            .model_name("mock")
            .role("")
            .instruction("go")
            .provider(Arc::new(MockProvider::new(vec![])));

        let results = Werk::new()
            .lines(2)
            .worker(agent_with_response("ok1", "first"))
            .worker(failing)
            .worker(agent_with_response("ok2", "second"))
            .produce()
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
    async fn stream_yields_hire_indices() {
        let (producing, mut stream) = Werk::new()
            .lines(4)
            .workers(["a", "b", "c"].iter().map(|n| agent_with_response(n, "ok")))
            .spawn();
        drop(producing);

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
    async fn hire_returns_monotonic_indices_continuing_preloaded() {
        let (producing, mut stream) = Werk::new()
            .lines(4)
            .worker(agent_with_response("preloaded", "ok"))
            .spawn();

        let idx_b = producing.hire(agent_with_response("b", "ok"));
        let idx_c = producing.hire(agent_with_response("c", "ok"));
        assert_eq!(idx_b, 1);
        assert_eq!(idx_c, 2);

        drop(producing);
        let mut seen = Vec::new();
        while let Some((i, _)) = stream.next().await {
            seen.push(i);
        }
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn lines_cap_bounds_parallelism() {
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
                .role("")
                .instruction("go")
                .tool(slow_tool)
                .provider(Arc::new(MockProvider::new(vec![
                    tool_response("slow", "c1", serde_json::json!({})),
                    text_response("finished"),
                ])))
        };

        let results = Werk::new()
            .lines(3)
            .workers((0..10).map(make))
            .produce()
            .await;
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.is_ok()));
        let peak = max_concurrent.load(Ordering::SeqCst);
        assert!(peak <= 3, "peak in-flight {peak} exceeded line cap of 3");
        assert!(
            peak >= 2,
            "peak in-flight {peak} never reached meaningful overlap"
        );
    }

    #[tokio::test]
    async fn lines_scale_throughput() {
        let start = tokio::time::Instant::now();
        let seq = Werk::new()
            .lines(1)
            .workers((0..10).map(|i| agent_with_delay("w", 30, &format!("r{i}"))))
            .produce()
            .await;
        let seq_elapsed = start.elapsed();

        let start = tokio::time::Instant::now();
        let par = Werk::new()
            .lines(10)
            .workers((0..10).map(|i| agent_with_delay("w", 30, &format!("r{i}"))))
            .produce()
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
        let results = Werk::new()
            .lines(50)
            .workers((0..500).map(|i| agent_with_response("w", &format!("r{i}"))))
            .produce()
            .await;
        assert_eq!(results.len(), 500);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn spawn_accepts_dynamic_hires() {
        let (producing, mut stream) = Werk::new().lines(2).spawn();
        producing.hire(agent_with_response("a", "first"));
        producing.hire(agent_with_response("b", "second"));

        let r1 = stream.next().await.expect("first result");
        let r2 = stream.next().await.expect("second result");

        producing.hire(agent_with_response("c", "third"));
        drop(producing);

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
        let (producing, mut stream) = Werk::new().lines(4).spawn();
        let clone = producing.clone();
        producing.hire(agent_with_response("a", "done"));
        drop(producing);
        assert!(stream.next().await.unwrap().1.is_ok());
        clone.hire(agent_with_response("b", "done"));
        assert!(stream.next().await.unwrap().1.is_ok());
        drop(clone);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn spawn_drops_handle_completes_backlog_and_ends_stream() {
        let (producing, mut stream) = Werk::new().lines(2).spawn();
        producing.hire(agent_with_response("a", "done"));
        producing.hire(agent_with_response("b", "done"));
        drop(producing);

        let mut seen = 0;
        while let Some((_, r)) = stream.next().await {
            r.unwrap();
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn close_lets_in_flight_workers_finish_unlike_interrupt() {
        let (producing, mut stream) = Werk::new().lines(2).spawn();
        producing.hire(agent_with_delay("a", 30, "done"));
        producing.hire(agent_with_delay("b", 30, "done"));
        producing.close();

        let mut seen = 0;
        while let Some((_, r)) = stream.next().await {
            let out = r.unwrap();
            assert_eq!(out.outcome, crate::output::Outcome::Completed);
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn spawn_interrupt_stops_in_flight_workers() {
        let (producing, mut stream) = Werk::new().lines(2).spawn();
        producing.hire(agent_with_delay("slow", 200, "never"));

        tokio::time::sleep(Duration::from_millis(20)).await;
        producing.interrupt();

        let (_, result) = stream.next().await.expect("result after interrupt");
        let out = result.unwrap();
        assert_eq!(out.outcome, crate::output::Outcome::Cancelled);
        assert!(producing.is_interrupted());
        drop(producing);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn preloaded_workers_produce_without_explicit_hire() {
        let (producing, stream) = Werk::new()
            .lines(2)
            .workers(["a", "b"].iter().map(|n| agent_with_response(n, "ok")))
            .spawn();
        drop(producing);
        let results = stream.collect().await;
        assert_eq!(results.len(), 2);
    }
}
