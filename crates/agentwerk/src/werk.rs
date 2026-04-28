//! Run many agents on a fixed number of production lines. `Werk::work` waits for a fixed cohort; `Werk::keep_working` hands back a [`Werking`] handle you can staff into while it's running.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_util::stream::{FuturesUnordered, Stream, StreamExt};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::error::Result;
use crate::output::Output;

const DEFAULT_LINES: usize = 1;

/// Workshop of agents capped to a fixed number of production lines. Build with
/// [`Werk::new`], chain [`lines`](Self::lines), then finish with
/// [`work`](Self::work) (bounded: hand in the workers, wait for all) or
/// [`keep_working`](Self::keep_working) (dynamic: get a handle and staff over time).
pub struct Werk {
    lines: usize,
    staff: Vec<Agent>,
    interrupt_signal: Option<Arc<AtomicBool>>,
}

impl Default for Werk {
    fn default() -> Self {
        Self {
            lines: DEFAULT_LINES,
            staff: Vec::new(),
            interrupt_signal: None,
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

    /// Share an external cancel signal with the workshop. Every staffed agent
    /// uses it, and [`Werking::interrupt`] writes to it. Useful when the
    /// caller already owns a signal (e.g. wired to Ctrl-C) and wants in-flight
    /// agents to observe it.
    pub fn interrupt_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.interrupt_signal = Some(signal);
        self
    }

    /// Run a fixed cohort of agents to completion under the configured line cap.
    /// Returns a [`HashMap`] keyed by each agent's name (`Agent::get_name()`).
    /// A failing agent does not abort the others. Caller is responsible for
    /// unique names — duplicates collapse, with the later result winning.
    pub async fn work<I>(mut self, agents: I) -> HashMap<String, Result<Output>>
    where
        I: IntoIterator<Item = Agent>,
    {
        self.staff.extend(agents);
        let (handle, stream) = self.keep_working(std::iter::empty::<Agent>());
        drop(handle);

        let mut results: HashMap<String, Result<Output>> = HashMap::new();
        for (name, result) in stream.collect().await {
            results.insert(name, result);
        }
        results
    }

    /// Open the workshop on a background tokio task and return a pair:
    ///
    /// - [`Werking`] — cheap, clonable handle for staffing more agents
    ///   or cancelling.
    /// - [`WerkOutputStream`] — yields `(String, Result<Output>)` in
    ///   completion order. The [`String`] is the agent's name. Ends once
    ///   all handles are dropped (let in-flight finish), or
    ///   [`interrupt`](Werking::interrupt) is called (stop in-flight) and
    ///   the backlog completes.
    ///
    /// Requires a running tokio runtime.
    pub fn keep_working<I>(self, initial: I) -> (Werking, WerkOutputStream)
    where
        I: IntoIterator<Item = Agent>,
    {
        let lines = self.lines;
        let (hire_tx, hire_rx) = mpsc::unbounded_channel::<(String, Agent)>();
        let (output_tx, output_rx) = mpsc::unbounded_channel::<(String, Result<Output>)>();
        let cancel = self
            .interrupt_signal
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

        for agent in self.staff.into_iter().chain(initial) {
            let name = agent.get_name().to_string();
            let _ = hire_tx.send((name, agent));
        }

        let dispatcher_cancel = cancel.clone();
        tokio::spawn(async move {
            dispatch(hire_rx, output_tx, lines, dispatcher_cancel).await;
        });

        let handle = Werking {
            sender: hire_tx,
            cancel,
        };
        let output = WerkOutputStream { rx: output_rx };
        (handle, output)
    }
}

async fn dispatch(
    mut hire_rx: mpsc::UnboundedReceiver<(String, Agent)>,
    output_tx: mpsc::UnboundedSender<(String, Result<Output>)>,
    lines: usize,
    cancel: Arc<AtomicBool>,
) {
    let mut in_flight: FuturesUnordered<tokio::task::JoinHandle<(String, Result<Output>)>> =
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
                // aborted — the agent's name is unrecoverable from the join error,
                // so the result is simply dropped.
                if let Ok(pair) = join {
                    let _ = output_tx.send(pair);
                }
            }
            maybe = hire_rx.recv(), if !closed && in_flight.len() < lines => {
                let Some((name, agent)) = maybe else {
                    closed = true;
                    continue;
                };
                let agent = agent.interrupt_signal(cancel.clone());
                in_flight.push(tokio::spawn(async move {
                    (name, agent.execute().await)
                }));
            }
            else => return,
        }
    }
}

/// Cheap, clonable handle to a running [`Werk`]. Obtained from
/// [`Werk::keep_working`].
///
/// While any clone of the handle is alive, the workshop accepts new agents.
/// Dropping the last clone closes the workshop gracefully: pending and
/// in-flight agents finish, then the output stream ends. Use
/// [`interrupt`](Self::interrupt) to stop in-flight agents instead.
#[derive(Clone)]
pub struct Werking {
    sender: mpsc::UnboundedSender<(String, Agent)>,
    cancel: Arc<AtomicBool>,
}

impl Werking {
    /// Staff the workshop with another agent. The agent is run as soon as a
    /// production line is free. If the workshop has already been cancelled or
    /// the dispatcher has exited the agent is silently dropped.
    pub fn staff(&self, agent: Agent) {
        let name = agent.get_name().to_string();
        let _ = self.sender.send((name, agent));
    }

    /// Staff the workshop with several agents at once.
    pub fn staff_more<I>(&self, agents: I)
    where
        I: IntoIterator<Item = Agent>,
    {
        for agent in agents {
            self.staff(agent);
        }
    }

    /// Signal all in-flight agents to stop (via their `interrupt_signal`) and
    /// stop the dispatcher from accepting new staff. In-flight agents
    /// observe the flag at their next step boundary; the stream ends once
    /// they complete.
    ///
    /// The workshop owns one cancel signal and sets it on every staffed agent,
    /// overriding any per-agent signal the caller attached. To share an
    /// external signal with the workshop, pass it to
    /// [`Werk::interrupt_signal`](Werk::interrupt_signal).
    pub fn interrupt(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if [`interrupt`](Self::interrupt) has been called.
    pub fn is_interrupted(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Stream of per-agent results from a [`Werk::keep_working`] workshop. Yields
/// `(String, Result<Output>)` in completion order. The [`String`] is the
/// agent's name (`Agent::get_name()`). Ends once the workshop is closed
/// (all handles dropped, or [`interrupt`](Werking::interrupt)ed) and the
/// backlog completes.
pub struct WerkOutputStream {
    rx: mpsc::UnboundedReceiver<(String, Result<Output>)>,
}

impl Stream for WerkOutputStream {
    type Item = (String, Result<Output>);

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl WerkOutputStream {
    /// Collect every remaining result in completion order.
    pub async fn collect(self) -> Vec<(String, Result<Output>)> {
        StreamExt::collect(self).await
    }

    /// Await the next result, or `None` once the workshop has closed.
    pub async fn next(&mut self) -> Option<(String, Result<Output>)> {
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
            .model("mock")
            .role("")
            .work("go")
            .provider(Arc::new(MockProvider::text(text)))
    }

    fn agent_with_delay(name: &str, delay_ms: u64, text: &str) -> Agent {
        let slow_tool = Tool::new("slow", "simulates work")
            .contract(serde_json::json!({"type": "object", "properties": {}}))
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
            .model("mock")
            .role("")
            .work("go")
            .tool(slow_tool)
            .provider(provider)
    }

    #[tokio::test]
    async fn empty_work_yields_empty_map() {
        let results = Werk::new().lines(4).work(std::iter::empty::<Agent>()).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn work_returns_results_keyed_by_name() {
        let results = Werk::new()
            .lines(4)
            .work(["a", "b", "c"].iter().map(|n| agent_with_response(n, "ok")))
            .await;
        assert_eq!(results.len(), 3);
        for name in ["a", "b", "c"] {
            let out = results
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .as_ref()
                .unwrap();
            assert_eq!(out.name, name);
        }
    }

    #[tokio::test]
    async fn work_completion_order_does_not_affect_lookup() {
        // First agent finishes last; lookup by name still works.
        let slow = agent_with_delay("slow", 80, "slow");
        let fast = agent_with_response("fast", "fast");

        let results = Werk::new().lines(4).work([slow, fast]).await;
        assert_eq!(results.get("slow").unwrap().as_ref().unwrap().name, "slow");
        assert_eq!(results.get("fast").unwrap().as_ref().unwrap().name, "fast");
    }

    #[tokio::test]
    async fn work_surfaces_failures_without_blocking_others() {
        let failing = Agent::new()
            .name("fail")
            .model("mock")
            .role("")
            .work("go")
            .provider(Arc::new(MockProvider::new(vec![])));

        let results = Werk::new()
            .lines(2)
            .work([
                agent_with_response("ok1", "first"),
                failing,
                agent_with_response("ok2", "second"),
            ])
            .await;
        assert_eq!(results.len(), 3);
        assert_eq!(
            results.get("ok1").unwrap().as_ref().unwrap().outcome,
            crate::output::Outcome::Completed
        );
        assert_eq!(
            results.get("fail").unwrap().as_ref().unwrap().outcome,
            crate::output::Outcome::Failed
        );
        assert_eq!(
            results.get("ok2").unwrap().as_ref().unwrap().outcome,
            crate::output::Outcome::Completed
        );
    }

    #[tokio::test]
    async fn stream_yields_agent_names() {
        let (handle, mut stream) = Werk::new()
            .lines(4)
            .keep_working(std::iter::empty::<Agent>());
        handle.staff_more(["a", "b", "c"].iter().map(|n| agent_with_response(n, "ok")));
        drop(handle);

        let mut seen: Vec<String> = Vec::new();
        while let Some((name, result)) = stream.next().await {
            seen.push(name);
            result.unwrap();
        }
        seen.sort();
        assert_eq!(seen, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn lines_cap_bounds_parallelism() {
        let running = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let make = |i: usize| {
            let r = running.clone();
            let m = max_concurrent.clone();
            let slow_tool = Tool::new("slow", "work")
                .contract(serde_json::json!({"type": "object", "properties": {}}))
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
                .model("mock")
                .role("")
                .work("go")
                .tool(slow_tool)
                .provider(Arc::new(MockProvider::new(vec![
                    tool_response("slow", "c1", serde_json::json!({})),
                    text_response("finished"),
                ])))
        };

        let results = Werk::new().lines(3).work((0..10).map(make)).await;
        assert_eq!(results.len(), 10);
        assert!(results.values().all(|r| r.is_ok()));
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
            .work((0..10).map(|i| agent_with_delay(&format!("seq{i}"), 30, &format!("r{i}"))))
            .await;
        let seq_elapsed = start.elapsed();

        let start = tokio::time::Instant::now();
        let par = Werk::new()
            .lines(10)
            .work((0..10).map(|i| agent_with_delay(&format!("par{i}"), 30, &format!("r{i}"))))
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
            .work((0..500).map(|i| agent_with_response(&format!("w{i}"), &format!("r{i}"))))
            .await;
        assert_eq!(results.len(), 500);
        assert!(results.values().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn keep_working_accepts_dynamic_staff() {
        let (handle, mut stream) = Werk::new()
            .lines(2)
            .keep_working(std::iter::empty::<Agent>());
        handle.staff(agent_with_response("a", "first"));
        handle.staff(agent_with_response("b", "second"));

        let r1 = stream.next().await.expect("first result");
        let r2 = stream.next().await.expect("second result");

        handle.staff(agent_with_response("c", "third"));
        drop(handle);

        let r3 = stream.next().await.expect("third result");
        assert!(stream.next().await.is_none(), "stream must end after drop");

        let mut names: Vec<String> = [r1, r2, r3].into_iter().map(|(name, _)| name).collect();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn keep_working_keeps_stream_open_while_any_handle_lives() {
        let (handle, mut stream) = Werk::new()
            .lines(4)
            .keep_working(std::iter::empty::<Agent>());
        let clone = handle.clone();
        handle.staff(agent_with_response("a", "done"));
        drop(handle);
        assert!(stream.next().await.unwrap().1.is_ok());
        clone.staff(agent_with_response("b", "done"));
        assert!(stream.next().await.unwrap().1.is_ok());
        drop(clone);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn keep_working_drops_handle_completes_backlog_and_ends_stream() {
        let (handle, mut stream) = Werk::new()
            .lines(2)
            .keep_working(std::iter::empty::<Agent>());
        handle.staff(agent_with_response("a", "done"));
        handle.staff(agent_with_response("b", "done"));
        drop(handle);

        let mut seen = 0;
        while let Some((_, r)) = stream.next().await {
            r.unwrap();
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn drop_lets_in_flight_agents_finish_unlike_interrupt() {
        let (handle, mut stream) = Werk::new()
            .lines(2)
            .keep_working(std::iter::empty::<Agent>());
        handle.staff(agent_with_delay("a", 30, "done"));
        handle.staff(agent_with_delay("b", 30, "done"));
        drop(handle);

        let mut seen = 0;
        while let Some((_, r)) = stream.next().await {
            let out = r.unwrap();
            assert_eq!(out.outcome, crate::output::Outcome::Completed);
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn keep_working_interrupt_stops_in_flight_agents() {
        let (handle, mut stream) = Werk::new()
            .lines(2)
            .keep_working(std::iter::empty::<Agent>());
        handle.staff(agent_with_delay("slow", 200, "never"));

        tokio::time::sleep(Duration::from_millis(20)).await;
        handle.interrupt();

        let (_, result) = stream.next().await.expect("result after interrupt");
        let out = result.unwrap();
        assert_eq!(out.outcome, crate::output::Outcome::Cancelled);
        assert!(handle.is_interrupted());
        drop(handle);
        assert!(stream.next().await.is_none());
    }
}
