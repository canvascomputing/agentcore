//! Execute multiple agents with controlled parallelism — dynamic edition.
//!
//! `AgentPool` takes already-configured `Agent`s. Jobs can be pushed while the
//! pool is running; results are consumed via `next()` (streaming) or `drain()`
//! (collect all). Ordering of results is controlled by `PoolOrdering`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;

use crate::error::{AgenticError, Result};

use super::output::AgentOutput;
use super::werk::Agent;

const DEFAULT_BATCH_SIZE: usize = 10;

pub type JobId = u64;

/// Order in which `next()` / `drain()` yields completed jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolOrdering {
    /// Yield as jobs complete — fully dynamic streaming. Default.
    Completion,
    /// Yield in the order jobs were `spawn`ed. Jobs that finish out-of-order
    /// are buffered until their predecessor has been yielded.
    Submission,
}

impl Default for PoolOrdering {
    fn default() -> Self {
        Self::Completion
    }
}

/// Controlled-parallelism executor for `Agent`s. Clone-safe? No — use one pool
/// per batch. Methods take `&self` so tasks can be spawned and consumed from
/// different code paths concurrently.
pub struct AgentPool {
    batch_size: usize,
    ordering: PoolOrdering,
    semaphore: Arc<Semaphore>,
    state: Mutex<PoolState>,
    next_id: AtomicU64,
}

struct PoolState {
    join_set: JoinSet<(JobId, Result<AgentOutput>)>,
    /// Used only when `ordering == Submission` — buffers completed jobs that
    /// arrived before their predecessor.
    buffer: BTreeMap<JobId, Result<AgentOutput>>,
    /// Next JobId expected by `Submission` ordering.
    next_expected: JobId,
}

impl AgentPool {
    pub fn new() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            ordering: PoolOrdering::default(),
            semaphore: Arc::new(Semaphore::new(DEFAULT_BATCH_SIZE)),
            state: Mutex::new(PoolState {
                join_set: JoinSet::new(),
                buffer: BTreeMap::new(),
                next_expected: 0,
            }),
            next_id: AtomicU64::new(0),
        }
    }

    /// Maximum number of jobs running concurrently.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self.semaphore = Arc::new(Semaphore::new(self.batch_size));
        self
    }

    /// Select how `next()` / `drain()` order results.
    pub fn ordering(mut self, o: PoolOrdering) -> Self {
        self.ordering = o;
        self
    }

    /// Submit a pre-configured agent. Awaits a permit if at capacity.
    pub async fn spawn(&self, agent: Agent) -> JobId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("pool semaphore closed unexpectedly");
        let mut st = self.state.lock().await;
        st.join_set.spawn(async move {
            let result = agent.run().await;
            drop(permit);
            (id, result)
        });
        id
    }

    /// Yield the next completed job per the configured ordering. `None` when
    /// no jobs are pending and no buffered entries remain.
    pub async fn next(&self) -> Option<(JobId, Result<AgentOutput>)> {
        match self.ordering {
            PoolOrdering::Completion => self.next_completion().await,
            PoolOrdering::Submission => self.next_submission().await,
        }
    }

    /// Drain every currently pending job, yielding per the configured ordering.
    pub async fn drain(&self) -> Vec<(JobId, Result<AgentOutput>)> {
        let mut out = Vec::new();
        while let Some(entry) = self.next().await {
            out.push(entry);
        }
        out
    }

    async fn next_completion(&self) -> Option<(JobId, Result<AgentOutput>)> {
        let mut st = self.state.lock().await;
        if st.join_set.is_empty() {
            return None;
        }
        match st.join_set.join_next().await {
            Some(Ok(pair)) => Some(pair),
            Some(Err(join_err)) => Some((
                u64::MAX,
                Err(AgenticError::Other(format!("task join error: {join_err}"))),
            )),
            None => None,
        }
    }

    async fn next_submission(&self) -> Option<(JobId, Result<AgentOutput>)> {
        loop {
            let mut st = self.state.lock().await;
            let next_id = st.next_expected;
            if let Some(r) = st.buffer.remove(&next_id) {
                st.next_expected = next_id + 1;
                return Some((next_id, r));
            }
            if st.join_set.is_empty() && st.buffer.is_empty() {
                return None;
            }
            let next = match st.join_set.join_next().await {
                Some(Ok(pair)) => pair,
                Some(Err(join_err)) => (
                    u64::MAX,
                    Err(AgenticError::Other(format!("task join error: {join_err}"))),
                ),
                None => return None,
            };
            st.buffer.insert(next.0, next.1);
            // Loop back to check if the just-buffered completion is the one we want.
        }
    }
}

impl Default for AgentPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{text_response, tool_response, MockProvider};
    use crate::tools::{ToolBuilder, ToolResult};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn agent_with_response(text: &str) -> Agent {
        let provider = Arc::new(MockProvider::text(text));
        Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(provider)
    }

    #[tokio::test]
    async fn pool_drain_submission_order() {
        let pool = AgentPool::new()
            .batch_size(2)
            .ordering(PoolOrdering::Submission);
        pool.spawn(agent_with_response("first")).await;
        pool.spawn(agent_with_response("second")).await;
        pool.spawn(agent_with_response("third")).await;

        let results = pool.drain().await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
        assert_eq!(results[2].0, 2);
        assert_eq!(results[0].1.as_ref().unwrap().response_raw, "first");
        assert_eq!(results[1].1.as_ref().unwrap().response_raw, "second");
        assert_eq!(results[2].1.as_ref().unwrap().response_raw, "third");
    }

    #[tokio::test]
    async fn pool_individual_failures() {
        let pool = AgentPool::new()
            .batch_size(2)
            .ordering(PoolOrdering::Submission);
        pool.spawn(agent_with_response("ok")).await;
        pool.spawn({
            let provider = Arc::new(MockProvider::new(vec![]));
            Agent::new()
                .name("fail")
                .model("mock")
                .identity_prompt("")
                .instruction_prompt("go")
                .provider(provider)
        })
        .await;
        pool.spawn(agent_with_response("also ok")).await;

        let results = pool.drain().await;

        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_ok());
        assert!(results[1].1.is_err());
        assert!(results[2].1.is_ok());
    }

    #[tokio::test]
    async fn pool_empty() {
        let pool = AgentPool::new();
        let results = pool.drain().await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn pool_runs_concurrently() {
        let running = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let pool = AgentPool::new().batch_size(3);

        for _ in 0..6 {
            let running = running.clone();
            let max_concurrent = max_concurrent.clone();

            let slow_tool = ToolBuilder::new("slow", "Simulates slow work")
                .schema(serde_json::json!({"type": "object", "properties": {}}))
                .handler(move |_, _| {
                    let running = running.clone();
                    let max_concurrent = max_concurrent.clone();
                    Box::pin(async move {
                        let current = running.fetch_add(1, Ordering::SeqCst) + 1;
                        max_concurrent.fetch_max(current, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        running.fetch_sub(1, Ordering::SeqCst);
                        Ok(ToolResult::success("done"))
                    })
                })
                .build();

            let provider = Arc::new(MockProvider::new(vec![
                tool_response("slow", "c1", serde_json::json!({})),
                text_response("finished"),
            ]));

            pool.spawn(
                Agent::new()
                    .name("worker")
                    .model("mock")
                    .identity_prompt("")
                    .instruction_prompt("go")
                    .tool(slow_tool)
                    .provider(provider),
            )
            .await;
        }

        let results = pool.drain().await;

        assert_eq!(results.len(), 6);
        assert!(results.iter().all(|r| r.1.is_ok()));
        assert!(
            max_concurrent.load(Ordering::SeqCst) >= 3,
            "Expected at least 3 concurrent agents, got {}",
            max_concurrent.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn pool_submission_ordering_buffers_fast_finishers() {
        // Agent A is slow (sleeps), Agent B is fast. Submit A then B.
        // Submission ordering should yield A first despite B completing first.
        let slow_tool = ToolBuilder::new("slow", "slow tool")
            .schema(serde_json::json!({"type": "object", "properties": {}}))
            .handler(|_, _| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    Ok(ToolResult::success("slow done"))
                })
            })
            .build();

        let a = Agent::new()
            .name("A")
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .tool(slow_tool)
            .provider(Arc::new(MockProvider::new(vec![
                tool_response("slow", "c1", serde_json::json!({})),
                text_response("A-done"),
            ])));

        let b = Agent::new()
            .name("B")
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(Arc::new(MockProvider::text("B-done")));

        let pool = AgentPool::new()
            .batch_size(2)
            .ordering(PoolOrdering::Submission);
        pool.spawn(a).await;
        pool.spawn(b).await;

        let results = pool.drain().await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1.as_ref().unwrap().response_raw, "A-done");
        assert_eq!(results[1].0, 1);
        assert_eq!(results[1].1.as_ref().unwrap().response_raw, "B-done");
    }

    #[tokio::test]
    async fn pool_dynamic_spawn_while_running() {
        // Submit one job, start consuming, spawn another mid-flight.
        let pool = Arc::new(AgentPool::new().batch_size(2));

        pool.spawn(agent_with_response("first")).await;

        let pool2 = pool.clone();
        let spawner = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            pool2.spawn(agent_with_response("second")).await;
        });

        let first = pool.next().await.unwrap();
        assert_eq!(first.1.as_ref().unwrap().response_raw, "first");

        spawner.await.unwrap();

        let second = pool.next().await.unwrap();
        assert_eq!(second.1.as_ref().unwrap().response_raw, "second");

        assert!(pool.next().await.is_none());
    }
}
