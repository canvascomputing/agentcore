//! Runs many agents with bounded concurrency. Turns the one-shot `Agent::run` into a streaming pool that accepts jobs while it is running.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;

use crate::error::{AgenticError, Result};

use super::agent::Agent;
use super::output::AgentOutput;

const DEFAULT_BATCH_SIZE: usize = 10;

pub type AgentJobId = u64;

/// Controls the order in which `next()` / `drain()` yield results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentPoolStrategy {
    /// Results are yielded as agents finish. An agent that completes
    /// earlier is returned before one spawned earlier. Default.
    CompletionOrder,
    /// Results are yielded in the order agents were spawned. Agents
    /// that finish out of order are buffered until their predecessor
    /// has been returned.
    SpawnOrder,
}

impl Default for AgentPoolStrategy {
    fn default() -> Self {
        Self::CompletionOrder
    }
}

/// Controlled-parallelism executor for `Agent`s. Clone-safe? No — use one pool
/// per batch. Methods take `&self` so tasks can be spawned and consumed from
/// different code paths concurrently.
pub struct AgentPool {
    batch_size: usize,
    ordering: AgentPoolStrategy,
    semaphore: Arc<Semaphore>,
    state: Mutex<PoolState>,
    next_id: AtomicU64,
}

struct PoolState {
    join_set: JoinSet<(AgentJobId, Result<AgentOutput>)>,
    /// Used only when `ordering == SpawnOrder` — buffers completed jobs that
    /// arrived before their predecessor.
    buffer: BTreeMap<AgentJobId, Result<AgentOutput>>,
    /// Next AgentJobId expected by `SpawnOrder` ordering.
    next_expected: AgentJobId,
}

impl AgentPool {
    pub fn new() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            ordering: AgentPoolStrategy::default(),
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
    pub fn ordering(mut self, o: AgentPoolStrategy) -> Self {
        self.ordering = o;
        self
    }

    /// Submit a pre-configured agent. Awaits a permit if at capacity.
    pub async fn spawn(&self, agent: Agent) -> AgentJobId {
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
    pub async fn next(&self) -> Option<(AgentJobId, Result<AgentOutput>)> {
        match self.ordering {
            AgentPoolStrategy::CompletionOrder => self.next_by_completion().await,
            AgentPoolStrategy::SpawnOrder => self.next_by_spawn_order().await,
        }
    }

    /// Drain every currently pending job, yielding per the configured ordering.
    pub async fn drain(&self) -> Vec<(AgentJobId, Result<AgentOutput>)> {
        let mut out = Vec::new();
        while let Some(entry) = self.next().await {
            out.push(entry);
        }
        out
    }

    async fn next_by_completion(&self) -> Option<(AgentJobId, Result<AgentOutput>)> {
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

    async fn next_by_spawn_order(&self) -> Option<(AgentJobId, Result<AgentOutput>)> {
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
    use crate::tools::{Tool, ToolResult};
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
    async fn pool_drain_spawn_order() {
        let pool = AgentPool::new()
            .batch_size(2)
            .ordering(AgentPoolStrategy::SpawnOrder);
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
            .ordering(AgentPoolStrategy::SpawnOrder);
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

            let slow_tool = Tool::new("slow", "Simulates slow work")
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
                });

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
    async fn pool_spawn_order_buffers_fast_finishers() {
        // Agent A is slow (sleeps), Agent B is fast. Spawn A then B.
        // SpawnOrder should yield A first despite B completing first.
        let slow_tool = Tool::new("slow", "slow tool")
            .schema(serde_json::json!({"type": "object", "properties": {}}))
            .handler(|_, _| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    Ok(ToolResult::success("slow done"))
                })
            });

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
            .ordering(AgentPoolStrategy::SpawnOrder);
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
    async fn pool_completion_order_yields_fast_first() {
        // Agent A is slow (sleeps), Agent B is fast. Spawn A then B.
        // CompletionOrder should yield B first because it finishes first.
        let slow_tool = Tool::new("slow", "slow tool")
            .schema(serde_json::json!({"type": "object", "properties": {}}))
            .handler(|_, _| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    Ok(ToolResult::success("slow done"))
                })
            });

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
            .ordering(AgentPoolStrategy::CompletionOrder);
        pool.spawn(a).await;
        pool.spawn(b).await;

        let results = pool.drain().await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.as_ref().unwrap().response_raw, "B-done");
        assert_eq!(results[1].1.as_ref().unwrap().response_raw, "A-done");
    }

    #[tokio::test]
    async fn pool_completion_order_failure_does_not_block() {
        let pool = AgentPool::new()
            .batch_size(2)
            .ordering(AgentPoolStrategy::CompletionOrder);
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
        pool.spawn(agent_with_response("ok")).await;

        let results = pool.drain().await;
        assert_eq!(results.len(), 2);
        let ok_count = results.iter().filter(|r| r.1.is_ok()).count();
        let err_count = results.iter().filter(|r| r.1.is_err()).count();
        assert_eq!(ok_count, 1);
        assert_eq!(err_count, 1);
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
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .tool(slow_tool)
            .provider(provider)
    }

    #[tokio::test]
    async fn perf_throughput_scales_with_batch_size() {
        // Sequential: 20 agents × 50ms each ≈ 1000ms
        let start = tokio::time::Instant::now();
        let pool = AgentPool::new().batch_size(1);
        for i in 0..20 {
            pool.spawn(agent_with_delay("w", 50, &format!("r{i}")))
                .await;
        }
        let seq_results = pool.drain().await;
        let seq_elapsed = start.elapsed();

        assert_eq!(seq_results.len(), 20);
        assert!(seq_results.iter().all(|r| r.1.is_ok()));

        // Parallel: 20 agents × 50ms with batch_size=20 ≈ 50ms
        let start = tokio::time::Instant::now();
        let pool = AgentPool::new().batch_size(20);
        for i in 0..20 {
            pool.spawn(agent_with_delay("w", 50, &format!("r{i}")))
                .await;
        }
        let par_results = pool.drain().await;
        let par_elapsed = start.elapsed();

        assert_eq!(par_results.len(), 20);
        assert!(par_results.iter().all(|r| r.1.is_ok()));

        assert!(
            seq_elapsed > par_elapsed * 3,
            "Sequential ({seq_elapsed:?}) should be at least 3x slower than parallel ({par_elapsed:?})"
        );
    }

    #[tokio::test]
    async fn perf_spawn_order_vs_completion_order() {
        let mut times = Vec::new();

        for strategy in [
            AgentPoolStrategy::CompletionOrder,
            AgentPoolStrategy::SpawnOrder,
        ] {
            let start = tokio::time::Instant::now();
            let pool = AgentPool::new().batch_size(10).ordering(strategy);
            for i in 0..50 {
                pool.spawn(agent_with_delay("w", 20, &format!("r{i}")))
                    .await;
            }
            let results = pool.drain().await;
            let elapsed = start.elapsed();

            assert_eq!(results.len(), 50);
            assert!(results.iter().all(|r| r.1.is_ok()));
            times.push(elapsed);
        }

        let diff = if times[0] > times[1] {
            times[0] - times[1]
        } else {
            times[1] - times[0]
        };
        assert!(
            diff < Duration::from_millis(500),
            "CompletionOrder ({:?}) and SpawnOrder ({:?}) should complete within 500ms of each other",
            times[0],
            times[1]
        );
    }

    #[tokio::test]
    async fn perf_high_job_count() {
        let pool = AgentPool::new().batch_size(50);
        for i in 0..500 {
            pool.spawn(agent_with_response(&format!("r{i}"))).await;
        }

        let start = tokio::time::Instant::now();
        let results = pool.drain().await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 500);
        assert!(results.iter().all(|r| r.1.is_ok()));
        assert!(
            elapsed < Duration::from_secs(5),
            "500 instant agents should drain in < 5s, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn perf_concurrent_spawn_and_consume() {
        let running = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(AgentPool::new().batch_size(5));

        // Spawner task: push 50 agents
        let pool_s = pool.clone();
        let running_s = running.clone();
        let max_s = max_concurrent.clone();
        let spawner = tokio::spawn(async move {
            for i in 0..50 {
                let r = running_s.clone();
                let m = max_s.clone();
                let slow_tool = Tool::new("slow", "work")
                    .schema(serde_json::json!({"type": "object", "properties": {}}))
                    .handler(move |_, _| {
                        let r = r.clone();
                        let m = m.clone();
                        Box::pin(async move {
                            let cur = r.fetch_add(1, Ordering::SeqCst) + 1;
                            m.fetch_max(cur, Ordering::SeqCst);
                            tokio::time::sleep(Duration::from_millis(10)).await;
                            r.fetch_sub(1, Ordering::SeqCst);
                            Ok(ToolResult::success("done"))
                        })
                    });

                let provider = Arc::new(MockProvider::new(vec![
                    tool_response("slow", "c1", serde_json::json!({})),
                    text_response(&format!("r{i}")),
                ]));

                pool_s
                    .spawn(
                        Agent::new()
                            .name("w")
                            .model("mock")
                            .identity_prompt("")
                            .instruction_prompt("go")
                            .tool(slow_tool)
                            .provider(provider),
                    )
                    .await;
            }
        });

        // Consumer: collect all results
        let mut collected = Vec::new();
        loop {
            match pool.next().await {
                Some(entry) => collected.push(entry),
                None => {
                    if spawner.is_finished() {
                        // Drain remaining after spawner is done
                        while let Some(entry) = pool.next().await {
                            collected.push(entry);
                        }
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        }
        spawner.await.unwrap();

        assert_eq!(collected.len(), 50);
        assert!(collected.iter().all(|r| r.1.is_ok()));
        assert!(
            max_concurrent.load(Ordering::SeqCst) <= 5,
            "Max concurrent ({}) should not exceed batch_size (5)",
            max_concurrent.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn perf_spawn_order_buffering_under_load() {
        let pool = AgentPool::new()
            .batch_size(10)
            .ordering(AgentPoolStrategy::SpawnOrder);

        for i in 0u64..100 {
            let delay = (i % 5) * 10; // 0, 10, 20, 30, 40ms
            pool.spawn(agent_with_delay("w", delay, &format!("r{i}")))
                .await;
        }

        let results = pool.drain().await;

        assert_eq!(results.len(), 100);
        for (idx, (job_id, result)) in results.iter().enumerate() {
            assert_eq!(
                *job_id, idx as u64,
                "Result at position {idx} has wrong job_id {job_id}"
            );
            assert!(result.is_ok(), "Job {job_id} failed unexpectedly");
        }
    }

    #[tokio::test]
    async fn perf_semaphore_backpressure() {
        let start = tokio::time::Instant::now();
        let pool = AgentPool::new().batch_size(2);

        for i in 0..10 {
            pool.spawn(agent_with_delay("w", 50, &format!("r{i}")))
                .await;
        }

        let results = pool.drain().await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.1.is_ok()));
        // 10 agents / 2 concurrent × 50ms = 250ms minimum
        assert!(
            elapsed >= Duration::from_millis(200),
            "Should take at least 200ms with batch_size=2, took {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(1000),
            "Should complete within 1s, took {elapsed:?}"
        );
    }
}
