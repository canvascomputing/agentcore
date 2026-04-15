//! Execute multiple agents with controlled parallelism.
//!
//! Each agent is a fully configured `AgentBuilder` with its own provider, prompts,
//! and tools. The pipeline controls how many run concurrently via a semaphore.
//! Results are returned in push order. Individual failures do not abort the pipeline.

use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::error::Result;
use super::builder::AgentBuilder;
use super::output::AgentOutput;

const DEFAULT_BATCH_SIZE: usize = 10;

/// Execute multiple agents with controlled parallelism.
pub struct Pipeline {
    batch_size: usize,
    agents: Vec<AgentBuilder>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            agents: Vec::new(),
        }
    }

    /// Maximum number of agents running concurrently.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Add a configured agent to the pipeline.
    pub fn push(&mut self, agent: AgentBuilder) {
        self.agents.push(agent);
    }

    /// Execute all queued agents and return results in push order.
    pub async fn run(self) -> Vec<Result<AgentOutput>> {
        let agent_count = self.agents.len();
        if agent_count == 0 {
            return Vec::new();
        }

        let semaphore = Arc::new(Semaphore::new(self.batch_size));
        let mut set = JoinSet::new();

        for (index, builder) in self.agents.into_iter().enumerate() {
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            set.spawn(async move {
                let result = builder.run().await;
                drop(permit);
                (index, result)
            });
        }

        let mut results: Vec<Option<Result<AgentOutput>>> = (0..agent_count).map(|_| None).collect();
        while let Some(join_result) = set.join_next().await {
            let (index, result) = join_result.unwrap();
            results[index] = Some(result);
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::MockProvider;

    fn agent_with_response(text: &str) -> AgentBuilder {
        let provider = Arc::new(MockProvider::text(text));
        AgentBuilder::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(provider)
    }

    #[tokio::test]
    async fn pipeline_executes_in_order() {
        let mut pipeline = Pipeline::new().batch_size(2);
        pipeline.push(agent_with_response("first"));
        pipeline.push(agent_with_response("second"));
        pipeline.push(agent_with_response("third"));

        let results = pipeline.run().await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap().response_raw, "first");
        assert_eq!(results[1].as_ref().unwrap().response_raw, "second");
        assert_eq!(results[2].as_ref().unwrap().response_raw, "third");
    }

    #[tokio::test]
    async fn pipeline_individual_failures() {
        let mut pipeline = Pipeline::new().batch_size(2);
        pipeline.push(agent_with_response("ok"));
        pipeline.push({
            let provider = Arc::new(MockProvider::new(vec![])); // no responses → error
            AgentBuilder::new()
                .name("fail")
                .model("mock")
                .identity_prompt("")
                .instruction_prompt("go")
                .provider(provider)
        });
        pipeline.push(agent_with_response("also ok"));

        let results = pipeline.run().await;

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok());
    }

    #[tokio::test]
    async fn pipeline_empty() {
        let pipeline = Pipeline::new();
        let results = pipeline.run().await;
        assert!(results.is_empty());
    }
}
