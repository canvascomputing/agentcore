use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use super::prompts::{BehaviorPrompt, ContextBuilder};
use crate::tools::ToolRegistry;

use super::context::InvocationContext;
use super::output::{AgentOutput, OutputSchema};

/// The single agent interface. Implemented by AgentLoop (via AgentBuilder)
/// and any user-defined agent.
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>>;
}

/// An LLM-powered agent. Calls an LLM in a loop, executing tools until done.
/// Created via `AgentBuilder::build()`.
pub(crate) struct AgentLoop {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) model: String,
    pub(crate) system_prompt: String,
    pub(crate) max_tokens: u32,
    pub(crate) max_turns: Option<u32>,
    pub(crate) max_budget: Option<f64>,
    pub(crate) output_schema: Option<OutputSchema>,
    pub(crate) max_schema_retries: u32,
    pub(crate) behavior_prompts: Vec<(BehaviorPrompt, String)>,
    pub(crate) context_builder: ContextBuilder,
    pub(crate) tools: ToolRegistry,
    #[allow(dead_code)]
    pub(crate) sub_agents: Vec<Arc<dyn Agent>>,
}

impl Agent for AgentLoop {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>> {
        Box::pin(async move { self.execute(ctx).await })
    }
}
