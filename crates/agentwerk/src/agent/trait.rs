use std::future::Future;
use std::pin::Pin;

use crate::error::Result;

use super::output::AgentOutput;
use super::context::InvocationContext;

/// The single agent interface. Implemented by AgentLoop (via AgentBuilder)
/// and any user-defined agent.
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>>;
}
