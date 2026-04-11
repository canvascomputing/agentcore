mod builder;
mod context;
mod event;
mod r#loop;
mod output;
mod queue;

use std::future::Future;
use std::pin::Pin;

use crate::error::Result;

pub use builder::AgentBuilder;
pub use context::{InvocationContext, generate_agent_id};
pub use event::Event;
pub use output::{AgentOutput, OutputSchema, validate_value};
pub use queue::{CommandQueue, CommandSource, QueuePriority, QueuedCommand};

/// The single agent interface.
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>>;
}
