//! Agent builder surface and execution loop. This is the package every user of the crate reaches into first.

pub(crate) mod agent;
mod batch;
pub(crate) mod compact;
mod event;
pub(crate) mod r#loop;
mod output;
pub(crate) mod prompts;
pub(crate) mod queue;
mod spawn;
pub(crate) mod spec;

pub use agent::Agent;
pub use batch::batch;
pub use compact::CompactReason;
pub use event::{Event, EventKind};
pub use output::{AgentOutput, AgentStatistics, AgentStatus};
pub use prompts::DEFAULT_BEHAVIOR_PROMPT;
pub(crate) use r#loop::LoopRuntime;
pub use spawn::{AgentHandle, AgentOutputFuture};
pub(crate) use spec::AgentSpec;
