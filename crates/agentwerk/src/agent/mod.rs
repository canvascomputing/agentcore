//! Agent builder surface and execution loop. This is the package every user of the crate reaches into first.

pub(crate) mod agent;
pub(crate) mod compact;
pub(crate) mod error;
pub(crate) mod r#loop;
pub(crate) mod prompts;
pub(crate) mod queue;
mod spawn;
pub(crate) mod spec;

pub use agent::Agent;
pub use error::AgentError;
pub use prompts::DEFAULT_BEHAVIOR_PROMPT;
pub(crate) use r#loop::LoopRuntime;
pub use spawn::{AgentHandle, OutputFuture};
pub(crate) use spec::AgentSpec;
