//! Agent builder surface and execution loop. This is the package every user of the crate reaches into first.

pub(crate) mod agent;
pub(crate) mod compact;
pub(crate) mod error;
pub(crate) mod r#loop;
pub(crate) mod prompts;
pub(crate) mod spec;
pub(crate) mod work;

pub use agent::{Agent, AgentWorking, IntoContract, IntoPrompt, OutputFuture};
pub use error::AgentError;
pub use prompts::DEFAULT_BEHAVIOR;
pub(crate) use r#loop::LoopRuntime;
pub(crate) use spec::AgentSpec;
