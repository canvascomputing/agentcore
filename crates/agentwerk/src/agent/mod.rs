pub(crate) mod compact;
mod event;
pub(crate) mod r#loop;
mod output;
mod pool;
pub(crate) mod prompts;
pub(crate) mod queue;
mod running;
pub(crate) mod werk;

pub use compact::CompactReason;
pub use event::{AgentEvent, AgentEventKind};
pub use output::{AgentOutput, AgentStatistics, AgentStatus};
pub use pool::{AgentPool, AgentJobId, AgentPoolStrategy};
pub use prompts::DEFAULT_BEHAVIOR_PROMPT;
pub use running::{AgentHandle, AgentOutputFuture};
pub use werk::Agent;
pub(crate) use r#loop::{LoopRuntime, LoopSpec};
