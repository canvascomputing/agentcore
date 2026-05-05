//! agentwerk: minimal Rust crate for building agentic workflows.

pub mod agents;
pub mod event;
pub mod prompts;
pub mod providers;
pub mod schemas;
pub mod tools;

pub use agents::{
    Agent, Memory, ResultRecord, Runnable, Stats, Status, Ticket, TicketSystem,
};
pub use event::{default_logger, Event, EventKind, PolicyKind, ToolFailureKind};
pub use schemas::{format_violations, Schema, SchemaParseError, SchemaViolation};
pub use tools::{MemoryTool, Tool, ToolContext, ToolLike, ToolResult};
