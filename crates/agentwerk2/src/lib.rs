//! agentwerk2: clean-slate rewrite of agentwerk.

pub mod agents;
pub mod event;
pub mod prompts;
pub mod providers;
pub mod schemas;
pub mod tools;

pub use event::{default_logger, Event, EventKind, PolicyKind, ToolFailureKind};
pub use schemas::{format_violations, Schema, SchemaParseError, SchemaViolation};
