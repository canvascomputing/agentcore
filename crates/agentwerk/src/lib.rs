//! agentwerk: minimal Rust crate for building agentic workflows.

pub mod agents;
pub mod event;
pub mod prompts;
pub mod providers;
pub mod schemas;
pub mod tools;

#[cfg(test)]
pub(crate) mod test_util;

// Builder, orchestrator, run handle
pub use agents::Agent;
pub use agents::Running;
pub use agents::TicketSystem;

// Tickets and results
pub use agents::Ticket;
pub use agents::TicketResults;

// Tuning, telemetry, durable state
pub use agents::Knowledge;
pub use agents::Policies;
pub use agents::Stats;

// Observation
pub use event::Event;
