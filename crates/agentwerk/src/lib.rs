//! agentwerk: minimal Rust crate for building agentic workflows.

pub mod agents;
pub mod event;
pub mod prompts;
pub mod providers;
pub mod schemas;
pub mod tools;

#[cfg(test)]
pub(crate) mod test_util;

// Workshop: agents pull tickets from the system
pub use agents::Agent;
pub use agents::Ticket;
pub use agents::TicketSystem;

// Tuning, telemetry, durable state
pub use agents::Knowledge;
pub use agents::Policies;
pub use agents::Stats;

// Observation
pub use event::Event;
