//! Agent: identity + prompt parts + reference to a shared `Runtime`.

use std::sync::{Arc, Mutex};

use crate::providers::Provider;

use super::r#loop::handle_tickets;
use super::tickets::TicketSystem;

pub const DEFAULT_BEHAVIOR: &str =
    "Be concise. Take the user's instruction at face value.";

/// Externals shared across all agents that drive the same ticket queue.
pub struct Runtime {
    pub provider: Arc<dyn Provider>,
    pub model: String,
    pub system: Arc<Mutex<TicketSystem>>,
}

#[derive(Clone)]
pub struct Agent {
    pub name: String,
    pub runtime: Arc<Runtime>,
    role: Option<String>,
    behavior: Option<String>,
    context: Option<String>,
}

impl Agent {
    pub fn new(name: impl Into<String>, runtime: Arc<Runtime>) -> Self {
        Self {
            name: name.into(),
            runtime,
            role: None,
            behavior: None,
            context: None,
        }
    }

    pub fn role(mut self, r: impl Into<String>) -> Self {
        self.role = Some(r.into());
        self
    }

    pub fn behavior(mut self, b: impl Into<String>) -> Self {
        self.behavior = Some(b.into());
        self
    }

    pub fn context(mut self, c: impl Into<String>) -> Self {
        self.context = Some(c.into());
        self
    }

    pub(super) fn system_prompt(&self) -> String {
        let behavior = self
            .behavior
            .clone()
            .unwrap_or_else(|| DEFAULT_BEHAVIOR.to_string());

        let mut parts = Vec::with_capacity(3);
        if let Some(r) = &self.role {
            parts.push(r.clone());
        }
        parts.push(behavior);
        if let Some(c) = &self.context {
            parts.push(c.clone());
        }
        parts.join("\n\n")
    }

    /// Drive a single agent's drain loop. Returns when the
    /// `interrupt_signal` on the runtime's `TicketSystem` fires or a
    /// policy limit is violated.
    pub async fn run(self) {
        handle_tickets(self).await;
    }
}
