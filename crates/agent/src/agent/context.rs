use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::cost::CostTracker;
use crate::provider::LlmProvider;
use crate::session::SessionStore;

use super::event::Event;
use super::queue::CommandQueue;

/// Runtime context passed to Agent::run().
#[derive(Clone)]
pub struct InvocationContext {
    pub input: String,
    pub state: HashMap<String, Value>,
    pub working_directory: PathBuf,
    pub provider: Arc<dyn LlmProvider>,
    pub cost_tracker: CostTracker,
    pub on_event: Arc<dyn Fn(Event) + Send + Sync>,
    pub cancelled: Arc<AtomicBool>,
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub command_queue: Option<Arc<CommandQueue>>,
    pub agent_id: String,
}

impl InvocationContext {
    pub fn child(&self, agent_name: &str) -> Self {
        let mut child = self.clone();
        child.agent_id = generate_agent_id(agent_name);
        child
    }

    pub fn with_input(&self, input: impl Into<String>) -> Self {
        let mut child = self.clone();
        child.input = input.into();
        child
    }
}

pub fn generate_agent_id(name: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{name}_{nanos}")
}

pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
