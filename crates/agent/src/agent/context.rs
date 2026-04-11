use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::provider::cost::CostTracker;
use crate::provider::LlmProvider;
use crate::persistence::session::SessionStore;

use super::event::Event;
use super::queue::CommandQueue;

/// Runtime context passed to Agent::run().
#[derive(Clone)]
pub struct InvocationContext {
    // Lifecycle
    pub agent_id: String,
    pub event_handler: Arc<dyn Fn(Event) + Send + Sync>,
    pub cancel_signal: Arc<AtomicBool>,

    // What to do
    pub prompt: String,
    pub template_variables: HashMap<String, Value>,
    pub working_directory: PathBuf,

    // LLM
    pub provider: Arc<dyn LlmProvider>,
    pub cost_tracker: CostTracker,

    // Optional persistence
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub command_queue: Option<Arc<CommandQueue>>,
}

impl InvocationContext {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            agent_id: generate_agent_id("agent"),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            prompt: String::new(),
            template_variables: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            provider,
            cost_tracker: CostTracker::new(),
            session_store: None,
            command_queue: None,
        }
    }

    pub fn child(&self, agent_name: &str) -> Self {
        let mut child = self.clone();
        child.agent_id = generate_agent_id(agent_name);
        child
    }

    pub fn with_input(&self, input: impl Into<String>) -> Self {
        let mut child = self.clone();
        child.prompt = input.into();
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
