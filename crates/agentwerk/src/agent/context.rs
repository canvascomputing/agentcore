use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::provider::LlmProvider;
use crate::persistence::session::SessionStore;

use super::event::Event;
use super::queue::CommandQueue;

/// Runtime context passed to Agent::run().
#[derive(Clone)]
pub struct InvocationContext {
    // Lifecycle
    pub agent_name: String,
    pub event_handler: Arc<dyn Fn(Event) + Send + Sync>,
    pub cancel_signal: Arc<AtomicBool>,

    // What to do
    pub instruction_prompt: String,
    pub template_variables: HashMap<String, Value>,
    pub working_directory: PathBuf,

    // LLM
    pub provider: Arc<dyn LlmProvider>,

    // Model for this context — sub-agents using Inherit resolve to this
    pub model: String,

    // Optional persistence
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub command_queue: Option<Arc<CommandQueue>>,
}

impl InvocationContext {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            agent_name: generate_agent_name("agent"),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            instruction_prompt: String::new(),
            template_variables: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            provider,
            model: String::new(),
            session_store: None,
            command_queue: None,
        }
    }

    pub fn instruction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.instruction_prompt = prompt.into();
        self
    }

    pub fn working_directory(mut self, dir: PathBuf) -> Self {
        self.working_directory = dir;
        self
    }

    pub fn template_variables(mut self, vars: HashMap<String, Value>) -> Self {
        self.template_variables = vars;
        self
    }

    pub fn template_variable(mut self, key: impl Into<String>, value: Value) -> Self {
        self.template_variables.insert(key.into(), value);
        self
    }

    pub fn event_handler(mut self, handler: Arc<dyn Fn(Event) + Send + Sync>) -> Self {
        self.event_handler = handler;
        self
    }

    pub fn cancel_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.cancel_signal = signal;
        self
    }

    pub fn session_store(mut self, store: Arc<Mutex<SessionStore>>) -> Self {
        self.session_store = Some(store);
        self
    }

    pub fn command_queue(mut self, queue: Arc<CommandQueue>) -> Self {
        self.command_queue = Some(queue);
        self
    }

    /// Set the model ID for this context. Sub-agents using `Inherit` resolve to this.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn agent_name(mut self, name: impl Into<String>) -> Self {
        self.agent_name = name.into();
        self
    }

    pub fn child(&self, name: &str) -> Self {
        let mut child = self.clone();
        child.agent_name = generate_agent_name(name);
        child
    }
}

pub(crate) fn generate_agent_name(name: &str) -> String {
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
