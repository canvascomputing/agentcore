use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::provider::LlmProvider;
use crate::persistence::session::SessionStore;
use crate::util::generate_agent_name;

use super::event::Event;
use super::queue::CommandQueue;

/// Runtime context passed to Agent::run().
#[derive(Clone)]
pub struct RuntimeContext {
    // Lifecycle
    pub(crate) agent_name: String,
    pub(crate) event_handler: Arc<dyn Fn(Event) + Send + Sync>,
    pub(crate) cancel_signal: Arc<AtomicBool>,

    // What to do
    pub(crate) instruction_prompt: String,
    pub(crate) template_variables: HashMap<String, Value>,
    pub(crate) working_directory: PathBuf,

    // LLM
    pub(crate) provider: Arc<dyn LlmProvider>,

    // Model for this context — sub-agents using Inherit resolve to this
    pub(crate) model: String,

    // Context
    pub(crate) environment_context: Option<String>,

    // Optional persistence
    pub(crate) session_store: Option<Arc<Mutex<SessionStore>>>,
    pub(crate) command_queue: Option<Arc<CommandQueue>>,
}

impl RuntimeContext {
    pub(crate) fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            agent_name: generate_agent_name("agent"),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            instruction_prompt: String::new(),
            template_variables: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            provider,
            model: String::new(),
            environment_context: None,
            session_store: None,
            command_queue: None,
        }
    }

    pub(crate) fn instruction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.instruction_prompt = prompt.into();
        self
    }

    pub(crate) fn working_directory(mut self, dir: PathBuf) -> Self {
        self.working_directory = dir;
        self
    }

    pub(crate) fn event_handler(mut self, handler: Arc<dyn Fn(Event) + Send + Sync>) -> Self {
        self.event_handler = handler;
        self
    }

    pub(crate) fn cancel_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.cancel_signal = signal;
        self
    }

    pub(crate) fn session_store(mut self, store: Arc<Mutex<SessionStore>>) -> Self {
        self.session_store = Some(store);
        self
    }

    pub(crate) fn command_queue(mut self, queue: Arc<CommandQueue>) -> Self {
        self.command_queue = Some(queue);
        self
    }

    /// Set the model ID for this context. Sub-agents using `Inherit` resolve to this.
    pub(crate) fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub(crate) fn environment_context(mut self, ctx: String) -> Self {
        self.environment_context = Some(ctx);
        self
    }

    #[cfg(test)]
    pub(crate) fn agent_name(mut self, name: impl Into<String>) -> Self {
        self.agent_name = name.into();
        self
    }

    pub(crate) fn child(&self, name: &str) -> Self {
        let mut child = self.clone();
        child.agent_name = generate_agent_name(name);
        child
    }
}

