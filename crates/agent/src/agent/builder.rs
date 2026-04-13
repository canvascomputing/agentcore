use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::provider::LlmProvider;
use crate::provider::model::ModelSpec;
use crate::persistence::session::SessionStore;
use super::context::{InvocationContext, generate_agent_name};
use super::event::Event;
use super::output::{AgentOutput, OutputSchema};
use super::prompts::{BehaviorPrompt, ContextBuilder, EnvironmentContext};
use super::queue::CommandQueue;
use super::r#loop::AgentLoop;
use super::r#trait::Agent;
use crate::tools::{Tool, ToolRegistry};

const DEFAULT_MAX_TOKENS: u32 = 4096;
const READ_ONLY_MAX_TOKENS: u32 = DEFAULT_MAX_TOKENS / 2;

#[derive(Clone)]
pub struct AgentBuilder {
    // Agent definition
    name: Option<String>,
    description: String,
    model: ModelSpec,
    system_prompt: String,
    max_tokens: u32,
    max_turns: Option<u32>,
    max_budget: Option<f64>,
    output_schema: Option<OutputSchema>,
    max_schema_retries: u32,
    behavior_prompts: Vec<(BehaviorPrompt, String)>,
    context_builder: ContextBuilder,
    tools: ToolRegistry,
    sub_agents: Vec<Arc<dyn Agent>>,

    // Runtime context
    provider: Option<Arc<dyn LlmProvider>>,
    prompt: String,
    template_variables: HashMap<String, Value>,
    working_directory: PathBuf,
    event_handler: Arc<dyn Fn(Event) + Send + Sync>,
    cancel_signal: Arc<AtomicBool>,
    session_store: Option<Arc<Mutex<SessionStore>>>,
    command_queue: Option<Arc<CommandQueue>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        let behavior_prompts = BehaviorPrompt::all()
            .iter()
            .map(|kind| (*kind, kind.default_content().to_string()))
            .collect();

        Self {
            name: None,
            description: String::new(),
            model: ModelSpec::Inherit,
            system_prompt: String::new(),
            max_tokens: DEFAULT_MAX_TOKENS,
            max_turns: None,
            max_budget: None,
            output_schema: None,
            max_schema_retries: 3,
            behavior_prompts,
            context_builder: ContextBuilder::new(),
            tools: ToolRegistry::new(),
            sub_agents: Vec::new(),

            provider: None,
            prompt: String::new(),
            template_variables: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            session_store: None,
            command_queue: None,
        }
    }

    // --- Agent definition ---

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the model ID. If not called, the agent inherits the parent's model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = ModelSpec::Exact(model.into());
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    pub fn max_turns(mut self, max: u32) -> Self {
        self.max_turns = Some(max);
        self
    }

    pub fn max_budget(mut self, budget: f64) -> Self {
        self.max_budget = Some(budget);
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(OutputSchema::new(schema).expect("invalid output schema"));
        self
    }

    pub fn max_schema_retries(mut self, retries: u32) -> Self {
        self.max_schema_retries = retries;
        self
    }

    pub fn behavior_prompt(mut self, kind: BehaviorPrompt, content: impl Into<String>) -> Self {
        if let Some(entry) = self.behavior_prompts.iter_mut().find(|(k, _)| *k == kind) {
            entry.1 = content.into();
        }
        self
    }

    pub fn environment_context(mut self, env: &EnvironmentContext) -> Self {
        self.context_builder.environment_context(env);
        self
    }

    pub fn instruction_files(mut self, cwd: &std::path::Path) -> Self {
        self.context_builder.instruction_files(cwd).ok();
        self
    }

    pub fn memory(mut self, memory_dir: &std::path::Path) -> Self {
        self.context_builder.memory(memory_dir).ok();
        self
    }

    pub fn user_context(mut self, context: impl Into<String>) -> Self {
        self.context_builder.user_context(context.into());
        self
    }

    pub fn sub_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.sub_agents.push(agent);
        self
    }

    /// Configure for read-only operation with minimal prompt overhead.
    pub fn read_only(mut self) -> Self {
        self.max_tokens = READ_ONLY_MAX_TOKENS;
        self.behavior_prompts.clear();
        self.context_builder = ContextBuilder::new();
        self
    }

    // --- Runtime context ---

    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    pub fn template_var(mut self, key: impl Into<String>, value: Value) -> Self {
        self.template_variables.insert(key.into(), value);
        self
    }

    pub fn template_variables(mut self, vars: HashMap<String, Value>) -> Self {
        self.template_variables = vars;
        self
    }

    pub fn working_directory(mut self, dir: PathBuf) -> Self {
        self.working_directory = dir;
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

    // --- Build & Run ---

    /// Build the agent without running it. Use when you need `Arc<dyn Agent>`
    /// (e.g., to register as a sub-agent).
    pub fn build(self) -> Result<Arc<dyn Agent>> {
        let name = self
            .name
            .unwrap_or_else(|| generate_agent_name("agent"));

        Ok(Arc::new(AgentLoop {
            name,
            description: self.description,
            model: self.model,
            system_prompt: self.system_prompt,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            max_budget: self.max_budget,
            output_schema: self.output_schema,
            max_schema_retries: self.max_schema_retries,
            behavior_prompts: self.behavior_prompts,
            context_builder: self.context_builder,
            tools: self.tools,
            sub_agents: self.sub_agents,
        }))
    }

    /// Build the agent and run it. Requires `.provider()` and `.prompt()`.
    pub async fn run(self) -> Result<AgentOutput> {
        let provider = self
            .provider
            .clone()
            .ok_or_else(|| AgenticError::Other("AgentBuilder::run() requires a provider".into()))?;

        if self.prompt.is_empty() {
            return Err(AgenticError::Other(
                "AgentBuilder::run() requires a prompt".into(),
            ));
        }

        let resolved_model = self.model.resolve(&String::new());
        let prompt = self.prompt.clone();
        let template_variables = self.template_variables.clone();
        let working_directory = self.working_directory.clone();
        let event_handler = self.event_handler.clone();
        let cancel_signal = self.cancel_signal.clone();
        let session_store = self.session_store.clone();
        let command_queue = self.command_queue.clone();

        let agent = self.build()?;

        let ctx = InvocationContext::new(provider)
            .prompt(prompt)
            .template_variables(template_variables)
            .working_directory(working_directory)
            .event_handler(event_handler)
            .cancel_signal(cancel_signal)
            .model(resolved_model);

        let ctx = match session_store {
            Some(s) => ctx.session_store(s),
            None => ctx,
        };
        let ctx = match command_queue {
            Some(q) => ctx.command_queue(q),
            None => ctx,
        };

        agent.run(ctx).await
    }
}
