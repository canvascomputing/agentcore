use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::provider::LlmProvider;
use crate::provider::model::ModelSpec;
use crate::provider::retry::{DEFAULT_MAX_REQUEST_RETRIES, DEFAULT_BACKOFF_MS};

use crate::persistence::session::SessionStore;
use super::context::{InvocationContext, generate_agent_name};
use super::event::Event;
use super::output::{AgentOutput, OutputSchema};
use super::prompts::{BehaviorPrompt, EnvironmentContext};
use super::queue::CommandQueue;
use super::r#loop::AgentLoop;
use super::r#trait::Agent;
use crate::tools::{Tool, ToolRegistry, SpawnAgentTool};

#[derive(Clone)]
pub struct AgentBuilder {
    // Agent definition
    name: Option<String>,
    model: ModelSpec,
    identity_prompt: String,
    max_tokens: u32,
    max_turns: u32,
    output_schema: Option<Value>,
    max_schema_retries: u32,
    behavior_prompts: Vec<(BehaviorPrompt, String)>,
    user_context_blocks: Vec<String>,
    environment_prompt: Option<String>,
    tools: ToolRegistry,
    pub(crate) max_request_retries: u32,
    pub(crate) request_retry_backoff_ms: u64,
    pub(crate) retries_customized: bool,
    sub_agents: Vec<Arc<dyn Agent>>,
    prompt_errors: Vec<String>,

    // Runtime context
    provider: Option<Arc<dyn LlmProvider>>,
    instruction_prompt: String,
    template_variables: HashMap<String, Value>,
    working_directory: PathBuf,
    event_handler: Arc<dyn Fn(Event) + Send + Sync>,
    cancel_signal: Arc<AtomicBool>,
    session_dir: Option<PathBuf>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        let behavior_prompts = BehaviorPrompt::all()
            .iter()
            .map(|kind| (*kind, kind.default_content().to_string()))
            .collect();

        Self {
            name: None,
            model: ModelSpec::Inherit,
            identity_prompt: String::new(),
            max_tokens: crate::UNLIMITED,
            max_turns: crate::UNLIMITED,
            output_schema: None,
            max_schema_retries: 10,
            behavior_prompts,
            user_context_blocks: Vec::new(),
            environment_prompt: None,
            tools: ToolRegistry::new(),
            max_request_retries: DEFAULT_MAX_REQUEST_RETRIES,
            request_retry_backoff_ms: DEFAULT_BACKOFF_MS,
            retries_customized: false,
            sub_agents: Vec::new(),
            prompt_errors: Vec::new(),

            provider: None,
            instruction_prompt: String::new(),
            template_variables: HashMap::new(),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            session_dir: None,
        }
    }

    // --- Agent definition ---

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the model ID. If not called, the agent inherits the parent's model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = ModelSpec::Exact(model.into());
        self
    }

    /// The agent's persistent identity — who it is and how it behaves.
    pub fn identity_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.identity_prompt = prompt.into();
        self
    }

    /// Load the identity prompt from a file.
    pub fn identity_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.identity_prompt = self.read_file(path.into());
        self
    }

    /// Maximum output tokens per LLM request. `UNLIMITED` (0) uses the provider default.
    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    /// Maximum agentic loop iterations. `UNLIMITED` (0) means no limit.
    pub fn max_turns(mut self, max: u32) -> Self {
        self.max_turns = max;
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Maximum retries for structured output compliance. `UNLIMITED` (0) retries indefinitely.
    pub fn max_schema_retries(mut self, retries: u32) -> Self {
        self.max_schema_retries = retries;
        self
    }

    /// Maximum retries for transient API errors (429, 529, network failures).
    pub fn max_request_retries(mut self, n: u32) -> Self {
        self.max_request_retries = n;
        self.retries_customized = true;
        self
    }

    /// Base delay in ms for exponential backoff on request retries (`backoff * 2^attempt`).
    pub fn request_retry_backoff_ms(mut self, ms: u64) -> Self {
        self.request_retry_backoff_ms = ms;
        self.retries_customized = true;
        self
    }

    pub fn behavior_prompt(mut self, kind: BehaviorPrompt, content: impl Into<String>) -> Self {
        if let Some(entry) = self.behavior_prompts.iter_mut().find(|(k, _)| *k == kind) {
            entry.1 = content.into();
        }
        self
    }

    /// Load a behavior prompt override from a file.
    pub fn behavior_prompt_file(mut self, kind: BehaviorPrompt, path: impl Into<PathBuf>) -> Self {
        let content = self.read_file(path.into());
        if let Some(entry) = self.behavior_prompts.iter_mut().find(|(k, _)| *k == kind) {
            entry.1 = content;
        }
        self
    }

    /// Append additional context alongside the instruction prompt. Can be called multiple times.
    pub fn context_prompt(mut self, content: impl Into<String>) -> Self {
        self.user_context_blocks.push(content.into());
        self
    }

    /// Append additional context from a file. Can be called multiple times.
    pub fn context_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        let content = self.read_file(path.into());
        self.user_context_blocks.push(content);
        self
    }

    /// Override the environment context (working directory, platform, OS version, date).
    pub fn environment_prompt(mut self, content: impl Into<String>) -> Self {
        self.environment_prompt = Some(content.into());
        self
    }

    /// Load the environment prompt override from a file.
    pub fn environment_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.environment_prompt = Some(self.read_file(path.into()));
        self
    }

    pub fn sub_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.sub_agents.push(agent);
        self
    }

    // --- Runtime context ---

    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// The task for this run — what to do right now.
    pub fn instruction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.instruction_prompt = prompt.into();
        self
    }

    /// Load the instruction prompt from a file.
    pub fn instruction_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.instruction_prompt = self.read_file(path.into());
        self
    }

    pub fn template_variable(mut self, key: impl Into<String>, value: Value) -> Self {
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

    /// Enable session transcript persistence to the given directory.
    pub fn session_dir(mut self, dir: PathBuf) -> Self {
        self.session_dir = Some(dir);
        self
    }

    // --- Internal helpers ---

    /// Read a file's contents, collecting errors for deferred reporting.
    fn read_file(&mut self, path: PathBuf) -> String {
        match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(err) => {
                self.prompt_errors.push(format!(
                    "Failed to read prompt from {}: {}",
                    path.display(),
                    err
                ));
                String::new()
            }
        }
    }

    fn check_prompt_errors(&self) -> Result<()> {
        if self.prompt_errors.is_empty() {
            return Ok(());
        }
        Err(AgenticError::Other(self.prompt_errors.join("; ")))
    }

    // --- Build & Run ---

    /// Build the agent without running it. Use when you need `Arc<dyn Agent>`
    /// (e.g., to register as a sub-agent).
    pub fn build(mut self) -> Result<Arc<dyn Agent>> {
        self.check_prompt_errors()?;

        let name = self
            .name
            .unwrap_or_else(|| generate_agent_name("agent"));

        if !self.sub_agents.is_empty() && self.tools.get("spawn_agent").is_none() {
            self.tools.register(SpawnAgentTool::new().sub_agents(self.sub_agents));
        }

        let output_schema = self.output_schema.map(OutputSchema::new).transpose()?;

        Ok(Arc::new(AgentLoop {
            name,
            model: self.model,
            identity_prompt: self.identity_prompt,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            output_schema,
            max_schema_retries: self.max_schema_retries,
            behavior_prompts: self.behavior_prompts,
            user_context_blocks: self.user_context_blocks,
            tools: self.tools,
            max_request_retries: self.max_request_retries,
            request_retry_backoff_ms: self.request_retry_backoff_ms,
        }))
    }

    /// Build the agent and run it. Requires `.provider()` and `.instruction_prompt()`.
    pub async fn run(self) -> Result<AgentOutput> {
        self.check_prompt_errors()?;

        let provider = self
            .provider
            .clone()
            .ok_or_else(|| AgenticError::Other("AgentBuilder::run() requires a provider".into()))?;

        if self.instruction_prompt.is_empty() {
            return Err(AgenticError::Other(
                "AgentBuilder::run() requires a prompt".into(),
            ));
        }

        let env_context = match self.environment_prompt {
            Some(ref custom) => custom.clone(),
            None => {
                let env = EnvironmentContext::collect(&self.working_directory);
                super::prompts::format_environment_context(&env)
            }
        };

        let resolved_model = self.model.resolve(&String::new());
        let prompt = self.instruction_prompt.clone();
        let template_variables = self.template_variables.clone();
        let working_directory = self.working_directory.clone();
        let event_handler = self.event_handler.clone();
        let cancel_signal = self.cancel_signal.clone();
        let session_dir = self.session_dir.clone();

        let agent = self.build()?;

        let mut ctx = InvocationContext::new(provider)
            .instruction_prompt(prompt)
            .template_variables(template_variables)
            .working_directory(working_directory)
            .event_handler(event_handler)
            .cancel_signal(cancel_signal)
            .model(resolved_model)
            .environment_context(env_context)
            .command_queue(Arc::new(CommandQueue::new()));

        if let Some(dir) = session_dir {
            let store = SessionStore::new(&dir, &generate_agent_name("session"));
            ctx = ctx.session_store(Arc::new(Mutex::new(store)));
        }

        agent.run(ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_prompt_file_loads_content() {
        let dir = std::env::temp_dir().join("agentwerk_test_builder");
        let path = dir.join("identity.txt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "You are a test agent").unwrap();

        let builder = AgentBuilder::new().identity_prompt_file(&path);
        assert_eq!(builder.identity_prompt, "You are a test agent");
        assert!(builder.prompt_errors.is_empty());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn instruction_prompt_file_loads_content() {
        let dir = std::env::temp_dir().join("agentwerk_test_builder_instr");
        let path = dir.join("instruction.txt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "Do the thing").unwrap();

        let builder = AgentBuilder::new().instruction_prompt_file(&path);
        assert_eq!(builder.instruction_prompt, "Do the thing");
        assert!(builder.prompt_errors.is_empty());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn file_prompt_missing_file_collects_error() {
        let builder = AgentBuilder::new()
            .identity_prompt_file("/nonexistent/prompt.txt");
        assert_eq!(builder.prompt_errors.len(), 1);
        assert!(builder.prompt_errors[0].contains("/nonexistent/prompt.txt"));
    }

    #[test]
    fn build_fails_on_prompt_file_error() {
        let result = AgentBuilder::new()
            .identity_prompt_file("/nonexistent/prompt.txt")
            .build();
        match result {
            Err(e) => assert!(e.to_string().contains("/nonexistent/prompt.txt")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn context_prompt_file_loads_content() {
        let dir = std::env::temp_dir().join("agentwerk_test_builder_ctx");
        let path = dir.join("context.txt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "Extra context here").unwrap();

        let builder = AgentBuilder::new().context_prompt_file(&path);
        assert!(builder.prompt_errors.is_empty());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn behavior_prompt_file_loads_content() {
        let dir = std::env::temp_dir().join("agentwerk_test_builder_bhv");
        let path = dir.join("task_exec.txt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "Custom task execution rules").unwrap();

        let builder = AgentBuilder::new()
            .behavior_prompt_file(BehaviorPrompt::TaskExecution, &path);
        let entry = builder
            .behavior_prompts
            .iter()
            .find(|(k, _)| *k == BehaviorPrompt::TaskExecution)
            .unwrap();
        assert_eq!(entry.1, "Custom task execution rules");
        assert!(builder.prompt_errors.is_empty());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[tokio::test]
    async fn sub_agent_auto_wires_spawn_tool() {
        use crate::testutil::*;

        let sub = AgentBuilder::new()
            .name("helper")
            .identity_prompt("I help.")
            .build()
            .unwrap();

        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("parent")
            .identity_prompt("I coordinate.")
            .sub_agent(sub)
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        assert!(req.tools.iter().any(|t| t.name == "spawn_agent"),
            "sub_agent() should auto-register a spawn_agent tool");
    }

    #[tokio::test]
    async fn sub_agent_does_not_clobber_user_spawn_tool() {
        use crate::testutil::*;

        let sub = AgentBuilder::new()
            .name("helper")
            .identity_prompt("I help.")
            .build()
            .unwrap();

        let user_spawn = SpawnAgentTool::new().default_model("custom-model");

        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("parent")
            .identity_prompt("I coordinate.")
            .tool(user_spawn)
            .sub_agent(sub)
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let spawn_tools: Vec<_> = req.tools.iter().filter(|t| t.name == "spawn_agent").collect();
        assert_eq!(spawn_tools.len(), 1,
            "should not duplicate spawn_agent when user already registered one");
    }

    #[test]
    fn invalid_output_schema_fails_at_build() {
        let result = AgentBuilder::new()
            .name("test")
            .identity_prompt("")
            .output_schema(serde_json::json!({"type": "string"}))
            .build();
        assert!(result.is_err(), "invalid schema should fail at build(), not panic");
    }
}
