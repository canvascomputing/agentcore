//! The crate's central user-facing type and its builder. Carries prompts, tools, and tuning knobs into the execution loop.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;
use crate::persistence::session::SessionStore;
use crate::provider::model::Model;
use crate::provider::Provider;
use crate::tools::{SpawnAgentTool, ToolLike, ToolRegistry};
use crate::util::generate_agent_name;

use crate::event::Event;
use crate::output::{Output, OutputSchema};

use super::queue::CommandQueue;
use super::r#loop::{run_loop, LoopRuntime, LoopState};
use super::spec::AgentSpec;

/// An agent. Cheap to clone: the static template is shared, per-run fields are not.
///
/// ```
/// use std::sync::Arc;
/// use agentwerk::Agent;
/// use agentwerk::testutil::MockProvider;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let provider = Arc::new(MockProvider::text("Hello!"));
///
/// let agent = Agent::new()
///     .provider(provider)
///     .model_name("claude-sonnet-4-20250514")
///     .identity_prompt("You are a helpful assistant.");
///
/// let first = agent.clone().instruction_prompt("Greet me.").run().await.unwrap();
/// assert_eq!(first.response_raw, "Hello!");
/// # });
/// ```
#[derive(Clone)]
pub struct Agent {
    pub(crate) spec: Arc<AgentSpec>,
    pub(crate) provider: Option<Arc<dyn Provider>>,
    pub(crate) instruction_prompt: String,
    pub(crate) template_variables: HashMap<String, Value>,
    pub(crate) working_dir: Option<PathBuf>,
    pub(crate) event_handler: Option<Arc<dyn Fn(Event) + Send + Sync>>,
    pub(crate) cancel_signal: Option<Arc<AtomicBool>>,
    pub(crate) command_queue: Option<Arc<CommandQueue>>,
    pub(crate) session_dir: Option<PathBuf>,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            spec: Arc::new(AgentSpec::default()),
            provider: None,
            instruction_prompt: String::new(),
            event_handler: None,
            command_queue: None,
            cancel_signal: None,
            working_dir: None,
            session_dir: None,
            template_variables: HashMap::new(),
        }
    }
}

fn load_prompt_file(path: PathBuf) -> String {
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read prompt file {}: {e}", path.display()))
}

fn append_context_block(context_prompt: &mut String, content: &str) {
    if !context_prompt.is_empty() {
        context_prompt.push_str("\n\n");
    }
    context_prompt.push_str("<context>\n");
    context_prompt.push_str(content);
    context_prompt.push_str("\n</context>");
}

fn load_json_file(path: PathBuf) -> Value {
    let content = load_prompt_file(path.clone());
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("invalid JSON in {}: {e}", path.display()))
}

impl Agent {
    /// Default number of retries for transient API errors.
    pub const DEFAULT_MAX_REQUEST_RETRIES: u32 = AgentSpec::DEFAULT_MAX_REQUEST_RETRIES;

    /// Default base delay for the exponential-backoff retry policy.
    pub const DEFAULT_REQUEST_RETRY_DELAY: Duration = AgentSpec::DEFAULT_REQUEST_RETRY_DELAY;

    /// A fresh agent with a generated `name`, no provider, no tools, and no prompts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mutate the shared `AgentSpec` via copy-on-write (`Arc::make_mut`).
    fn with_spec<F: FnOnce(&mut AgentSpec)>(mut self, f: F) -> Self {
        f(Arc::make_mut(&mut self.spec));
        self
    }

    /// Replace `{key}` placeholders in `template` with this agent's template variables.
    pub(crate) fn interpolate(&self, template: &str) -> String {
        let mut result = template.to_string();
        for (key, value) in &self.template_variables {
            let replacement = match value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            result = result.replace(&format!("{{{key}}}"), &replacement);
        }
        result
    }

    /// Override the generated name.
    pub fn name(self, n: impl Into<String>) -> Self {
        self.with_spec(|c| c.name = n.into())
    }

    /// Set the model by name. Context window size is auto-detected; use [`Agent::model`] to override.
    pub fn model_name(self, name: impl Into<String>) -> Self {
        self.with_spec(|c| c.model = Some(Model::from_name(name)))
    }

    /// Set the full [`Model`] — name plus capability overrides.
    pub fn model(self, model: Model) -> Self {
        self.with_spec(|c| c.model = Some(model))
    }

    /// The agent's persistent identity — who it is and how it behaves.
    pub fn identity_prompt(self, p: impl Into<String>) -> Self {
        self.with_spec(|c| c.identity_prompt = p.into())
    }

    /// Load the identity prompt from a file.
    pub fn identity_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let s = load_prompt_file(path.into());
        self.with_spec(|c| c.identity_prompt = s)
    }

    /// Maximum output tokens per request (`max_tokens` on the wire).
    pub fn max_request_tokens(self, n: u32) -> Self {
        self.with_spec(|c| c.max_request_tokens = Some(n))
    }

    /// Maximum agentic loop iterations.
    pub fn max_turns(self, n: u32) -> Self {
        self.with_spec(|c| c.max_turns = Some(n))
    }

    /// Maximum cumulative input tokens across the run.
    pub fn max_input_tokens(self, n: u64) -> Self {
        self.with_spec(|c| c.max_input_tokens = Some(n))
    }

    /// Maximum cumulative output tokens across the run.
    pub fn max_output_tokens(self, n: u64) -> Self {
        self.with_spec(|c| c.max_output_tokens = Some(n))
    }

    /// Register a tool.
    pub fn tool(self, tool: impl ToolLike + 'static) -> Self {
        self.with_spec(|c| c.tool_registry.register(tool))
    }

    /// Register a structured output schema. Panics if the schema is invalid.
    pub fn output_schema(self, schema: Value) -> Self {
        let schema =
            OutputSchema::new(schema).unwrap_or_else(|e| panic!("invalid output schema: {e}"));
        self.with_spec(|c| c.output_schema = Some(schema))
    }

    /// Load a structured output schema from a JSON file.
    pub fn output_schema_file(self, path: impl Into<PathBuf>) -> Self {
        self.output_schema(load_json_file(path.into()))
    }

    /// Maximum retries for structured output compliance. Default is 10.
    pub fn max_schema_retries(self, n: u32) -> Self {
        self.with_spec(|c| c.max_schema_retries = Some(n))
    }

    /// Maximum retries for transient API errors (429, 529, network failures).
    pub fn max_request_retries(self, n: u32) -> Self {
        self.with_spec(|c| c.max_request_retries = n)
    }

    /// Base delay for exponential backoff on request retries.
    pub fn request_retry_delay(self, delay: Duration) -> Self {
        self.with_spec(|c| c.request_retry_delay = delay)
    }

    /// Park the agent idle after a terminal output until a peer message arrives or `cancel_signal` fires.
    ///
    /// [`Agent::spawn`] sets this implicitly. Call it only on a sub-agent template
    /// that should idle in the background after the orchestrator spawns it.
    pub fn keep_alive(self) -> Self {
        self.with_spec(|c| c.keep_alive = true)
    }

    /// Override the default behavior prompt.
    pub fn behavior_prompt(self, content: impl Into<String>) -> Self {
        let content = content.into();
        self.with_spec(|c| c.behavior_prompt = content)
    }

    /// Load a behavior prompt override from a file.
    pub fn behavior_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let content = load_prompt_file(path.into());
        self.with_spec(|c| c.behavior_prompt = content)
    }

    /// Append additional context alongside the instruction prompt.
    pub fn context_prompt(self, content: impl Into<String>) -> Self {
        let content = content.into();
        self.with_spec(|c| append_context_block(&mut c.context_prompt, &content))
    }

    /// Append additional context from a file.
    pub fn context_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let content = load_prompt_file(path.into());
        self.with_spec(|c| append_context_block(&mut c.context_prompt, &content))
    }

    /// Register agents callable by name as sub-agents.
    pub fn sub_agents(self, agents: impl IntoIterator<Item = Agent>) -> Self {
        let agents: Vec<_> = agents.into_iter().collect();
        self.with_spec(|c| c.sub_agents.extend(agents))
    }

    /// Install the provider this agent calls out to.
    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    /// Resolve the provider from environment variables. See [`crate::provider::from_env`].
    pub fn provider_from_env(self) -> Result<Self> {
        Ok(self.provider(crate::provider::from_env()?))
    }

    /// Resolve the model from environment variables.
    ///
    /// Priority: `MODEL` → `*_MODEL` (provider-prefixed) → hosted default.
    pub fn model_from_env(self) -> Result<Self> {
        Ok(self.model_name(crate::provider::environment::model_from_env()?))
    }

    /// The task for this run — what to do right now.
    pub fn instruction_prompt(mut self, p: impl Into<String>) -> Self {
        self.instruction_prompt = p.into();
        self
    }

    /// Load the instruction prompt from a file.
    pub fn instruction_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.instruction_prompt = load_prompt_file(path.into());
        self
    }

    /// Bind `{key}` to `value` for placeholder substitution in all prompts before the run.
    pub fn template_variable(mut self, key: impl Into<String>, value: Value) -> Self {
        self.template_variables.insert(key.into(), value);
        self
    }

    /// Working directory surfaced to tools and the environment prompt. Defaults to the process cwd.
    pub fn working_dir(mut self, d: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(d.into());
        self
    }

    /// Observe loop activity. The handler must be cheap and non-blocking.
    pub fn event_handler(mut self, h: Arc<dyn Fn(Event) + Send + Sync>) -> Self {
        self.event_handler = Some(h);
        self
    }

    /// Drop every event, opting out of the default stderr logger.
    pub fn silent(mut self) -> Self {
        self.event_handler = Some(Arc::new(|_| {}));
        self
    }

    /// Share a cancel flag. Setting it to `true` stops the loop at the next safe point.
    pub fn cancel_signal(mut self, s: Arc<AtomicBool>) -> Self {
        self.cancel_signal = Some(s);
        self
    }

    /// Install an externally-owned command queue so an `AgentHandle` can inject instructions.
    pub(crate) fn command_queue(mut self, q: Arc<CommandQueue>) -> Self {
        self.command_queue = Some(q);
        self
    }

    /// Enable session transcript persistence to the given directory.
    pub fn session_dir(mut self, d: impl Into<PathBuf>) -> Self {
        self.session_dir = Some(d.into());
        self
    }

    /// The agent's name.
    pub fn get_name(&self) -> &str {
        &self.spec.name
    }

    /// Drive the loop to completion and return the agent's output.
    ///
    /// Requires `.provider()` (or [`Agent::provider_from_env`]), `.model_name()`
    /// (or [`Agent::model_from_env`]), and `.instruction_prompt()`.
    pub async fn run(&self) -> Result<Output> {
        let (spec, runtime) = self.compile(None);
        let runtime = Arc::new(runtime);
        let instruction = self.interpolate(&self.instruction_prompt);
        let context_prompt = spec.context_prompt(runtime.environment.as_deref());
        let state = LoopState::initial(context_prompt, instruction);
        run_loop(runtime, spec, state).await
    }

    /// Run as a child under a parent's run-tree. `parent_spec` supplies the model fallback.
    pub(crate) async fn run_child(
        &self,
        parent_spec: &AgentSpec,
        parent_runtime: &LoopRuntime,
    ) -> Result<Output> {
        let (spec, runtime) = self.compile(Some((parent_spec, parent_runtime)));
        let runtime = Arc::new(runtime);
        let instruction = self.interpolate(&self.instruction_prompt);
        let context_prompt = spec.context_prompt(runtime.environment.as_deref());
        let state = LoopState::initial(context_prompt, instruction);
        run_loop(runtime, spec, state).await
    }

    /// Apply LLM-supplied JSON overrides. Missing keys are left alone, unknown keys ignored.
    pub(crate) fn apply_overrides(mut self, overrides: &Value) -> Self {
        if let Some(m) = overrides.get("model").and_then(Value::as_str) {
            self = self.model_name(m);
        }
        if let Some(i) = overrides.get("identity").and_then(Value::as_str) {
            self = self.identity_prompt(i);
        }
        if let Some(t) = overrides.get("max_request_tokens").and_then(Value::as_u64) {
            self = self.max_request_tokens(t as u32);
        }
        if let Some(t) = overrides.get("max_input_tokens").and_then(Value::as_u64) {
            self = self.max_input_tokens(t);
        }
        if let Some(t) = overrides.get("max_output_tokens").and_then(Value::as_u64) {
            self = self.max_output_tokens(t);
        }
        if let Some(mt) = overrides.get("max_turns").and_then(Value::as_u64) {
            self = self.max_turns(mt as u32);
        }
        if let Some(sr) = overrides.get("max_schema_retries").and_then(Value::as_u64) {
            self = self.max_schema_retries(sr as u32);
        }
        if let Some(rr) = overrides.get("max_request_retries").and_then(Value::as_u64) {
            self = self.max_request_retries(rr as u32);
        }
        if let Some(ms) = overrides.get("request_retry_delay").and_then(Value::as_u64) {
            self = self.request_retry_delay(Duration::from_millis(ms));
        }
        if let Some(schema) = overrides.get("output_schema").cloned() {
            self = self.output_schema(schema);
        }
        self
    }

    /// Compile into the `(spec, runtime)` pair the loop consumes.
    ///
    /// Root runs (`parent = None`) require an explicit model. Sub-agents
    /// (`parent = Some(...)`) inherit the model and externals from the parent.
    pub(crate) fn compile(
        &self,
        parent: Option<(&AgentSpec, &LoopRuntime)>,
    ) -> (Arc<AgentSpec>, LoopRuntime) {
        let resolved_model = match (self.spec.model.as_ref(), parent) {
            (Some(m), _) => m.clone(),
            (None, Some((parent_spec, _))) => parent_spec.model().clone(),
            (None, None) => panic!(
                "Agent::run() requires .model() / .model_name() on root agents (sub-agents inherit)"
            ),
        };

        let mut spec = Arc::clone(&self.spec);
        Arc::make_mut(&mut spec).model = Some(resolved_model);

        let runtime = match parent {
            Some((_, parent_runtime)) => self.inherit_runtime(parent_runtime, &spec),
            None => self.build_runtime(&spec),
        };

        (spec, runtime)
    }

    /// Build the root `LoopRuntime`. Requires `self.provider` to be set.
    fn build_runtime(&self, spec: &AgentSpec) -> LoopRuntime {
        let provider = self.provider.clone().unwrap_or_else(|| {
            panic!("Agent::run() requires .provider() (or .provider_from_env()) on root agents")
        });

        let working_dir = self
            .working_dir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let event_handler: Arc<dyn Fn(Event) + Send + Sync> = self
            .event_handler
            .clone()
            .unwrap_or_else(Event::default_logger);

        let cancel_signal = self
            .cancel_signal
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

        // Every root run carries a command queue so background sub-agents can post
        // notifications back. An externally supplied queue wins so a handle can reach the loop.
        let command_queue = Some(
            self.command_queue
                .clone()
                .unwrap_or_else(|| Arc::new(CommandQueue::new())),
        );

        let session_store = self.session_dir.as_ref().map(|dir| {
            let store = SessionStore::new(dir, &generate_agent_name("session"));
            Arc::new(Mutex::new(store))
        });

        let environment = Some(LoopRuntime::environment(&working_dir));

        LoopRuntime {
            provider,
            event_handler,
            cancel_signal,
            working_dir,
            command_queue,
            session_store,
            environment,
            tool_registry: build_tools(spec),
            template_variables: self.template_variables.clone(),
        }
    }

    /// Build a child `LoopRuntime`: parent externals, with this agent's per-run fields overriding.
    fn inherit_runtime(&self, parent: &LoopRuntime, spec: &AgentSpec) -> LoopRuntime {
        LoopRuntime {
            provider: self
                .provider
                .clone()
                .unwrap_or_else(|| parent.provider.clone()),
            event_handler: self
                .event_handler
                .clone()
                .unwrap_or_else(|| parent.event_handler.clone()),
            cancel_signal: self
                .cancel_signal
                .clone()
                .unwrap_or_else(|| parent.cancel_signal.clone()),
            working_dir: self
                .working_dir
                .clone()
                .unwrap_or_else(|| parent.working_dir.clone()),
            command_queue: parent.command_queue.clone(),
            session_store: parent.session_store.clone(),
            environment: parent.environment.clone(),
            tool_registry: build_tools(spec),
            template_variables: self.template_variables.clone(),
        }
    }
}

/// Clone `spec.tools`, auto-wiring `SpawnAgentTool` when sub-agents exist and the slot is free.
fn build_tools(spec: &AgentSpec) -> Arc<ToolRegistry> {
    let mut tools = spec.tool_registry.clone();
    if !spec.sub_agents.is_empty() && tools.get("spawn_agent").is_none() {
        tools.register(SpawnAgentTool);
    }
    Arc::new(tools)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventKind;
    use crate::output::Outcome;

    #[test]
    fn silent_sets_a_no_op_handler() {
        let agent = Agent::new().silent();
        let handler = agent
            .event_handler
            .as_ref()
            .expect(".silent() must install a handler")
            .clone();
        handler(Event::new(
            "t",
            EventKind::AgentFinished {
                turns: 1,
                outcome: Outcome::Completed,
            },
        ));
    }

    #[test]
    fn default_logger_is_used_when_no_handler_is_set() {
        let agent = Agent::new()
            .name("t")
            .model_name("mock")
            .identity_prompt("")
            .provider(std::sync::Arc::new(crate::testutil::MockProvider::text(
                "ok",
            )));
        assert!(agent.event_handler.is_none());
        let _ = agent.compile(None);
    }

    #[test]
    fn identity_prompt_file_loads_content() {
        let dir = std::env::temp_dir().join("agentwerk_test_werk_identity");
        let path = dir.join("identity.txt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "You are a test agent").unwrap();

        let agent = Agent::new().identity_prompt_file(&path);
        assert_eq!(agent.spec.identity_prompt, "You are a test agent");

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "failed to read prompt file")]
    fn missing_prompt_file_panics() {
        let _ = Agent::new().identity_prompt_file("/nonexistent/xxx.txt");
    }

    #[test]
    fn output_schema_file_loads_valid_schema() {
        let dir = std::env::temp_dir().join("agentwerk_test_werk_schema");
        let path = dir.join("schema.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            &path,
            r#"{"type":"object","properties":{"answer":{"type":"string"}}}"#,
        )
        .unwrap();

        let agent = Agent::new().output_schema_file(&path);
        assert!(agent.spec.output_schema.is_some());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "failed to read prompt file")]
    fn output_schema_file_missing_file_panics() {
        let _ = Agent::new().output_schema_file("/nonexistent/schema.json");
    }

    #[test]
    #[should_panic(expected = "invalid output schema")]
    fn invalid_output_schema_panics() {
        let _ = Agent::new()
            .name("test")
            .identity_prompt("")
            .output_schema(serde_json::json!({"type": "string"}));
    }

    #[tokio::test]
    async fn apply_overrides_applies_json_fields() {
        let base = Agent::new().name("x").model_name("original").max_turns(3);
        let applied = base.apply_overrides(&serde_json::json!({
            "model": "overridden",
            "max_turns": 7,
            "max_request_tokens": 256,
            "max_input_tokens": 4000,
            "max_output_tokens": 5000
        }));
        assert_eq!(applied.spec.max_turns, Some(7));
        assert_eq!(applied.spec.max_request_tokens, Some(256));
        assert_eq!(applied.spec.max_input_tokens, Some(4000));
        assert_eq!(applied.spec.max_output_tokens, Some(5000));
        match &applied.spec.model {
            Some(m) => assert_eq!(m.name, "overridden"),
            None => panic!("expected a resolved model"),
        }
    }

    #[tokio::test]
    #[should_panic(expected = ".provider()")]
    async fn missing_provider_panics_on_run() {
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("x")
            .instruction_prompt("do");
        let _ = agent.run().await;
    }
}
