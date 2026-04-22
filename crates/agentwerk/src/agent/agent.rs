//! The crate's central user-facing type and its builder. Carries prompts, tools, and tuning knobs into the execution loop.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::persistence::session::SessionStore;
use crate::provider::model::Model;
use crate::provider::Provider;
use crate::tools::{SpawnAgentTool, ToolRegistry, Toolable};
use crate::util::generate_agent_name;

use super::event::Event;
use super::output::{Output, OutputSchema};
use super::queue::CommandQueue;
use super::r#loop::{run_loop, LoopRuntime, LoopState};
use super::spec::{build_context_prompt, AgentSpec};

/// An LLM-powered agent. Cheap to clone (internally `Arc`-wrapped spec + a
/// handful of small per-run fields).
///
/// Build with `Agent::new()` and chain builder methods; call `.run()` to
/// execute. The same `Agent` can be `run()` multiple times. Cloning an agent
/// and mutating per-run fields (e.g. `.instruction_prompt`) does not clone
/// the static template (tools, sub-agents, behavior prompts).
#[derive(Clone)]
pub struct Agent {
    pub(crate) spec: Arc<AgentSpec>,
    pub(crate) provider: Option<Arc<dyn Provider>>,
    pub(crate) instruction_prompt: String,
    pub(crate) template_variables: HashMap<String, Value>,
    pub(crate) working_directory: Option<PathBuf>,
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
            template_variables: HashMap::new(),
            working_directory: None,
            event_handler: None,
            cancel_signal: None,
            command_queue: None,
            session_dir: None,
        }
    }
}

fn load_prompt_file(path: PathBuf) -> String {
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read prompt file {}: {e}", path.display()))
}

fn load_json_file(path: PathBuf) -> Value {
    let content = load_prompt_file(path.clone());
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("invalid JSON in {}: {e}", path.display()))
}

impl Agent {
    /// Default number of retries for transient API errors.
    pub const DEFAULT_MAX_REQUEST_RETRIES: u32 = AgentSpec::DEFAULT_MAX_REQUEST_RETRIES;

    /// Default base delay (ms) for the exponential-backoff retry policy.
    pub const DEFAULT_BACKOFF_MS: u64 = AgentSpec::DEFAULT_BACKOFF_MS;

    pub fn new() -> Self {
        Self::default()
    }

    /// Mutate the shared `AgentSpec` via copy-on-write (`Arc::make_mut`).
    /// Per-run fields are owned and mutate directly at their call sites.
    fn with_spec<F: FnOnce(&mut AgentSpec)>(mut self, f: F) -> Self {
        f(Arc::make_mut(&mut self.spec));
        self
    }

    /// Replace `{key}` placeholders in `template` with this agent's template
    /// variables.
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

    /// Set the agent's name. `Agent::new()` pre-fills it with a generated
    /// value like `agent_<nanos>`; this method overwrites.
    pub fn name(self, n: impl Into<String>) -> Self {
        self.with_spec(|c| c.name = n.into())
    }

    /// Set the model by name. Sugar for `.model(Model::from_name(name))`. If
    /// not called, the agent inherits the parent's model. The context window
    /// size is auto-detected from the built-in registry (see
    /// [`Model::from_name`]); use `.model(Model::from_name(name).context_window_size(n))`
    /// to override.
    pub fn model_name(self, name: impl Into<String>) -> Self {
        self.with_spec(|c| c.model = Some(Model::from_name(name)))
    }

    /// Set the full [`Model`] — name plus any capability overrides (e.g.
    /// explicit context window size for local proxies or private deployments).
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

    /// Maximum output tokens per LLM request (serialized as `max_tokens` on the
    /// wire). Omit to use the provider default.
    pub fn max_request_tokens(self, n: u32) -> Self {
        self.with_spec(|c| c.max_request_tokens = Some(n))
    }

    /// Maximum agentic loop iterations. Omit for no limit.
    pub fn max_turns(self, n: u32) -> Self {
        self.with_spec(|c| c.max_turns = Some(n))
    }

    /// Maximum cumulative input tokens across the whole run before the agent stops.
    pub fn max_input_tokens(self, n: u64) -> Self {
        self.with_spec(|c| c.max_input_tokens = Some(n))
    }

    /// Maximum cumulative output tokens across the whole run before the agent stops.
    pub fn max_output_tokens(self, n: u64) -> Self {
        self.with_spec(|c| c.max_output_tokens = Some(n))
    }

    /// Register a tool.
    pub fn tool(self, tool: impl Toolable + 'static) -> Self {
        self.with_spec(|c| c.tools.register(tool))
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

    /// Base delay in milliseconds for exponential backoff on request retries.
    pub fn request_retry_delay(self, ms: u64) -> Self {
        self.with_spec(|c| c.request_retry_delay = ms)
    }

    /// Keep the agent alive after a terminal output, parking it idle until a
    /// peer message arrives or `cancel_signal` fires.
    ///
    /// Most users won't call this directly: [`Agent::spawn`] sets it
    /// implicitly. You only need it when declaring a sub-agent *template*
    /// (via `.sub_agents([...])`) that should idle while waiting for peer
    /// messages after the orchestrator spawns it in the background.
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
        self.with_spec(|c| c.context_prompts.push(content))
    }

    /// Append additional context from a file.
    pub fn context_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let content = load_prompt_file(path.into());
        self.with_spec(|c| c.context_prompts.push(content))
    }

    /// Register one or more agents as sub-agents. The LLM can call them by
    /// name once this agent runs.
    pub fn sub_agents(self, agents: impl IntoIterator<Item = Agent>) -> Self {
        let agents: Vec<_> = agents.into_iter().collect();
        self.with_spec(|c| c.sub_agents.extend(agents))
    }

    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    /// Resolve the provider from environment variables (`*_API_KEY` +
    /// `*_BASE_URL`). See [`crate::provider::from_env`] for the detection
    /// matrix. Pair with [`Agent::model_from_env`] to also pick a model, or
    /// call [`Agent::model_name`] / [`Agent::model`] explicitly.
    pub fn provider_from_env(self) -> Result<Self> {
        Ok(self.provider(crate::provider::from_env()?))
    }

    /// Resolve the model from environment variables.
    ///
    /// Priority: `MODEL` (generic override) → `*_MODEL` (provider-prefixed,
    /// same detection matrix as [`Agent::provider_from_env`]) → hosted
    /// default for the detected provider.
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

    pub fn template_variable(mut self, key: impl Into<String>, value: Value) -> Self {
        self.template_variables.insert(key.into(), value);
        self
    }

    pub fn working_directory(mut self, d: impl Into<PathBuf>) -> Self {
        self.working_directory = Some(d.into());
        self
    }

    pub fn event_handler(mut self, h: Arc<dyn Fn(Event) + Send + Sync>) -> Self {
        self.event_handler = Some(h);
        self
    }

    /// Drop every event. Opts out of the default stderr logger installed when no
    /// handler is set.
    pub fn silent(mut self) -> Self {
        self.event_handler = Some(Arc::new(|_| {}));
        self
    }

    pub fn cancel_signal(mut self, s: Arc<AtomicBool>) -> Self {
        self.cancel_signal = Some(s);
        self
    }

    /// Install an externally-owned command queue. Only used by `Agent::spawn`
    /// so the returned `AgentHandle` can inject instructions into the loop.
    pub(crate) fn command_queue(mut self, q: Arc<CommandQueue>) -> Self {
        self.command_queue = Some(q);
        self
    }

    /// Enable session transcript persistence to the given directory.
    pub fn session_dir(mut self, d: impl Into<PathBuf>) -> Self {
        self.session_dir = Some(d.into());
        self
    }

    /// Returns the agent's name (always set — eagerly generated by `Agent::new()`
    /// and overwritten by `.name(...)`).
    pub fn name_ref(&self) -> &str {
        &self.spec.name
    }

    /// Execute this agent to completion. Requires `.provider()` and `.instruction_prompt()`.
    pub async fn run(&self) -> Result<Output> {
        let (spec, runtime) = self.compile(None)?;
        let runtime = Arc::new(runtime);
        let instruction = self.interpolate(&self.instruction_prompt);
        let context_prompt =
            build_context_prompt(&spec.context_prompts, runtime.metadata.as_deref());
        let state = LoopState::initial(context_prompt, instruction);
        run_loop(runtime, spec, state, None).await
    }

    /// Crate-internal: run this agent as a child under a parent's run-tree.
    /// The `parent_spec` supplies the model fallback for `model: None`
    /// (id *and* context window size both inherit). `description` becomes the
    /// `AgentStart` event's human-readable label.
    pub(crate) async fn run_child(
        &self,
        parent_spec: &AgentSpec,
        parent_runtime: &LoopRuntime,
        description: Option<String>,
    ) -> Result<Output> {
        let (spec, runtime) = self.compile(Some((parent_spec, parent_runtime)))?;
        let runtime = Arc::new(runtime);
        let instruction = self.interpolate(&self.instruction_prompt);
        let context_prompt =
            build_context_prompt(&spec.context_prompts, runtime.metadata.as_deref());
        let state = LoopState::initial(context_prompt, instruction);
        run_loop(runtime, spec, state, description).await
    }

    /// Apply LLM-supplied JSON overrides for any Agent field a tool can
    /// legitimately set. Missing keys are left alone; unknown keys are
    /// silently ignored. Single source of truth for tool-driven config updates.
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
        if let Some(bo) = overrides.get("request_retry_delay").and_then(Value::as_u64) {
            self = self.request_retry_delay(bo);
        }
        if let Some(schema) = overrides.get("output_schema").cloned() {
            self = self.output_schema(schema);
        }
        self
    }

    /// Compile this agent into the pair of inputs the loop consumes.
    ///
    /// `parent = None` starts a root run — requires an explicit `.model()` /
    /// `.model_name()` and uses the agent's own per-run fields (provider,
    /// cancel signal, etc.) falling back to sensible defaults.
    ///
    /// `parent = Some((parent_spec, parent_runtime))` spawns a sub-agent —
    /// externals inherit from `parent_runtime` (child's own per-run fields
    /// override on a per-field basis), and `self.spec.model = None` resolves
    /// against `parent_spec.model()`.
    ///
    /// The returned `AgentSpec` is a CoW clone of `self.spec` with
    /// `model: Some(resolved)` filled in. `self.spec` itself is untouched.
    pub(crate) fn compile(
        &self,
        parent: Option<(&AgentSpec, &LoopRuntime)>,
    ) -> Result<(Arc<AgentSpec>, LoopRuntime)> {
        // Resolve the model first — root runs require an explicit model.
        let resolved_model = match (self.spec.model.as_ref(), parent) {
            (Some(m), _) => m.clone(),
            (None, Some((parent_spec, _))) => parent_spec.model().clone(),
            (None, None) => {
                return Err(AgenticError::Other(
                    "root agent requires an explicit .model() / .model_name() (or must be spawned as a child)"
                        .into(),
                ));
            }
        };

        // CoW-clone the spec and fill in the resolved model.
        let mut spec = Arc::clone(&self.spec);
        Arc::make_mut(&mut spec).model = Some(resolved_model);

        // Build the runtime: inherit from parent or build fresh externals.
        let runtime = match parent {
            Some((_, parent_runtime)) => self.inherit_runtime(parent_runtime, &spec),
            None => self.build_runtime(&spec)?,
        };

        Ok((spec, runtime))
    }

    /// Build the root `LoopRuntime` from this agent's per-run fields plus
    /// reasonable defaults. Requires `self.provider` to be set.
    fn build_runtime(&self, spec: &AgentSpec) -> Result<LoopRuntime> {
        let provider = self
            .provider
            .clone()
            .ok_or_else(|| AgenticError::Other("Agent::run() requires a provider".into()))?;

        let working_directory = self
            .working_directory
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

        // Every root run gets a command queue so background sub-agents can post
        // notifications back to the parent. An externally supplied queue
        // (e.g. from `Agent::spawn`) wins so the handle can reach the loop.
        let command_queue = Some(
            self.command_queue
                .clone()
                .unwrap_or_else(|| Arc::new(CommandQueue::new())),
        );

        let session_store = self.session_dir.as_ref().map(|dir| {
            let store = SessionStore::new(dir, &generate_agent_name("session"));
            Arc::new(Mutex::new(store))
        });

        let metadata = Some(LoopRuntime::environment(&working_directory));

        Ok(LoopRuntime {
            provider,
            event_handler,
            cancel_signal,
            working_directory,
            command_queue,
            session_store,
            metadata,
            discovered_tools: Arc::new(Mutex::new(HashSet::new())),
            tools: build_tools(spec),
            template_variables: self.template_variables.clone(),
        })
    }

    /// Build a child `LoopRuntime`: inherit externals from the parent, let this
    /// agent's own per-run fields override any that it set explicitly.
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
            working_directory: self
                .working_directory
                .clone()
                .unwrap_or_else(|| parent.working_directory.clone()),
            command_queue: parent.command_queue.clone(),
            session_store: parent.session_store.clone(),
            metadata: parent.metadata.clone(),
            discovered_tools: parent.discovered_tools.clone(),
            tools: build_tools(spec),
            template_variables: self.template_variables.clone(),
        }
    }
}

/// Build the runtime's `Arc<ToolRegistry>`: a clone of `spec.tools` with
/// `SpawnAgentTool` auto-wired when `sub_agents` is non-empty and the user
/// hasn't registered a conflicting tool.
fn build_tools(spec: &AgentSpec) -> Arc<ToolRegistry> {
    let mut tools = spec.tools.clone();
    if !spec.sub_agents.is_empty() && tools.get("spawn_agent").is_none() {
        tools.register(SpawnAgentTool);
    }
    Arc::new(tools)
}

#[cfg(test)]
mod tests {
    use super::super::event::EventKind;
    use super::super::output::Status;
    use super::*;
    use crate::error::AgenticError;

    #[test]
    fn silent_sets_a_no_op_handler() {
        let agent = Agent::new().silent();
        let handler = agent
            .event_handler
            .as_ref()
            .expect(".silent() must install a handler")
            .clone();
        // Every variant passes through without panicking; no output is asserted —
        // the point is that a handler is present and benign.
        handler(Event::new(
            "t",
            EventKind::AgentFinished {
                turns: 1,
                status: Status::Completed,
            },
        ));
    }

    #[test]
    fn default_logger_is_used_when_no_handler_is_set() {
        // No `.event_handler(...)` call — the runtime built by `compile` must
        // carry the default logger, not a no-op. We can't compare function
        // pointers across clones, so we assert the default is present by
        // exercising the build path.
        let agent = Agent::new()
            .name("t")
            .model_name("mock")
            .identity_prompt("")
            .provider(std::sync::Arc::new(crate::testutil::MockProvider::text(
                "ok",
            )));
        assert!(agent.event_handler.is_none());
        // `compile` must succeed without a user-set handler — proves the
        // default path is wired up.
        let _ = agent.compile(None).expect("compile with default logger");
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
    async fn missing_provider_fails_run() {
        let agent = Agent::new()
            .name("test")
            .model_name("mock")
            .identity_prompt("x")
            .instruction_prompt("do");
        let err = agent.run().await.unwrap_err();
        match err {
            AgenticError::Other(msg) => assert!(msg.contains("provider"), "got: {msg}"),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}
