//! The crate's central user-facing type and its builder. Carries prompts, tools, and tuning knobs into the execution loop.

use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;
use crate::persistence::session::SessionStore;
use crate::provider::model::Model;
use crate::provider::Provider;
use crate::tools::{AgentTool, ToolLike, ToolRegistry};
use crate::util::generate_agent_name;

use crate::event::{default_logger, Event};
use crate::output::{Output, OutputSchema};

use super::prompts;
use super::r#loop::{run_loop, LoopRuntime, LoopState};
use super::spec::AgentSpec;
use super::work::Work;

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
///     .model("claude-sonnet-4-20250514")
///     .role("You are a helpful assistant.");
///
/// let first = agent.clone().task("Greet me.").await.unwrap();
/// assert_eq!(first.response_raw, "Hello!");
/// # });
/// ```
#[derive(Clone)]
pub struct Agent {
    pub(crate) spec: Arc<AgentSpec>,
    pub(crate) provider: Option<Arc<dyn Provider>>,
    pub(crate) task: String,
    pub(crate) templates: HashMap<String, Value>,
    pub(crate) working_dir: Option<PathBuf>,
    pub(crate) event_handler: Option<Arc<dyn Fn(Event) + Send + Sync>>,
    pub(crate) interrupt_signal: Option<Arc<AtomicBool>>,
    pub(crate) incoming_work: Option<Arc<Work>>,
    pub(crate) session_dir: Option<PathBuf>,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            spec: Arc::new(AgentSpec::default()),
            provider: None,
            task: String::new(),
            event_handler: None,
            incoming_work: None,
            interrupt_signal: None,
            working_dir: None,
            session_dir: None,
            templates: HashMap::new(),
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

    /// Replace `{key}` placeholders in `template` with this agent's templates.
    pub(crate) fn interpolate(&self, template: &str) -> String {
        let mut result = template.to_string();
        for (key, value) in &self.templates {
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

    /// Set the model. Pass a name (`&str` / `String`) for registry-backed
    /// auto-detection of the context window, or a [`Model`] to override
    /// capabilities.
    pub fn model(self, model: impl Into<Model>) -> Self {
        self.with_spec(|c| c.model = Some(model.into()))
    }

    /// The agent's persistent role — who it is and how it behaves.
    pub fn role(self, p: impl Into<String>) -> Self {
        self.with_spec(|c| c.role = p.into())
    }

    /// Load the role prompt from a file.
    pub fn role_file(self, path: impl Into<PathBuf>) -> Self {
        let s = load_prompt_file(path.into());
        self.with_spec(|c| c.role = s)
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

    /// Register a structured output contract (JSON Schema). Panics if invalid.
    pub fn contract(self, value: Value) -> Self {
        let contract =
            OutputSchema::new(value).unwrap_or_else(|e| panic!("invalid output contract: {e}"));
        self.with_spec(|c| c.contract = Some(contract))
    }

    /// Load a structured output contract (JSON Schema) from a file.
    pub fn contract_file(self, path: impl Into<PathBuf>) -> Self {
        self.contract(load_json_file(path.into()))
    }

    /// Maximum retries for structured output compliance. Default is 10.
    pub fn max_contract_retries(self, n: u32) -> Self {
        self.with_spec(|c| c.max_contract_retries = Some(n))
    }

    /// Maximum retries for transient API errors (429, 529, network failures).
    pub fn max_request_retries(self, n: u32) -> Self {
        self.with_spec(|c| c.max_request_retries = n)
    }

    /// Base delay for exponential backoff on request retries.
    pub fn request_retry_delay(self, delay: Duration) -> Self {
        self.with_spec(|c| c.request_retry_delay = delay)
    }

    /// Park the agent idle after a terminal output until a peer message arrives or `interrupt_signal` fires.
    ///
    /// [`Agent::retain`] sets this implicitly. Call it only on a sub-agent template
    /// that should idle in the background after the orchestrator spawns it.
    pub fn keep_alive(self) -> Self {
        self.with_spec(|c| c.keep_alive = true)
    }

    /// Override the default behavior prompt.
    pub fn behavior(self, content: impl Into<String>) -> Self {
        let content = content.into();
        self.with_spec(|c| c.behavior = content)
    }

    /// Load a behavior prompt override from a file.
    pub fn behavior_file(self, path: impl Into<PathBuf>) -> Self {
        let content = load_prompt_file(path.into());
        self.with_spec(|c| c.behavior = content)
    }

    /// Override the context prompt sent as the first user message.
    ///
    /// Passing a non-empty string replaces the default environment block verbatim;
    /// passing `""` opts out of the context message entirely. Compose on top of
    /// the default via [`Agent::default_context`].
    pub fn context(self, content: impl Into<String>) -> Self {
        self.with_spec(|c| c.context = Some(content.into()))
    }

    /// Load a context prompt override from a file.
    pub fn context_file(self, path: impl Into<PathBuf>) -> Self {
        let content = load_prompt_file(path.into());
        self.with_spec(|c| c.context = Some(content))
    }

    /// The default context prompt: environment metadata (working directory,
    /// platform, OS version, date) wrapped in an `<environment>` block.
    /// Uses the process cwd. Override with [`Agent::context`].
    pub fn default_context() -> String {
        let cwd = std::env::current_dir().unwrap_or_default();
        prompts::default_context(&cwd)
    }

    /// Hire one sub-agent, callable by name from this agent.
    pub fn hire(self, worker: Agent) -> Self {
        self.with_spec(|c| c.hires.push(worker))
    }

    /// Hire many sub-agents at once. Equivalent to chaining `.hire(w)` for each.
    pub fn hire_all<I>(self, workers: I) -> Self
    where
        I: IntoIterator<Item = Agent>,
    {
        self.with_spec(|c| c.hires.extend(workers))
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
        Ok(self.model(crate::provider::environment::model_from_env()?))
    }

    /// The task for this run — what to do right now.
    pub fn task(mut self, p: impl Into<String>) -> Self {
        self.task = p.into();
        self
    }

    /// Load the task from a file.
    pub fn task_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.task = load_prompt_file(path.into());
        self
    }

    /// Bind `{key}` to `value` for placeholder substitution in all prompts before the run.
    pub fn template(mut self, key: impl Into<String>, value: Value) -> Self {
        self.templates.insert(key.into(), value);
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
    pub fn interrupt_signal(mut self, s: Arc<AtomicBool>) -> Self {
        self.interrupt_signal = Some(s);
        self
    }

    /// Install an externally-owned work inbox so an `AgentWorking` can inject instructions.
    pub(crate) fn incoming_work(mut self, w: Arc<Work>) -> Self {
        self.incoming_work = Some(w);
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

    /// Drive the loop to completion and return the agent's output. Awaiting an
    /// `Agent` (via the [`IntoFuture`] impl) is the public entry point.
    ///
    /// Requires `.provider()` (or [`Agent::provider_from_env`]), `.model()`
    /// (or [`Agent::model_from_env`]), and `.task()`.
    pub(crate) async fn work(&self) -> Result<Output> {
        let (spec, runtime) = self.compile(None);
        let runtime = Arc::new(runtime);
        let task = self.interpolate(&self.task);
        let context = spec.context(&runtime.default_context);
        let state = LoopState::initial(context, task);
        run_loop(runtime, spec, state).await
    }

    /// Work as a child under a parent's run-tree. `parent_spec` supplies the model fallback.
    pub(crate) async fn work_child(
        &self,
        parent_spec: &AgentSpec,
        parent_runtime: &LoopRuntime,
    ) -> Result<Output> {
        let (spec, runtime) = self.compile(Some((parent_spec, parent_runtime)));
        let runtime = Arc::new(runtime);
        let task = self.interpolate(&self.task);
        let context = spec.context(&runtime.default_context);
        let state = LoopState::initial(context, task);
        run_loop(runtime, spec, state).await
    }

    /// Apply LLM-supplied JSON overrides. Missing keys are left alone, unknown keys ignored.
    pub(crate) fn apply_overrides(mut self, overrides: &Value) -> Self {
        if let Some(m) = overrides.get("model").and_then(Value::as_str) {
            self = self.model(m);
        }
        if let Some(i) = overrides.get("identity").and_then(Value::as_str) {
            self = self.role(i);
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
        if let Some(sr) = overrides
            .get("max_contract_retries")
            .and_then(Value::as_u64)
        {
            self = self.max_contract_retries(sr as u32);
        }
        if let Some(rr) = overrides.get("max_request_retries").and_then(Value::as_u64) {
            self = self.max_request_retries(rr as u32);
        }
        if let Some(ms) = overrides.get("request_retry_delay").and_then(Value::as_u64) {
            self = self.request_retry_delay(Duration::from_millis(ms));
        }
        if let Some(contract) = overrides.get("contract").cloned() {
            self = self.contract(contract);
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
                "Agent::task(...).await requires .model(...) on root agents (sub-agents inherit)"
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
            panic!("Agent::task(...).await requires .provider() (or .provider_from_env()) on root agents")
        });

        let working_dir = self
            .working_dir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let event_handler: Arc<dyn Fn(Event) + Send + Sync> =
            self.event_handler.clone().unwrap_or_else(default_logger);

        let interrupt_signal = self
            .interrupt_signal
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

        // Every root run carries a work inbox so background sub-agents can post
        // notifications back. An externally supplied inbox wins so a handle can reach the loop.
        let incoming_work = Some(
            self.incoming_work
                .clone()
                .unwrap_or_else(|| Arc::new(Work::new())),
        );

        let session_store = self.session_dir.as_ref().map(|dir| {
            let store = SessionStore::new(dir, &generate_agent_name("session"));
            Arc::new(Mutex::new(store))
        });

        let default_context = prompts::default_context(&working_dir);

        LoopRuntime {
            provider,
            event_handler,
            interrupt_signal,
            working_dir,
            incoming_work,
            session_store,
            default_context,
            tool_registry: build_tools(spec),
            templates: self.templates.clone(),
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
            interrupt_signal: self
                .interrupt_signal
                .clone()
                .unwrap_or_else(|| parent.interrupt_signal.clone()),
            working_dir: self
                .working_dir
                .clone()
                .unwrap_or_else(|| parent.working_dir.clone()),
            incoming_work: parent.incoming_work.clone(),
            session_store: parent.session_store.clone(),
            default_context: parent.default_context.clone(),
            tool_registry: build_tools(spec),
            templates: self.templates.clone(),
        }
    }
}

/// `Agent` is awaitable: `agent.await` drives the loop and yields the
/// [`Output`]. This is the public terminal — the chain ends on
/// `.task(...).await`.
impl IntoFuture for Agent {
    type Output = Result<Output>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<Output>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.work().await })
    }
}

/// Clone `spec.tools`, auto-wiring `AgentTool` when sub-agents exist and the slot is free.
fn build_tools(spec: &AgentSpec) -> Arc<ToolRegistry> {
    let mut tools = spec.tool_registry.clone();
    if !spec.hires.is_empty() && tools.get("agent").is_none() {
        tools.register(AgentTool);
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
            .model("mock")
            .role("")
            .provider(std::sync::Arc::new(crate::testutil::MockProvider::text(
                "ok",
            )));
        assert!(agent.event_handler.is_none());
        let _ = agent.compile(None);
    }

    #[test]
    fn role_file_loads_content() {
        let dir = std::env::temp_dir().join("agentwerk_test_werk_role");
        let path = dir.join("role.txt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "You are a test agent").unwrap();

        let agent = Agent::new().role_file(&path);
        assert_eq!(agent.spec.role, "You are a test agent");

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "failed to read prompt file")]
    fn missing_prompt_file_panics() {
        let _ = Agent::new().role_file("/nonexistent/xxx.txt");
    }

    #[test]
    fn hire_all_extends_hires_in_order() {
        let a = Agent::new().name("a").model("mock");
        let b = Agent::new().name("b").model("mock");
        let agent = Agent::new().hire_all([a, b]);
        let names: Vec<String> = agent
            .spec
            .hires
            .iter()
            .map(|h| h.spec.name.clone())
            .collect();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn contract_file_loads_valid_schema() {
        let dir = std::env::temp_dir().join("agentwerk_test_werk_contract");
        let path = dir.join("contract.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            &path,
            r#"{"type":"object","properties":{"answer":{"type":"string"}}}"#,
        )
        .unwrap();

        let agent = Agent::new().contract_file(&path);
        assert!(agent.spec.contract.is_some());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "failed to read prompt file")]
    fn contract_file_missing_file_panics() {
        let _ = Agent::new().contract_file("/nonexistent/contract.json");
    }

    #[test]
    #[should_panic(expected = "invalid output contract")]
    fn invalid_contract_panics() {
        let _ = Agent::new()
            .name("test")
            .role("")
            .contract(serde_json::json!({"type": "string"}));
    }

    #[tokio::test]
    async fn apply_overrides_applies_json_fields() {
        let base = Agent::new().name("x").model("original").max_turns(3);
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
    async fn missing_provider_panics_on_await() {
        let agent = Agent::new().name("test").model("mock").role("x").task("do");
        let _ = agent.await;
    }
}
