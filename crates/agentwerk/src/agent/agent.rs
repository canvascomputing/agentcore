//! The crate's central user-facing type and its builder. Carries prompts, tools, and tuning knobs into the execution loop.

use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::task::JoinHandle;

use crate::error::Result;
use crate::persistence::session::SessionStore;
use crate::provider::model::Model;
use crate::provider::Provider;
use crate::tools::{AgentTool, ToolLike, ToolRegistry};
use crate::util::generate_agent_name;

use crate::event::{default_logger, Event};
use crate::output::{Output, OutputSchema};

use super::error::AgentError;
use super::prompts;
use super::r#loop::{run_loop, LoopRuntime, LoopState};
use super::spec::AgentSpec;
use super::work::{Task, TaskSource, Work, WorkPriority};

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
/// let first = agent.clone().work("Greet me.").await.unwrap();
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
    pub fn max_steps(self, n: u32) -> Self {
        self.with_spec(|c| c.max_steps = Some(n))
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
    /// [`Agent::keep_working`] sets this implicitly. Call it only on a sub-agent template
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

    /// Register one sub-agent, callable by name from this agent.
    pub fn staff(self, sub: Agent) -> Self {
        self.with_spec(|c| c.staff.push(sub))
    }

    /// Register many sub-agents at once. Equivalent to chaining `.staff(s)` for each.
    pub fn staff_more<I>(self, subs: I) -> Self
    where
        I: IntoIterator<Item = Agent>,
    {
        self.with_spec(|c| c.staff.extend(subs))
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
    pub fn work(mut self, p: impl Into<String>) -> Self {
        self.task = p.into();
        self
    }

    /// Load the task from a file.
    pub fn work_file(mut self, path: impl Into<PathBuf>) -> Self {
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

    /// Install an externally-owned work inbox so a [`Working`](crate::Working) handle can inject instructions.
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
    /// (or [`Agent::model_from_env`]), and `.work(...)`.
    pub(crate) async fn execute(&self) -> Result<Output> {
        let (spec, runtime) = self.compile(None);
        let runtime = Arc::new(runtime);
        let task = self.interpolate(&self.task);
        let context = spec.context(&runtime.default_context);
        let state = LoopState::initial(context, task);
        run_loop(runtime, spec, state).await
    }

    /// Execute as a child under a parent's run-tree. `parent_spec` supplies the model fallback.
    pub(crate) async fn execute_child(
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
        if let Some(mt) = overrides.get("max_steps").and_then(Value::as_u64) {
            self = self.max_steps(mt as u32);
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
                "Agent::work(...).await requires .model(...) on root agents (sub-agents inherit)"
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
            panic!("Agent::work(...).await requires .provider() (or .provider_from_env()) on root agents")
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
/// `.work(...).await`.
impl IntoFuture for Agent {
    type Output = Result<Output>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<Output>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.execute().await })
    }
}

/// Clone `spec.tools`, auto-wiring `AgentTool` when sub-agents exist and the slot is free.
fn build_tools(spec: &AgentSpec) -> Arc<ToolRegistry> {
    let mut tools = spec.tool_registry.clone();
    if !spec.staff.is_empty() && tools.get("agent").is_none() {
        tools.register(AgentTool);
    }
    Arc::new(tools)
}

/// RAII token: flips the shared cancel flag when its last clone drops, so
/// abandoning the handle without an explicit `.interrupt()` still unblocks
/// the loop.
struct CancelGuard {
    cancel: Arc<AtomicBool>,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Cheap, clonable handle to an agent whose loop runs on a background tokio
/// task. Obtained from [`Agent::keep_working`].
///
/// While any clone of the handle is alive, the loop idles after producing
/// output; dropping the last clone (or calling [`interrupt`](Self::interrupt))
/// signals the loop to exit.
#[derive(Clone)]
pub struct AgentWorking {
    work: Arc<Work>,
    cancel: Arc<AtomicBool>,
    #[allow(dead_code)]
    guard: Arc<CancelGuard>,
}

impl AgentWorking {
    /// Hand the running agent another task. Picked up at the next step
    /// boundary, or immediately if the agent is parked idle.
    pub fn work(&self, task: impl Into<String>) {
        self.work.add(Task {
            content: task.into(),
            priority: WorkPriority::Next,
            source: TaskSource::UserInput,
            agent_name: None,
        });
    }

    /// Queue several tasks at once. Order is preserved; the loop picks them
    /// up one by one at step boundaries.
    pub fn work_more<I>(&self, tasks: I)
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        for task in tasks {
            self.work(task);
        }
    }

    /// Signal the agent to stop. The loop observes this at the next step
    /// boundary or idle-wait poll and exits.
    pub fn interrupt(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if an interrupt signal has been raised (explicitly via
    /// [`interrupt`](Self::interrupt) or implicitly via the last handle being
    /// dropped).
    pub fn is_interrupted(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// Resolves to the agent's final [`Output`](crate::output::Output) once the
/// background loop exits.
///
/// Only [`AgentWorking`] clones keep the agent alive; dropping this
/// (without awaiting) just abandons the result. Whether the loop keeps
/// running is decided by whether any handles remain.
///
/// Implements [`IntoFuture`]: `.await` consumes the value by move, so a
/// double-await is a compile error rather than a runtime failure.
pub struct OutputFuture {
    join: JoinHandle<Result<Output>>,
}

impl IntoFuture for OutputFuture {
    type Output = Result<Output>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<Output>> + Send + 'static>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            match self.join.await {
                Ok(result) => result,
                Err(e) => Err(AgentError::AgentCrashed {
                    message: e.to_string(),
                }
                .into()),
            }
        })
    }
}

impl Agent {
    /// Start the agent on a background tokio task and return a pair:
    ///
    /// - [`AgentWorking`]: cheap, clonable handle for injecting new
    ///   instructions, cancelling, or inspecting state.
    /// - [`OutputFuture`]: resolves to the final
    ///   [`Output`](crate::output::Output) once the loop exits.
    ///
    /// The loop idles after each terminal output as long as any handle is
    /// alive. Dropping the last handle calls [`AgentWorking::interrupt`] for you
    /// (RAII safety); an explicit `.interrupt()` does the same thing. For a
    /// pure one-shot run without a handle, await the agent directly: a
    /// `agent.work(task).await` runs the loop synchronously.
    ///
    /// Requires a running tokio runtime (`tokio::spawn` is invoked
    /// synchronously). Requires `.provider()` and either an initial
    /// `.work(...)` set in the builder or follow-up
    /// [`AgentWorking::work`] calls on the returned handle.
    pub fn keep_working(self) -> (AgentWorking, OutputFuture) {
        let work = Arc::new(Work::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let guard = Arc::new(CancelGuard {
            cancel: cancel.clone(),
        });

        let prepared = self
            .interrupt_signal(cancel.clone())
            .incoming_work(work.clone())
            .keep_alive();

        let join = tokio::spawn(async move { prepared.execute().await });

        (
            AgentWorking {
                work,
                cancel,
                guard,
            },
            OutputFuture { join },
        )
    }
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
                steps: 1,
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
    fn staff_more_extends_staff_in_order() {
        let a = Agent::new().name("a").model("mock");
        let b = Agent::new().name("b").model("mock");
        let agent = Agent::new().staff_more([a, b]);
        let names: Vec<String> = agent
            .spec
            .staff
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
        let base = Agent::new().name("x").model("original").max_steps(3);
        let applied = base.apply_overrides(&serde_json::json!({
            "model": "overridden",
            "max_steps": 7,
            "max_request_tokens": 256,
            "max_input_tokens": 4000,
            "max_output_tokens": 5000
        }));
        assert_eq!(applied.spec.max_steps, Some(7));
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
        let agent = Agent::new().name("test").model("mock").role("x").work("do");
        let _ = agent.await;
    }
}

#[cfg(test)]
mod keep_working_tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    use crate::event::EventKind;
    use crate::output::Outcome;
    use crate::provider::types::{ContentBlock, Message, ModelResponse};
    use crate::provider::ModelRequest;
    use crate::testutil::{text_response, MockProvider};

    #[tokio::test]
    async fn keep_working_returns_handle_and_future() {
        let (handle, output) = one_shot_agent("hello");
        let clone = handle.clone();
        // AgentWorking is Clone; OutputFuture is a Future. Interrupt so the
        // keep-alive loop terminates.
        clone.interrupt();
        let _: Result<Output> = output.await;
    }

    #[tokio::test]
    async fn keep_working_starts_loop_immediately() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);
        // AgentStarted is the first event emitted by run_loop: observable
        // before we await the future.
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentStarted { .. }))
            .await;
        handle.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn work_adds_user_input_work() {
        let (handle, output) = one_shot_agent("done");
        handle.work("hi");
        let task = handle
            .work
            .take_if(Some("anyone"), |_| true)
            .expect("pending task");
        assert_eq!(task.content, "hi");
        assert!(matches!(task.priority, WorkPriority::Next));
        assert!(matches!(task.source, TaskSource::UserInput));
        assert!(task.agent_name.is_none());
        handle.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn work_more_queues_tasks_in_order() {
        let (handle, output) = one_shot_agent("done");
        handle.work_more(["one", "two", "three"]);
        let mut seen = Vec::new();
        while let Some(task) = handle.work.take_if(Some("anyone"), |_| true) {
            seen.push(task.content);
        }
        assert_eq!(seen, vec!["one", "two", "three"]);
        handle.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn work_reaches_next_provider_request() {
        let events = EventLog::new();
        let (provider, handle, output) = keep_alive_agent_with_provider(
            vec![text_response("first"), text_response("second")],
            &events,
        );

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        handle.work("follow-up");
        wait_until(|| provider.requests() >= 2).await;

        let second = provider.last_request().expect("second request");
        let last_user = last_user_text(&second).expect("user message in second request");
        assert!(
            last_user.contains("follow-up"),
            "injected instruction must appear in step 2's user message; got {last_user:?}",
        );

        handle.interrupt();
        let out = output.await.expect("output");
        assert!(matches!(
            out.outcome,
            Outcome::Completed | Outcome::Cancelled
        ));
    }

    #[tokio::test]
    async fn clone_shares_work() {
        let (handle, output) = one_shot_agent("done");
        let other = handle.clone();
        other.work("relay");
        let task = handle
            .work
            .take_if(Some("anyone"), |_| true)
            .expect("pending task");
        assert_eq!(task.content, "relay");
        handle.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn clone_shares_interrupt() {
        let (handle, output) = one_shot_agent("done");
        let other = handle.clone();
        assert!(!handle.is_interrupted());
        other.interrupt();
        assert!(handle.is_interrupted() && other.is_interrupted());
        let _ = output.await;
    }

    #[tokio::test]
    async fn interrupt_during_idle_preserves_completed_status() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        handle.interrupt();
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentResumed))
            .await;
        let out = output.await.expect("output");
        assert_eq!(out.outcome, Outcome::Completed);
    }

    #[tokio::test]
    async fn interrupt_from_spawned_task() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        let interrupter = handle.clone();
        tokio::spawn(async move {
            interrupter.interrupt();
        });
        let _ = output.await.expect("output");
    }

    #[tokio::test]
    async fn dropping_last_handle_triggers_interrupt() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        drop(handle);
        let out = output.await.expect("output");
        assert_eq!(out.outcome, Outcome::Completed);
    }

    #[tokio::test]
    async fn dropping_one_of_two_handles_does_not_interrupt() {
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        let survivor = handle.clone();
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        drop(handle);
        // Interrupt is NOT set while another handle is alive.
        assert!(!survivor.is_interrupted());
        // cleanup
        survivor.interrupt();
        let _ = output.await;
    }

    #[tokio::test]
    async fn dropping_future_alone_does_not_interrupt() {
        // The future holds no CancelGuard, so dropping it doesn't interrupt. The
        // loop keeps running: cleanup belongs to the handle.
        let events = EventLog::new();
        let (handle, output) = keep_alive_agent(vec![text_response("first")], &events);

        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        drop(output);
        assert!(!handle.is_interrupted());
        handle.interrupt();
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentFinished { .. }))
            .await;
    }

    #[tokio::test]
    async fn keep_alive_idle_and_resumed_events_still_fire() {
        let events = EventLog::new();
        let (provider, handle, output) = keep_alive_agent_with_provider(
            vec![text_response("first"), text_response("second")],
            &events,
        );
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentPaused))
            .await;
        handle.work("wake up");
        wait_until(|| provider.requests() >= 2).await;
        events
            .wait_for(|e| matches!(e.kind, EventKind::AgentResumed))
            .await;
        handle.interrupt();
        let _ = output.await;
    }

    fn one_shot_agent(text: &str) -> (AgentWorking, OutputFuture) {
        Agent::new()
            .name("demo")
            .model("mock")
            .provider(Arc::new(MockProvider::text(text)))
            .role("")
            .work("x")
            .keep_working()
    }

    fn keep_alive_agent(
        responses: Vec<ModelResponse>,
        events: &EventLog,
    ) -> (AgentWorking, OutputFuture) {
        let (_, h, o) = keep_alive_agent_with_provider(responses, events);
        (h, o)
    }

    fn keep_alive_agent_with_provider(
        responses: Vec<ModelResponse>,
        events: &EventLog,
    ) -> (Arc<MockProvider>, AgentWorking, OutputFuture) {
        let provider = Arc::new(MockProvider::new(responses));
        let (h, o) = Agent::new()
            .name("root")
            .model("mock")
            .provider(provider.clone())
            .role("")
            .work("initial")
            .event_handler(events.handler())
            .keep_working();
        (provider, h, o)
    }

    struct EventLog {
        events: Arc<StdMutex<Vec<Event>>>,
    }

    impl EventLog {
        fn new() -> Self {
            Self {
                events: Arc::new(StdMutex::new(Vec::new())),
            }
        }

        fn handler(&self) -> Arc<dyn Fn(Event) + Send + Sync> {
            let events = self.events.clone();
            Arc::new(move |e| events.lock().unwrap().push(e))
        }

        async fn wait_for<F: Fn(&Event) -> bool>(&self, pred: F) {
            for _ in 0..200 {
                if self.events.lock().unwrap().iter().any(&pred) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            let seen: Vec<_> = self
                .events
                .lock()
                .unwrap()
                .iter()
                .map(|e| format!("{}:{:?}", e.agent_name, e.kind))
                .collect();
            panic!("timed out after 5s waiting for event; saw: {seen:#?}");
        }
    }

    async fn wait_until<F: FnMut() -> bool>(mut pred: F) {
        for _ in 0..200 {
            if pred() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("timed out after 5s waiting for condition");
    }

    fn last_user_text(req: &ModelRequest) -> Option<String> {
        req.messages.iter().rev().find_map(|m| match m {
            Message::User { content } => Some(
                content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
    }
}
