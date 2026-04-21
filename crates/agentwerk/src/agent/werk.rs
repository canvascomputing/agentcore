//! Agent definition + builder surface.
//!
//! Holds:
//! - `Agent` (public) — the single user-facing type. Internally split into a
//!   shared `Arc<AgentConfig>` (static template: tools, prompts, sub-agents,
//!   tuning knobs) and an owned `AgentRuntime` (per-run: provider, instruction
//!   prompt, working directory, event handler). Builder methods route through
//!   `with_config` (copy-on-write via `Arc::make_mut`) or `with_runtime`
//!   (direct mutation), so `template.clone().provider(p).instruction_prompt(t)`
//!   never copies the heavy template.
//! - `Agent::compile` — single entry point that produces the pair
//!   `(LoopRuntime, LoopSpec)` the execution loop consumes. Handles both
//!   root runs (`parent = None`) and sub-agent spawns (`parent = Some((rt, spec))`).
//!
//! The actual loop machinery lives in `super::r#loop`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::persistence::session::SessionStore;
use crate::provider::model::{Model, ModelSpec};
use crate::tools::{SpawnAgentTool, Toolable, ToolRegistry};
use crate::provider::ToolChoice;
use crate::util::generate_agent_name;

use super::event::AgentEvent;
use super::output::{AgentOutput, OutputSchema};
use super::prompts::{self as prompts, DEFAULT_BEHAVIOR_PROMPT};
use super::queue::CommandQueue;
use super::r#loop::{run_loop, LoopRuntime, LoopSpec};
use super::running::{AgentHandle, AgentOutputFuture, HandleState, LifeToken};
use crate::provider::Provider;

// ---------------------------------------------------------------------------
// Agent — the single public user-facing type
// ---------------------------------------------------------------------------

/// An LLM-powered agent. Cheap to clone (internally `Arc`-wrapped config + a
/// small owned runtime struct).
///
/// Build with `Agent::new()` and chain builder methods; call `.run()` to
/// execute. The same `Agent` can be `run()` multiple times. Cloning an agent
/// and mutating per-run fields (e.g. `.instruction_prompt`) does not clone
/// the static template (tools, sub-agents, behavior prompts).
#[derive(Clone, Default)]
pub struct Agent {
    pub(crate) config: Arc<AgentConfig>,
    pub(crate) runtime: AgentRuntime,
}

/// Immutable agent definition. Shared across clones via `Arc`; changes trigger COW.
pub(crate) struct AgentConfig {
    pub name: Option<String>,
    pub model: ModelSpec,
    pub identity_prompt: String,
    pub behavior_prompt: String,
    pub context_prompts: Vec<String>,
    pub tools: ToolRegistry,
    pub sub_agents: Vec<Agent>,
    pub output_schema: Option<OutputSchema>,
    pub max_output_tokens: Option<u32>,
    pub max_input_tokens: Option<u64>,
    pub max_turns: Option<u32>,
    pub max_schema_retries: Option<u32>,
    pub max_request_retries: u32,
    pub request_retry_backoff_ms: u64,
    pub keep_alive: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: None,
            model: ModelSpec::Inherit,
            identity_prompt: String::new(),
            behavior_prompt: DEFAULT_BEHAVIOR_PROMPT.to_string(),
            context_prompts: Vec::new(),
            tools: ToolRegistry::new(),
            sub_agents: Vec::new(),
            output_schema: None,
            max_output_tokens: None,
            max_input_tokens: None,
            max_turns: None,
            max_schema_retries: Some(10),
            max_request_retries: Agent::DEFAULT_MAX_REQUEST_RETRIES,
            request_retry_backoff_ms: Agent::DEFAULT_BACKOFF_MS,
            keep_alive: false,
        }
    }
}

impl Clone for AgentConfig {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            model: self.model.clone(),
            identity_prompt: self.identity_prompt.clone(),
            behavior_prompt: self.behavior_prompt.clone(),
            context_prompts: self.context_prompts.clone(),
            tools: self.tools.clone(),
            sub_agents: self.sub_agents.clone(),
            output_schema: self.output_schema.clone(),
            max_output_tokens: self.max_output_tokens,
            max_input_tokens: self.max_input_tokens,
            max_turns: self.max_turns,
            max_schema_retries: self.max_schema_retries,
            max_request_retries: self.max_request_retries,
            request_retry_backoff_ms: self.request_retry_backoff_ms,
            keep_alive: self.keep_alive,
        }
    }
}

/// Per-run configuration. Owned per `Agent` clone — no COW, cheap direct mutation.
#[derive(Clone, Default)]
pub(crate) struct AgentRuntime {
    pub provider: Option<Arc<dyn Provider>>,
    pub instruction_prompt: String,
    pub template_variables: HashMap<String, Value>,
    pub working_directory: Option<PathBuf>,
    pub event_handler: Option<Arc<dyn Fn(AgentEvent) + Send + Sync>>,
    pub cancel_signal: Option<Arc<AtomicBool>>,
    pub command_queue: Option<Arc<CommandQueue>>,
    pub session_dir: Option<PathBuf>,
}

impl AgentRuntime {
    /// Replace `{key}` placeholders in `template` with the runtime's
    /// template variables.
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
    pub const DEFAULT_MAX_REQUEST_RETRIES: u32 = 3;

    /// Default base delay (ms) for the exponential-backoff retry policy.
    pub const DEFAULT_BACKOFF_MS: u64 = 10_000;

    pub fn new() -> Self {
        Self::default()
    }

    // --- Internal mutators ---

    fn with_config<F: FnOnce(&mut AgentConfig)>(mut self, f: F) -> Self {
        f(Arc::make_mut(&mut self.config));
        self
    }

    fn with_runtime<F: FnOnce(&mut AgentRuntime)>(mut self, f: F) -> Self {
        f(&mut self.runtime);
        self
    }

    // --- Definition (static) builders — route through with_config ---

    /// Set the agent's name. If unset, a generated name like `agent-a3f1` is used.
    pub fn name(self, n: impl Into<String>) -> Self {
        self.with_config(|c| c.name = Some(n.into()))
    }

    /// Set the model ID. If not called, the agent inherits the parent's model.
    /// The model's context window size is auto-detected from the built-in
    /// registry (see `Model::from_id`); use `.model_with_context_window_size`
    /// to override.
    pub fn model(self, m: impl Into<String>) -> Self {
        self.with_config(|c| c.model = ModelSpec::Exact(Model::from_id(m)))
    }

    /// Set the model ID together with an explicit context window size.
    /// Use this for local proxies, private deployments, or any id the
    /// built-in registry doesn't cover.
    pub fn model_with_context_window_size(
        self,
        id: impl Into<String>,
        context_window_size: u64,
    ) -> Self {
        let model = Model::from_id(id).with_context_window_size(Some(context_window_size));
        self.with_config(|c| c.model = ModelSpec::Exact(model))
    }

    /// The agent's persistent identity — who it is and how it behaves.
    pub fn identity_prompt(self, p: impl Into<String>) -> Self {
        self.with_config(|c| c.identity_prompt = p.into())
    }

    /// Load the identity prompt from a file.
    pub fn identity_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let s = load_prompt_file(path.into());
        self.with_config(|c| c.identity_prompt = s)
    }

    /// Maximum output tokens per LLM request. Omit to use the provider default.
    pub fn max_output_tokens(self, n: u32) -> Self {
        self.with_config(|c| c.max_output_tokens = Some(n))
    }

    /// Maximum agentic loop iterations. Omit for no limit.
    pub fn max_turns(self, n: u32) -> Self {
        self.with_config(|c| c.max_turns = Some(n))
    }

    /// Maximum cumulative input tokens before the agent stops.
    pub fn max_input_tokens(self, n: u64) -> Self {
        self.with_config(|c| c.max_input_tokens = Some(n))
    }

    /// Register a tool.
    pub fn tool(self, tool: impl Toolable + 'static) -> Self {
        self.with_config(|c| c.tools.register(tool))
    }

    /// Register a structured output schema. Panics if the schema is invalid.
    pub fn output_schema(self, schema: Value) -> Self {
        let schema =
            OutputSchema::new(schema).unwrap_or_else(|e| panic!("invalid output schema: {e}"));
        self.with_config(|c| c.output_schema = Some(schema))
    }

    /// Load a structured output schema from a JSON file.
    pub fn output_schema_file(self, path: impl Into<PathBuf>) -> Self {
        self.output_schema(load_json_file(path.into()))
    }

    /// Maximum retries for structured output compliance. Default is 10.
    pub fn max_schema_retries(self, n: u32) -> Self {
        self.with_config(|c| c.max_schema_retries = Some(n))
    }

    /// Maximum retries for transient API errors (429, 529, network failures).
    pub fn max_request_retries(self, n: u32) -> Self {
        self.with_config(|c| c.max_request_retries = n)
    }

    /// Base delay in ms for exponential backoff on request retries.
    pub fn request_retry_backoff_ms(self, ms: u64) -> Self {
        self.with_config(|c| c.request_retry_backoff_ms = ms)
    }

    /// Keep the agent alive after a terminal output, parking it idle until a
    /// peer message arrives or `cancel_signal` fires.
    ///
    /// Most users won't call this directly: [`Agent::spawn`] sets it
    /// implicitly. You only need it when declaring a sub-agent *template*
    /// (via `.sub_agents([...])`) that should idle while waiting for peer
    /// messages after the orchestrator spawns it in the background.
    pub fn keep_alive(self) -> Self {
        self.with_config(|c| c.keep_alive = true)
    }

    /// Override the default behavior prompt.
    pub fn behavior_prompt(self, content: impl Into<String>) -> Self {
        let content = content.into();
        self.with_config(|c| c.behavior_prompt = content)
    }

    /// Load a behavior prompt override from a file.
    pub fn behavior_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let content = load_prompt_file(path.into());
        self.with_config(|c| c.behavior_prompt = content)
    }

    /// Append additional context alongside the instruction prompt.
    pub fn context_prompt(self, content: impl Into<String>) -> Self {
        let content = content.into();
        self.with_config(|c| c.context_prompts.push(content))
    }

    /// Append additional context from a file.
    pub fn context_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let content = load_prompt_file(path.into());
        self.with_config(|c| c.context_prompts.push(content))
    }

    /// Register one or more agents as sub-agents. The LLM can call them by
    /// name once this agent runs.
    pub fn sub_agents(self, agents: impl IntoIterator<Item = Agent>) -> Self {
        let agents: Vec<_> = agents.into_iter().collect();
        self.with_config(|c| c.sub_agents.extend(agents))
    }

    // --- Per-run (runtime) builders — route through with_runtime ---

    pub fn provider(self, p: Arc<dyn Provider>) -> Self {
        self.with_runtime(|r| r.provider = Some(p))
    }

    /// Resolve the provider + default model from environment variables and
    /// apply both in one call. See [`crate::provider::from_env`] for the
    /// detection order.
    pub fn provider_from_env(self) -> Result<Self> {
        let (provider, model) = crate::provider::from_env()?;
        Ok(self.provider(provider).model(model))
    }

    /// The task for this run — what to do right now.
    pub fn instruction_prompt(self, p: impl Into<String>) -> Self {
        self.with_runtime(|r| r.instruction_prompt = p.into())
    }

    /// Load the instruction prompt from a file.
    pub fn instruction_prompt_file(self, path: impl Into<PathBuf>) -> Self {
        let s = load_prompt_file(path.into());
        self.with_runtime(|r| r.instruction_prompt = s)
    }

    pub fn template_variable(self, key: impl Into<String>, value: Value) -> Self {
        let key = key.into();
        self.with_runtime(|r| {
            r.template_variables.insert(key, value);
        })
    }

    pub fn working_directory(self, d: impl Into<PathBuf>) -> Self {
        let d = d.into();
        self.with_runtime(|r| r.working_directory = Some(d))
    }

    pub fn event_handler(self, h: Arc<dyn Fn(AgentEvent) + Send + Sync>) -> Self {
        self.with_runtime(|r| r.event_handler = Some(h))
    }

    /// Drop every event. Opts out of the default stderr logger installed when no
    /// handler is set.
    pub fn silent(self) -> Self {
        self.with_runtime(|r| r.event_handler = Some(Arc::new(|_| {})))
    }

    pub fn cancel_signal(self, s: Arc<AtomicBool>) -> Self {
        self.with_runtime(|r| r.cancel_signal = Some(s))
    }

    /// Install an externally-owned command queue. Only used by `Agent::spawn`
    /// so the returned `AgentHandle` can inject instructions into the loop.
    pub(crate) fn command_queue(self, q: Arc<CommandQueue>) -> Self {
        self.with_runtime(|r| r.command_queue = Some(q))
    }

    /// Enable session transcript persistence to the given directory.
    pub fn session_dir(self, d: impl Into<PathBuf>) -> Self {
        let d = d.into();
        self.with_runtime(|r| r.session_dir = Some(d))
    }

    // --- Accessors ---

    /// Returns the agent's configured name, if any.
    pub fn name_ref(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    // --- Terminal ---

    /// Execute this agent to completion. Requires `.provider()` and `.instruction_prompt()`.
    pub async fn run(&self) -> Result<AgentOutput> {
        let (runtime, spec) = self.compile(None)?;
        run_loop(Arc::new(runtime), Arc::new(spec), None).await
    }

    /// Start the agent on a background tokio task and return a pair:
    ///
    /// - [`AgentHandle`] — cheap, clonable handle for injecting new
    ///   instructions, cancelling, or inspecting state.
    /// - [`AgentOutputFuture`] — resolves to the final
    ///   [`AgentOutput`](crate::agent::AgentOutput) once the loop exits.
    ///
    /// The loop idles after each terminal output as long as any handle is
    /// alive. Dropping the last handle calls [`AgentHandle::cancel`] for you
    /// (RAII safety); an explicit `.cancel()` does the same thing. For a
    /// pure one-shot run without a handle, use [`Agent::run`] instead — a
    /// `let (_, out) = agent.spawn(); out.await?` pattern will cancel
    /// before the first turn completes.
    ///
    /// Requires a running tokio runtime (`tokio::spawn` is invoked
    /// synchronously). Requires `.provider()` and `.instruction_prompt()`.
    pub fn spawn(self) -> (AgentHandle, AgentOutputFuture) {
        let queue = Arc::new(CommandQueue::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let life = LifeToken::new(cancel.clone());

        let prepared = self
            .cancel_signal(cancel.clone())
            .command_queue(queue.clone())
            .keep_alive();

        let stopped_for_task = stopped.clone();
        let join = tokio::spawn(async move {
            let result = prepared.run().await;
            stopped_for_task.store(true, Ordering::Relaxed);
            result
        });

        let state = Arc::new(HandleState { queue, cancel, stopped });
        let handle = AgentHandle::new(state, life);
        let output = AgentOutputFuture::new(join);
        (handle, output)
    }

    /// Crate-internal: run this agent as a child under a parent's run-tree.
    /// The `parent_spec` supplies the model fallback for `ModelSpec::Inherit`
    /// (id *and* context window size both inherit). `description` becomes the
    /// `AgentStart` event's human-readable label.
    pub(crate) async fn run_child(
        &self,
        parent_runtime: &LoopRuntime,
        parent_spec: &LoopSpec,
        description: Option<String>,
    ) -> Result<AgentOutput> {
        let (runtime, spec) = self.compile(Some((parent_runtime, parent_spec)))?;
        run_loop(Arc::new(runtime), Arc::new(spec), description).await
    }

    /// Apply LLM-supplied JSON overrides for any Agent field a tool can
    /// legitimately set. Missing keys are left alone; unknown keys are
    /// silently ignored. Single source of truth for tool-driven config updates.
    pub(crate) fn apply_overrides(mut self, overrides: &Value) -> Self {
        if let Some(m) = overrides.get("model").and_then(Value::as_str) {
            self = self.model(m);
        }
        if let Some(i) = overrides.get("identity").and_then(Value::as_str) {
            self = self.identity_prompt(i);
        }
        if let Some(t) = overrides.get("max_output_tokens").and_then(Value::as_u64) {
            self = self.max_output_tokens(t as u32);
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
        if let Some(bo) = overrides.get("request_retry_backoff_ms").and_then(Value::as_u64) {
            self = self.request_retry_backoff_ms(bo);
        }
        if let Some(schema) = overrides.get("output_schema").cloned() {
            self = self.output_schema(schema);
        }
        self
    }

    // --- Compilation ---

    /// Compile this agent into the pair of inputs the loop consumes.
    ///
    /// `parent = None` starts a root run — requires an explicit `.model()`
    /// and uses the agent's own per-run fields (provider, cancel signal, etc.)
    /// falling back to sensible defaults.
    ///
    /// `parent = Some((runtime, spec))` spawns a sub-agent — externals
    /// inherit from `runtime` (child's own runtime fields override on a
    /// per-field basis), and `ModelSpec::Inherit` resolves against
    /// `spec.model`.
    pub(crate) fn compile(
        &self,
        parent: Option<(&LoopRuntime, &LoopSpec)>,
    ) -> Result<(LoopRuntime, LoopSpec)> {
        let runtime = match parent {
            Some((parent_runtime, _)) => self.inherit_runtime(parent_runtime),
            None => self.build_runtime()?,
        };
        let fallback_model = parent.map(|(_, ps)| &ps.model);
        let spec = self.compile_spec(&runtime, fallback_model)?;
        Ok((runtime, spec))
    }

    /// Compile just the `LoopSpec` against a pre-built runtime. Used by
    /// `compile` and by tests that need to exercise spec composition against
    /// a hand-built runtime (e.g. with synthetic metadata).
    pub(crate) fn compile_spec(
        &self,
        runtime: &LoopRuntime,
        fallback_model: Option<&Model>,
    ) -> Result<LoopSpec> {
        let name = self
            .config
            .name
            .clone()
            .unwrap_or_else(|| generate_agent_name("agent"));

        let model = match (&self.config.model, fallback_model) {
            (ModelSpec::Inherit, None) => {
                return Err(AgenticError::Other(
                    "root agent requires an explicit .model() (or must be spawned as a child)"
                        .into(),
                ));
            }
            (spec, Some(parent)) => spec.resolve(parent),
            (ModelSpec::Exact(m), None) => m.clone(),
        };

        let (tools, tool_choice) = compile_tools(self);

        let system_prompt = compile_system_prompt(self);
        let instruction_prompt = self.runtime.interpolate(&self.runtime.instruction_prompt);
        let context_prompt = compile_context_prompt(runtime, self);

        Ok(LoopSpec {
            name,
            model,
            system_prompt,
            instruction_prompt,
            context_prompt,
            tools,
            tool_choice,
            sub_agents: self.config.sub_agents.clone(),
            output_schema: self.config.output_schema.clone(),
            max_output_tokens: self.config.max_output_tokens,
            max_input_tokens: self.config.max_input_tokens,
            max_turns: self.config.max_turns,
            max_schema_retries: self.config.max_schema_retries,
            max_request_retries: self.config.max_request_retries,
            request_retry_backoff_ms: self.config.request_retry_backoff_ms,
            keep_alive: self.config.keep_alive,
        })
    }

    /// Build the root `LoopRuntime` from this agent's per-run fields plus
    /// reasonable defaults. Requires `self.runtime.provider` to be set.
    fn build_runtime(&self) -> Result<LoopRuntime> {
        let provider = self
            .runtime
            .provider
            .clone()
            .ok_or_else(|| AgenticError::Other("Agent::run() requires a provider".into()))?;

        let working_directory = self
            .runtime
            .working_directory
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let event_handler: Arc<dyn Fn(AgentEvent) + Send + Sync> = self
            .runtime
            .event_handler
            .clone()
            .unwrap_or_else(AgentEvent::default_logger);

        let cancel_signal = self
            .runtime
            .cancel_signal
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

        // Every root run gets a command queue so background sub-agents can post
        // notifications back to the parent. An externally supplied queue
        // (e.g. from `Agent::create`) wins so the handle can reach the loop.
        let command_queue = Some(
            self.runtime
                .command_queue
                .clone()
                .unwrap_or_else(|| Arc::new(CommandQueue::new())),
        );

        let session_store = self.runtime.session_dir.as_ref().map(|dir| {
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
        })
    }

    /// Build a child `LoopRuntime`: inherit externals from the parent, let this
    /// agent's own per-run fields override any that it set explicitly.
    fn inherit_runtime(&self, parent: &LoopRuntime) -> LoopRuntime {
        let overrides = &self.runtime;
        LoopRuntime {
            provider: overrides
                .provider
                .clone()
                .unwrap_or_else(|| parent.provider.clone()),
            event_handler: overrides
                .event_handler
                .clone()
                .unwrap_or_else(|| parent.event_handler.clone()),
            cancel_signal: overrides
                .cancel_signal
                .clone()
                .unwrap_or_else(|| parent.cancel_signal.clone()),
            working_directory: overrides
                .working_directory
                .clone()
                .unwrap_or_else(|| parent.working_directory.clone()),
            command_queue: parent.command_queue.clone(),
            session_store: parent.session_store.clone(),
            metadata: parent.metadata.clone(),
            discovered_tools: parent.discovered_tools.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Spec-compilation helpers (private to werk.rs — used by Agent::compile_spec)
// ---------------------------------------------------------------------------

fn compile_tools(agent: &Agent) -> (Arc<ToolRegistry>, Option<ToolChoice>) {
    let mut tools = agent.config.tools.clone();

    // If the agent declared sub-agents, make sure a SpawnAgentTool is
    // available so the LLM can call them. Skip when the user registered one
    // themselves.
    if !agent.config.sub_agents.is_empty() && tools.get("spawn_agent").is_none() {
        tools.register(SpawnAgentTool);
    }

    (Arc::new(tools), None)
}

fn compile_system_prompt(agent: &Agent) -> String {
    let mut s = agent.runtime.interpolate(&agent.config.identity_prompt);
    if !agent.config.behavior_prompt.is_empty() {
        s.push_str("\n\n");
        s.push_str(&agent.config.behavior_prompt);
    }
    if agent.config.output_schema.is_some() {
        s.push_str(prompts::STRUCTURED_OUTPUT_INSTRUCTION);
    }
    s
}

fn compile_context_prompt(runtime: &LoopRuntime, agent: &Agent) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(meta) = &runtime.metadata {
        parts.push(meta.clone());
    }
    for block in &agent.config.context_prompts {
        parts.push(format!("<context>\n{block}\n</context>"));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

// ---------------------------------------------------------------------------
// Tests — builder-only. Loop-behavior tests live in `super::r#loop`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::event::AgentEventKind;
    use super::super::output::AgentStatus;
    use crate::error::AgenticError;

    #[test]
    fn silent_sets_a_no_op_handler() {
        let agent = Agent::new().silent();
        let handler = agent
            .runtime
            .event_handler
            .as_ref()
            .expect(".silent() must install a handler")
            .clone();
        // Every variant passes through without panicking; no output is asserted —
        // the point is that a handler is present and benign.
        handler(AgentEvent::new(
            "t",
            AgentEventKind::AgentEnd {
                turns: 1,
                status: AgentStatus::Completed,
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
            .model("mock")
            .identity_prompt("")
            .provider(std::sync::Arc::new(crate::testutil::MockProvider::text("ok")));
        assert!(agent.runtime.event_handler.is_none());
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
        assert_eq!(agent.config.identity_prompt, "You are a test agent");

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
        assert!(agent.config.output_schema.is_some());

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
        let base = Agent::new().name("x").model("original").max_turns(3);
        let applied = base.apply_overrides(&serde_json::json!({
            "model": "overridden",
            "max_turns": 7
        }));
        assert_eq!(applied.config.max_turns, Some(7));
        match &applied.config.model {
            ModelSpec::Exact(m) => assert_eq!(m.id, "overridden"),
            _ => panic!("expected Exact model"),
        }
    }

    #[tokio::test]
    async fn missing_provider_fails_run() {
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("x")
            .instruction_prompt("do");
        let err = agent.run().await.unwrap_err();
        match err {
            AgenticError::Other(msg) => assert!(msg.contains("provider"), "got: {msg}"),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}
