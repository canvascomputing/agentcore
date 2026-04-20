//! Core agent type and execution loop.
//!
//! Holds:
//! - `Agent` (public) — the single user-facing type. Internally split into a
//!   shared `Arc<AgentConfig>` (static template: tools, prompts, sub-agents,
//!   tuning knobs) and an owned `AgentRuntime` (per-run: provider, instruction
//!   prompt, working directory, event handler). Builder methods route through
//!   `with_config` (copy-on-write via `Arc::make_mut`) or `with_runtime`
//!   (direct mutation), so `template.clone().provider(p).instruction_prompt(t)`
//!   never copies the heavy template.
//! - `LoopRuntime` / `AgentSpec` / `LoopState` (pub(crate)) — the three internal
//!   structs the loop works with. See their doc comments for the split.
//! - `run_loop` (pub(crate) free function) — consumes the three structs.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::persistence::session::{TranscriptEntryType, SessionStore, TranscriptEntry};
use crate::provider::model::{Model, ModelSpec};
use crate::provider::retry::compute_delay;
use crate::provider::types::{
    ContentBlock, Message, CompletionResponse, ResponseStatus, StreamEvent, TokenUsage,
};
use crate::provider::{CompletionRequest, Provider, ProviderError, ToolChoice};
use crate::tools::{SpawnAgentTool, Tool, ToolCall, ToolContext, ToolRegistry};
use crate::util::{generate_agent_name, now_millis};

use super::compact;
use super::event::{AgentEvent, AgentEventKind};
use super::output::{AgentOutput, AgentStatus, OutputSchema, AgentStatistics};
use super::prompts::{self as prompts, DEFAULT_BEHAVIOR_PROMPT};
use super::queue::{CommandQueue, QueuePriority};
use super::running::RunningAgent;

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
    pub(crate) prompt_errors: Vec<String>,
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
    pub keep_alive_ms: Option<u64>,
}

/// Sentinel value for "keep_alive with no timeout". ~584 million years in ms
/// — effectively unbounded. Exposed only via `Agent::keep_alive_unlimited()`,
/// never a user-facing magic number.
const KEEP_ALIVE_UNLIMITED: u64 = u64::MAX;

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
            keep_alive_ms: None,
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
            keep_alive_ms: self.keep_alive_ms,
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

    fn read_file(&mut self, path: PathBuf) -> String {
        match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(err) => {
                self.prompt_errors
                    .push(format!("Failed to read prompt from {}: {}", path.display(), err));
                String::new()
            }
        }
    }

    fn read_json_file(&mut self, path: PathBuf) -> Option<Value> {
        let content = self.read_file(path);
        if content.is_empty() {
            return None;
        }
        match serde_json::from_str(&content) {
            Ok(v) => Some(v),
            Err(e) => {
                self.prompt_errors.push(format!("invalid JSON: {e}"));
                None
            }
        }
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
    pub fn identity_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        let s = self.read_file(path.into());
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
    pub fn tool(self, tool: impl Tool + 'static) -> Self {
        self.with_config(|c| c.tools.register(tool))
    }

    /// Register a structured output schema. Invalid schemas surface as an
    /// error at `run()` time.
    pub fn output_schema(mut self, schema: Value) -> Self {
        match OutputSchema::new(schema) {
            Ok(s) => self.with_config(|c| c.output_schema = Some(s)),
            Err(e) => {
                self.prompt_errors.push(format!("invalid output schema: {e}"));
                self
            }
        }
    }

    /// Load a structured output schema from a JSON file.
    pub fn output_schema_file(mut self, path: impl Into<PathBuf>) -> Self {
        match self.read_json_file(path.into()) {
            Some(schema) => self.output_schema(schema),
            None => self,
        }
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

    /// Keep the agent alive after a benign text-only response for up to `ms`
    /// milliseconds, listening for peer messages. Wakes on any message visible
    /// to this agent, or on `cancel_signal`. Default: unset (one-shot).
    pub fn keep_alive_ms(self, ms: u64) -> Self {
        self.with_config(|c| c.keep_alive_ms = Some(ms))
    }

    /// Keep the agent alive indefinitely after a benign text-only response.
    /// Only the `cancel_signal` (or process exit) will stop the wait.
    pub fn keep_alive_unlimited(self) -> Self {
        self.with_config(|c| c.keep_alive_ms = Some(KEEP_ALIVE_UNLIMITED))
    }

    /// Override the default behavior prompt.
    pub fn behavior_prompt(self, content: impl Into<String>) -> Self {
        let content = content.into();
        self.with_config(|c| c.behavior_prompt = content)
    }

    /// Load a behavior prompt override from a file.
    pub fn behavior_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        let content = self.read_file(path.into());
        self.with_config(|c| c.behavior_prompt = content)
    }

    /// Append additional context alongside the instruction prompt.
    pub fn context_prompt(self, content: impl Into<String>) -> Self {
        let content = content.into();
        self.with_config(|c| c.context_prompts.push(content))
    }

    /// Append additional context from a file.
    pub fn context_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        let content = self.read_file(path.into());
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
    pub fn instruction_prompt_file(mut self, path: impl Into<PathBuf>) -> Self {
        let s = self.read_file(path.into());
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

    pub fn cancel_signal(self, s: Arc<AtomicBool>) -> Self {
        self.with_runtime(|r| r.cancel_signal = Some(s))
    }

    /// Install an externally-owned command queue. Only used by `Agent::create`
    /// so the returned `RunningAgent` can inject instructions into the loop.
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
        self.check_prompt_errors()?;
        let runtime = LoopRuntime::from_agent(self)?;
        let spec = AgentSpec::compile(self, &runtime, None)?;
        run_loop(Arc::new(runtime), Arc::new(spec), None).await
    }

    /// Start the agent on a background tokio task and return a
    /// [`RunningAgent`] handle for injecting new instructions, cancelling, or
    /// awaiting the final output.
    ///
    /// Requires a running tokio runtime (`tokio::spawn` is invoked
    /// synchronously). Requires `.provider()` and `.instruction_prompt()`.
    pub fn create(self) -> RunningAgent {
        let queue = Arc::new(CommandQueue::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let prepared = self
            .cancel_signal(cancel.clone())
            .command_queue(queue.clone());
        let join = tokio::spawn(async move { prepared.run().await });
        RunningAgent::new(queue, cancel, join)
    }

    /// Crate-internal: run this agent as a child under a parent's `LoopRuntime`.
    /// `parent_model` resolves `ModelSpec::Inherit` on the child (id *and*
    /// context window size both inherit). `description` becomes the
    /// `AgentStart` event's human-readable label.
    pub(crate) async fn run_child(
        &self,
        parent_runtime: &LoopRuntime,
        parent_model: &Model,
        description: Option<String>,
    ) -> Result<AgentOutput> {
        self.check_prompt_errors()?;
        let runtime = parent_runtime.inherit(self);
        let spec = AgentSpec::compile(self, &runtime, Some(parent_model))?;
        run_loop(Arc::new(runtime), Arc::new(spec), description).await
    }

    /// Test-only escape hatch: run with fully constructed LoopRuntime + spec.
    #[cfg(test)]
    pub(crate) async fn run_with_parts(
        &self,
        runtime: Arc<LoopRuntime>,
        spec: Arc<AgentSpec>,
    ) -> Result<AgentOutput> {
        run_loop(runtime, spec, None).await
    }

    fn check_prompt_errors(&self) -> Result<()> {
        if self.prompt_errors.is_empty() {
            Ok(())
        } else {
            Err(AgenticError::Other(self.prompt_errors.join("; ")))
        }
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
}

// ---------------------------------------------------------------------------
// LoopRuntime — external services & I/O handles
// ---------------------------------------------------------------------------

/// External services and I/O handles for one run-tree (the root agent + every
/// sub-agent spawned transitively). Shared as `Arc<LoopRuntime>`. Read-only from
/// the loop's perspective — mutability is only via the interior atomics and
/// mutexes inside.
pub(crate) struct LoopRuntime {
    pub provider: Arc<dyn Provider>,
    pub event_handler: Arc<dyn Fn(AgentEvent) + Send + Sync>,
    pub cancel_signal: Arc<AtomicBool>,
    pub working_directory: PathBuf,
    pub command_queue: Option<Arc<CommandQueue>>,
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub metadata: Option<String>,
}

impl LoopRuntime {
    /// Build the root `LoopRuntime` from an Agent's per-run fields plus reasonable defaults.
    /// Requires `agent.runtime.provider` to be set.
    pub(crate) fn from_agent(agent: &Agent) -> Result<Self> {
        let provider = agent
            .runtime
            .provider
            .clone()
            .ok_or_else(|| AgenticError::Other("Agent::run() requires a provider".into()))?;

        let working_directory = agent
            .runtime
            .working_directory
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let event_handler: Arc<dyn Fn(AgentEvent) + Send + Sync> = agent
            .runtime
            .event_handler
            .clone()
            .unwrap_or_else(|| Arc::new(|_| {}));

        let cancel_signal = agent
            .runtime
            .cancel_signal
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

        // Every root run gets a command queue so background sub-agents can post
        // notifications back to the parent. An externally supplied queue
        // (e.g. from `Agent::create`) wins so the handle can reach the loop.
        let command_queue = Some(
            agent
                .runtime
                .command_queue
                .clone()
                .unwrap_or_else(|| Arc::new(CommandQueue::new())),
        );

        let session_store = agent.runtime.session_dir.as_ref().map(|dir| {
            let store = SessionStore::new(dir, &generate_agent_name("session"));
            Arc::new(Mutex::new(store))
        });

        let metadata = Some(LoopRuntime::environment(&working_directory));

        Ok(Self {
            provider,
            event_handler,
            cancel_signal,
            working_directory,
            command_queue,
            session_store,
            metadata,
        })
    }

    /// Build the environment metadata block — working directory, platform, OS
    /// version, and current date — for prepending to the first user message.
    pub(crate) fn environment(working_directory: &std::path::Path) -> String {
        let working_directory = working_directory.display();
        let platform = std::env::consts::OS;
        let os_version = std::process::Command::new("uname")
            .arg("-r")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let date = format_current_date();
        format!(
            "<environment>\nWorking directory: {working_directory}\nPlatform: {platform}\nOS version: {os_version}\nDate: {date}\n</environment>"
        )
    }

    /// Build a child LoopRuntime: inherit externals from the parent, let the
    /// child's own per-run fields override any that the child set explicitly.
    pub(crate) fn inherit(&self, child: &Agent) -> Self {
        let overrides = &child.runtime;
        Self {
            provider: overrides
                .provider
                .clone()
                .unwrap_or_else(|| self.provider.clone()),
            event_handler: overrides
                .event_handler
                .clone()
                .unwrap_or_else(|| self.event_handler.clone()),
            cancel_signal: overrides
                .cancel_signal
                .clone()
                .unwrap_or_else(|| self.cancel_signal.clone()),
            working_directory: overrides
                .working_directory
                .clone()
                .unwrap_or_else(|| self.working_directory.clone()),
            command_queue: self.command_queue.clone(),
            session_store: self.session_store.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Convert epoch seconds to a date string using the civil-from-days algorithm.
/// http://howardhinnant.github.io/date_algorithms.html
fn format_current_date() -> String {
    let epoch_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = epoch_secs / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}")
}

// ---------------------------------------------------------------------------
// AgentSpec — compiled per-agent blueprint
// ---------------------------------------------------------------------------

/// Compiled blueprint for one `Agent`'s execution. Built once by
/// `AgentSpec::compile` at the start of a run and never mutated. Per-agent:
/// child agents compile their own spec.
pub(crate) struct AgentSpec {
    pub name: String,
    pub model: Model,
    pub system_prompt: String,
    pub instruction_prompt: String,
    pub context_prompt: Option<String>,
    pub tools: Arc<ToolRegistry>,
    pub tool_choice: Option<ToolChoice>,
    pub sub_agents: Vec<Agent>,
    pub output_schema: Option<OutputSchema>,
    pub max_output_tokens: Option<u32>,
    pub max_input_tokens: Option<u64>,
    pub max_turns: Option<u32>,
    pub max_schema_retries: Option<u32>,
    pub max_request_retries: u32,
    pub request_retry_backoff_ms: u64,
    pub keep_alive_ms: Option<u64>,
}

impl AgentSpec {
    pub(crate) fn compile(
        agent: &Agent,
        runtime: &LoopRuntime,
        fallback_model: Option<&Model>,
    ) -> Result<Self> {
        let name = agent
            .config
            .name
            .clone()
            .unwrap_or_else(|| generate_agent_name("agent"));

        let model = match (&agent.config.model, fallback_model) {
            (ModelSpec::Inherit, None) => {
                return Err(AgenticError::Other(
                    "root agent requires an explicit .model() (or must be spawned as a child)"
                        .into(),
                ));
            }
            (spec, Some(parent)) => spec.resolve(parent),
            (ModelSpec::Exact(m), None) => m.clone(),
        };

        let (tools, tool_choice) = compile_tools(agent);

        let system_prompt = compile_system_prompt(agent);
        let instruction_prompt = agent.runtime.interpolate(&agent.runtime.instruction_prompt);
        let context_prompt = compile_context_prompt(runtime, agent);

        Ok(Self {
            name,
            model,
            system_prompt,
            instruction_prompt,
            context_prompt,
            tools,
            tool_choice,
            sub_agents: agent.config.sub_agents.clone(),
            output_schema: agent.config.output_schema.clone(),
            max_output_tokens: agent.config.max_output_tokens,
            max_input_tokens: agent.config.max_input_tokens,
            max_turns: agent.config.max_turns,
            max_schema_retries: agent.config.max_schema_retries,
            max_request_retries: agent.config.max_request_retries,
            request_retry_backoff_ms: agent.config.request_retry_backoff_ms,
            keep_alive_ms: agent.config.keep_alive_ms,
        })
    }
}

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
// LoopState — mutable per-agent state
// ---------------------------------------------------------------------------

/// Everything the loop mutates. Created fresh for each agent run.
#[derive(Default)]
pub(crate) struct LoopState {
    pub messages: Vec<Message>,
    pub total_usage: TokenUsage,
    pub request_count: u64,
    pub tool_call_count: u64,
    pub turn: u32,
    pub schema_retries: u32,
    pub discovered_tools: HashSet<String>,
    pub is_idle: bool,
}

impl LoopState {
    pub(crate) fn init(spec: &AgentSpec) -> Self {
        let mut messages = Vec::new();
        if let Some(cp) = &spec.context_prompt {
            messages.push(Message::user(cp.clone()));
        }
        messages.push(Message::user(spec.instruction_prompt.clone()));
        Self {
            messages,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Execution loop — run_loop + its per-turn helpers (guards, provider call,
// compaction seams, tool execution, completion, transcript, events)
// ---------------------------------------------------------------------------

/// The agent execution loop — one turn per iteration.
///
/// Each turn sends the conversation to the model and processes the reply.
/// The loop **continues** when more work is needed and **exits** when the
/// agent has nothing left to do.
///
/// Continues when:
/// - the model called tools (run them and feed results back)
/// - the reply was truncated (ask the model to keep going)
/// - a peer agent or background task queued a message
/// - the agent was parked idle and a message woke it
/// - a required output schema wasn't met (retry)
///
/// Exits when:
/// - the model stopped and any required schema validates — return the
///   final answer
/// - a guard fired (cancel, turn limit, input-token budget) — return a
///   partial output with the reason attached
///
/// Context-window overflow is handled transparently: compaction is
/// triggered proactively when the estimated next request would overflow,
/// and reactively when the provider reports overflow mid-turn.
pub(crate) fn run_loop(
    runtime: Arc<LoopRuntime>,
    spec: Arc<AgentSpec>,
    description: Option<String>,
) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send>> {
    Box::pin(async move {
        runtime.provider.prewarm().await;
        let mut state = LoopState::init(&spec);
        record_transcript(
            &runtime,
            TranscriptEntryType::UserMessage,
            state.messages.last().unwrap(),
            None,
        );
        emit_agent_start(&runtime, &spec, description);

        loop {
            if let Some(status) = check_guards(&runtime, &spec, &state) {
                return Ok(finish_early(&runtime, &spec, &mut state, status));
            }

            state.turn += 1;
            let turn = state.turn;

            emit_turn_start(&runtime, &spec, turn);
            emit_request_start(&runtime, &spec);

            let response = call_provider_with_retry(&runtime, &spec, &mut state, turn).await?;

            emit_request_end(&runtime, &spec);
            record_usage(&runtime, &spec, &mut state, &response);

            let (text, tool_calls) = parse_response(&response);
            state.messages.push(Message::Assistant {
                content: response.content.clone(),
            });
            record_transcript(
                &runtime,
                TranscriptEntryType::AssistantMessage,
                state.messages.last().unwrap(),
                Some((&response.usage, &response.model)),
            );

            // Mid-generation overflow: same reactive seam as the pre-flight error.
            if response.status == ResponseStatus::ContextWindowExceeded
                && spec.model.context_window_size.is_some()
            {
                compact::trigger_reactive(&runtime, &spec, &mut state, turn).await?;
            }

            compact::trigger_if_over_threshold(&runtime, &spec, &mut state).await?;

            let tool_use_ready =
                response.status == ResponseStatus::ToolUse && !tool_calls.is_empty();
            let response_truncated =
                response.status == ResponseStatus::OutputTruncated && tool_calls.is_empty();

            if tool_use_ready {
                let results = execute_tools(&runtime, &spec, &mut state, &tool_calls).await;
                state.messages.push(Message::User { content: results });
                record_transcript(
                    &runtime,
                    TranscriptEntryType::ToolResult,
                    state.messages.last().unwrap(),
                    None,
                );
                drain_command_queue(&runtime, &spec, &mut state);
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            if response_truncated {
                emit_output_truncated(&runtime, &spec, turn);
                state
                    .messages
                    .push(Message::user(prompts::MAX_TOKENS_CONTINUATION));
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            let drain_found_messages = drain_pending_messages(&runtime, &spec, &mut state);
            if drain_found_messages {
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            // Short-circuit: idle_until_message only runs when keep_alive is
            // enabled, preserving its side effects (emit Idle/Resumed).
            let keep_alive_enabled = spec.keep_alive_ms.is_some();
            let idle_found_message = keep_alive_enabled
                && idle_until_message(&runtime, &spec, &mut state).await;
            if idle_found_message {
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            let output_validation = match &spec.output_schema {
                None => Ok(None),
                Some(schema) => schema.validate(&text).map(Some),
            };

            if let Err(detail) = output_validation.as_ref() {
                state.schema_retries += 1;

                let retry_limit_exceeded = spec
                    .max_schema_retries
                    .filter(|&limit| state.schema_retries > limit);
                if let Some(limit) = retry_limit_exceeded {
                    return Err(AgenticError::SchemaRetryExhausted { retries: limit });
                }

                let retry_prompt = OutputSchema::retry_message(detail);
                state.messages.push(Message::user(retry_prompt));
                emit_turn_end(&runtime, &spec, turn);
                continue;
            }

            let validated = output_validation.expect("Err handled above");
            let agent_end = AgentEventKind::AgentEnd {
                turns: state.turn,
                status: AgentStatus::Completed,
            };

            emit(&runtime, &spec, agent_end);
            emit_turn_end(&runtime, &spec, turn);
            return Ok(build_output(&state, text, validated, AgentStatus::Completed));
        }
    })
}

fn finish_early(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    status: AgentStatus,
) -> AgentOutput {
    let text = last_assistant_text(&state.messages);
    emit(
        runtime,
        spec,
        AgentEventKind::AgentEnd {
            turns: state.turn,
            status: status.clone(),
        },
    );
    build_output(state, text, None, status)
}

fn check_guards(runtime: &LoopRuntime, spec: &AgentSpec, state: &LoopState) -> Option<AgentStatus> {
    if runtime.cancel_signal.load(Ordering::Relaxed) {
        return Some(AgentStatus::Cancelled);
    }
    if let Some(limit) = spec.max_turns {
        if state.turn >= limit {
            return Some(AgentStatus::TurnLimitReached { limit });
        }
    }
    if let Some(limit) = spec.max_input_tokens {
        if state.total_usage.input_tokens >= limit {
            return Some(AgentStatus::BudgetExhausted {
                usage: state.total_usage.input_tokens,
                limit,
            });
        }
    }
    None
}

async fn call_provider(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &LoopState,
) -> Result<CompletionResponse> {
    let tool_defs = spec.tools.definitions(&state.discovered_tools);
    let request = CompletionRequest {
        model: spec.model.id.clone(),
        system_prompt: spec.system_prompt.clone(),
        messages: state.messages.clone(),
        tools: tool_defs,
        max_output_tokens: spec.max_output_tokens,
        tool_choice: spec.tool_choice.clone(),
    };

    let event_handler = runtime.event_handler.clone();
    let agent_name = spec.name.clone();
    let on_event = Arc::new(move |event: StreamEvent| {
        if let StreamEvent::TextDelta { text, .. } = &event {
            event_handler(AgentEvent::new(
                agent_name.clone(),
                AgentEventKind::ResponseTextChunk {
                    content: text.clone(),
                },
            ));
        }
    });

    runtime
        .provider
        .complete_streaming(request, on_event)
        .await
        .map_err(AgenticError::from)
}

/// One turn's provider call with two built-in resilience seams:
/// - transient errors (429, 529, 5xx) retry up to `spec.max_request_retries`
/// - a provider-reported `ContextWindowExceeded` fires
///   [`compact::trigger_reactive`] before propagating the original error
async fn call_provider_with_retry(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    turn: u32,
) -> Result<CompletionResponse> {
    let mut last_err = None;
    for attempt in 0..=spec.max_request_retries {
        match call_provider(runtime, spec, state).await {
            Ok(response) => return Ok(response),
            Err(AgenticError::Provider(ProviderError::ContextWindowExceeded {
                provider_message,
            })) if spec.model.context_window_size.is_some() => {
                compact::trigger_reactive(runtime, spec, state, turn).await?;
                // compact::run returns NotImplemented today; once
                // implemented this branch will retry the turn instead of
                // surfacing the error.
                return Err(AgenticError::Provider(
                    ProviderError::ContextWindowExceeded { provider_message },
                ));
            }
            Err(e) if e.is_retryable() && attempt < spec.max_request_retries => {
                let delay_ms =
                    compute_delay(spec.request_retry_backoff_ms, attempt, e.retry_after_ms());
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| AgenticError::Other("retry loop ended unexpectedly".into())))
}

fn record_usage(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
    response: &CompletionResponse,
) {
    state.total_usage += &response.usage;
    state.request_count += 1;
    emit(
        runtime,
        spec,
        AgentEventKind::TokenUsage {
            model: response.model.clone(),
            usage: response.usage.clone(),
        },
    );
}

fn parse_response(response: &CompletionResponse) -> (String, Vec<ToolCall>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in &response.content {
        match block {
            ContentBlock::Text { text: chunk } => text.push_str(chunk),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            }),
            _ => {}
        }
    }
    (text, tool_calls)
}

async fn execute_tools(
    runtime: &Arc<LoopRuntime>,
    spec: &Arc<AgentSpec>,
    state: &mut LoopState,
    calls: &[ToolCall],
) -> Vec<ContentBlock> {
    state.tool_call_count += calls.len() as u64;
    for call in calls {
        emit(
            runtime,
            spec,
            AgentEventKind::ToolCallStart {
                tool_name: call.name.clone(),
                call_id: call.id.clone(),
                input: call.input.clone(),
            },
        );
    }

    let tool_ctx = ToolContext::new(runtime.working_directory.clone())
        .registry(Arc::clone(&spec.tools))
        .runtime(Arc::clone(runtime))
        .caller_spec(Arc::clone(spec));

    let raw = spec.tools.execute(calls, &tool_ctx).await;
    let mut blocks = Vec::with_capacity(raw.len());
    for (block, result) in raw {
        for t in result.discovered_tools {
            state.discovered_tools.insert(t);
        }

        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } = &block
        {
            let tool_name = calls
                .iter()
                .find(|c| c.id == *tool_use_id)
                .map(|c| c.name.clone())
                .unwrap_or_default();
            emit(
                runtime,
                spec,
                AgentEventKind::ToolCallEnd {
                    tool_name,
                    call_id: tool_use_id.clone(),
                    output: content.clone(),
                    is_error: *is_error,
                },
            );
        }

        blocks.push(block);
    }
    blocks
}

fn drain_command_queue(runtime: &LoopRuntime, spec: &AgentSpec, state: &mut LoopState) {
    let Some(queue) = runtime.command_queue.as_ref() else {
        return;
    };
    while let Some(cmd) = queue.dequeue_if(Some(&spec.name), |c| c.priority != QueuePriority::Later)
    {
        state.messages.push(Message::user(cmd.as_user_message()));
    }
}

/// Drain the command queue into `state.messages` and report whether anything
/// arrived. Used by `run_loop` to decide whether to continue the loop without
/// a new LLM request when the model gave no actionable output.
fn drain_pending_messages(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
) -> bool {
    let before = state.messages.len();
    drain_command_queue(runtime, spec, state);
    state.messages.len() > before
}

/// Park the agent as idle, poll for incoming messages, then emit the resume
/// event. Returns `true` if a message arrived (loop should continue),
/// `false` on timeout or cancel (loop should finalize).
async fn idle_until_message(
    runtime: &LoopRuntime,
    spec: &AgentSpec,
    state: &mut LoopState,
) -> bool {
    state.is_idle = true;
    emit_agent_idle(runtime, spec);
    let woken = wait_for_message(runtime, spec, state).await;
    state.is_idle = false;
    emit_agent_resumed(runtime, spec);
    woken
}

/// Park the agent between turn chains, polling the command queue for messages
/// visible to `spec.name`. Returns `true` if a message arrived (agent should
/// resume), `false` on timeout or cancel (agent should finalize).
///
/// Only called via `idle_until_message` when `spec.keep_alive_ms.is_some()` —
/// `None` is handled as an early return for defensive coding.
async fn wait_for_message(runtime: &LoopRuntime, spec: &AgentSpec, state: &mut LoopState) -> bool {
    const POLL_INTERVAL: Duration = Duration::from_millis(100);
    let deadline = match spec.keep_alive_ms {
        Some(KEEP_ALIVE_UNLIMITED) => None,
        Some(ms) => Some(Instant::now() + Duration::from_millis(ms)),
        None => return false,
    };
    loop {
        if runtime.cancel_signal.load(Ordering::Relaxed) {
            return false;
        }
        let before = state.messages.len();
        drain_command_queue(runtime, spec, state);
        if state.messages.len() > before {
            return true;
        }
        match deadline {
            Some(d) if Instant::now() >= d => return false,
            Some(d) => {
                let remaining = d - Instant::now();
                tokio::time::sleep(POLL_INTERVAL.min(remaining)).await;
            }
            None => tokio::time::sleep(POLL_INTERVAL).await,
        }
    }
}

fn emit(runtime: &LoopRuntime, spec: &AgentSpec, kind: AgentEventKind) {
    (runtime.event_handler)(AgentEvent::new(spec.name.clone(), kind));
}

fn emit_agent_start(runtime: &LoopRuntime, spec: &AgentSpec, description: Option<String>) {
    emit(runtime, spec, AgentEventKind::AgentStart { description });
}

fn emit_turn_start(runtime: &LoopRuntime, spec: &AgentSpec, turn: u32) {
    emit(runtime, spec, AgentEventKind::TurnStart { turn });
}

fn emit_turn_end(runtime: &LoopRuntime, spec: &AgentSpec, turn: u32) {
    emit(runtime, spec, AgentEventKind::TurnEnd { turn });
}

fn emit_request_start(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(
        runtime,
        spec,
        AgentEventKind::RequestStart {
            model: spec.model.id.clone(),
        },
    );
}

fn emit_request_end(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(
        runtime,
        spec,
        AgentEventKind::RequestEnd {
            model: spec.model.id.clone(),
        },
    );
}

fn emit_output_truncated(runtime: &LoopRuntime, spec: &AgentSpec, turn: u32) {
    emit(runtime, spec, AgentEventKind::OutputTruncated { turn });
}

fn emit_agent_idle(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(runtime, spec, AgentEventKind::AgentIdle);
}

fn emit_agent_resumed(runtime: &LoopRuntime, spec: &AgentSpec) {
    emit(runtime, spec, AgentEventKind::AgentResumed);
}

fn record_transcript(
    runtime: &LoopRuntime,
    entry_type: TranscriptEntryType,
    message: &Message,
    usage_and_model: Option<(&TokenUsage, &str)>,
) {
    let Some(ref store) = runtime.session_store else {
        return;
    };
    store
        .lock()
        .unwrap()
        .record(TranscriptEntry {
            recorded_at: now_millis(),
            entry_type,
            message: message.clone(),
            usage: usage_and_model.map(|(u, _)| u.clone()),
            model: usage_and_model.map(|(_, m)| m.to_string()),
        })
        .ok();
}

fn build_output(
    state: &LoopState,
    text: String,
    response: Option<Value>,
    status: AgentStatus,
) -> AgentOutput {
    AgentOutput {
        response,
        response_raw: text,
        statistics: AgentStatistics {
            input_tokens: state.total_usage.input_tokens,
            output_tokens: state.total_usage.output_tokens,
            requests: state.request_count,
            tool_calls: state.tool_call_count,
            turns: state.turn,
        },
        status,
    }
}

fn last_assistant_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::Assistant { content } => {
                let text: String = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                Some(text)
            }
            _ => None,
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::queue::{CommandSource, QueuedCommand};
    use crate::error::AgenticError;
    use crate::provider::types::ContentBlock;
    use crate::testutil::*;

    fn simple_agent() -> Agent {
        Agent::new()
            .name("test-agent")
            .model("mock-model")
            .identity_prompt("You are a test assistant.")
    }

    fn assert_lifecycle_events(harness: &TestHarness, output: &AgentOutput) {
        let events = harness.events().all();

        let agent_end_status = events.iter().find_map(|e| match &e.kind {
            AgentEventKind::AgentEnd { status, .. } => Some(status.clone()),
            _ => None,
        });
        assert_eq!(
            agent_end_status.as_ref(),
            Some(&output.status),
            "AgentEnd status must match output.status"
        );

        let last_significant = events
            .iter()
            .rev()
            .find(|e| !matches!(e.kind, AgentEventKind::TurnEnd { .. }));
        assert!(
            matches!(
                last_significant.map(|e| &e.kind),
                Some(AgentEventKind::AgentEnd { .. })
            ),
            "AgentEnd must be the last significant event"
        );

        for (i, event) in events.iter().enumerate() {
            if matches!(event.kind, AgentEventKind::OutputTruncated { .. }) {
                let after_agent_end = events[..i]
                    .iter()
                    .any(|e| matches!(e.kind, AgentEventKind::AgentEnd { .. }));
                assert!(!after_agent_end, "OutputTruncated at {i} after AgentEnd");
            }
        }
    }

    #[tokio::test]
    async fn simple_text_response() {
        let harness = TestHarness::new(MockProvider::text("Hello, world!"));
        let output = harness.run_agent(&simple_agent(), "Hi").await.unwrap();
        assert_eq!(output.response_raw, "Hello, world!");
        assert!(output.response.is_none());
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn simple_tool_execution() {
        let provider =
            MockProvider::tool_then_text("echo_tool", serde_json::json!({"text": "ping"}), "Done!");
        let agent = Agent::new()
            .name("test-agent")
            .model("mock-model")
            .identity_prompt("You are helpful.")
            .tool(MockTool::new("echo_tool", false, "pong"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "Echo test").await.unwrap();
        assert_eq!(output.response_raw, "Done!");
        assert_eq!(harness.provider().request_count(), 2);
    }

    #[tokio::test]
    async fn guard_max_turns() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            tool_response("t", "c2", serde_json::json!({})),
            tool_response("t", "c3", serde_json::json!({})),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_turns(2)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.status, AgentStatus::TurnLimitReached { limit: 2 });
        assert_eq!(output.statistics.turns, 2);
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn guard_cancellation() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("done"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        harness.cancel();
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.status, AgentStatus::Cancelled);
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn template_variable_interpolates_in_system_prompt() {
        let provider = MockProvider::text("Answer about rust");
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("You are an expert on {topic}.");

        let harness = TestHarness::new(provider).with_state("topic", serde_json::json!("rust"));
        harness.run_agent(&agent, "Tell me").await.unwrap();

        let prompts = harness.provider().system_prompts();
        assert!(prompts[0].contains("expert on rust"));
    }

    #[tokio::test]
    async fn events_emitted() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "Done");
        let agent = Agent::new()
            .name("assistant")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("read", true, "file contents"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "read it").await.unwrap();

        let events = harness.events();
        assert_eq!(events.agent_starts(), vec!["assistant"]);
        assert!(!events.tool_starts().is_empty());
        assert!(events.texts().contains(&"Done".to_string()));
        assert_eq!(events.agent_ends().len(), 1);
    }

    #[tokio::test]
    async fn command_queue_drains_next_priority() {
        use std::sync::Arc;
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "extra instruction".into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: Some("test".into()),
        });

        let harness = TestHarness::with_provider_and_queue(Arc::new(provider), queue);
        let output = harness.run_agent(&agent, "start").await.unwrap();
        assert_eq!(output.response_raw, "final");

        let requests = harness.provider().requests.lock().unwrap();
        let has_extra = requests[1].messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains("extra instruction"),
                _ => false,
            }),
            _ => false,
        });
        assert!(has_extra);
    }

    #[tokio::test]
    async fn command_queue_requeues_later_priority() {
        use std::sync::Arc;
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "later task".into(),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification {
                task_id: "42".into(),
            },
            agent_name: Some("test".into()),
        });

        let harness = TestHarness::with_provider_and_queue(Arc::new(provider), queue.clone());
        harness.run_agent(&agent, "start").await.unwrap();

        let cmd = queue.dequeue_if(Some("test"), |_| true);
        assert!(cmd.is_some());
        assert_eq!(cmd.unwrap().content, "later task");
    }

    #[tokio::test]
    async fn deferred_tool_filtering() {
        let provider = MockProvider::text("ok");
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("always", true, "ok"))
            .tool(DeferredMockTool::new("deferred"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let deferred_def = req.tools.iter().find(|t| t.name == "deferred").unwrap();
        assert!(deferred_def.description.is_empty());
    }

    #[tokio::test]
    async fn structured_output_extracted() {
        let schema_input = serde_json::json!({"category": "billing", "priority": "high"});
        let provider = MockProvider::new(vec![text_response(&schema_input.to_string())]);
        let agent = Agent::new()
            .name("classifier")
            .model("mock")
            .identity_prompt("Classify.")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": { "category": {"type": "string"}, "priority": {"type": "string"} },
                "required": ["category", "priority"]
            }));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "ticket").await.unwrap();
        let so = output.response.unwrap();
        assert_eq!(so["category"], "billing");
        assert_eq!(so["priority"], "high");
    }

    #[tokio::test]
    async fn structured_output_retry_exhausted() {
        let provider = MockProvider::new(vec![
            text_response("nope"),
            text_response("still nope"),
            text_response("nope again"),
            text_response("last nope"),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .max_schema_retries(3);

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(&agent, "go").await.unwrap_err();
        assert!(matches!(
            err,
            AgenticError::SchemaRetryExhausted { retries: 3 }
        ));
    }

    #[tokio::test]
    async fn sub_agents_auto_wire_spawn_tool() {
        let sub = Agent::new()
            .name("helper")
            .model("mock")
            .identity_prompt("I help.");

        let provider = MockProvider::text("ok");
        let agent = Agent::new()
            .name("parent")
            .model("mock")
            .identity_prompt("I coordinate.")
            .sub_agents([sub]);

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        assert!(
            req.tools.iter().any(|t| t.name == "spawn_agent"),
            ".sub_agents() should register spawn_agent automatically"
        );
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

    #[test]
    fn identity_prompt_file_loads_content() {
        let dir = std::env::temp_dir().join("agentwerk_test_werk_identity");
        let path = dir.join("identity.txt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "You are a test agent").unwrap();

        let agent = Agent::new().identity_prompt_file(&path);
        assert_eq!(agent.config.identity_prompt, "You are a test agent");
        assert!(agent.prompt_errors.is_empty());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn missing_prompt_file_surfaces_at_run() {
        let agent = Agent::new().identity_prompt_file("/nonexistent/xxx.txt");
        assert_eq!(agent.prompt_errors.len(), 1);
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
        assert!(agent.prompt_errors.is_empty());
        assert!(agent.config.output_schema.is_some());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn output_schema_file_missing_file_surfaces_error() {
        let agent = Agent::new().output_schema_file("/nonexistent/schema.json");
        assert_eq!(agent.prompt_errors.len(), 1);
    }

    #[test]
    fn invalid_output_schema_surfaces_as_prompt_error() {
        let agent = Agent::new()
            .name("test")
            .identity_prompt("")
            .output_schema(serde_json::json!({"type": "string"}));
        assert_eq!(agent.prompt_errors.len(), 1);
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

    fn runtime_with_metadata(meta: &str) -> LoopRuntime {
        LoopRuntime {
            provider: Arc::new(MockProvider::text("ok")),
            event_handler: Arc::new(|_| {}),
            cancel_signal: Arc::new(AtomicBool::new(false)),
            working_directory: PathBuf::from("/tmp"),
            command_queue: None,
            session_store: None,
            metadata: Some(meta.to_string()),
        }
    }

    #[tokio::test]
    async fn simple_text_response_status_completed() {
        let harness = TestHarness::new(MockProvider::text("Hello!"));
        let output = harness.run_agent(&simple_agent(), "Hi").await.unwrap();
        assert_eq!(output.status, AgentStatus::Completed);
        assert_lifecycle_events(&harness, &output);
    }

    #[tokio::test]
    async fn max_tokens_auto_continuation() {
        let provider = MockProvider::new(vec![
            truncated_response("partial response..."),
            text_response("...completed response"),
        ]);
        let harness = TestHarness::new(provider);
        let output = harness
            .run_agent(&simple_agent(), "write a long essay")
            .await
            .unwrap();

        assert_eq!(output.status, AgentStatus::Completed);
        assert_eq!(output.response_raw, "...completed response");
        assert_eq!(harness.provider().request_count(), 2);
        assert_lifecycle_events(&harness, &output);

        let req = &harness.provider().requests.lock().unwrap()[1];
        let has_continuation = req.messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains("cut off"),
                _ => false,
            }),
            _ => false,
        });
        assert!(has_continuation);
    }

    #[tokio::test]
    async fn max_tokens_continuation_events() {
        let provider =
            MockProvider::new(vec![truncated_response("partial"), text_response("done")]);
        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&simple_agent(), "go").await.unwrap();
        assert_lifecycle_events(&harness, &output);

        let truncated: Vec<u32> = harness
            .events()
            .all()
            .iter()
            .filter_map(|e| match &e.kind {
                AgentEventKind::OutputTruncated { turn } => Some(*turn),
                _ => None,
            })
            .collect();
        assert_eq!(truncated, vec![1]);
    }

    #[tokio::test]
    async fn token_budget_guard() {
        let mut response = tool_response("t", "c1", serde_json::json!({}));
        response.usage = TokenUsage {
            input_tokens: 5000,
            output_tokens: 100,
            ..Default::default()
        };
        let provider = MockProvider::new(vec![response, text_response("done")]);

        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_input_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(
            output.status,
            AgentStatus::BudgetExhausted {
                usage: 5000,
                limit: 4000
            }
        );
        assert_lifecycle_events(&harness, &output);
    }

    #[test]
    fn context_prompt_appended_after_metadata() {
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .context_prompt("user-provided context");

        let runtime = runtime_with_metadata("<environment>\ntest metadata\n</environment>");
        let spec = AgentSpec::compile(&agent, &runtime, None).unwrap();

        let ctx = spec.context_prompt.unwrap();
        let meta_pos = ctx.find("<environment>").expect("metadata missing");
        let user_pos = ctx
            .find("<context>\nuser-provided context\n</context>")
            .expect("context_prompt missing");
        assert!(
            meta_pos < user_pos,
            "metadata should appear before context_prompt:\n{ctx}"
        );
    }

    #[test]
    fn multiple_context_prompts_stacked() {
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .context_prompt("first block")
            .context_prompt("second block");

        let runtime = runtime_with_metadata("<environment>\nmetadata\n</environment>");
        let spec = AgentSpec::compile(&agent, &runtime, None).unwrap();

        let ctx = spec.context_prompt.unwrap();
        let meta_pos = ctx.find("<environment>").unwrap();
        let first_pos = ctx.find("<context>\nfirst block\n</context>").unwrap();
        let second_pos = ctx.find("<context>\nsecond block\n</context>").unwrap();
        assert!(meta_pos < first_pos, "metadata before first context");
        assert!(
            first_pos < second_pos,
            "first context before second context"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // keep_alive / idle wait — matrix-driven test suite
    //
    // Wake sources (W1-W6):   does a queue item wake an idle listener?
    // Lifecycle (L1-L6):      do one-shot / timeout / cancel / events behave?
    // ──────────────────────────────────────────────────────────────────────

    use crate::agent::queue::CommandQueue;

    const AGENT_NAME: &str = "test-agent";

    fn peer_msg(target: Option<&str>, from: &str, content: &str) -> QueuedCommand {
        QueuedCommand {
            content: content.into(),
            priority: QueuePriority::Next,
            source: CommandSource::PeerMessage {
                from: from.into(),
                summary: None,
            },
            agent_name: target.map(|s| s.into()),
        }
    }

    fn task_notification(target: Option<&str>, content: &str) -> QueuedCommand {
        QueuedCommand {
            content: content.into(),
            priority: QueuePriority::Next,
            source: CommandSource::TaskNotification {
                task_id: "task-1".into(),
            },
            agent_name: target.map(|s| s.into()),
        }
    }

    fn user_input(target: Option<&str>, content: &str) -> QueuedCommand {
        QueuedCommand {
            content: content.into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: target.map(|s| s.into()),
        }
    }

    /// Build a harness whose agent name is `AGENT_NAME` and which shares a
    /// fresh queue we return to the test for direct manipulation.
    fn listener_harness(provider: Arc<MockProvider>) -> (TestHarness, Arc<CommandQueue>) {
        let queue = Arc::new(CommandQueue::new());
        let harness = TestHarness::with_provider_and_queue(provider, queue.clone());
        (harness, queue)
    }

    /// Enqueue `cmd` after `delay_ms`. Used to drive wake-during-wait tests.
    fn enqueue_after(queue: &Arc<CommandQueue>, delay_ms: u64, cmd: QueuedCommand) {
        let q = queue.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            q.enqueue(cmd);
        });
    }

    fn two_text_responses() -> Arc<MockProvider> {
        Arc::new(MockProvider::new(vec![
            text_response("first"),
            text_response("second"),
        ]))
    }

    // ----- Wake-source matrix -----

    #[tokio::test]
    async fn wake_on_peer_message_targeted_at_me() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, peer_msg(Some(AGENT_NAME), "peer", "hi"));

        let agent = simple_agent().keep_alive_ms(5_000);
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(output.statistics.turns, 2, "peer message should wake the listener");
    }

    #[tokio::test]
    async fn wake_on_task_notification_broadcast() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, task_notification(None, "Task foo completed"));

        let agent = simple_agent().keep_alive_ms(5_000);
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "broadcast task notification should wake the listener"
        );
    }

    #[tokio::test]
    async fn wake_on_user_input_targeted_at_me() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, user_input(Some(AGENT_NAME), "hello"));

        let agent = simple_agent().keep_alive_ms(5_000);
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(output.statistics.turns, 2, "user input (targeted) should wake the listener");
    }

    #[tokio::test]
    async fn wake_on_user_input_broadcast() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, user_input(None, "anyone?"));

        let agent = simple_agent().keep_alive_ms(5_000);
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(output.statistics.turns, 2, "user input (broadcast) should wake the listener");
    }

    #[tokio::test]
    async fn ignores_peer_message_targeted_at_other_agent() {
        let (harness, queue) = listener_harness(Arc::new(MockProvider::text("done")));
        enqueue_after(&queue, 120, peer_msg(Some("someone-else"), "peer", "not for you"));

        let agent = simple_agent().keep_alive_ms(400);
        let t0 = std::time::Instant::now();
        let output = harness.run_agent(&agent, "hi").await.unwrap();
        let elapsed = t0.elapsed();

        assert_eq!(output.statistics.turns, 1, "message for another agent must not wake us");
        assert!(
            elapsed >= std::time::Duration::from_millis(300),
            "should have waited the full timeout, elapsed = {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn times_out_when_queue_stays_empty() {
        let (harness, _queue) = listener_harness(Arc::new(MockProvider::text("done")));

        let agent = simple_agent().keep_alive_ms(200);
        let t0 = std::time::Instant::now();
        let output = harness.run_agent(&agent, "hi").await.unwrap();
        let elapsed = t0.elapsed();

        assert_eq!(output.statistics.turns, 1);
        assert_eq!(output.status, AgentStatus::Completed);
        assert!(
            elapsed >= std::time::Duration::from_millis(150),
            "must have waited the timeout, elapsed = {elapsed:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(1500),
            "must have exited near the timeout, elapsed = {elapsed:?}"
        );
    }

    // ----- Lifecycle matrix -----

    #[tokio::test]
    async fn one_shot_when_keep_alive_unset() {
        let harness = TestHarness::new(MockProvider::text("done"));
        let agent = simple_agent();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(output.status, AgentStatus::Completed);
        assert_eq!(output.statistics.turns, 1);
    }

    #[tokio::test]
    async fn cancel_interrupts_keep_alive_ms() {
        let (harness, _queue) = listener_harness(Arc::new(MockProvider::text("done")));

        let cancel = harness.cancel_signal_for_test();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            cancel.store(true, Ordering::Relaxed);
        });

        let agent = simple_agent().keep_alive_ms(10_000);
        let t0 = std::time::Instant::now();
        let _ = harness.run_agent(&agent, "hi").await.unwrap();
        let elapsed = t0.elapsed();

        assert!(
            elapsed < std::time::Duration::from_millis(1_500),
            "cancel must interrupt the long timeout promptly, elapsed = {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn cancel_interrupts_keep_alive_unlimited() {
        let (harness, _queue) = listener_harness(Arc::new(MockProvider::text("done")));

        let cancel = harness.cancel_signal_for_test();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            cancel.store(true, Ordering::Relaxed);
        });

        let agent = simple_agent().keep_alive_unlimited();
        let t0 = std::time::Instant::now();
        let _ = harness.run_agent(&agent, "hi").await.unwrap();
        let elapsed = t0.elapsed();

        assert!(
            elapsed >= std::time::Duration::from_millis(100),
            "should have actually waited, elapsed = {elapsed:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(1_500),
            "should have exited promptly on cancel, elapsed = {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn idle_and_resumed_events_fire_in_order() {
        let (harness, queue) = listener_harness(two_text_responses());
        enqueue_after(&queue, 120, peer_msg(Some(AGENT_NAME), "peer", "hi"));

        let agent = simple_agent().keep_alive_ms(300);
        let _ = harness.run_agent(&agent, "hi").await.unwrap();

        let kinds: Vec<&'static str> = harness
            .events()
            .all()
            .iter()
            .filter_map(|e| match &e.kind {
                AgentEventKind::AgentIdle => Some("idle"),
                AgentEventKind::AgentResumed => Some("resumed"),
                _ => None,
            })
            .collect();
        let first_idle = kinds.iter().position(|k| *k == "idle").expect("idle fired");
        let first_resumed = kinds.iter().position(|k| *k == "resumed").expect("resumed fired");
        assert!(first_idle < first_resumed, "idle must precede resumed: {kinds:?}");
    }

    #[tokio::test]
    async fn drain_before_exit_picks_up_preloaded_message() {
        let (harness, queue) = listener_harness(two_text_responses());
        queue.enqueue(peer_msg(Some(AGENT_NAME), "peer", "pre-loaded"));

        // No keep_alive — the drain-before-exit safety net must still catch it.
        let agent = simple_agent();
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "drain-before-exit must inject the preloaded message and force a second turn"
        );
    }

    #[tokio::test]
    async fn drains_batch_of_messages_into_one_turn() {
        let (harness, queue) = listener_harness(two_text_responses());
        // Preload two messages. Both must arrive in a single drained turn,
        // not in two separate turns.
        queue.enqueue(peer_msg(Some(AGENT_NAME), "alice", "first"));
        queue.enqueue(peer_msg(Some(AGENT_NAME), "bob", "second"));

        let agent = simple_agent().keep_alive_ms(200);
        let output = harness.run_agent(&agent, "hi").await.unwrap();

        assert_eq!(
            output.statistics.turns, 2,
            "two pending messages should drain into ONE additional turn, not two"
        );
    }

    #[test]
    fn from_agent_uses_externally_supplied_queue_and_cancel() {
        let queue = Arc::new(CommandQueue::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let agent = Agent::new()
            .provider(Arc::new(MockProvider::text("x")))
            .instruction_prompt("")
            .cancel_signal(cancel.clone())
            .command_queue(queue.clone());

        let rt = LoopRuntime::from_agent(&agent).unwrap();

        assert!(
            Arc::ptr_eq(rt.command_queue.as_ref().unwrap(), &queue),
            "LoopRuntime should reuse the externally supplied queue"
        );
        assert!(
            Arc::ptr_eq(&rt.cancel_signal, &cancel),
            "LoopRuntime should reuse the externally supplied cancel signal"
        );
    }

    #[test]
    fn from_agent_allocates_default_queue_when_none_supplied() {
        let agent = Agent::new()
            .provider(Arc::new(MockProvider::text("x")))
            .instruction_prompt("");

        let rt = LoopRuntime::from_agent(&agent).unwrap();
        assert!(
            rt.command_queue.is_some(),
            "default queue must be allocated so peer messaging still works"
        );
    }
}

#[cfg(test)]
mod retry_and_events_tests {
    use super::*;
    use crate::error::AgenticError;
    use crate::provider::ProviderError;
    use crate::testutil::*;

    fn rate_limit_error() -> ProviderError {
        ProviderError::RateLimited {
            message: "rate limited".into(),
            status: 429,
            retry_after_ms: None,
        }
    }

    #[tokio::test]
    async fn retry_succeeds_after_rate_limit() {
        let provider = MockProvider::with_results(vec![
            Err(rate_limit_error()),
            Err(rate_limit_error()),
            Ok(text_response("hello")),
        ]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_backoff_ms(10);

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.response_raw, "hello");
        assert_eq!(harness.provider().request_count(), 3);
    }

    #[tokio::test]
    async fn no_retry_on_auth_error() {
        let provider = MockProvider::with_results(vec![Err(ProviderError::AuthenticationFailed {
            provider_message: "unauthorized".into(),
        })]);
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .max_request_retries(3)
            .request_retry_backoff_ms(10);

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(&agent, "go").await.unwrap_err();
        assert!(matches!(
            err,
            AgenticError::Provider(ProviderError::AuthenticationFailed { .. })
        ));
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn event_sequence_complete() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "done");
        let agent = Agent::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .tool(MockTool::new("read", true, "file contents"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        let names: Vec<&str> = events.iter().map(event_name).collect();
        assert_eq!(
            names,
            vec![
                "AgentStart",
                "TurnStart",
                "RequestStart",
                "RequestEnd",
                "TokenUsage",
                "ToolCallStart",
                "ToolCallEnd",
                "TurnEnd",
                "TurnStart",
                "RequestStart",
                "ResponseTextChunk",
                "RequestEnd",
                "TokenUsage",
                "AgentEnd",
                "TurnEnd",
            ]
        );
    }

    fn event_name(event: &AgentEvent) -> &'static str {
        match &event.kind {
            AgentEventKind::AgentStart { .. } => "AgentStart",
            AgentEventKind::AgentEnd { .. } => "AgentEnd",
            AgentEventKind::TurnStart { .. } => "TurnStart",
            AgentEventKind::TurnEnd { .. } => "TurnEnd",
            AgentEventKind::RequestStart { .. } => "RequestStart",
            AgentEventKind::RequestEnd { .. } => "RequestEnd",
            AgentEventKind::ResponseTextChunk { .. } => "ResponseTextChunk",
            AgentEventKind::ToolCallStart { .. } => "ToolCallStart",
            AgentEventKind::ToolCallEnd { .. } => "ToolCallEnd",
            AgentEventKind::TokenUsage { .. } => "TokenUsage",
            AgentEventKind::OutputTruncated { .. } => "OutputTruncated",
            AgentEventKind::CompactTriggered { .. } => "CompactTriggered",
            AgentEventKind::AgentIdle => "AgentIdle",
            AgentEventKind::AgentResumed => "AgentResumed",
        }
    }
}
