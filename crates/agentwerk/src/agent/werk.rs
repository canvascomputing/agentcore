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
//! - `Runtime` / `AgentSpec` / `LoopState` (pub(crate)) — the three internal
//!   structs the loop works with. See their doc comments for the split.
//! - `run_loop` (pub(crate) free function) — consumes the three structs.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::persistence::session::{EntryType, SessionStore, TranscriptEntry};
use crate::provider::model::ModelSpec;
use crate::provider::retry::{compute_delay, DEFAULT_BACKOFF_MS, DEFAULT_MAX_REQUEST_RETRIES};
use crate::provider::types::{ContentBlock, Message, ModelResponse, ResponseStatus, StreamEvent, TokenUsage};
use crate::provider::{CompletionRequest, LlmProvider, ToolChoice};
use crate::tools::{execute_tool_calls, SpawnAgentTool, Tool, ToolCall, ToolContext, ToolRegistry};
use crate::util::{generate_agent_name, now_millis};

use super::event::{Event, EventKind};
use super::output::{AgentOutput, OutputSchema, Statistics, Status, StructuredOutputTool};
use super::prompts::{
    self as prompts, interpolate, DEFAULT_BEHAVIOR_PROMPT, STRUCTURED_OUTPUT_TOOL_NAME,
};
use super::queue::{CommandQueue, QueuePriority};

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
    pub max_continuations: Option<u32>,
    pub max_schema_retries: Option<u32>,
    pub max_request_retries: u32,
    pub request_retry_backoff_ms: u64,
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
            max_continuations: Some(5),
            max_schema_retries: Some(10),
            max_request_retries: DEFAULT_MAX_REQUEST_RETRIES,
            request_retry_backoff_ms: DEFAULT_BACKOFF_MS,
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
            max_continuations: self.max_continuations,
            max_schema_retries: self.max_schema_retries,
            max_request_retries: self.max_request_retries,
            request_retry_backoff_ms: self.request_retry_backoff_ms,
        }
    }
}

/// Per-run configuration. Owned per `Agent` clone — no COW, cheap direct mutation.
#[derive(Clone, Default)]
pub(crate) struct AgentRuntime {
    pub provider: Option<Arc<dyn LlmProvider>>,
    pub instruction_prompt: String,
    pub template_variables: HashMap<String, Value>,
    pub working_directory: Option<PathBuf>,
    pub event_handler: Option<Arc<dyn Fn(Event) + Send + Sync>>,
    pub cancel_signal: Option<Arc<AtomicBool>>,
    pub session_dir: Option<PathBuf>,
}

impl Agent {
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
    pub fn model(self, m: impl Into<String>) -> Self {
        self.with_config(|c| c.model = ModelSpec::Exact(m.into()))
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

    /// Maximum auto-continuations when the model output is truncated. Default: 5.
    pub fn max_continuations(self, n: u32) -> Self {
        self.with_config(|c| c.max_continuations = Some(n))
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

    pub fn provider(self, p: Arc<dyn LlmProvider>) -> Self {
        self.with_runtime(|r| r.provider = Some(p))
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

    pub fn event_handler(self, h: Arc<dyn Fn(Event) + Send + Sync>) -> Self {
        self.with_runtime(|r| r.event_handler = Some(h))
    }

    pub fn cancel_signal(self, s: Arc<AtomicBool>) -> Self {
        self.with_runtime(|r| r.cancel_signal = Some(s))
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
        let runtime = Runtime::from_agent(self)?;
        let spec = AgentSpec::compile(self, &runtime, None)?;
        run_loop(Arc::new(runtime), Arc::new(spec), None).await
    }

    /// Crate-internal: run this agent as a child under a parent's `Runtime`.
    /// `parent_model` resolves `ModelSpec::Inherit` on the child. `description`
    /// becomes the `AgentStart` event's human-readable label.
    pub(crate) async fn run_child(
        &self,
        parent_runtime: &Runtime,
        parent_model: &str,
        description: Option<String>,
    ) -> Result<AgentOutput> {
        self.check_prompt_errors()?;
        let runtime = parent_runtime.inherit(self);
        let spec = AgentSpec::compile(self, &runtime, Some(parent_model))?;
        run_loop(Arc::new(runtime), Arc::new(spec), description).await
    }

    /// Test-only escape hatch: run with fully constructed Runtime + spec.
    #[cfg(test)]
    pub(crate) async fn run_with_parts(
        &self,
        runtime: Arc<Runtime>,
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
        self
    }
}

// ---------------------------------------------------------------------------
// Runtime — external services & I/O handles
// ---------------------------------------------------------------------------

/// External services and I/O handles for one run-tree (the root agent + every
/// sub-agent spawned transitively). Shared as `Arc<Runtime>`. Read-only from
/// the loop's perspective — mutability is only via the interior atomics and
/// mutexes inside.
pub(crate) struct Runtime {
    pub provider: Arc<dyn LlmProvider>,
    pub event_handler: Arc<dyn Fn(Event) + Send + Sync>,
    pub cancel_signal: Arc<AtomicBool>,
    pub working_directory: PathBuf,
    pub command_queue: Option<Arc<CommandQueue>>,
    pub session_store: Option<Arc<Mutex<SessionStore>>>,
    pub metadata: Option<String>,
}

impl Runtime {
    /// Build the root `Runtime` from an Agent's per-run fields plus reasonable defaults.
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

        let event_handler: Arc<dyn Fn(Event) + Send + Sync> = agent
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
        // notifications back to the parent.
        let command_queue = Some(Arc::new(CommandQueue::new()));

        let session_store = agent.runtime.session_dir.as_ref().map(|dir| {
            let store = SessionStore::new(dir, &generate_agent_name("session"));
            Arc::new(Mutex::new(store))
        });

        let metadata = Some(prompts::collect_metadata(&working_directory));

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

    /// Build a child Runtime: inherit externals from the parent, let the
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

// ---------------------------------------------------------------------------
// AgentSpec — compiled per-agent blueprint
// ---------------------------------------------------------------------------

/// Compiled blueprint for one `Agent`'s execution. Built once by
/// `AgentSpec::compile` at the start of a run and never mutated. Per-agent:
/// child agents compile their own spec.
pub(crate) struct AgentSpec {
    pub name: String,
    pub model: String,
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
    pub max_continuations: Option<u32>,
    pub max_schema_retries: Option<u32>,
    pub max_request_retries: u32,
    pub request_retry_backoff_ms: u64,
}

impl AgentSpec {
    pub(crate) fn compile(
        agent: &Agent,
        runtime: &Runtime,
        fallback_model: Option<&str>,
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
            (spec, fallback) => spec.resolve(fallback.unwrap_or("")),
        };

        let (tools, tool_choice) = compile_tools(agent);

        let system_prompt = compile_system_prompt(agent);
        let instruction_prompt =
            interpolate(&agent.runtime.instruction_prompt, &agent.runtime.template_variables);
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
            max_continuations: agent.config.max_continuations,
            max_schema_retries: agent.config.max_schema_retries,
            max_request_retries: agent.config.max_request_retries,
            request_retry_backoff_ms: agent.config.request_retry_backoff_ms,
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

    // If the agent demands structured output, register the matching tool and
    // (when no other tools exist) force the LLM to call it.
    let tool_choice = if let Some(ref schema) = agent.config.output_schema {
        let user_tools_empty = tools.is_empty();
        tools.register(StructuredOutputTool::new(schema.clone()));
        if user_tools_empty {
            Some(ToolChoice::Specific {
                name: STRUCTURED_OUTPUT_TOOL_NAME.into(),
            })
        } else {
            None
        }
    } else {
        None
    };

    (Arc::new(tools), tool_choice)
}

fn compile_system_prompt(agent: &Agent) -> String {
    let mut s = interpolate(&agent.config.identity_prompt, &agent.runtime.template_variables);
    if !agent.config.behavior_prompt.is_empty() {
        s.push_str("\n\n");
        s.push_str(&agent.config.behavior_prompt);
    }
    if agent.config.output_schema.is_some() {
        s.push_str(prompts::STRUCTURED_OUTPUT_INSTRUCTION);
    }
    s
}

fn compile_context_prompt(runtime: &Runtime, agent: &Agent) -> Option<String> {
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
    pub continuations: u32,
    pub paused: bool,
    pub discovered_tools: HashSet<String>,
    pub structured_output: Option<Value>,
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
// run_loop — the execution loop itself
// ---------------------------------------------------------------------------

/// The agent execution loop.
///
/// Each iteration is one "turn": call the LLM, process its response, decide
/// whether to continue.
///
/// ```text
/// loop {
///     // 1. GUARD CHECK — can we even start this turn?
///     //    Check: cancel_signal, max_turns, token_budget
///     //    If any guard fires → exit with partial output + reason
///
///     // 2. CALL LLM — get the model's response
///     //    The model returns: text content, tool_use blocks, and a status
///
///     // 3. COMPLETION DECISION — based on what the model returned:
///     //
///     //    ┌─ ResponseStatus::ToolUse + non-empty tool calls
///     //    │   → Execute tools, push results to messages, CONTINUE loop
///     //    │
///     //    ├─ ResponseStatus::OutputTruncated + no tool calls
///     //    │   → Model was CUT OFF mid-response (did NOT decide to stop)
///     //    │   → Send continuation message, CONTINUE loop
///     //    │   → Give up after max_continuations
///     //    │
///     //    ├─ ResponseStatus::ContextWindowExceeded
///     //    │   → Input too large — EXIT with error
///     //    │
///     //    ├─ ResponseStatus::Refused
///     //    │   → Model declined on safety grounds — EXIT with error
///     //    │
///     //    └─ ResponseStatus::EndTurn / StopSequence + no tool calls
///     //        → Model DECIDED it is finished → EXIT with output
/// }
/// ```
///
/// **The fundamental signal is: the model responds without making tool calls
/// and status is `EndTurn`.** This means the model has nothing left to do —
/// no files to read, no commands to run, no sub-agents to spawn. It has
/// completed its task (or believes it has).
///
/// `OutputTruncated` is NOT a completion signal — it means the model ran out
/// of output space. `ToolUse` is NOT a completion signal — it means the model
/// needs tools to make progress. Only `EndTurn` (with no tool calls) means
/// the model chose to stop.
pub(crate) fn run_loop(
    runtime: Arc<Runtime>,
    spec: Arc<AgentSpec>,
    description: Option<String>,
) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send>> {
    Box::pin(async move {
        runtime.provider.prewarm().await;
        let mut state = LoopState::init(&spec);
        record_transcript(&runtime, EntryType::UserMessage, state.messages.last().unwrap(), None);
        emit(&runtime, &spec.name, EventKind::AgentStart { description });

        loop {
            // Guards: cancel signal, max turns, token budget
            if let Some(status) = check_guards(&runtime, &spec, &state) {
                return Ok(finish_early(&runtime, &spec, &mut state, status));
            }

            state.turn += 1;
            let turn = state.turn;

            // LLM request
            let model = spec.model.clone();
            emit(&runtime, &spec.name, EventKind::TurnStart { turn });
            emit(&runtime, &spec.name, EventKind::RequestStart { model: model.clone() });
            let response = call_llm_with_retry(&runtime, &spec, &state).await?;
            emit(&runtime, &spec.name, EventKind::RequestEnd { model });
            record_usage(&runtime, &spec, &mut state, &response);

            // Record assistant message
            let (text, tool_calls) = parse_response(&response);
            state.messages.push(Message::Assistant { content: response.content.clone() });
            record_transcript(
                &runtime, EntryType::AssistantMessage,
                state.messages.last().unwrap(), Some((&response.usage, &response.model)),
            );

            // Decision: what did the model return?
            let has_tool_calls = response.status == ResponseStatus::ToolUse && !tool_calls.is_empty();
            let is_truncated = response.status == ResponseStatus::OutputTruncated && tool_calls.is_empty();

            if has_tool_calls {
                let results = execute_tools(&runtime, &spec, &mut state, &tool_calls).await;
                state.messages.push(Message::User { content: results });
                record_transcript(&runtime, EntryType::ToolResult, state.messages.last().unwrap(), None);
                drain_command_queue(&runtime, &spec, &mut state);
            } else if is_truncated {
                emit(&runtime, &spec.name, EventKind::TurnPaused { turn });
                state.paused = true;
                state.continuations += 1;
                if spec.max_continuations.is_some_and(|limit| state.continuations > limit) {
                    return Ok(finish_early(&runtime, &spec, &mut state, Status::OutputLimitReached));
                }
                debug_assert!(state.paused, "TurnResumed without preceding TurnPaused");
                state.paused = false;
                emit(&runtime, &spec.name, EventKind::TurnResumed { turn, continuation: state.continuations });
                state.messages.push(Message::user(prompts::MAX_TOKENS_CONTINUATION));
            } else if let Some(output) = try_finish(&runtime, &spec, &mut state, text)? {
                emit(&runtime, &spec.name, EventKind::TurnEnd { turn });
                return Ok(output);
            }

            emit(&runtime, &spec.name, EventKind::TurnEnd { turn });
        }
    })
}

fn finish_early(runtime: &Runtime, spec: &AgentSpec, state: &mut LoopState, status: Status) -> AgentOutput {
    let text = last_assistant_text(&state.messages);
    emit(runtime, &spec.name, EventKind::AgentEnd { turns: state.turn, status: status.clone() });
    build_output(state, text, status)
}

fn check_guards(runtime: &Runtime, spec: &AgentSpec, state: &LoopState) -> Option<Status> {
    if runtime.cancel_signal.load(Ordering::Relaxed) {
        return Some(Status::Cancelled);
    }
    if let Some(limit) = spec.max_turns {
        if state.turn >= limit {
            return Some(Status::TurnLimitReached { limit });
        }
    }
    if let Some(limit) = spec.max_input_tokens {
        if state.total_usage.input_tokens >= limit {
            return Some(Status::BudgetExhausted {
                usage: state.total_usage.input_tokens,
                limit,
            });
        }
    }
    None
}

async fn call_llm(runtime: &Runtime, spec: &AgentSpec, state: &LoopState) -> Result<ModelResponse> {
    let tool_defs = spec.tools.definitions(&state.discovered_tools);
    let request = CompletionRequest {
        model: spec.model.clone(),
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
            event_handler(Event::new(
                agent_name.clone(),
                EventKind::ResponseTextChunk { content: text.clone() },
            ));
        }
    });

    runtime.provider.complete_streaming(request, on_event).await
}

async fn call_llm_with_retry(
    runtime: &Runtime,
    spec: &AgentSpec,
    state: &LoopState,
) -> Result<ModelResponse> {
    let mut last_err = None;
    for attempt in 0..=spec.max_request_retries {
        match call_llm(runtime, spec, state).await {
            Ok(response) => return Ok(response),
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

fn record_usage(runtime: &Runtime, spec: &AgentSpec, state: &mut LoopState, response: &ModelResponse) {
    state.total_usage += &response.usage;
    state.request_count += 1;
    emit(
        runtime,
        &spec.name,
        EventKind::TokenUsage {
            model: response.model.clone(),
            usage: response.usage.clone(),
        },
    );
}

fn parse_response(response: &ModelResponse) -> (String, Vec<ToolCall>) {
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

fn try_finish(
    runtime: &Runtime,
    spec: &AgentSpec,
    state: &mut LoopState,
    text: String,
) -> Result<Option<AgentOutput>> {
    if spec.output_schema.is_some() && state.structured_output.is_none() {
        state.schema_retries += 1;
        if let Some(limit) = spec.max_schema_retries {
            if state.schema_retries > limit {
                return Err(AgenticError::SchemaRetryExhausted { retries: limit });
            }
        }
        state.messages.push(Message::user(prompts::STRUCTURED_OUTPUT_RETRY));
        return Ok(None);
    }

    emit(runtime, &spec.name, EventKind::AgentEnd { turns: state.turn, status: Status::Completed });
    Ok(Some(build_output(state, text, Status::Completed)))
}

async fn execute_tools(
    runtime: &Arc<Runtime>,
    spec: &Arc<AgentSpec>,
    state: &mut LoopState,
    calls: &[ToolCall],
) -> Vec<ContentBlock> {
    state.tool_call_count += calls.len() as u64;
    for call in calls {
        emit(
            runtime,
            &spec.name,
            EventKind::ToolCallStart {
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

    let raw = execute_tool_calls(calls, &tool_ctx).await;
    let mut blocks = Vec::with_capacity(raw.len());
    for (block, result) in raw {
        if let Some(v) = result.structured_output {
            state.structured_output = Some(v);
        }
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
                &spec.name,
                EventKind::ToolCallEnd {
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

fn drain_command_queue(runtime: &Runtime, spec: &AgentSpec, state: &mut LoopState) {
    let Some(queue) = runtime.command_queue.as_ref() else {
        return;
    };
    while let Some(cmd) = queue.dequeue_if(Some(&spec.name), |c| c.priority != QueuePriority::Later) {
        state.messages.push(Message::user(cmd.content));
    }
}

fn emit(runtime: &Runtime, agent_name: &str, kind: EventKind) {
    (runtime.event_handler)(Event::new(agent_name, kind));
}

fn record_transcript(
    runtime: &Runtime,
    entry_type: EntryType,
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

fn build_output(state: &mut LoopState, text: String, status: Status) -> AgentOutput {
    AgentOutput {
        response: state.structured_output.take(),
        response_raw: text,
        statistics: Statistics {
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
        let provider = MockProvider::tool_then_text(
            "echo_tool",
            serde_json::json!({"text": "ping"}),
            "Done!",
        );
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
            .name("test").model("mock").identity_prompt("")
            .max_turns(2)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.status, Status::TurnLimitReached { limit: 2 });
        assert_eq!(output.statistics.turns, 2);
    }

    #[tokio::test]
    async fn guard_cancellation() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("done"),
        ]);
        let agent = Agent::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        harness.cancel();
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.status, Status::Cancelled);
    }

    #[tokio::test]
    async fn template_variable_interpolates_in_system_prompt() {
        let provider = MockProvider::text("Answer about rust");
        let agent = Agent::new()
            .name("test").model("mock")
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
            .name("assistant").model("mock").identity_prompt("")
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
            .name("test").model("mock").identity_prompt("")
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
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("t", false, "ok"));

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "later task".into(),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification { task_id: "42".into() },
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
            .name("test").model("mock").identity_prompt("")
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
        let provider = MockProvider::new(vec![
            tool_response(STRUCTURED_OUTPUT_TOOL_NAME, "so1", schema_input),
            text_response("done"),
        ]);
        let agent = Agent::new()
            .name("classifier").model("mock").identity_prompt("Classify.")
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
            .name("test").model("mock").identity_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .max_schema_retries(3);

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(&agent, "go").await.unwrap_err();
        assert!(matches!(err, AgenticError::SchemaRetryExhausted { retries: 3 }));
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
        assert!(req.tools.iter().any(|t| t.name == "spawn_agent"),
            ".sub_agents() should register spawn_agent automatically");
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
            ModelSpec::Exact(m) => assert_eq!(m, "overridden"),
            _ => panic!("expected Exact model"),
        }
    }

    fn runtime_with_metadata(meta: &str) -> Runtime {
        Runtime {
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
        assert_eq!(output.status, Status::Completed);
    }

    #[tokio::test]
    async fn max_tokens_auto_continuation() {
        let provider = MockProvider::new(vec![
            truncated_response("partial response..."),
            text_response("...completed response"),
        ]);
        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&simple_agent(), "write a long essay").await.unwrap();

        assert_eq!(output.status, Status::Completed);
        assert_eq!(output.response_raw, "...completed response");
        assert_eq!(harness.provider().request_count(), 2);

        // Verify continuation message was injected
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
    async fn max_tokens_continuation_exhausted() {
        let provider = MockProvider::new(vec![
            truncated_response("part 1"),
            truncated_response("part 2"),
            truncated_response("part 3"),
        ]);
        let agent = Agent::new()
            .name("test").model("mock").identity_prompt("")
            .max_continuations(2);
        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();

        assert_eq!(output.status, Status::OutputLimitReached);
    }

    #[tokio::test]
    async fn max_tokens_continuation_events() {
        let provider = MockProvider::new(vec![
            truncated_response("partial"),
            text_response("done"),
        ]);
        let harness = TestHarness::new(provider);
        harness.run_agent(&simple_agent(), "go").await.unwrap();

        let events = harness.events().all();
        let paused: Vec<u32> = events.iter().filter_map(|e| match &e.kind {
            EventKind::TurnPaused { turn } => Some(*turn),
            _ => None,
        }).collect();
        let resumed: Vec<(u32, u32)> = events.iter().filter_map(|e| match &e.kind {
            EventKind::TurnResumed { turn, continuation } => Some((*turn, *continuation)),
            _ => None,
        }).collect();
        assert_eq!(paused, vec![1]);
        assert_eq!(resumed, vec![(1, 1)]);
    }

    #[tokio::test]
    async fn token_budget_guard() {
        let mut response = tool_response("t", "c1", serde_json::json!({}));
        response.usage = TokenUsage { input_tokens: 5000, output_tokens: 100, ..Default::default() };
        let provider = MockProvider::new(vec![response, text_response("done")]);

        let agent = Agent::new()
            .name("test").model("mock").identity_prompt("")
            .max_input_tokens(4000)
            .tool(MockTool::new("t", false, "ok"));

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.status, Status::BudgetExhausted { usage: 5000, limit: 4000 });
    }

    #[tokio::test]
    async fn agent_end_event_carries_status() {
        let provider = MockProvider::text("done");
        let harness = TestHarness::new(provider);
        harness.run_agent(&simple_agent(), "go").await.unwrap();

        let ends = harness.events().agent_ends();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0].2, Status::Completed);
    }

    #[tokio::test]
    async fn turn_resumed_always_follows_turn_paused() {
        let provider = MockProvider::new(vec![
            truncated_response("p1"),
            truncated_response("p2"),
            text_response("done"),
        ]);
        let harness = TestHarness::new(provider);
        harness.run_agent(&simple_agent(), "go").await.unwrap();

        let events = harness.events().all();
        for (i, event) in events.iter().enumerate() {
            if matches!(event.kind, EventKind::TurnResumed { .. }) {
                assert!(
                    i > 0 && matches!(events[i - 1].kind, EventKind::TurnPaused { .. }),
                    "TurnResumed at index {i} not preceded by TurnPaused"
                );
            }
        }
    }

    #[test]
    fn context_prompt_appended_after_metadata() {
        let agent = Agent::new()
            .name("test").model("mock").identity_prompt("")
            .context_prompt("user-provided context");

        let runtime = runtime_with_metadata("<environment>\ntest metadata\n</environment>");
        let spec = AgentSpec::compile(&agent, &runtime, None).unwrap();

        let ctx = spec.context_prompt.unwrap();
        let meta_pos = ctx.find("<environment>").expect("metadata missing");
        let user_pos = ctx.find("<context>\nuser-provided context\n</context>")
            .expect("context_prompt missing");
        assert!(meta_pos < user_pos,
            "metadata should appear before context_prompt:\n{ctx}");
    }

    #[test]
    fn multiple_context_prompts_stacked() {
        let agent = Agent::new()
            .name("test").model("mock").identity_prompt("")
            .context_prompt("first block")
            .context_prompt("second block");

        let runtime = runtime_with_metadata("<environment>\nmetadata\n</environment>");
        let spec = AgentSpec::compile(&agent, &runtime, None).unwrap();

        let ctx = spec.context_prompt.unwrap();
        let meta_pos = ctx.find("<environment>").unwrap();
        let first_pos = ctx.find("<context>\nfirst block\n</context>").unwrap();
        let second_pos = ctx.find("<context>\nsecond block\n</context>").unwrap();
        assert!(meta_pos < first_pos, "metadata before first context");
        assert!(first_pos < second_pos, "first context before second context");
    }
}

#[cfg(test)]
mod retry_and_events_tests {
    use super::*;
    use crate::error::AgenticError;
    use crate::testutil::*;

    fn rate_limit_error() -> AgenticError {
        AgenticError::Api {
            message: "rate limited".into(),
            status: Some(429),
            retryable: true,
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
            .name("test").model("mock").identity_prompt("")
            .max_request_retries(3).request_retry_backoff_ms(10);

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert_eq!(output.response_raw, "hello");
        assert_eq!(harness.provider().request_count(), 3);
    }

    #[tokio::test]
    async fn no_retry_on_auth_error() {
        let provider = MockProvider::with_results(vec![Err(AgenticError::Api {
            message: "unauthorized".into(),
            status: Some(401),
            retryable: false,
            retry_after_ms: None,
        })]);
        let agent = Agent::new()
            .name("test").model("mock").identity_prompt("")
            .max_request_retries(3).request_retry_backoff_ms(10);

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(&agent, "go").await.unwrap_err();
        assert!(matches!(err, AgenticError::Api { status: Some(401), .. }));
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn event_sequence_complete() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "done");
        let agent = Agent::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("read", true, "file contents"));

        let harness = TestHarness::new(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let events = harness.events().all();
        let names: Vec<&str> = events.iter().map(event_name).collect();
        assert_eq!(names, vec![
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
        ]);
    }

    fn event_name(event: &Event) -> &'static str {
        match &event.kind {
            EventKind::AgentStart { .. } => "AgentStart",
            EventKind::AgentEnd { .. } => "AgentEnd",
            EventKind::TurnStart { .. } => "TurnStart",
            EventKind::TurnEnd { .. } => "TurnEnd",
            EventKind::RequestStart { .. } => "RequestStart",
            EventKind::RequestEnd { .. } => "RequestEnd",
            EventKind::ResponseTextChunk { .. } => "ResponseTextChunk",
            EventKind::ToolCallStart { .. } => "ToolCallStart",
            EventKind::ToolCallEnd { .. } => "ToolCallEnd",
            EventKind::TokenUsage { .. } => "TokenUsage",
            EventKind::TurnPaused { .. } => "TurnPaused",
            EventKind::TurnResumed { .. } => "TurnResumed",
        }
    }
}
