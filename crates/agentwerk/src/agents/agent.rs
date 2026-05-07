//! Agent: identity + prompt parts + provider/model + a bound ticket
//! system. Holds a `Weak<TicketSystem>`; `Default` produces a dangling
//! `Weak`, and `tickets.agent(agent)` (or `agent.ticket_system(&shared)`)
//! stamps the system's `Weak<Self>` onto the agent. The loop upgrades it
//! once at the start of `handle_tickets` and accesses `tickets`,
//! `policies`, `stats`, and `interrupt_signal` through the resulting
//! `Arc<TicketSystem>`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use serde::Serialize;

use crate::event::{default_logger, Event};
use crate::prompts::{default_context, PromptBuilder, Section};
use crate::providers::{Provider, ProviderToolDefinition};
use crate::tools::{MemoryTool, ToolLike, ToolRegistry, WriteResultTool};

use super::memory::{IntoMemory, Memory};

use super::policy::Policies;
use super::stats::Stats;
use super::tickets::{TicketResult, Ticket, TicketSystem};

static AGENT_COUNTER: AtomicU64 = AtomicU64::new(0);

fn default_agent_name() -> String {
    let n = AGENT_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("agent-{n}")
}

#[derive(Clone)]
pub struct Agent {
    pub(crate) name: String,
    provider: Option<Arc<dyn Provider>>,
    model: Option<String>,
    role: Option<String>,
    context: Option<String>,
    pub(crate) labels: Vec<String>,
    template_variables: Vec<(String, String)>,
    tools: ToolRegistry,
    working_dir: Option<PathBuf>,
    event_handler: Option<Arc<dyn Fn(Event) + Send + Sync>>,
    memory: Option<Arc<Memory>>,
    pub(crate) ticket_system: Weak<TicketSystem>,
}

impl Default for Agent {
    fn default() -> Self {
        let mut tools = ToolRegistry::default();
        tools.register(WriteResultTool);
        Self {
            name: default_agent_name(),
            provider: None,
            model: None,
            role: None,
            context: None,
            labels: Vec::new(),
            template_variables: Vec::new(),
            tools,
            working_dir: None,
            event_handler: None,
            memory: None,
            ticket_system: Weak::new(),
        }
    }
}

impl Agent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, n: impl Into<String>) -> Self {
        self.name = n.into();
        self
    }

    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }

    pub fn role(mut self, r: impl Into<String>) -> Self {
        self.role = Some(r.into());
        self
    }

    pub fn context(mut self, c: impl Into<String>) -> Self {
        self.context = Some(c.into());
        self
    }

    /// Add a single label to the agent's scope. Use [`Self::labels`] to
    /// add several at once.
    pub fn label(mut self, l: impl Into<String>) -> Self {
        self.labels.push(l.into());
        self
    }

    /// Add many labels at once.
    pub fn labels<I, S>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.labels.extend(iter.into_iter().map(Into::into));
        self
    }

    /// Bind `{key}` to `value`. The placeholder is substituted in the
    /// agent's `role`, `context`, and any string-typed `Ticket::task`
    /// enqueued through this agent. Unresolved placeholders are left
    /// verbatim.
    pub fn template_variable(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.template_variables.push((key.into(), value.into()));
        self
    }

    /// Bind many `{key} → value` pairs at once.
    pub fn template_variables<I, K, V>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.template_variables
            .extend(vars.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Register a single tool the agent may call.
    pub fn tool(mut self, tool: impl ToolLike + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    /// Register many tools at once.
    pub fn tools<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: ToolLike + 'static,
    {
        for t in tools {
            self.tools.register(t);
        }
        self
    }

    /// Working directory tools resolve filesystem paths against. Defaults
    /// to the process's current directory when unset.
    pub fn working_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(p.into());
        self
    }

    /// Install an event observer. The handler must be cheap and non-blocking.
    /// When not set, [`default_logger`] is used.
    pub fn event_handler(mut self, h: Arc<dyn Fn(Event) + Send + Sync>) -> Self {
        self.event_handler = Some(h);
        self
    }

    /// Bind this agent to a `Memory` store. Accepts either an `&Arc<Memory>`
    /// (to share one store across multiple agents, the same way
    /// `ticket_system(&shared)` shares a queue) or a path to a directory the
    /// store should be rooted at (opens a fresh store under the hood, mirroring
    /// `TicketSystem::workspace(dir)`). Registers `MemoryTool` on the agent's
    /// tool registry and arranges for the store's entries to be concatenated
    /// into the system prompt under `## Memory` at the top of every ticket.
    /// Off by default; each ticket starts without a memory section when no
    /// store is bound. Panics on IO failure when opening from a path.
    pub fn memory<M: IntoMemory>(mut self, store: M) -> Self {
        let store = store.into_memory().expect("open memory store");
        self.tools.register(MemoryTool::new(Arc::clone(&store)));
        self.memory = Some(store);
        self
    }

    pub(super) fn memory_handle(&self) -> Option<Arc<Memory>> {
        self.memory.clone()
    }

    /// Bind this agent to a shared `TicketSystem`. Drains any tickets
    /// the agent had already enqueued in its prior store into `sys`,
    /// stamps `sys`'s `Weak<Self>` onto `self.ticket_system`, and
    /// registers a clone of `self` into `sys`'s agents list so the
    /// loop will dispatch this agent at `run` / `run_dry` time.
    pub fn ticket_system(mut self, sys: &Arc<TicketSystem>) -> Self {
        sys.bind_agent(&mut self);
        self
    }

    pub(super) fn get_name(&self) -> &str {
        &self.name
    }

    pub(super) fn resolve_event_handler(&self) -> Arc<dyn Fn(Event) + Send + Sync> {
        self.event_handler.clone().unwrap_or_else(default_logger)
    }

    /// Returns true when the agent's label scope intersects the ticket's
    /// labels, OR when one of the ticket's labels equals the agent's name
    /// (name acts as an implicit self-label, so labelling a ticket with an
    /// agent's name pins it to that agent). Empty agent labels mean
    /// "default scope": tickets with no labels match.
    pub(super) fn handles_labels(&self, ticket_labels: &[String]) -> bool {
        if ticket_labels.iter().any(|l| l == &self.name) {
            return true;
        }
        if self.labels.is_empty() {
            ticket_labels.is_empty()
        } else {
            self.labels
                .iter()
                .any(|l| ticket_labels.iter().any(|t| t == l))
        }
    }

    pub(super) fn tool_definitions(&self) -> Vec<ProviderToolDefinition> {
        self.tools.definitions()
    }

    pub(super) fn tool_registry(&self) -> &ToolRegistry {
        &self.tools
    }

    pub(super) fn working_dir_or_default(&self) -> PathBuf {
        self.working_dir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    pub(super) fn provider_handle(&self) -> Arc<dyn Provider> {
        Arc::clone(
            self.provider
                .as_ref()
                .expect("Agent::run requires .provider(...) to be set"),
        )
    }

    pub(super) fn model_str(&self) -> &str {
        self.model
            .as_deref()
            .expect("Agent::run requires .model(...) to be set")
    }

    /// Build the system prompt. `memory` is the body the loop captured at the
    /// top of the current ticket, or `None` if [`Self::memory`] was not set.
    /// Tests may pass `None`.
    pub(super) fn system_prompt(&self, memory: Option<&str>) -> String {
        let mut b = PromptBuilder::default();
        if let Some(role) = &self.role {
            b = b.role(self.interpolate(role));
        }
        if let Some(snap) = memory.filter(|s| !s.is_empty()) {
            b = b.memory(snap.to_string());
        }
        b.build().system
    }

    /// Render the context block pushed as the first user message in the
    /// loop. Falls back to [`default_context`] (working directory, platform,
    /// OS version, date, plus a `… remaining` line for each configured
    /// policy budget) when [`Self::context`] was not set. A custom context
    /// is left byte-exact: `policies` and `stats` are ignored on that
    /// branch.
    pub(super) fn context_message(
        &self,
        policies: &Policies,
        stats: &Stats,
    ) -> Option<String> {
        match &self.context {
            Some(body) => Some(Section::context(self.interpolate(body)).render()),
            None => Some(default_context(
                &self.working_dir_or_default(),
                policies,
                stats,
            )),
        }
    }

    fn interpolate(&self, s: &str) -> String {
        if self.template_variables.is_empty() {
            return s.to_string();
        }
        let mut out = s.to_string();
        for (key, value) in &self.template_variables {
            out = out.replace(&format!("{{{key}}}"), value);
        }
        out
    }

    /// Enqueue a ticket carrying `task` as its body. Always available
    /// (the agent has a bound ticket system from construction onward).
    /// Returns `&Self` for chaining.
    pub fn task<T: Serialize>(&self, task: T) -> &Self {
        let ticket = Ticket::new(task);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a ticket carrying `task` and attached to `label` for
    /// Path B routing. To pin a ticket directly to an agent, label it
    /// with the agent's name: `agent.ticket(Ticket::new(...).label("alice"))`.
    pub fn task_labeled<T: Serialize>(&self, task: T, label: impl Into<String>) -> &Self {
        let ticket = Ticket::new(task).label(label);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a ticket carrying `task` plus a `schema` the agent's final
    /// `done` result must validate against.
    pub fn task_schema<T: Serialize>(&self, task: T, schema: crate::schemas::Schema) -> &Self {
        let ticket = Ticket::new(task).schema(schema);
        self.dispatch(ticket);
        self
    }

    /// `task_schema` + `task_labeled` combined.
    pub fn task_schema_labeled<T: Serialize>(
        &self,
        task: T,
        schema: crate::schemas::Schema,
        label: impl Into<String>,
    ) -> &Self {
        let ticket = Ticket::new(task).schema(schema).label(label);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a fully-built `Ticket`. System-managed fields (key,
    /// reporter, created_at, status, result) are overwritten. To pin the
    /// ticket to a specific agent, label it with the agent's name.
    pub fn ticket(&self, ticket: Ticket) -> &Self {
        self.dispatch(ticket);
        self
    }

    fn dispatch(&self, mut ticket: Ticket) {
        let sys = self
            .ticket_system
            .upgrade()
            .expect("Agent::task requires a bound TicketSystem");
        if let serde_json::Value::String(s) = &ticket.task {
            ticket.task = serde_json::Value::String(self.interpolate(s));
        }
        sys.insert(ticket, self.name.clone());
    }

    /// Drive the agent's bound `TicketSystem` until the queue settles
    /// Start the agent loop on a background tokio task and return a
    /// [`Running`] handle. Forwards to the bound `TicketSystem`.
    /// Tickets queued afterwards are picked up within
    /// ~`IDLE_POLL_INTERVAL`. Finish with [`Running::run_dry`] to wait
    /// for the queue to drain.
    pub fn run(&self) -> super::running::Running {
        let sys = self
            .ticket_system
            .upgrade()
            .expect("Agent::run requires a bound TicketSystem");
        sys.run()
    }

    /// Start a background run and wait for the queue to drain.
    /// Returns every finished ticket's [`TicketResult`], in creation
    /// order. Equivalent to `self.run().run_dry().await`.
    pub async fn run_dry(&self) -> Vec<TicketResult> {
        let sys = self
            .ticket_system
            .upgrade()
            .expect("Agent::run_dry requires a bound TicketSystem");
        sys.run_dry().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::stats::LoopStats;

    #[test]
    fn handles_labels_default_scope_only_picks_unlabeled_tickets() {
        let agent = Agent::new();
        assert!(agent.handles_labels(&[]));
        assert!(!agent.handles_labels(&["research".into()]));
    }

    #[test]
    fn handles_labels_with_labels_intersects_ticket_labels() {
        let agent = Agent::new().label("research").label("urgent");
        assert!(agent.handles_labels(&["research".into()]));
        assert!(agent.handles_labels(&["urgent".into(), "other".into()]));
        assert!(!agent.handles_labels(&["report".into()]));
        assert!(!agent.handles_labels(&[]));
    }

    #[test]
    fn handles_labels_matches_when_ticket_label_equals_agent_name() {
        // Default-scope agent: a ticket labelled with the agent's name
        // routes here even though the agent has no other labels.
        let agent = Agent::new().name("alice");
        assert!(agent.handles_labels(&["alice".into()]));
        assert!(agent.handles_labels(&["alice".into(), "other".into()]));
        // Same holds when the agent does carry topical labels.
        let agent = Agent::new().name("alice").label("math");
        assert!(agent.handles_labels(&["alice".into()]));
        assert!(agent.handles_labels(&["math".into()]));
        assert!(!agent.handles_labels(&["report".into()]));
    }

    #[test]
    fn get_name_returns_configured_name() {
        let agent = Agent::new().name("alice");
        assert_eq!(agent.get_name(), "alice");
    }

    #[test]
    fn default_name_is_unique_per_agent() {
        let a = Agent::new();
        let b = Agent::new();
        assert_ne!(a.get_name(), b.get_name());
        assert!(a.get_name().starts_with("agent-"));
        assert!(b.get_name().starts_with("agent-"));
    }

    #[test]
    fn context_message_falls_back_to_default_when_unset() {
        let agent = Agent::new().role("R");
        let policies = Policies::default();
        let stats = Stats::new();
        let rendered = agent
            .context_message(&policies, &stats)
            .expect("default context");
        assert!(rendered.starts_with("## Context\n\n"));
        assert!(rendered.contains("- Working directory: "));
        assert!(rendered.contains("- Platform: "));
        assert!(rendered.contains("- Date: "));
    }

    #[test]
    fn context_message_renders_h2_heading_when_set() {
        let agent = Agent::new().context("- Working directory: /tmp");
        let policies = Policies::default();
        let stats = Stats::new();
        assert_eq!(
            agent.context_message(&policies, &stats).as_deref(),
            Some("## Context\n\n- Working directory: /tmp"),
        );
    }

    #[test]
    fn context_message_appends_runtime_lines_when_policy_budgets_are_set() {
        let agent = Agent::new().working_dir("/tmp/check");
        let policies = Policies {
            max_steps: Some(3),
            max_input_tokens: Some(1_000),
            ..Policies::default()
        };
        let stats = Stats::new();
        stats.record_step();
        stats.record_request(250, 0);

        let rendered = agent
            .context_message(&policies, &stats)
            .expect("default context");

        let expected = format!(
            "{static_prefix}\n\
             - Steps remaining: 2\n\
             - Input tokens remaining: 750",
            static_prefix =
                default_context(&PathBuf::from("/tmp/check"), &Policies::default(), &Stats::new()),
        );
        assert_eq!(rendered, expected);
    }

    #[test]
    fn context_message_ignores_runtime_args_for_custom_context() {
        // Custom contexts stay byte-exact regardless of policy/stats —
        // the caller opted out of the default scaffolding entirely.
        let agent = Agent::new().context("- Note: custom");
        let policies = Policies {
            max_steps: Some(3),
            ..Policies::default()
        };
        let stats = Stats::new();
        stats.record_step();
        assert_eq!(
            agent.context_message(&policies, &stats).as_deref(),
            Some("## Context\n\n- Note: custom"),
        );
    }

    #[test]
    fn system_prompt_does_not_include_context() {
        let agent = Agent::new().role("ROLE").context("CTX");
        let prompt = agent.system_prompt(None);
        assert!(prompt.contains("ROLE"));
        assert!(!prompt.contains("CTX"));
        assert!(!prompt.contains("## Context"));
    }

    #[test]
    fn system_prompt_is_role_only() {
        let agent = Agent::new().role("ROLE");
        let prompt = agent.system_prompt(None);
        assert_eq!(prompt, "ROLE");
    }

    #[test]
    fn system_prompt_empty_when_role_unset() {
        let agent = Agent::new();
        assert!(agent.system_prompt(None).is_empty());
    }

    #[test]
    fn new_agent_has_write_result_registered() {
        let agent = Agent::new();
        let names: Vec<String> = agent
            .tool_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(names.iter().any(|n| n == "write_result_tool"));
    }

    #[test]
    fn system_prompt_interpolates_role_placeholders() {
        let agent = Agent::new()
            .role("You are {persona}.")
            .template_variable("persona", "a senior reviewer");
        assert_eq!(agent.system_prompt(None), "You are a senior reviewer.");
    }

    #[test]
    fn context_message_interpolates_context_placeholders() {
        let agent = Agent::new()
            .context("- Topic: {topic}")
            .template_variable("topic", "Rust generics");
        let policies = Policies::default();
        let stats = Stats::new();
        assert_eq!(
            agent.context_message(&policies, &stats).as_deref(),
            Some("## Context\n\n- Topic: Rust generics"),
        );
    }

    #[test]
    fn unresolved_placeholders_pass_through() {
        let agent = Agent::new()
            .role("Hi {missing}.")
            .context("- Note: {also_missing}");
        let policies = Policies::default();
        let stats = Stats::new();
        assert_eq!(agent.system_prompt(None), "Hi {missing}.");
        assert_eq!(
            agent.context_message(&policies, &stats).as_deref(),
            Some("## Context\n\n- Note: {also_missing}"),
        );
    }

    #[test]
    fn multiple_variables_substitute_independently() {
        let agent = Agent::new()
            .role("{greeting}, {name}.")
            .template_variables([("greeting", "Hello"), ("name", "Alice")]);
        assert_eq!(agent.system_prompt(None), "Hello, Alice.");
    }

    #[test]
    fn no_variables_renders_role_unchanged() {
        let agent = Agent::new().role("You are a senior reviewer.");
        assert_eq!(agent.system_prompt(None), "You are a senior reviewer.");
    }

    #[tokio::test]
    async fn dispatch_interpolates_string_task_body() {
        let sys = crate::agents::TicketSystem::new();
        let agent = Agent::new()
            .template_variable("topic", "rust")
            .ticket_system(&sys);
        agent.task("Search {topic} forums.");
        let stored = sys.first().expect("ticket should have been enqueued");
        assert_eq!(
            stored.task,
            serde_json::Value::String("Search rust forums.".into()),
        );
    }

    #[tokio::test]
    async fn dispatch_leaves_object_task_unchanged() {
        let sys = crate::agents::TicketSystem::new();
        let agent = Agent::new()
            .template_variable("topic", "rust")
            .ticket_system(&sys);
        let value = serde_json::json!({"q": "Find {topic}"});
        agent.ticket(Ticket::new(value.clone()));
        let stored = sys.first().expect("ticket should have been enqueued");
        assert_eq!(stored.task, value);
    }

    #[test]
    fn memory_defaults_to_none() {
        let agent = Agent::new();
        assert!(agent.memory_handle().is_none());
    }

    #[test]
    fn memory_registers_memory_tool_on_the_agent() {
        let dir = tempfile::tempdir().unwrap();
        let store = Memory::open(dir.path()).unwrap();
        let agent = Agent::new().memory(&store);
        let names: Vec<String> = agent
            .tool_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(
            names.iter().any(|n| n == "memory_tool"),
            "memory_tool should be registered: {names:?}"
        );
    }

    #[test]
    fn memory_opens_a_fresh_store_when_passed_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::new().memory(dir.path());
        let names: Vec<String> = agent
            .tool_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(names.iter().any(|n| n == "memory_tool"));
        agent.memory_handle().unwrap().add("from path").unwrap();
        assert!(dir.path().join("memory.jsonl").exists());
    }

    #[test]
    fn cloned_agent_observes_writes_through_original_handle() {
        let dir = tempfile::tempdir().unwrap();
        let store = Memory::open(dir.path()).unwrap();
        let agent = Agent::new().memory(&store);
        let cloned = agent.clone();
        agent.memory_handle().unwrap().add("via original").unwrap();
        assert_eq!(cloned.memory_handle().unwrap().entries().join("\n§\n"), "via original");
    }

    #[test]
    fn two_agents_bound_to_one_store_see_each_others_writes() {
        let dir = tempfile::tempdir().unwrap();
        let store = Memory::open(dir.path()).unwrap();
        let alice = Agent::new().memory(&store);
        let bob = Agent::new().memory(&store);
        alice.memory_handle().unwrap().add("from alice").unwrap();
        assert_eq!(bob.memory_handle().unwrap().entries().join("\n§\n"), "from alice");
    }

    #[test]
    fn system_prompt_renders_memory_section_when_body_present() {
        let agent = Agent::new().role("R");
        let prompt = agent.system_prompt(Some("note one\n§\nnote two"));
        assert!(prompt.contains("R"));
        assert!(prompt.contains("## Memory\n\nnote one\n§\nnote two"));
    }

    #[test]
    fn system_prompt_omits_memory_when_body_empty() {
        let agent = Agent::new().role("R");
        assert_eq!(agent.system_prompt(Some("")), "R");
    }
}
