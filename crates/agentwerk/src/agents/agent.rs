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
use crate::tools::{KnowledgeTool, ToolLike, ToolRegistry, WriteResultTool};

use super::knowledge::Knowledge;

use super::policy::Policies;
use super::stats::Stats;
use super::tickets::{Ticket, TicketSystem};

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
    dir: Option<PathBuf>,
    event_handler: Option<Arc<dyn Fn(Event) + Send + Sync>>,
    knowledge: Option<Arc<Knowledge>>,
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
            dir: None,
            event_handler: None,
            knowledge: None,
            ticket_system: Weak::new(),
        }
    }
}

impl Agent {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an `Agent` with no tools pre-registered. Use this
    /// when the agent must not have `WriteResultTool` available — for
    /// example, a researcher in a chain that should only ever call
    /// `WriteHandoverTool`. The caller is responsible for registering
    /// at least one finisher tool (`WriteResultTool` or
    /// `WriteHandoverTool`) via [`Self::tool`].
    pub fn empty() -> Self {
        Self {
            name: default_agent_name(),
            provider: None,
            model: None,
            role: None,
            context: None,
            labels: Vec::new(),
            template_variables: Vec::new(),
            tools: ToolRegistry::default(),
            dir: None,
            event_handler: None,
            knowledge: None,
            ticket_system: Weak::new(),
        }
    }

    pub fn name(mut self, n: impl Into<String>) -> Self {
        self.name = n.into();
        self
    }

    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    /// Detect the provider from environment variables. Panics if no provider env var is set.
    pub fn provider_from_env(self) -> Self {
        let provider = crate::providers::provider_from_env()
            .expect("LLM provider required: set ANTHROPIC_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY, or LITELLM_API_KEY");
        self.provider(provider)
    }

    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }

    /// Read the model name from environment variables. Panics if no provider can be detected.
    pub fn model_from_env(self) -> Self {
        let model = crate::providers::model_from_env().expect("model name required");
        self.model(model)
    }

    /// Detect both the provider and the model from environment variables.
    /// Equivalent to `provider_from_env().model_from_env()`. Panics if no
    /// provider env var is set.
    pub fn from_env(self) -> Self {
        self.provider_from_env().model_from_env()
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

    /// Directory tools resolve filesystem paths against. Defaults
    /// to the process's current directory when unset.
    pub fn dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.dir = Some(p.into());
        self
    }

    /// Install an event observer. The handler must be cheap and non-blocking.
    /// When not set, [`default_logger`] is used.
    pub fn event_handler(mut self, h: Arc<dyn Fn(Event) + Send + Sync>) -> Self {
        self.event_handler = Some(h);
        self
    }

    /// Knowledge store the agent uses for its long-term memory. Share
    /// one store across multiple agents the same way
    /// `ticket_system(&shared)` shares a queue. Defaults to a fresh
    /// store rooted at `./.agentwerk` when unset, mirroring how
    /// [`Self::dir`] defaults to the current working directory; the
    /// default store is opened lazily when the agent is bound to a
    /// `TicketSystem`. Registers `KnowledgeTool` on the agent's tool
    /// registry and arranges for the store's index to be injected into
    /// the system prompt under `## Knowledge` at the top of every
    /// ticket.
    pub fn knowledge(mut self, store: &Arc<Knowledge>) -> Self {
        self.tools.register(KnowledgeTool::new(Arc::clone(store)));
        self.knowledge = Some(Arc::clone(store));
        self
    }

    /// Configured knowledge store, or a freshly-opened store rooted at
    /// `./.agentwerk` when unset. Parallel to [`Self::dir_or_default`].
    /// Panics on IO failure when opening the default.
    pub(super) fn knowledge_or_default(&self) -> Arc<Knowledge> {
        self.knowledge
            .clone()
            .unwrap_or_else(|| Knowledge::open(".agentwerk").expect("open knowledge store"))
    }

    /// Materialize the default knowledge store and register
    /// `KnowledgeTool` if `.knowledge(...)` was not invoked. Called by
    /// `TicketSystem::bind_agent` so every running agent has a store
    /// without the caller having to wire one up.
    pub(super) fn ensure_knowledge_bound(&mut self) {
        if self.knowledge.is_some() {
            return;
        }
        let store = Knowledge::open(".agentwerk").expect("open knowledge store");
        self.tools.register(KnowledgeTool::new(Arc::clone(&store)));
        self.knowledge = Some(store);
    }

    /// Bind this agent to a shared `TicketSystem`. Drains any tickets
    /// the agent had already enqueued in its prior store into `sys`,
    /// stamps `sys`'s `Weak<Self>` onto `self.ticket_system`, and
    /// registers a clone of `self` into `sys`'s agents list so the
    /// loop will dispatch this agent at `run` / `finish` time.
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

    pub(super) fn dir_or_default(&self) -> PathBuf {
        self.dir
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

    /// Build the system prompt. `knowledge` is the index body the loop
    /// captured at the top of the current ticket, or `None` if
    /// [`Self::knowledge`] was not set. Tests may pass `None`.
    pub(super) fn system_prompt(&self, knowledge: Option<&str>) -> String {
        let mut b = PromptBuilder::default();
        if let Some(role) = &self.role {
            b = b.role(self.interpolate(role));
        }
        if let Some(snap) = knowledge.filter(|s| !s.is_empty()) {
            b = b.knowledge(snap.to_string());
        }
        b.build().system
    }

    /// Render the context block pushed as the first user message in the
    /// loop. Falls back to [`default_context`] (working directory, platform,
    /// OS version, date, plus a `… remaining` line for each configured
    /// policy budget) when [`Self::context`] was not set. A custom context
    /// is left byte-exact: `policies` and `stats` are ignored on that
    /// branch.
    pub(super) fn context_message(&self, policies: &Policies, stats: &Stats) -> Option<String> {
        match &self.context {
            Some(body) => Some(Section::context(self.interpolate(body)).render()),
            None => Some(default_context(&self.dir_or_default(), policies, stats)),
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

    /// Enqueue a fully-built `Ticket`. System-managed fields (key,
    /// reporter, created_at, status, result) are overwritten. To pin the
    /// ticket to a specific agent, label it with the agent's name.
    /// Compose schema and label via `Ticket::new(...).schema(...).label(...)`
    /// or `Ticket::new(...).schema_as::<R>()`.
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

    /// Start the agent loop on a background tokio task. Forwards to
    /// the bound `TicketSystem`. Returns the bound system so callers
    /// can `stop().await` or read results on the same value.
    pub fn start(&self) -> Arc<TicketSystem> {
        let sys = self
            .ticket_system
            .upgrade()
            .expect("Agent::start requires a bound TicketSystem");
        sys.start();
        sys
    }

    /// Start a background run and wait for every queued ticket to
    /// finish. Returns the bound system so the caller can read results
    /// via [`TicketSystem::last_result`] etc.
    pub async fn finish(&self) -> Arc<TicketSystem> {
        let sys = self
            .ticket_system
            .upgrade()
            .expect("Agent::finish requires a bound TicketSystem");
        let _ = sys.finish().await;
        sys
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
        let agent = Agent::new().dir("/tmp/check");
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
            static_prefix = default_context(
                &PathBuf::from("/tmp/check"),
                &Policies::default(),
                &Stats::new()
            ),
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
    fn knowledge_registers_knowledge_tool_on_the_agent() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(dir.path()).unwrap();
        let agent = Agent::new().knowledge(&store);
        let names: Vec<String> = agent
            .tool_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(
            names.iter().any(|n| n == "knowledge_tool"),
            "knowledge_tool should be registered: {names:?}"
        );
    }

    #[test]
    fn knowledge_binds_the_passed_store() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(dir.path()).unwrap();
        let agent = Agent::new().knowledge(&store);
        let names: Vec<String> = agent
            .tool_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(names.iter().any(|n| n == "knowledge_tool"));
        agent
            .knowledge_or_default()
            .write_page("from-store", "From store", "# From Store", &[])
            .unwrap();
        assert!(dir.path().join("pages").join("from-store.md").exists());
    }

    #[test]
    fn cloned_agent_observes_writes_through_original_handle() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(dir.path()).unwrap();
        let agent = Agent::new().knowledge(&store);
        let cloned = agent.clone();
        agent
            .knowledge_or_default()
            .write_page("shared", "Shared note", "# Shared", &[])
            .unwrap();
        assert!(cloned.knowledge_or_default().index().contains("shared"));
    }

    #[test]
    fn two_agents_bound_to_one_store_see_each_others_writes() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(dir.path()).unwrap();
        let alice = Agent::new().knowledge(&store);
        let bob = Agent::new().knowledge(&store);
        alice
            .knowledge_or_default()
            .write_page("from-alice", "From Alice", "# Alice", &[])
            .unwrap();
        assert!(bob.knowledge_or_default().index().contains("from-alice"));
    }

    #[test]
    fn system_prompt_renders_knowledge_section_when_body_present() {
        let agent = Agent::new().role("R");
        let prompt = agent.system_prompt(Some("- **config** — Port 8080"));
        assert!(prompt.contains("R"));
        assert!(prompt.contains("## Knowledge\n\n- **config** — Port 8080"));
    }

    #[test]
    fn system_prompt_omits_knowledge_when_body_empty() {
        let agent = Agent::new().role("R");
        assert_eq!(agent.system_prompt(Some("")), "R");
    }

    #[test]
    fn unbound_default_agent_does_not_register_knowledge_tool() {
        let agent = Agent::new();
        let names: Vec<String> = agent
            .tool_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(
            !names.iter().any(|n| n == "knowledge_tool"),
            "knowledge_tool must not appear before binding: {names:?}"
        );
    }

    #[test]
    fn binding_default_agent_materializes_knowledge_store() {
        let sys = crate::agents::TicketSystem::new();
        let agent = Agent::new().ticket_system(&sys);
        let names: Vec<String> = agent
            .tool_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(
            names.iter().any(|n| n == "knowledge_tool"),
            "knowledge_tool should be registered after binding: {names:?}"
        );
    }

    #[test]
    fn binding_agent_with_explicit_knowledge_keeps_explicit_store() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::open(dir.path()).unwrap();
        let sys = crate::agents::TicketSystem::new();
        let agent = Agent::new().knowledge(&store).ticket_system(&sys);
        assert!(Arc::ptr_eq(&store, &agent.knowledge_or_default()));
    }
}
