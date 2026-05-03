//! Agent: identity + prompt parts + provider/model + a bound ticket
//! system. Always carries an `Arc<Mutex<TicketSystemState>>`; the default
//! is a private one until `.ticket_system(&shared)` (or
//! `tickets.add(agent)`) hands it a shared one.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::event::{default_logger, Event};
use crate::prompts::{PromptBuilder, Section};
use crate::providers::{Provider, ProviderToolDefinition};
use crate::tools::{ToolLike, ToolRegistry};

use super::tickets::{Ticket, TicketSystem, TicketSystemState};

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
    tools: ToolRegistry,
    working_dir: Option<PathBuf>,
    current_ticket: Arc<Mutex<Option<String>>>,
    event_handler: Option<Arc<dyn Fn(Event) + Send + Sync>>,
    pub(crate) ticket_system: Arc<Mutex<TicketSystemState>>,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            name: default_agent_name(),
            provider: None,
            model: None,
            role: None,
            context: None,
            labels: Vec::new(),
            tools: ToolRegistry::default(),
            working_dir: None,
            current_ticket: Arc::new(Mutex::new(None)),
            event_handler: None,
            ticket_system: Arc::new(Mutex::new(TicketSystemState::default())),
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

    /// Drop every event, opting out of the default stderr logger.
    pub fn silent(mut self) -> Self {
        self.event_handler = Some(Arc::new(|_: Event| {}));
        self
    }

    /// Bind this agent to a shared `TicketSystem`. Drains any tickets the
    /// agent had already enqueued in its (private) default system into the
    /// shared one, then registers `self.clone()` into the shared system's
    /// agents list so the loop will dispatch this agent at `run_dry` time.
    pub fn ticket_system(mut self, sys: &TicketSystem) -> Self {
        sys.bind_agent(&mut self);
        self
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Labels the agent declared. Empty means "default scope" — the agent
    /// handles only tickets with no labels.
    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }

    /// The key of the ticket currently being processed, if any.
    pub fn current_ticket(&self) -> Option<String> {
        self.current_ticket.lock().unwrap().clone()
    }

    pub(super) fn set_current_ticket(&self, key: Option<String>) {
        *self.current_ticket.lock().unwrap() = key;
    }

    pub(super) fn resolve_event_handler(&self) -> Arc<dyn Fn(Event) + Send + Sync> {
        self.event_handler.clone().unwrap_or_else(default_logger)
    }

    /// Returns true when the agent's label scope intersects the ticket's
    /// labels. Empty agent labels mean "default scope" — only tickets with
    /// no labels match.
    pub fn handles(&self, ticket_labels: &[String]) -> bool {
        if self.labels.is_empty() {
            ticket_labels.is_empty()
        } else {
            self.labels.iter().any(|l| ticket_labels.iter().any(|t| t == l))
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

    pub(super) fn system_prompt(&self) -> String {
        let mut b = PromptBuilder::default();
        if let Some(role) = &self.role {
            b = b.role(role.clone());
        }
        b.build().system
    }

    /// Render the agent's `context` (if set) as a `## Context\n\n…`
    /// section, ready to be pushed as the first user message in the
    /// loop. Returns `None` when no context was configured.
    pub(super) fn context_message(&self) -> Option<String> {
        self.context
            .as_ref()
            .map(|body| Section::context(body.clone()).render())
    }

    /// Enqueue a ticket carrying `value` as its task body. Always available
    /// (the agent has a bound ticket system from construction onward).
    /// Returns `&Self` for chaining.
    pub fn task<T: Serialize>(&self, value: T) -> &Self {
        let ticket = Ticket::new(value);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a ticket carrying `value` and pinned to `assign` — either
    /// a label (Path B routing) or an agent name (Path A routing); the
    /// ticket system disambiguates at insertion time.
    pub fn task_assigned<T: Serialize>(&self, value: T, assign: impl Into<String>) -> &Self {
        let ticket = Ticket::new(value).assign(assign);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a ticket carrying `value` plus a `schema` the agent's final
    /// `done` result must validate against.
    pub fn task_schema<T: Serialize>(&self, value: T, schema: crate::schemas::Schema) -> &Self {
        let ticket = Ticket::new(value).schema(schema);
        self.dispatch(ticket);
        self
    }

    /// `task_schema` + `task_assigned` combined.
    pub fn task_schema_assigned<T: Serialize>(
        &self,
        value: T,
        schema: crate::schemas::Schema,
        assign: impl Into<String>,
    ) -> &Self {
        let ticket = Ticket::new(value).schema(schema).assign(assign);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a fully-built `Ticket`. The ticket system overrides the
    /// system-managed fields (key, status, created_at, reporter, assignee,
    /// result) at insertion time; only `task`, `labels`, `schema`, and the
    /// pending-assignee slot survive verbatim.
    pub fn create(&self, ticket: Ticket) -> &Self {
        self.dispatch(ticket);
        self
    }

    fn dispatch(&self, ticket: Ticket) {
        let mut state = self.ticket_system.lock().unwrap();
        state.insert(ticket, self.name.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_default_scope_only_picks_unlabeled_tickets() {
        let agent = Agent::new();
        assert!(agent.handles(&[]));
        assert!(!agent.handles(&["research".into()]));
    }

    #[test]
    fn handles_with_labels_intersects_ticket_labels() {
        let agent = Agent::new().label("research").label("urgent");
        assert!(agent.handles(&["research".into()]));
        assert!(agent.handles(&["urgent".into(), "other".into()]));
        assert!(!agent.handles(&["report".into()]));
        assert!(!agent.handles(&[]));
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
    fn context_message_returns_none_when_unset() {
        let agent = Agent::new().role("R");
        assert!(agent.context_message().is_none());
    }

    #[test]
    fn context_message_renders_h2_heading_when_set() {
        let agent = Agent::new().context("- Working directory: /tmp");
        assert_eq!(
            agent.context_message().as_deref(),
            Some("## Context\n\n- Working directory: /tmp"),
        );
    }

    #[test]
    fn system_prompt_does_not_include_context() {
        let agent = Agent::new().role("ROLE").context("CTX");
        let prompt = agent.system_prompt();
        assert!(prompt.contains("ROLE"));
        assert!(!prompt.contains("CTX"));
        assert!(!prompt.contains("## Context"));
    }

    #[test]
    fn system_prompt_is_role_only() {
        let agent = Agent::new().role("ROLE");
        let prompt = agent.system_prompt();
        assert_eq!(prompt, "ROLE");
    }

    #[test]
    fn system_prompt_empty_when_role_unset() {
        let agent = Agent::new();
        assert!(agent.system_prompt().is_empty());
    }
}
