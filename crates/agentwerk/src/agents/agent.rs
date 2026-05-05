//! Agent: identity + prompt parts + provider/model + a bound ticket
//! system. Holds a `Weak<TicketSystem>`; `Default` produces a dangling
//! `Weak`, and `tickets.add(agent)` (or `agent.ticket_system(&shared)`)
//! stamps the system's `Weak<Self>` onto the agent. The loop upgrades it
//! once at the start of `handle_tickets` and accesses `tickets`,
//! `policies`, `stats`, and `interrupt_signal` through the resulting
//! `Arc<TicketSystem>`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use serde::Serialize;

use crate::event::{default_logger, Event};
use crate::prompts::{default_context, PromptBuilder, Section};
use crate::providers::{Message, Provider, ProviderToolDefinition};
use crate::tools::{ToolLike, ToolRegistry, WriteResultTool};

use super::r#loop::Runnable;
use super::tickets::{insert_ticket, Ticket, TicketSystem};

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
    history: Option<Arc<Mutex<Vec<Message>>>>,
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
            history: None,
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

    /// Drop every event, opting out of the default stderr logger.
    pub fn silent(mut self) -> Self {
        self.event_handler = Some(Arc::new(|_: Event| {}));
        self
    }

    /// Carry this agent's conversation transcript across every ticket
    /// it handles, including across separate `run` / `run_dry` calls
    /// on the same `TicketSystem`. Off by default; each ticket starts
    /// from a blank message vector. When set, prior assistant turns,
    /// tool calls, and tool results are replayed as context for the
    /// next ticket. Only flushed at terminal ticket status (`Done` or
    /// `Failed`); mid-cycle aborts (cancel, request failure) are not
    /// persisted. Storage is shared by every clone of the agent: the
    /// `bind_agent` clone in `TicketSystem.agents` writes back to the
    /// same slot the caller's original handle reads from.
    pub fn remember_history(mut self) -> Self {
        self.history = Some(Arc::new(Mutex::new(Vec::new())));
        self
    }

    /// Read the agent's stored conversation history. Returns `None`
    /// when [`Self::remember_history`] was not set on this agent. The
    /// returned vector is a clone; callers cannot mutate the slot
    /// through it.
    pub fn history(&self) -> Option<Vec<Message>> {
        self.history
            .as_ref()
            .map(|slot| slot.lock().unwrap().clone())
    }

    /// Drop everything in the agent's stored history slot. No-op when
    /// [`Self::remember_history`] was not set on this agent.
    pub fn clear_history(&self) {
        if let Some(slot) = self.history.as_ref() {
            slot.lock().unwrap().clear();
        }
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

    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Labels the agent declared. Empty means "default scope" — the agent
    /// handles only tickets with no labels.
    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }

    pub(super) fn resolve_event_handler(&self) -> Arc<dyn Fn(Event) + Send + Sync> {
        self.event_handler.clone().unwrap_or_else(default_logger)
    }

    pub(super) fn replace_history(&self, messages: Vec<Message>) {
        if let Some(slot) = self.history.as_ref() {
            *slot.lock().unwrap() = messages;
        }
    }

    /// Returns true when the agent's label scope intersects the ticket's
    /// labels. Empty agent labels mean "default scope" — only tickets with
    /// no labels match.
    pub fn handles(&self, ticket_labels: &[String]) -> bool {
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

    pub(super) fn system_prompt(&self) -> String {
        let mut b = PromptBuilder::default();
        if let Some(role) = &self.role {
            b = b.role(self.interpolate(role));
        }
        b.build().system
    }

    /// Render the context block pushed as the first user message in the
    /// loop. Falls back to [`default_context`] (working directory, platform,
    /// OS version, date) when [`Self::context`] was not set.
    pub(super) fn context_message(&self) -> Option<String> {
        match &self.context {
            Some(body) => Some(Section::context(self.interpolate(body)).render()),
            None => Some(default_context(&self.working_dir_or_default())),
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

    /// Enqueue a ticket carrying `value` as its task body. Always available
    /// (the agent has a bound ticket system from construction onward).
    /// Returns `&Self` for chaining.
    pub fn task<T: Serialize>(&self, value: T) -> &Self {
        let ticket = Ticket::new(value);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a ticket carrying `value` and attached to `label` for
    /// Path B routing. To pin a ticket directly to an agent (Path A),
    /// build it explicitly: `agent.create(Ticket::new(...).assign_to("alice"))`.
    pub fn task_assigned<T: Serialize>(&self, value: T, label: impl Into<String>) -> &Self {
        let ticket = Ticket::new(value).label(label);
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
        label: impl Into<String>,
    ) -> &Self {
        let ticket = Ticket::new(value).schema(schema).label(label);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a fully-built `Ticket`. System-managed fields (key,
    /// reporter, created_at, status, result) are overwritten unless the
    /// caller set `assignee` on the ticket — that case births the ticket
    /// `InProgress` to enable Path A routing.
    pub fn create(&self, ticket: Ticket) -> &Self {
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
        insert_ticket(&sys, ticket, self.name.clone());
    }

    /// Drive the agent's bound `TicketSystem` until the queue settles
    /// (drain mode). Returns the most recent `Done` ticket's `result`,
    /// or an empty string if no ticket reached `Done`. For runs where
    /// tickets keep arriving over time, drop down to
    /// `TicketSystem::run` directly.
    pub async fn run(&self) -> String {
        let sys = self
            .ticket_system
            .upgrade()
            .expect("Agent::run requires a bound TicketSystem");
        Runnable::run_dry(&*sys).await
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
    fn context_message_falls_back_to_default_when_unset() {
        let agent = Agent::new().role("R");
        let rendered = agent.context_message().expect("default context");
        assert!(rendered.starts_with("## Context\n\n"));
        assert!(rendered.contains("- Working directory: "));
        assert!(rendered.contains("- Platform: "));
        assert!(rendered.contains("- Date: "));
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
        assert_eq!(agent.system_prompt(), "You are a senior reviewer.");
    }

    #[test]
    fn context_message_interpolates_context_placeholders() {
        let agent = Agent::new()
            .context("- Topic: {topic}")
            .template_variable("topic", "Rust generics");
        assert_eq!(
            agent.context_message().as_deref(),
            Some("## Context\n\n- Topic: Rust generics"),
        );
    }

    #[test]
    fn unresolved_placeholders_pass_through() {
        let agent = Agent::new()
            .role("Hi {missing}.")
            .context("- Note: {also_missing}");
        assert_eq!(agent.system_prompt(), "Hi {missing}.");
        assert_eq!(
            agent.context_message().as_deref(),
            Some("## Context\n\n- Note: {also_missing}"),
        );
    }

    #[test]
    fn multiple_variables_substitute_independently() {
        let agent = Agent::new()
            .role("{greeting}, {name}.")
            .template_variables([("greeting", "Hello"), ("name", "Alice")]);
        assert_eq!(agent.system_prompt(), "Hello, Alice.");
    }

    #[test]
    fn no_variables_renders_role_unchanged() {
        let agent = Agent::new().role("You are a senior reviewer.");
        assert_eq!(agent.system_prompt(), "You are a senior reviewer.");
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
        agent.create(Ticket::new(value.clone()));
        let stored = sys.first().expect("ticket should have been enqueued");
        assert_eq!(stored.task, value);
    }

    #[test]
    fn history_defaults_to_none() {
        let agent = Agent::new();
        assert!(agent.history().is_none());
    }

    #[test]
    fn remember_history_turns_history_on() {
        let agent = Agent::new().remember_history();
        assert!(agent.history().is_some());
    }

    #[test]
    fn history_storage_is_shared_across_clones() {
        let agent = Agent::new().remember_history();
        let cloned = agent.clone();

        // bind_agent relies on this: the loop's clone writes the slot
        // the caller's original handle later reads from.
        cloned.replace_history(vec![Message::user("via clone")]);
        let snap = agent.history().expect("opted in");
        assert_eq!(snap.len(), 1);
    }

    #[test]
    fn history_returns_clone_when_opted_in() {
        let agent = Agent::new().remember_history();
        agent.replace_history(vec![Message::user("hello")]);
        let snap = agent.history().expect("opted in");
        assert_eq!(snap.len(), 1);
    }

    #[test]
    fn clear_history_empties_the_slot() {
        let agent = Agent::new().remember_history();
        agent.replace_history(vec![Message::user("hello")]);
        agent.clear_history();
        assert_eq!(agent.history().unwrap().len(), 0);
    }

    #[test]
    fn clear_history_is_no_op_when_history_off() {
        let agent = Agent::new();
        agent.clear_history();
        assert!(agent.history().is_none());
    }

    #[test]
    fn replace_history_is_no_op_when_history_off() {
        let agent = Agent::new();
        agent.replace_history(vec![Message::user("ignored")]);
        assert!(agent.history().is_none());
    }
}
