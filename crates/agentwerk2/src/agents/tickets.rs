//! Ticket queue, run policies, and per-run metrics. Doubles as the
//! orchestrator: binding (or `add`-ing) an `Agent` and calling
//! `run_dry().await` drives every staged agent until the queue is drained,
//! the cancel signal fires, a policy is violated, or the configured
//! timeout elapses.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::event::PolicyKind;
use crate::providers::{AsUserMessage, Message, ProviderError, TokenUsage};

use super::agent::Agent;
use super::policy::Policies;
use super::r#loop::{run_main_loop, Runnable};

#[derive(Debug, Clone)]
pub struct Ticket {
    pub task: serde_json::Value,
    pub labels: Vec<String>,
    pub schema: Option<crate::schemas::Schema>,
    pub(crate) key: String,
    pub(crate) status: Status,
    pub(crate) assignee: Option<String>,
    pub(crate) reporter: String,
    pub(crate) created_at: u64,
    pub(crate) result: Option<String>,
    /// Deferred routing slot: filled by `Ticket::assign(...)` and
    /// resolved by `TicketSystemState::insert` to either `assignee` (when
    /// the value matches a registered agent name) or appended to `labels`.
    pub(crate) pending_assignee: Option<String>,
}

impl Ticket {
    /// New ticket carrying `task` as its body. Use the chainable helpers
    /// (`label`, `labels`, `schema`, `assign`) to populate caller-settable
    /// fields. System-managed fields (`key`, `status`, `assignee`,
    /// `reporter`, `created_at`, `result`) are stamped by the ticket
    /// system at insertion time; the placeholders set here are
    /// overwritten.
    pub fn new<T: Serialize>(task: T) -> Self {
        let value = serde_json::to_value(task)
            .expect("Ticket::new: value must serialize to JSON");
        Self {
            task: value,
            labels: Vec::new(),
            schema: None,
            key: String::new(),
            status: Status::Todo,
            assignee: None,
            reporter: String::new(),
            created_at: 0,
            result: None,
            pending_assignee: None,
        }
    }

    /// Add a single label. Use [`Self::labels`] to add several at once.
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

    pub fn schema(mut self, schema: crate::schemas::Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Assigns the ticket. If `who` matches a registered agent at insert
    /// time, sets `assignee` and births the ticket `InProgress` (Path A
    /// routing). Otherwise appends `who` as a label (Path B routing).
    pub fn assign(mut self, who: impl Into<String>) -> Self {
        self.pending_assignee = Some(who.into());
        self
    }

    // ---- read-only accessors for system-managed fields ----

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn assignee(&self) -> Option<&str> {
        self.assignee.as_deref()
    }

    pub fn reporter(&self) -> &str {
        &self.reporter
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }
}

impl AsUserMessage for Ticket {
    fn as_user_message(&self) -> Message {
        let body = match &self.task {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string_pretty(other).unwrap_or_default(),
        };
        Message::user(body)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Todo,
    InProgress,
    Done,
    Failed,
}

#[derive(Debug)]
pub enum TicketError {
    TicketMissing { key: String },
    TransitionRejected { from: Status, to: Status },
}

impl fmt::Display for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TicketMissing { key } => write!(f, "Ticket {key} not found"),
            Self::TransitionRejected { from, to } => {
                write!(f, "Illegal transition {from:?} -> {to:?}")
            }
        }
    }
}

impl std::error::Error for TicketError {}

#[derive(Default)]
struct LoopMetrics {
    steps: u64,
    requests: u64,
    tool_calls: u64,
    errors: u64,
    input_tokens: u64,
    output_tokens: u64,
}

/// Mutable inner state of a ticket system. Held behind `Arc<Mutex<...>>`
/// by both the public `TicketSystem` handle and every `Agent` bound to
/// the system, so caller code, the loop, and individual agents share one
/// queue.
pub struct TicketSystemState {
    pub(crate) tickets: HashMap<String, Ticket>,
    next_id: u32,
    #[allow(dead_code)]
    directory: PathBuf,
    metrics: LoopMetrics,
    interrupt_signal: Arc<AtomicBool>,
    policies: Policies,
    timeout: Option<Duration>,
    pub(crate) agents: Vec<Agent>,
}

impl Default for TicketSystemState {
    fn default() -> Self {
        Self {
            tickets: HashMap::new(),
            next_id: 1,
            directory: PathBuf::from("./tickets"),
            metrics: LoopMetrics::default(),
            interrupt_signal: Arc::new(AtomicBool::new(false)),
            policies: Policies::default(),
            timeout: None,
            agents: Vec::new(),
        }
    }
}

impl TicketSystemState {
    /// Insert a caller-built ticket. Stamps every system-managed field
    /// (key, created_at, status, assignee, reporter, result) and resolves
    /// `pending_assignee` against the registered-agents set.
    pub(crate) fn insert(&mut self, mut ticket: Ticket, reporter: String) -> &Ticket {
        ticket.key = format!("TICKET-{}", self.next_id);
        self.next_id += 1;
        ticket.created_at = now_millis();
        ticket.reporter = reporter;
        ticket.result = None;

        if let Some(who) = ticket.pending_assignee.take() {
            if self.agents.iter().any(|a| a.get_name() == who) {
                ticket.assignee = Some(who);
                ticket.status = Status::InProgress;
            } else {
                ticket.labels.push(who);
                ticket.assignee = None;
                ticket.status = Status::Todo;
            }
        } else if ticket.assignee.is_some() {
            ticket.status = Status::InProgress;
        } else {
            ticket.status = Status::Todo;
        }

        let key = ticket.key.clone();
        self.tickets.insert(key.clone(), ticket);
        &self.tickets[&key]
    }

    pub(crate) fn get(&self, key: &str) -> Option<&Ticket> {
        self.tickets.get(key)
    }

    pub(crate) fn update_status(
        &mut self,
        key: &str,
        status: Status,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        if !is_allowed_transition(ticket.status, status) {
            return Err(TicketError::TransitionRejected {
                from: ticket.status,
                to: status,
            });
        }
        ticket.status = status;
        Ok(())
    }

    pub(crate) fn assign_to(
        &mut self,
        key: &str,
        assignee: impl Into<String>,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        ticket.assignee = Some(assignee.into());
        Ok(())
    }

    /// Bypass the state machine. Reserved for the loop's recovery paths
    /// (e.g. `MaxSchemaRetries` trip → Failed) so a stuck ticket doesn't
    /// get re-picked indefinitely via Path A.
    pub(crate) fn force_status(
        &mut self,
        key: &str,
        status: Status,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        ticket.status = status;
        Ok(())
    }

    pub(crate) fn edit_ticket(
        &mut self,
        key: &str,
        task: Option<serde_json::Value>,
        labels: Option<Vec<String>>,
        schema: Option<Option<crate::schemas::Schema>>,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        if let Some(t) = task {
            ticket.task = t;
        }
        if let Some(l) = labels {
            ticket.labels = l;
        }
        if let Some(s) = schema {
            ticket.schema = s;
        }
        Ok(())
    }

    pub(crate) fn set_result(
        &mut self,
        key: &str,
        result: String,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        ticket.result = Some(result);
        Ok(())
    }

    pub(crate) fn list_by_assignee(&self, assignee: &str) -> Vec<&Ticket> {
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| t.assignee.as_deref() == Some(assignee))
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub(crate) fn list_by_status(&self, status: Status) -> Vec<&Ticket> {
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| t.status == status)
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub(crate) fn search(&self, query: &str) -> Vec<&Ticket> {
        let needle = query.to_lowercase();
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| match &t.task {
                serde_json::Value::String(s) => s.to_lowercase().contains(&needle),
                other => other.to_string().to_lowercase().contains(&needle),
            })
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub(crate) fn pending(&self) -> usize {
        self.tickets
            .values()
            .filter(|t| t.status == Status::Todo)
            .count()
    }

    pub(crate) fn record_step(&mut self, _agent: &str) {
        self.metrics.steps += 1;
    }

    pub(crate) fn record_request(&mut self, _agent: &str, usage: &TokenUsage) {
        self.metrics.requests += 1;
        self.metrics.input_tokens += usage.input_tokens;
        self.metrics.output_tokens += usage.output_tokens;
    }

    pub(crate) fn record_tool_calls(&mut self, _agent: &str, n: u64) {
        self.metrics.tool_calls += n;
    }

    pub(crate) fn record_error(
        &mut self,
        _agent: &str,
        _ticket_key: &str,
        _err: &ProviderError,
    ) {
        self.metrics.errors += 1;
    }

    pub(crate) fn interrupt_signal_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.interrupt_signal)
    }

    pub(crate) fn is_interrupted(&self) -> bool {
        self.interrupt_signal.load(Ordering::Relaxed)
    }

    pub(crate) fn policies(&self) -> &Policies {
        &self.policies
    }

    pub(crate) fn policy_violated(&self) -> bool {
        let p = &self.policies;
        if let Some(limit) = p.max_steps {
            if self.metrics.steps >= u64::from(limit) {
                return true;
            }
        }
        if let Some(limit) = p.max_input_tokens {
            if self.metrics.input_tokens >= limit {
                return true;
            }
        }
        if let Some(limit) = p.max_output_tokens {
            if self.metrics.output_tokens >= limit {
                return true;
            }
        }
        false
    }

    pub(crate) fn policy_violated_kind(&self) -> Option<(PolicyKind, u64)> {
        let p = &self.policies;
        if let Some(limit) = p.max_steps {
            if self.metrics.steps >= u64::from(limit) {
                return Some((PolicyKind::Steps, u64::from(limit)));
            }
        }
        if let Some(limit) = p.max_input_tokens {
            if self.metrics.input_tokens >= limit {
                return Some((PolicyKind::InputTokens, limit));
            }
        }
        if let Some(limit) = p.max_output_tokens {
            if self.metrics.output_tokens >= limit {
                return Some((PolicyKind::OutputTokens, limit));
            }
        }
        None
    }
}

/// Public handle to a ticket system. A thin wrapper over an
/// `Arc<Mutex<TicketSystemState>>`; the same Arc is cloned into every
/// agent bound to this system.
#[derive(Default, Clone)]
pub struct TicketSystem {
    pub(crate) state: Arc<Mutex<TicketSystemState>>,
}

impl TicketSystem {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- policy builders ----

    pub fn max_steps(self, n: u32) -> Self {
        self.state.lock().unwrap().policies.max_steps = Some(n);
        self
    }

    pub fn max_input_tokens(self, n: u64) -> Self {
        self.state.lock().unwrap().policies.max_input_tokens = Some(n);
        self
    }

    pub fn max_output_tokens(self, n: u64) -> Self {
        self.state.lock().unwrap().policies.max_output_tokens = Some(n);
        self
    }

    pub fn max_request_tokens(self, n: u32) -> Self {
        self.state.lock().unwrap().policies.max_request_tokens = Some(n);
        self
    }

    pub fn max_schema_retries(self, n: u32) -> Self {
        self.state.lock().unwrap().policies.max_schema_retries = Some(n);
        self
    }

    pub fn max_request_retries(self, n: u32) -> Self {
        self.state.lock().unwrap().policies.max_request_retries = n;
        self
    }

    pub fn request_retry_delay(self, d: Duration) -> Self {
        self.state.lock().unwrap().policies.request_retry_delay = d;
        self
    }

    /// Maximum wall-clock duration `run_dry` will wait before tripping
    /// the interrupt signal and returning. Independent from the
    /// policy-violation cap surface — a timeout is a graceful stop, not
    /// a `PolicyViolated` event.
    pub fn timeout(self, d: Duration) -> Self {
        self.state.lock().unwrap().timeout = Some(d);
        self
    }

    pub fn interrupt_signal(self, signal: Arc<AtomicBool>) -> Self {
        self.state.lock().unwrap().interrupt_signal = signal;
        self
    }

    // ---- ticket-creation API mirrored on Agent ----

    /// Enqueue a ticket carrying `value` as its task body.
    pub fn task<T: Serialize>(&self, value: T) -> &Self {
        let ticket = Ticket::new(value);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a ticket carrying `value`, pinned to `assign` (label or
    /// agent name; the ticket system disambiguates at insertion).
    pub fn task_assigned<T: Serialize>(&self, value: T, assign: impl Into<String>) -> &Self {
        let ticket = Ticket::new(value).assign(assign);
        self.dispatch(ticket);
        self
    }

    /// Enqueue a ticket whose final `done` result must validate against
    /// `schema`.
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

    /// Enqueue a fully-built `Ticket`. System fields are overwritten.
    pub fn create(&self, ticket: Ticket) -> &Self {
        self.dispatch(ticket);
        self
    }

    fn dispatch(&self, ticket: Ticket) {
        let mut state = self.state.lock().unwrap();
        state.insert(ticket, "user".to_string());
    }

    // ---- read-side accessors ----

    pub fn steps(&self) -> u64 {
        self.state.lock().unwrap().metrics.steps
    }

    pub fn requests(&self) -> u64 {
        self.state.lock().unwrap().metrics.requests
    }

    pub fn tool_calls(&self) -> u64 {
        self.state.lock().unwrap().metrics.tool_calls
    }

    pub fn errors(&self) -> u64 {
        self.state.lock().unwrap().metrics.errors
    }

    pub fn input_tokens(&self) -> u64 {
        self.state.lock().unwrap().metrics.input_tokens
    }

    pub fn output_tokens(&self) -> u64 {
        self.state.lock().unwrap().metrics.output_tokens
    }

    /// Returns a clone of the ticket at `key`, if any.
    pub fn get(&self, key: &str) -> Option<Ticket> {
        self.state.lock().unwrap().get(key).cloned()
    }

    /// Snapshot of every ticket, sorted by creation time then numeric key.
    pub fn tickets(&self) -> Vec<Ticket> {
        let state = self.state.lock().unwrap();
        let mut out: Vec<Ticket> = state.tickets.values().cloned().collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    /// Earliest ticket by creation time, if any.
    pub fn first(&self) -> Option<Ticket> {
        self.tickets().into_iter().next()
    }

    /// Snapshot of every ticket in `Status::Done`.
    pub fn done(&self) -> Vec<Ticket> {
        self.list_by_status(Status::Done)
    }

    /// Snapshot of every ticket in `Status::Failed`.
    pub fn failed(&self) -> Vec<Ticket> {
        self.list_by_status(Status::Failed)
    }

    /// Snapshot of every ticket in `status`, ordered by creation time.
    pub fn list_by_status(&self, status: Status) -> Vec<Ticket> {
        self.state
            .lock()
            .unwrap()
            .list_by_status(status)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Snapshot of every ticket assigned to `assignee`.
    pub fn list_by_assignee(&self, assignee: &str) -> Vec<Ticket> {
        self.state
            .lock()
            .unwrap()
            .list_by_assignee(assignee)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Substring search over the task body.
    pub fn search(&self, query: &str) -> Vec<Ticket> {
        self.state
            .lock()
            .unwrap()
            .search(query)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn pending(&self) -> usize {
        self.state.lock().unwrap().pending()
    }

    /// Wire `agent.ticket_system` to this system, draining any tickets
    /// the agent had already enqueued in its prior (private) system, and
    /// register `agent.clone()` into this system's agents list. Used by
    /// both `Agent::ticket_system(&sys)` and `TicketSystem::add(agent)`.
    pub(crate) fn bind_agent(&self, agent: &mut Agent) {
        if !Arc::ptr_eq(&agent.ticket_system, &self.state) {
            let drained: Vec<Ticket> = {
                let mut old = agent.ticket_system.lock().unwrap();
                std::mem::take(&mut old.tickets).into_values().collect()
            };
            let reporter = agent.name.clone();
            let mut new_state = self.state.lock().unwrap();
            for ticket in drained {
                new_state.insert(ticket, reporter.clone());
            }
            drop(new_state);
            agent.ticket_system = Arc::clone(&self.state);
        }
        self.state.lock().unwrap().agents.push(agent.clone());
    }
}

impl Runnable for TicketSystem {
    fn add(&self, mut agent: Agent) -> Agent {
        self.bind_agent(&mut agent);
        agent
    }

    async fn run(&self) {
        let agents = self.state.lock().unwrap().agents_take();
        run_main_loop(agents, Arc::clone(&self.state)).await;
    }

    async fn run_dry(&self) -> Option<String> {
        let agents = self.state.lock().unwrap().agents_take();
        let signal = self.state.lock().unwrap().interrupt_signal_handle();
        let timeout = self.state.lock().unwrap().timeout;

        let watcher_state = Arc::clone(&self.state);
        let watcher_signal = Arc::clone(&signal);
        let started = Instant::now();
        let watcher = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(20)).await;
                let sys = watcher_state.lock().unwrap();
                if sys.policy_violated() {
                    watcher_signal.store(true, Ordering::Relaxed);
                    return;
                }
                if let Some(limit) = timeout {
                    if started.elapsed() >= limit {
                        watcher_signal.store(true, Ordering::Relaxed);
                        return;
                    }
                }
                let pending = sys.list_by_status(Status::Todo).len()
                    + sys.list_by_status(Status::InProgress).len();
                if pending == 0 {
                    watcher_signal.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });
        run_main_loop(agents, Arc::clone(&self.state)).await;
        let _ = watcher.await;
        self.last_done_result()
    }
}

impl TicketSystem {
    /// Result of the most recently created `Status::Done` ticket, or
    /// `None` when no ticket has reached `Done` (or its `result` is
    /// unset).
    fn last_done_result(&self) -> Option<String> {
        self.done().last().and_then(|t| t.result().map(String::from))
    }
}

impl TicketSystemState {
    fn agents_take(&mut self) -> Vec<Agent> {
        std::mem::take(&mut self.agents)
    }
}

// `Default::default()` on `Vec<Agent>` is needed for `mem::take` on
// `agents` — `Vec` already implements `Default`, no extra work.

fn is_allowed_transition(from: Status, to: Status) -> bool {
    matches!(
        (from, to),
        (Status::Todo, Status::InProgress)
            | (Status::InProgress, Status::Done)
            | (Status::InProgress, Status::Failed)
    )
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn numeric_id(key: &str) -> u32 {
    key.rsplit('-')
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task_ticket(label: &str) -> Ticket {
        Ticket::new(format!("body-{label}")).label(label)
    }

    #[test]
    fn task_creates_ticket_with_user_reporter() {
        let sys = TicketSystem::new();
        sys.task("hello");
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.task, serde_json::Value::String("hello".into()));
        assert_eq!(t.reporter(), "user");
        assert_eq!(t.status(), Status::Todo);
        assert!(t.assignee().is_none());
    }

    #[test]
    fn task_assigned_with_unknown_target_treats_it_as_a_label() {
        let sys = TicketSystem::new();
        sys.task_assigned("hello", "research");
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.labels, vec!["research".to_string()]);
        assert_eq!(t.status(), Status::Todo);
        assert!(t.assignee().is_none());
    }

    #[test]
    fn task_assigned_with_known_agent_name_sets_assignee_and_inprogress() {
        let sys = TicketSystem::new();
        let _alice = sys.add(Agent::new().name("alice"));
        sys.task_assigned("hello", "alice");
        let t = sys.get("TICKET-2").unwrap_or_else(|| sys.get("TICKET-1").unwrap());
        assert_eq!(t.assignee(), Some("alice"));
        assert_eq!(t.status(), Status::InProgress);
    }

    #[test]
    fn create_with_label_and_schema_is_stored_verbatim() {
        let sys = TicketSystem::new();
        let schema = crate::schemas::Schema::parse(serde_json::json!({"type": "string"})).unwrap();
        sys.create(Ticket::new("x").label("urgent").schema(schema));
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.labels, vec!["urgent".to_string()]);
        assert!(t.schema.is_some());
    }

    #[test]
    fn allowed_transitions_match_state_machine() {
        assert!(is_allowed_transition(Status::Todo, Status::InProgress));
        assert!(is_allowed_transition(Status::InProgress, Status::Done));
        assert!(is_allowed_transition(Status::InProgress, Status::Failed));
        assert!(!is_allowed_transition(Status::Todo, Status::Done));
        assert!(!is_allowed_transition(Status::InProgress, Status::Todo));
        assert!(!is_allowed_transition(Status::Done, Status::Failed));
        assert!(!is_allowed_transition(Status::Failed, Status::Done));
    }

    #[test]
    fn ticket_system_handle_is_shared_between_caller_and_added_agent() {
        let sys = TicketSystem::new();
        let alice = sys.add(Agent::new().name("alice"));
        // Alice's task lands in the same queue.
        alice.task("from alice");
        sys.task("from system");
        let all_keys: Vec<String> = sys
            .list_by_status(Status::Todo)
            .iter()
            .map(|t| t.key().to_string())
            .collect();
        assert_eq!(all_keys.len(), 2);
    }

    #[test]
    fn agent_default_system_drains_into_shared_at_add_time() {
        let alice = Agent::new().name("alice");
        alice.task("private one").task("private two");
        let sys = TicketSystem::new();
        let _alice = sys.add(alice);
        assert_eq!(sys.list_by_status(Status::Todo).len(), 2);
    }

    #[test]
    fn search_matches_string_task_case_insensitively() {
        let sys = TicketSystem::new();
        sys.task("Fix Login");
        sys.task("Other thing");
        let hits = sys.search("login");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn ticket_label_helpers_compose() {
        let t = task_ticket("research").label("urgent");
        assert_eq!(t.labels, vec!["research".to_string(), "urgent".to_string()]);
    }

    #[test]
    fn set_result_updates_ticket() {
        let mut state = TicketSystemState::default();
        state.insert(Ticket::new("hi"), "user".into());
        state.set_result("TICKET-1", "answer".into()).unwrap();
        assert_eq!(state.get("TICKET-1").unwrap().result(), Some("answer"));
    }

    #[test]
    fn first_returns_none_on_empty_system() {
        let sys = TicketSystem::new();
        assert!(sys.first().is_none());
        assert!(sys.tickets().is_empty());
    }

    #[test]
    fn first_returns_earliest_ticket_by_creation() {
        let sys = TicketSystem::new();
        sys.task("first").task("second").task("third");
        let first = sys.first().unwrap();
        assert_eq!(first.key(), "TICKET-1");
        assert_eq!(first.task, serde_json::Value::String("first".into()));
    }

    #[test]
    fn tickets_returns_all_in_creation_order() {
        let sys = TicketSystem::new();
        sys.task("a").task("b").task("c");
        let all = sys.tickets();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].key(), "TICKET-1");
        assert_eq!(all[1].key(), "TICKET-2");
        assert_eq!(all[2].key(), "TICKET-3");
    }

    #[test]
    fn done_and_failed_filter_by_status() {
        let sys = TicketSystem::new();
        sys.task("ok").task("oops").task("pending");
        // Force-set statuses through the state for the snapshot test.
        {
            let mut state = sys.state.lock().unwrap();
            state.update_status("TICKET-1", Status::InProgress).unwrap();
            state.update_status("TICKET-1", Status::Done).unwrap();
            state.force_status("TICKET-2", Status::Failed).unwrap();
        }
        let done = sys.done();
        let failed = sys.failed();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].key(), "TICKET-1");
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].key(), "TICKET-2");
    }
}
