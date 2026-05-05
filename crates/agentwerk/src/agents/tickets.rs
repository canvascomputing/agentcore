//! Ticket queue and run orchestration. `TicketSystem` owns the shared
//! ticket store, the registered agents, the active policies, the
//! interrupt signal, and the run-time [`Stats`] object.
//! `bind_agent` stamps the ticket Arc, policies, stats, and signal onto
//! each agent at add time; `run` / `run_dry` then drive the bound
//! agents.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::providers::{AsUserMessage, Message};

use super::agent::Agent;
use super::policy::Policies;
use super::r#loop::{run_main_loop, Runnable};
use super::stats::{Stats, TicketStats};

/// A ticket. Caller-settable fields: `task`, `labels`, `schema`,
/// `assignee`. System-managed fields (`key`, `status`, `reporter`,
/// `created_at`, `result`) are stamped at insertion time.
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
    /// Set when the ticket transitions `Todo → InProgress`. Millis
    /// since epoch.
    pub(crate) started_at: Option<u64>,
    /// Set when the ticket reaches `Status::Done`. Millis since epoch.
    /// Mutually exclusive with `failed_at`.
    pub(crate) finished_at: Option<u64>,
    /// Set when the ticket reaches `Status::Failed`. Millis since
    /// epoch. Mutually exclusive with `finished_at`.
    pub(crate) failed_at: Option<u64>,
    pub(crate) result: Option<ResultRecord>,
}

/// Record an agent writes when it finishes a ticket. Carries the source
/// agent name, the ticket key, and the agent's result payload (a string
/// for tickets without a schema, otherwise the validated JSON value).
/// Same shape as the line `WriteResultTool` appends to `results.jsonl`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResultRecord {
    pub agent: String,
    pub ticket: String,
    pub result: serde_json::Value,
}

impl ResultRecord {
    /// Result rendered as a String: the raw text for string payloads,
    /// canonical JSON for everything else. Lossless re-parsing is not
    /// required.
    pub fn result_string(&self) -> String {
        match &self.result {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }
}

impl Ticket {
    /// New ticket carrying `task` as its body. Use the chainable helpers
    /// (`label`, `labels`, `schema`, `assignee`) to populate
    /// caller-settable fields. System-managed fields are stamped by the
    /// ticket system at insertion time; the placeholders set here are
    /// overwritten.
    pub fn new<T: Serialize>(task: T) -> Self {
        let value = serde_json::to_value(task).expect("Ticket::new: value must serialize to JSON");
        Self {
            task: value,
            labels: Vec::new(),
            schema: None,
            key: String::new(),
            status: Status::Todo,
            assignee: None,
            reporter: String::new(),
            created_at: 0,
            started_at: None,
            finished_at: None,
            failed_at: None,
            result: None,
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

    /// Pin the ticket directly to an agent by name. The ticket is born
    /// `InProgress` and Path A on the loop side picks it up. There is no
    /// auto-resolution between assignee and label — the caller must know
    /// which they want.
    pub fn assign_to(mut self, name: impl Into<String>) -> Self {
        self.assignee = Some(name.into());
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

    pub fn started_at(&self) -> Option<u64> {
        self.started_at
    }

    pub fn finished_at(&self) -> Option<u64> {
        self.finished_at
    }

    pub fn failed_at(&self) -> Option<u64> {
        self.failed_at
    }

    /// Wall-clock duration from creation to terminal status (Done or
    /// Failed), `None` while the ticket has not yet reached one.
    pub fn duration(&self) -> Option<Duration> {
        let terminal = self.finished_at.or(self.failed_at)?;
        Some(Duration::from_millis(
            terminal.saturating_sub(self.created_at),
        ))
    }

    pub fn result(&self) -> Option<&ResultRecord> {
        self.result.as_ref()
    }

    /// Result payload rendered as a String, or `None` when the ticket has
    /// no recorded result. Convenience for callers that want a flat
    /// string view of the result regardless of the underlying JSON shape.
    pub fn result_string(&self) -> Option<String> {
        self.result.as_ref().map(ResultRecord::result_string)
    }

    // ---- predicate helpers (compose with TicketSystem::filter / find / count) ----

    pub fn is_todo(&self) -> bool {
        self.status == Status::Todo
    }

    pub fn is_in_progress(&self) -> bool {
        self.status == Status::InProgress
    }

    pub fn is_done(&self) -> bool {
        self.status == Status::Done
    }

    pub fn is_failed(&self) -> bool {
        self.status == Status::Failed
    }

    pub fn is_assigned_to(&self, name: &str) -> bool {
        self.assignee.as_deref() == Some(name)
    }

    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l == label)
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

/// Public ticket system. Owns the shared ticket store, the registered
/// agents, the policies, the interrupt signal, and the run stats.
/// Always lives behind `Arc<TicketSystem>` — `new()` returns
/// `Arc<Self>` so each bound `Agent` can hold a `Weak<TicketSystem>`
/// without creating an Arc cycle through the system's `Vec<Agent>`.
pub struct TicketSystem {
    weak_self: Weak<TicketSystem>,
    pub(crate) tickets: Mutex<HashMap<String, Ticket>>,
    agents: Mutex<Vec<Agent>>,
    policies: Mutex<Policies>,
    pub(crate) interrupt_signal: Mutex<Arc<AtomicBool>>,
    pub(crate) stats: Stats,
    results_dir: Mutex<Option<PathBuf>>,
}

impl TicketSystem {
    /// Build a fresh `TicketSystem` and return it inside an `Arc`. The
    /// system captures its own `Weak<Self>` via `Arc::new_cyclic` so
    /// `bind_agent` can hand out the back-reference each `Agent` needs
    /// at run time.
    pub fn new() -> Arc<Self> {
        Arc::new_cyclic(|weak| Self {
            weak_self: weak.clone(),
            tickets: Mutex::new(HashMap::new()),
            agents: Mutex::new(Vec::new()),
            policies: Mutex::new(Policies::default()),
            interrupt_signal: Mutex::new(Arc::new(AtomicBool::new(false))),
            stats: Stats::new(),
            results_dir: Mutex::new(None),
        })
    }

    /// Run-time counters. Read after `run` / `run_dry` returns.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    // ---- policy builders ----

    pub fn max_steps(self: Arc<Self>, n: u32) -> Arc<Self> {
        self.policies.lock().unwrap().max_steps = Some(n);
        self
    }

    pub fn max_input_tokens(self: Arc<Self>, n: u64) -> Arc<Self> {
        self.policies.lock().unwrap().max_input_tokens = Some(n);
        self
    }

    pub fn max_output_tokens(self: Arc<Self>, n: u64) -> Arc<Self> {
        self.policies.lock().unwrap().max_output_tokens = Some(n);
        self
    }

    pub fn max_request_tokens(self: Arc<Self>, n: u32) -> Arc<Self> {
        self.policies.lock().unwrap().max_request_tokens = Some(n);
        self
    }

    pub fn max_schema_retries(self: Arc<Self>, n: u32) -> Arc<Self> {
        self.policies.lock().unwrap().max_schema_retries = Some(n);
        self
    }

    pub fn max_request_retries(self: Arc<Self>, n: u32) -> Arc<Self> {
        self.policies.lock().unwrap().max_request_retries = n;
        self
    }

    pub fn request_retry_delay(self: Arc<Self>, d: Duration) -> Arc<Self> {
        self.policies.lock().unwrap().request_retry_delay = d;
        self
    }

    /// Maximum elapsed duration `run_dry` will wait before tripping
    /// the interrupt signal and returning. Hitting the cap is a
    /// graceful stop, not a `PolicyViolated` event.
    pub fn max_time(self: Arc<Self>, d: Duration) -> Arc<Self> {
        self.policies.lock().unwrap().max_time = Some(d);
        self
    }

    /// Override the cancel signal. Useful when a caller wants to share
    /// one `Arc<AtomicBool>` across multiple subsystems.
    pub fn interrupt_signal(self: Arc<Self>, signal: Arc<AtomicBool>) -> Arc<Self> {
        *self.interrupt_signal.lock().unwrap() = signal;
        self
    }

    /// Directory where `WriteResultTool` appends `results.jsonl`. When
    /// unset, the tool falls back to the calling agent's working
    /// directory.
    pub fn results_dir(self: Arc<Self>, dir: impl Into<PathBuf>) -> Arc<Self> {
        *self.results_dir.lock().unwrap() = Some(dir.into());
        self
    }

    pub(crate) fn results_dir_value(&self) -> Option<PathBuf> {
        self.results_dir.lock().unwrap().clone()
    }

    // ---- ticket-creation API mirrored on Agent ----

    /// Enqueue a ticket carrying `value` as its task body.
    pub fn task<T: Serialize>(&self, value: T) -> &Self {
        self.dispatch(Ticket::new(value));
        self
    }

    /// Enqueue a ticket carrying `value`, attached to `label` for Path B
    /// routing.
    pub fn task_labeled<T: Serialize>(&self, value: T, label: impl Into<String>) -> &Self {
        self.dispatch(Ticket::new(value).label(label));
        self
    }

    /// Enqueue a ticket whose final `done` result must validate against
    /// `schema`.
    pub fn task_schema<T: Serialize>(&self, value: T, schema: crate::schemas::Schema) -> &Self {
        self.dispatch(Ticket::new(value).schema(schema));
        self
    }

    /// `task_schema` + `task_labeled` combined.
    pub fn task_schema_labeled<T: Serialize>(
        &self,
        value: T,
        schema: crate::schemas::Schema,
        label: impl Into<String>,
    ) -> &Self {
        self.dispatch(Ticket::new(value).schema(schema).label(label));
        self
    }

    /// Enqueue a fully-built `Ticket`. System-managed fields (key,
    /// reporter, created_at, status, result) are overwritten unless
    /// `assignee` was explicitly set on the ticket — that case births the
    /// ticket `InProgress` to enable Path A routing.
    pub fn create(&self, ticket: Ticket) -> &Self {
        self.dispatch(ticket);
        self
    }

    fn dispatch(&self, ticket: Ticket) {
        self.insert(ticket, "user".to_string());
    }

    // ---- inherent ticket-store methods ----

    /// Insert `ticket`, stamping system fields. If `ticket.assignee` was
    /// preset, the ticket is born `InProgress`; otherwise `Todo`. Returns
    /// the inserted ticket's key.
    pub(crate) fn insert(&self, mut ticket: Ticket, reporter: String) -> String {
        let mut store = self.tickets.lock().unwrap();
        let id = store.len() + 1;
        ticket.key = format!("TICKET-{id}");
        ticket.created_at = now_millis();
        ticket.reporter = reporter;
        ticket.result = None;
        ticket.status = if ticket.assignee.is_some() {
            Status::InProgress
        } else {
            Status::Todo
        };
        let key = ticket.key.clone();
        let labels = ticket.labels.clone();
        store.insert(key.clone(), ticket);
        drop(store);
        TicketStats::record_created(&self.stats);
        for l in &labels {
            let slice = self.stats.stats_for_label(l);
            TicketStats::record_created(&*slice);
        }
        key
    }

    /// Returns a clone of the ticket at `key`, if any.
    pub fn get(&self, key: &str) -> Option<Ticket> {
        self.tickets.lock().unwrap().get(key).cloned()
    }

    /// State-machine-checked status transition. Records a ticket-stats
    /// event when the transition reaches Done or Failed.
    pub fn update_status(&self, key: &str, status: Status) -> Result<(), TicketError> {
        let now = now_millis();
        let outcome = {
            let mut store = self.tickets.lock().unwrap();
            let ticket = store
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
            let prev = ticket.status;
            stamp_transition_timestamps(ticket, status, now);
            ticket.status = status;
            let durations = terminal_durations(ticket);
            let labels = ticket.labels.clone();
            (prev, durations, labels)
        };
        fire_transition_recorder(&self.stats, outcome.0, status, now, outcome.1);
        fire_label_transition(&self.stats, status, outcome.1, &outcome.2);
        Ok(())
    }

    /// Bypass the state machine. Reserved for the loop's recovery paths
    /// (e.g. `MaxSchemaRetries` trip → Failed) so a stuck ticket doesn't
    /// get re-picked indefinitely via Path A.
    pub fn force_status(&self, key: &str, status: Status) -> Result<(), TicketError> {
        let now = now_millis();
        let outcome = {
            let mut store = self.tickets.lock().unwrap();
            let ticket = store
                .get_mut(key)
                .ok_or_else(|| TicketError::TicketMissing {
                    key: key.to_string(),
                })?;
            let prev = ticket.status;
            stamp_transition_timestamps(ticket, status, now);
            ticket.status = status;
            let durations = terminal_durations(ticket);
            let labels = ticket.labels.clone();
            (prev, durations, labels)
        };
        fire_transition_recorder(&self.stats, outcome.0, status, now, outcome.1);
        fire_label_transition(&self.stats, status, outcome.1, &outcome.2);
        Ok(())
    }

    /// Set a ticket's assignee. Used by the loop to pin a Path-B ticket
    /// to the agent that just claimed it.
    pub(crate) fn set_assignee(
        &self,
        key: &str,
        assignee: impl Into<String>,
    ) -> Result<(), TicketError> {
        let mut store = self.tickets.lock().unwrap();
        let ticket = store
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        ticket.assignee = Some(assignee.into());
        Ok(())
    }

    /// Attach a result record to the ticket at `key`.
    pub(crate) fn set_result(
        &self,
        key: &str,
        record: ResultRecord,
    ) -> Result<(), TicketError> {
        let mut store = self.tickets.lock().unwrap();
        let ticket = store
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        ticket.result = Some(record);
        Ok(())
    }

    /// Edit caller-settable fields. Each `Some` overwrites; `None`
    /// leaves the field untouched. The `Option<Option<Schema>>` shape on
    /// `schema` lets callers explicitly clear it via `Some(None)`.
    pub(crate) fn edit(
        &self,
        key: &str,
        task: Option<serde_json::Value>,
        labels: Option<Vec<String>>,
        schema: Option<Option<crate::schemas::Schema>>,
    ) -> Result<(), TicketError> {
        let mut store = self.tickets.lock().unwrap();
        let ticket = store
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

    /// Snapshot of every ticket, sorted by creation time then numeric key.
    pub fn tickets(&self) -> Vec<Ticket> {
        let tickets = self.tickets.lock().unwrap();
        let mut out: Vec<Ticket> = tickets.values().cloned().collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    /// Earliest ticket by creation time, if any.
    pub fn first(&self) -> Option<Ticket> {
        self.tickets().into_iter().next()
    }

    /// Every `Done` ticket's `ResultRecord`, in ticket creation order.
    /// Tickets that finished without a recorded result are skipped.
    pub fn results(&self) -> Vec<ResultRecord> {
        self.filter(Ticket::is_done)
            .into_iter()
            .filter_map(|t| t.result)
            .collect()
    }

    /// Most recently created `Done` ticket's `ResultRecord`, or `None`.
    /// Structured analogue of `run_dry`'s String return.
    pub fn last_result(&self) -> Option<ResultRecord> {
        self.filter(Ticket::is_done)
            .into_iter()
            .rev()
            .find_map(|t| t.result)
    }

    /// Substring search over the task body, case-insensitive.
    pub fn search(&self, query: &str) -> Vec<Ticket> {
        let needle = query.to_lowercase();
        let store = self.tickets.lock().unwrap();
        let mut out: Vec<Ticket> = store
            .values()
            .filter(|t| match &t.task {
                serde_json::Value::String(s) => s.to_lowercase().contains(&needle),
                other => other.to_string().to_lowercase().contains(&needle),
            })
            .cloned()
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    /// Tickets matching `predicate`, sorted by creation time then numeric key.
    ///
    /// The predicate runs while `self.tickets` is locked. It MUST NOT call
    /// other `TicketSystem` methods that lock the same `Mutex` — deadlock.
    pub fn filter<F>(&self, predicate: F) -> Vec<Ticket>
    where
        F: Fn(&Ticket) -> bool,
    {
        let store = self.tickets.lock().unwrap();
        let mut out: Vec<Ticket> = store.values().filter(|t| predicate(t)).cloned().collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    /// First ticket matching `predicate`, by creation order. Short-circuits.
    ///
    /// The predicate runs while `self.tickets` is locked. It MUST NOT call
    /// other `TicketSystem` methods that lock the same `Mutex` — deadlock.
    pub fn find<F>(&self, predicate: F) -> Option<Ticket>
    where
        F: Fn(&Ticket) -> bool,
    {
        let store = self.tickets.lock().unwrap();
        let mut matching: Vec<&Ticket> = store.values().filter(|t| predicate(t)).collect();
        matching.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        matching.into_iter().next().cloned()
    }

    /// Count of tickets matching `predicate`. Does not allocate.
    ///
    /// The predicate runs while `self.tickets` is locked. It MUST NOT call
    /// other `TicketSystem` methods that lock the same `Mutex` — deadlock.
    pub fn count<F>(&self, predicate: F) -> usize
    where
        F: Fn(&Ticket) -> bool,
    {
        self.tickets
            .lock()
            .unwrap()
            .values()
            .filter(|t| predicate(t))
            .count()
    }

    /// Snapshot of the active policies for the loop's per-step guards.
    pub(crate) fn policies(&self) -> Policies {
        self.policies.lock().unwrap().clone()
    }

    /// Wire `agent` to this system. Drains any tickets the agent had
    /// queued in its private default system into this one, then stamps
    /// the system's `Weak<Self>` onto `agent.ticket_system`.
    pub(crate) fn bind_agent(&self, agent: &mut Agent) {
        if let Some(prior) = agent.ticket_system.upgrade() {
            if !Arc::ptr_eq(
                &prior,
                &self
                    .weak_self
                    .upgrade()
                    .expect("self Arc dropped during bind"),
            ) {
                let drained: Vec<Ticket> = {
                    let mut old = prior.tickets.lock().unwrap();
                    std::mem::take(&mut *old).into_values().collect()
                };
                let reporter = agent.name.clone();
                for ticket in drained {
                    self.insert(ticket, reporter.clone());
                }
            }
        }
        agent.ticket_system = self.weak_self.clone();
        self.agents.lock().unwrap().push(agent.clone());
    }
}

impl TicketSystem {
    /// Clone of the currently registered agent list. The list is
    /// append-only by invariant: `bind_agent` is the sole mutator and
    /// only calls `push`. `run_main_loop` relies on element indices
    /// being stable across calls. Any new mutator that removes or
    /// reorders entries would silently break late-add detection: route
    /// additions through `bind_agent` only.
    pub(super) fn clone_agents(&self) -> Vec<Agent> {
        self.agents.lock().unwrap().clone()
    }
}

impl Runnable for TicketSystem {
    fn add(&self, mut agent: Agent) -> Agent {
        self.bind_agent(&mut agent);
        agent
    }

    async fn run(&self) {
        run_main_loop(self).await;
    }

    async fn run_dry(&self) -> String {
        let signal = Arc::clone(&self.interrupt_signal.lock().unwrap());

        let watcher_system = self
            .weak_self
            .upgrade()
            .expect("TicketSystem dropped during run_dry");
        let watcher_signal = Arc::clone(&signal);
        let watcher_policies = self.policies();
        let started = Instant::now();
        let watcher = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(20)).await;
                // External cancel: if the signal flipped (test or
                // Ctrl-C), exit even when tickets are still
                // InProgress. Without this, `run_dry` hangs because
                // `pending_count` only drops when the loop settles
                // tickets, which a cancelled run won't do.
                if watcher_signal.load(Ordering::Relaxed) {
                    watcher_system.stats.mark_finished(now_millis());
                    return;
                }
                if policy_violated(&watcher_policies, &watcher_system.stats) {
                    watcher_signal.store(true, Ordering::Relaxed);
                    watcher_system.stats.mark_finished(now_millis());
                    return;
                }
                if let Some(limit) = watcher_policies.max_time {
                    if started.elapsed() >= limit {
                        watcher_signal.store(true, Ordering::Relaxed);
                        watcher_system.stats.mark_finished(now_millis());
                        return;
                    }
                }
                let pending = pending_count(&watcher_system);
                if pending == 0 {
                    watcher_signal.store(true, Ordering::Relaxed);
                    watcher_system.stats.mark_finished(now_millis());
                    return;
                }
            }
        });
        run_main_loop(self).await;
        let _ = watcher.await;
        self.last_done_result()
    }
}

impl TicketSystem {
    /// Result of the most recently created `Status::Done` ticket, or
    /// an empty string when no ticket has reached `Done` (or its
    /// `result` is unset). Strings are returned as-is; structured
    /// payloads are rendered as canonical JSON.
    fn last_done_result(&self) -> String {
        self.filter(Ticket::is_done)
            .last()
            .and_then(|t| t.result_string())
            .unwrap_or_default()
    }
}

/// Whether the run-wide policies have been exceeded by the current
/// stats reading. Used by the `run_dry` watcher and by the per-agent
/// loop's pre-claim check.
pub(crate) fn policy_violated(policies: &Policies, stats: &Stats) -> bool {
    if let Some(limit) = policies.max_steps {
        if stats.steps() >= u64::from(limit) {
            return true;
        }
    }
    if let Some(limit) = policies.max_input_tokens {
        if stats.input_tokens() >= limit {
            return true;
        }
    }
    if let Some(limit) = policies.max_output_tokens {
        if stats.output_tokens() >= limit {
            return true;
        }
    }
    false
}

/// Same as [`policy_violated`] but returns which policy tripped and its
/// configured limit, for the `PolicyViolated` event.
pub(crate) fn policy_violated_kind(
    policies: &Policies,
    stats: &Stats,
) -> Option<(crate::event::PolicyKind, u64)> {
    use crate::event::PolicyKind;
    if let Some(limit) = policies.max_steps {
        if stats.steps() >= u64::from(limit) {
            return Some((PolicyKind::Steps, u64::from(limit)));
        }
    }
    if let Some(limit) = policies.max_input_tokens {
        if stats.input_tokens() >= limit {
            return Some((PolicyKind::InputTokens, limit));
        }
    }
    if let Some(limit) = policies.max_output_tokens {
        if stats.output_tokens() >= limit {
            return Some((PolicyKind::OutputTokens, limit));
        }
    }
    None
}

fn pending_count(ticket_system: &TicketSystem) -> usize {
    ticket_system
        .tickets
        .lock()
        .unwrap()
        .values()
        .filter(|t| matches!(t.status, Status::Todo | Status::InProgress))
        .count()
}

/// Stamp `started_at` / `finished_at` / `failed_at` on a ticket whose
/// status is about to flip. Called inside the locked critical section.
fn stamp_transition_timestamps(ticket: &mut Ticket, next: Status, now: u64) {
    if ticket.status == Status::Todo && next == Status::InProgress {
        ticket.started_at = Some(now);
    }
    match next {
        Status::Done => {
            ticket.finished_at = Some(now);
        }
        Status::Failed => {
            ticket.failed_at = Some(now);
        }
        _ => {}
    }
}

/// Compute (ticket_duration, work_duration) for a ticket that just
/// reached a terminal status. `ticket_duration` is creation→terminal;
/// `work_duration` is started→terminal. Both default to zero if the
/// relevant timestamps aren't both set.
fn terminal_durations(ticket: &Ticket) -> (Duration, Duration) {
    let ticket_duration = ticket.duration().unwrap_or_default();
    let work_duration = match (ticket.started_at, ticket.finished_at.or(ticket.failed_at)) {
        (Some(start), Some(end)) => Duration::from_millis(end.saturating_sub(start)),
        _ => Duration::ZERO,
    };
    (ticket_duration, work_duration)
}

/// Fire the appropriate recorder hook for a status transition. Called
/// after the lock is released.
fn fire_transition_recorder(
    stats: &Stats,
    prev: Status,
    next: Status,
    now: u64,
    (ticket_duration, work_duration): (Duration, Duration),
) {
    if prev == next {
        return;
    }
    if prev == Status::Todo && next == Status::InProgress {
        stats.record_started(now);
    }
    match next {
        Status::Done => stats.record_done(ticket_duration, work_duration),
        Status::Failed => stats.record_failed(ticket_duration, work_duration),
        _ => {}
    }
}

/// Mirror a terminal transition onto every per-label slice the ticket
/// carries. `record_started` is intentionally not mirrored: per-label
/// `started_at` stays zero so `run_duration()` reads `None` on a slice.
fn fire_label_transition(
    stats: &Stats,
    next: Status,
    (ticket_duration, work_duration): (Duration, Duration),
    labels: &[String],
) {
    if !matches!(next, Status::Done | Status::Failed) {
        return;
    }
    for l in labels {
        let slice = stats.stats_for_label(l);
        match next {
            Status::Done => slice.record_done(ticket_duration, work_duration),
            Status::Failed => slice.record_failed(ticket_duration, work_duration),
            _ => unreachable!(),
        }
    }
}

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

    fn attach_done_record(sys: &TicketSystem, key: &str, agent: &str, result: &str) {
        sys.set_result(
            key,
            ResultRecord {
                agent: agent.into(),
                ticket: key.into(),
                result: serde_json::Value::String(result.into()),
            },
        )
        .unwrap();
        sys.force_status(key, Status::Done).unwrap();
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
    fn task_labeled_attaches_label_and_leaves_status_todo() {
        let sys = TicketSystem::new();
        sys.task_labeled("hello", "research");
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.labels, vec!["research".to_string()]);
        assert_eq!(t.status(), Status::Todo);
        assert!(t.assignee().is_none());
    }

    #[test]
    fn create_with_explicit_assignee_births_ticket_inprogress() {
        let sys = TicketSystem::new();
        sys.create(Ticket::new("specific work for alice").assign_to("alice"));
        let t = sys.get("TICKET-1").unwrap();
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
            .filter(Ticket::is_todo)
            .iter()
            .map(|t| t.key().to_string())
            .collect();
        assert_eq!(all_keys.len(), 2);
    }

    #[test]
    fn agent_must_be_bound_before_task() {
        let alice = Agent::new().name("alice");
        let sys = TicketSystem::new();
        let alice = sys.add(alice);
        // Bound — task() works, lands in the shared queue.
        alice.task("first").task("second");
        assert_eq!(sys.count(Ticket::is_todo), 2);
    }

    #[test]
    #[should_panic(expected = "Agent::task requires a bound TicketSystem")]
    fn unbound_agent_task_panics() {
        let alice = Agent::new().name("alice");
        alice.task("never lands");
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
        let sys = TicketSystem::new();
        sys.task("hi");
        sys.set_result(
            "TICKET-1",
            ResultRecord {
                agent: "tester".into(),
                ticket: "TICKET-1".into(),
                result: serde_json::Value::String("answer".into()),
            },
        )
        .unwrap();
        let stored = sys.get("TICKET-1").unwrap();
        let record = stored.result().unwrap();
        assert_eq!(record.agent, "tester");
        assert_eq!(record.ticket, "TICKET-1");
        assert_eq!(record.result, serde_json::Value::String("answer".into()));
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
    fn results_returns_done_records_in_creation_order() {
        let sys = TicketSystem::new();
        sys.task("a").task("b").task("c");
        attach_done_record(&sys, "TICKET-1", "alice", "first");
        attach_done_record(&sys, "TICKET-3", "alice", "third");
        let results = sys.results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].ticket, "TICKET-1");
        assert_eq!(results[0].result, serde_json::Value::String("first".into()));
        assert_eq!(results[1].ticket, "TICKET-3");
        assert_eq!(results[1].result, serde_json::Value::String("third".into()));
    }

    #[test]
    fn last_result_returns_most_recently_created_done_record() {
        let sys = TicketSystem::new();
        sys.task("a").task("b");
        attach_done_record(&sys, "TICKET-2", "alice", "second");
        attach_done_record(&sys, "TICKET-1", "alice", "first");
        let last = sys.last_result().expect("expected last_result");
        assert_eq!(last.ticket, "TICKET-2");
        assert_eq!(last.result, serde_json::Value::String("second".into()));
    }

    #[test]
    fn results_and_last_result_are_empty_when_nothing_done() {
        let sys = TicketSystem::new();
        sys.task("pending");
        assert!(sys.results().is_empty());
        assert!(sys.last_result().is_none());
    }

    #[test]
    fn done_and_failed_filter_by_status() {
        let sys = TicketSystem::new();
        sys.task("ok").task("oops").task("pending");
        sys.update_status("TICKET-1", Status::InProgress).unwrap();
        sys.update_status("TICKET-1", Status::Done).unwrap();
        sys.force_status("TICKET-2", Status::Failed).unwrap();
        let done = sys.filter(Ticket::is_done);
        let failed = sys.filter(Ticket::is_failed);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].key(), "TICKET-1");
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].key(), "TICKET-2");
    }

    #[test]
    fn ticket_status_transitions_record_stats() {
        let sys = TicketSystem::new();
        sys.task("a").task("b").task("c");
        // Created 3 tickets.
        assert_eq!(sys.stats().tickets_created(), 3);
        sys.update_status("TICKET-1", Status::InProgress).unwrap();
        sys.update_status("TICKET-1", Status::Done).unwrap();
        sys.update_status("TICKET-2", Status::InProgress).unwrap();
        sys.update_status("TICKET-2", Status::Failed).unwrap();
        assert_eq!(sys.stats().tickets_done(), 1);
        assert_eq!(sys.stats().tickets_failed(), 1);
    }

    #[test]
    fn stats_for_label_counts_creation_per_label() {
        let sys = TicketSystem::new();
        sys.create(Ticket::new("a").labels(["scan", "high"]));
        sys.create(Ticket::new("b").label("scan"));
        sys.create(Ticket::new("c"));
        let stats = sys.stats();
        assert_eq!(stats.tickets_created(), 3);
        assert_eq!(stats.stats_for_label("scan").tickets_created(), 2);
        assert_eq!(stats.stats_for_label("high").tickets_created(), 1);
        assert_eq!(stats.stats_for_label("never-used").tickets_created(), 0);
    }

    #[test]
    fn stats_for_label_counts_terminal_transitions_per_label() {
        let sys = TicketSystem::new();
        sys.create(Ticket::new("a").labels(["scan", "high"]));
        sys.create(Ticket::new("b").label("scan"));
        sys.update_status("TICKET-1", Status::InProgress).unwrap();
        sys.update_status("TICKET-1", Status::Done).unwrap();
        sys.update_status("TICKET-2", Status::InProgress).unwrap();
        sys.update_status("TICKET-2", Status::Failed).unwrap();
        let stats = sys.stats();
        let scan = stats.stats_for_label("scan");
        let high = stats.stats_for_label("high");
        assert_eq!(scan.tickets_done(), 1);
        assert_eq!(scan.tickets_failed(), 1);
        assert_eq!(scan.success_rate(), Some(0.5));
        assert_eq!(high.tickets_done(), 1);
        assert_eq!(high.tickets_failed(), 0);
        assert_eq!(high.success_rate(), Some(1.0));
    }

    #[test]
    fn stats_for_label_force_status_path_records_per_label() {
        let sys = TicketSystem::new();
        sys.create(Ticket::new("a").label("scan"));
        sys.force_status("TICKET-1", Status::Failed).unwrap();
        assert_eq!(sys.stats().stats_for_label("scan").tickets_failed(), 1);
    }

    #[test]
    fn stats_for_label_unaffected_by_no_label_ticket() {
        let sys = TicketSystem::new();
        sys.create(Ticket::new("a"));
        sys.update_status("TICKET-1", Status::InProgress).unwrap();
        sys.update_status("TICKET-1", Status::Done).unwrap();
        assert_eq!(sys.stats().tickets_done(), 1);
        assert_eq!(sys.stats().stats_for_label("scan").tickets_done(), 0);
        assert_eq!(sys.stats().stats_for_label("scan").tickets_created(), 0);
    }
}
