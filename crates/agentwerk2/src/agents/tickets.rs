//! Ticket queue, run policies, and per-run metrics. Doubles as the
//! orchestrator: assigning an `Agent` runs its loop against this system
//! until the queue is drained, the cancel signal fires, or a policy is
//! violated.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::providers::{ProviderError, TokenUsage};

use super::agent::Agent;
use super::policy::{Policies, PolicyConform};
use super::r#loop::{run_main_loop, Runnable};

pub type TicketType = String;

pub struct TicketSystem {
    tickets: HashMap<String, Ticket>,
    next_id: u32,
    #[allow(dead_code)]
    directory: PathBuf,
    metrics: LoopMetrics,
    interrupt_signal: Arc<AtomicBool>,
    policies: Policies,
    agents: Vec<Agent>,
}

#[derive(Default)]
struct LoopMetrics {
    steps: u64,
    requests: u64,
    input_tokens: u64,
    output_tokens: u64,
}

impl PolicyConform for TicketSystem {
    fn policies(&self) -> &Policies {
        &self.policies
    }
}

#[derive(Debug)]
pub struct Ticket {
    pub key: String,
    pub summary: String,
    pub description: String,
    pub status: Status,
    pub r#type: TicketType,
    pub assignee: Option<String>,
    pub reporter: String,
    pub comments: Vec<Comment>,
    pub attachments: Vec<Attachment>,
    pub created_at: u64,
}

#[derive(Debug)]
pub struct Comment {
    pub author: String,
    pub body: String,
    pub created_at: u64,
}

#[derive(Debug)]
pub struct Attachment {
    pub filename: String,
    pub path: PathBuf,
    pub schema: String,
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

impl Default for TicketSystem {
    fn default() -> Self {
        Self {
            tickets: HashMap::new(),
            next_id: 1,
            directory: PathBuf::from("./tickets"),
            metrics: LoopMetrics::default(),
            interrupt_signal: Arc::new(AtomicBool::new(false)),
            policies: Policies::default(),
            agents: Vec::new(),
        }
    }
}

impl TicketSystem {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- policy builders ----

    pub fn max_steps(mut self, n: u32) -> Self {
        self.policies.max_steps = Some(n);
        self
    }

    pub fn max_input_tokens(mut self, n: u64) -> Self {
        self.policies.max_input_tokens = Some(n);
        self
    }

    pub fn max_output_tokens(mut self, n: u64) -> Self {
        self.policies.max_output_tokens = Some(n);
        self
    }

    pub fn max_request_tokens(mut self, n: u32) -> Self {
        self.policies.max_request_tokens = Some(n);
        self
    }

    pub fn max_schema_retries(mut self, n: u32) -> Self {
        self.policies.max_schema_retries = Some(n);
        self
    }

    pub fn max_request_retries(mut self, n: u32) -> Self {
        self.policies.max_request_retries = n;
        self
    }

    pub fn request_retry_delay(mut self, d: Duration) -> Self {
        self.policies.request_retry_delay = d;
        self
    }

    pub fn interrupt_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.interrupt_signal = signal;
        self
    }

    // ---- queue ops ----

    pub fn create(
        &mut self,
        summary: impl Into<String>,
        description: impl Into<String>,
        ticket_type: impl Into<TicketType>,
        reporter: impl Into<String>,
    ) -> &Ticket {
        let key = format!("TICKET-{}", self.next_id);
        self.next_id += 1;
        let ticket = Ticket {
            key: key.clone(),
            summary: summary.into(),
            description: description.into(),
            status: Status::Todo,
            r#type: ticket_type.into(),
            assignee: None,
            reporter: reporter.into(),
            comments: Vec::new(),
            attachments: Vec::new(),
            created_at: now_millis(),
        };
        self.tickets.insert(key.clone(), ticket);
        &self.tickets[&key]
    }

    pub fn get(&self, key: &str) -> Option<&Ticket> {
        self.tickets.get(key)
    }

    pub fn update_status(&mut self, key: &str, status: Status) -> Result<(), TicketError> {
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
        if status == Status::Todo {
            ticket.assignee = None;
        }
        ticket.status = status;
        Ok(())
    }

    pub fn assign_to(
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

    pub fn list_by_assignee(&self, assignee: &str) -> Vec<&Ticket> {
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| t.assignee.as_deref() == Some(assignee))
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub fn list_by_status(&self, status: Status) -> Vec<&Ticket> {
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| t.status == status)
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub fn search(&self, query: &str) -> Vec<&Ticket> {
        let needle = query.to_lowercase();
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| {
                t.summary.to_lowercase().contains(&needle)
                    || t.description.to_lowercase().contains(&needle)
            })
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub fn add_comment(
        &mut self,
        key: &str,
        author: impl Into<String>,
        body: impl Into<String>,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        ticket.comments.push(Comment {
            author: author.into(),
            body: body.into(),
            created_at: now_millis(),
        });
        Ok(())
    }

    pub fn add_attachment(
        &mut self,
        key: &str,
        attachment: Attachment,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing {
                key: key.to_string(),
            })?;
        ticket.attachments.push(attachment);
        Ok(())
    }

    pub fn pending(&self) -> usize {
        self.tickets
            .values()
            .filter(|t| t.status == Status::Todo)
            .count()
    }

    // ---- metric reads ----

    pub fn steps(&self) -> u64 {
        self.metrics.steps
    }

    pub fn requests(&self) -> u64 {
        self.metrics.requests
    }

    pub fn input_tokens(&self) -> u64 {
        self.metrics.input_tokens
    }

    pub fn output_tokens(&self) -> u64 {
        self.metrics.output_tokens
    }

    // ---- metric writes (called by the loop) ----

    pub(super) fn record_step(&mut self, _agent: &str) {
        self.metrics.steps += 1;
    }

    pub(super) fn record_request(&mut self, _agent: &str, usage: &TokenUsage) {
        self.metrics.requests += 1;
        self.metrics.input_tokens += usage.input_tokens;
        self.metrics.output_tokens += usage.output_tokens;
    }

    pub(super) fn record_error(&mut self, agent: &str, ticket_key: &str, err: &ProviderError) {
        let _ = self.add_comment(ticket_key, agent, format!("error: {err}"));
    }

    // ---- loop helpers ----

    pub(super) fn is_interrupted(&self) -> bool {
        self.interrupt_signal.load(Ordering::Relaxed)
    }

    pub(super) fn policy_violated(&self) -> bool {
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
}

impl Runnable for TicketSystem {
    fn assign(mut self, agent: Agent) -> Self {
        self.agents.push(agent);
        self
    }

    async fn run(mut self) -> Self {
        let agents = std::mem::take(&mut self.agents);
        let shared = Arc::new(Mutex::new(self));
        run_main_loop(agents, Arc::clone(&shared)).await;
        Arc::try_unwrap(shared)
            .ok()
            .expect("TicketSystem has remaining shared references")
            .into_inner()
            .expect("TicketSystem mutex was poisoned")
    }

    async fn run_until_empty(mut self) -> Self {
        let agents = std::mem::take(&mut self.agents);

        // What ticket types this set of agents can handle. If any agent
        // has an empty allow-list, they handle every type.
        let any_handles_all = agents.iter().any(|a| a.handles_any_type());
        let handled: HashSet<String> = agents
            .iter()
            .flat_map(|a| a.allowed_ticket_types().iter().cloned())
            .collect();

        let shared = Arc::new(Mutex::new(self));
        let signal = shared.lock().unwrap().interrupt_signal.clone();
        let watcher_shared = Arc::clone(&shared);
        let watcher_signal = Arc::clone(&signal);
        let watcher = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(20)).await;
                let sys = watcher_shared.lock().unwrap();

                if sys.policy_violated() {
                    watcher_signal.store(true, Ordering::Relaxed);
                    return;
                }
                let in_progress = sys.list_by_status(Status::InProgress).len();
                let any_workable = sys
                    .list_by_status(Status::Todo)
                    .into_iter()
                    .any(|t| any_handles_all || handled.contains(&t.r#type));
                // Stop when no work is in flight AND nothing pending is
                // workable by the assigned agents.
                if in_progress == 0 && !any_workable {
                    watcher_signal.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });
        run_main_loop(agents, Arc::clone(&shared)).await;
        let _ = watcher.await;
        Arc::try_unwrap(shared)
            .ok()
            .expect("TicketSystem has remaining shared references")
            .into_inner()
            .expect("TicketSystem mutex was poisoned")
    }
}

fn is_allowed_transition(from: Status, to: Status) -> bool {
    matches!(
        (from, to),
        (Status::Todo, Status::InProgress)
            | (Status::InProgress, Status::Todo)
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
    key.strip_prefix("TICKET-")
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

impl fmt::Display for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TicketMissing { key } => write!(f, "ticket {key} not found"),
            Self::TransitionRejected { from, to } => {
                write!(f, "cannot transition ticket from {from:?} to {to:?}")
            }
        }
    }
}

impl std::error::Error for TicketError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(system: &mut TicketSystem, summary: &str) -> String {
        system.create(summary, "", "task", "tester").key.clone()
    }

    fn finish(system: &mut TicketSystem, key: &str) {
        system.update_status(key, Status::InProgress).unwrap();
        system.update_status(key, Status::Done).unwrap();
    }

    fn attachment(name: &str) -> Attachment {
        Attachment {
            filename: name.to_string(),
            path: PathBuf::from(format!("/tmp/{name}")),
            schema: "file".to_string(),
        }
    }

    #[test]
    fn create_assigns_sequential_ticket_keys() {
        let mut system = TicketSystem::default();
        let first = task(&mut system, "first");
        let second = task(&mut system, "second");
        assert_eq!(first, "TICKET-1");
        assert_eq!(second, "TICKET-2");
    }

    #[test]
    fn create_starts_ticket_in_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "new work");
        assert_eq!(system.get(&key).unwrap().status, Status::Todo);
    }

    #[test]
    fn pending_is_zero_for_default_system() {
        let system = TicketSystem::default();
        assert_eq!(system.pending(), 0);
    }

    #[test]
    fn pending_counts_only_todo_tickets() {
        let mut system = TicketSystem::default();
        let claimed = task(&mut system, "claim me");
        let _waiting = task(&mut system, "wait");
        let finished = task(&mut system, "finish me");
        system.update_status(&claimed, Status::InProgress).unwrap();
        finish(&mut system, &finished);
        assert_eq!(system.pending(), 1);
    }

    #[test]
    fn update_status_transitions_todo_to_in_progress() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "claim");
        system.update_status(&key, Status::InProgress).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::InProgress);
    }

    #[test]
    fn update_status_transitions_in_progress_to_done() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "complete");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Done).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::Done);
    }

    #[test]
    fn update_status_transitions_in_progress_to_failed() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "give up");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Failed).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::Failed);
    }

    #[test]
    fn update_status_transitions_in_progress_back_to_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "release");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Todo).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::Todo);
    }

    #[test]
    fn update_status_rejects_todo_to_done() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "skip");
        let err = system.update_status(&key, Status::Done).unwrap_err();
        assert!(matches!(
            err,
            TicketError::TransitionRejected {
                from: Status::Todo,
                to: Status::Done
            }
        ));
        assert_eq!(system.get(&key).unwrap().status, Status::Todo);
    }

    #[test]
    fn update_status_rejects_done_to_in_progress() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "done");
        finish(&mut system, &key);
        let err = system.update_status(&key, Status::InProgress).unwrap_err();
        assert!(matches!(
            err,
            TicketError::TransitionRejected {
                from: Status::Done,
                to: Status::InProgress
            }
        ));
        assert_eq!(system.get(&key).unwrap().status, Status::Done);
    }

    #[test]
    fn update_status_rejects_failed_to_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "failed");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Failed).unwrap();
        let err = system.update_status(&key, Status::Todo).unwrap_err();
        assert!(matches!(
            err,
            TicketError::TransitionRejected {
                from: Status::Failed,
                to: Status::Todo
            }
        ));
        assert_eq!(system.get(&key).unwrap().status, Status::Failed);
    }

    #[test]
    fn update_status_clears_assignee_when_returning_to_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "release me");
        system.assign_to(&key, "alice").unwrap();
        system.update_status(&key, Status::InProgress).unwrap();
        assert_eq!(
            system.get(&key).unwrap().assignee.as_deref(),
            Some("alice")
        );
        system.update_status(&key, Status::Todo).unwrap();
        assert_eq!(system.get(&key).unwrap().assignee, None);
    }

    #[test]
    fn update_status_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .update_status("TICKET-999", Status::InProgress)
            .unwrap_err();
        let TicketError::TicketMissing { key } = err else {
            panic!("expected TicketMissing");
        };
        assert_eq!(key, "TICKET-999");
    }

    #[test]
    fn assign_to_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .assign_to("TICKET-999", "alice")
            .unwrap_err();
        assert!(matches!(err, TicketError::TicketMissing { .. }));
    }

    #[test]
    fn add_comment_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .add_comment("TICKET-999", "alice", "hi")
            .unwrap_err();
        assert!(matches!(err, TicketError::TicketMissing { .. }));
    }

    #[test]
    fn add_attachment_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .add_attachment("TICKET-999", attachment("a.txt"))
            .unwrap_err();
        assert!(matches!(err, TicketError::TicketMissing { .. }));
    }

    #[test]
    fn add_comment_appends_to_ticket() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "discuss");
        system
            .add_comment(&key, "alice", "looks good")
            .unwrap();
        system
            .add_comment(&key, "bob", "agreed")
            .unwrap();
        let comments = &system.get(&key).unwrap().comments;
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].author, "alice");
        assert_eq!(comments[1].author, "bob");
    }

    #[test]
    fn add_attachment_appends_to_ticket() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "with files");
        system.add_attachment(&key, attachment("a.txt")).unwrap();
        system.add_attachment(&key, attachment("b.txt")).unwrap();
        let attachments = &system.get(&key).unwrap().attachments;
        assert_eq!(attachments.len(), 2);
        assert_eq!(attachments[0].filename, "a.txt");
        assert_eq!(attachments[1].filename, "b.txt");
    }

    #[test]
    fn get_returns_none_for_unknown_key() {
        let system = TicketSystem::default();
        assert!(system.get("TICKET-999").is_none());
    }

    #[test]
    fn list_by_status_returns_matching_tickets_in_creation_order() {
        let mut system = TicketSystem::default();
        let _a = task(&mut system, "a");
        let b = task(&mut system, "b");
        let _c = task(&mut system, "c");
        system.update_status(&b, Status::InProgress).unwrap();
        let todos = system.list_by_status(Status::Todo);
        let summaries: Vec<&str> = todos.iter().map(|t| t.summary.as_str()).collect();
        assert_eq!(summaries, vec!["a", "c"]);
    }

    #[test]
    fn list_by_status_returns_empty_when_no_match() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "still todo");
        assert!(system.list_by_status(Status::Done).is_empty());
    }

    #[test]
    fn list_by_assignee_returns_tickets_for_named_assignee() {
        let mut system = TicketSystem::default();
        let mine_a = task(&mut system, "mine a");
        let theirs = task(&mut system, "theirs");
        let mine_b = task(&mut system, "mine b");
        system.assign_to(&mine_a, "alice").unwrap();
        system.assign_to(&theirs, "bob").unwrap();
        system.assign_to(&mine_b, "alice").unwrap();
        let alice = system.list_by_assignee("alice");
        let summaries: Vec<&str> = alice.iter().map(|t| t.summary.as_str()).collect();
        assert_eq!(summaries, vec!["mine a", "mine b"]);
    }

    #[test]
    fn list_by_assignee_returns_empty_when_no_match() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "unassigned");
        assert!(system.list_by_assignee("nobody").is_empty());
    }

    #[test]
    fn search_matches_summary_case_insensitively() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "Fix Login Bug");
        let _ = task(&mut system, "rewrite docs");
        let hits = system.search("login");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].summary, "Fix Login Bug");
    }

    #[test]
    fn search_matches_description_field() {
        let mut system = TicketSystem::default();
        system.create("summary", "secret keyword inside body", "task", "tester");
        let hits = system.search("keyword");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn search_returns_empty_when_no_match() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "alpha");
        let _ = task(&mut system, "beta");
        assert!(system.search("gamma").is_empty());
    }

    #[test]
    fn defaults_match_documented_values() {
        let system = TicketSystem::new();
        let p = system.policies();
        assert_eq!(p.max_steps, None);
        assert_eq!(p.max_input_tokens, None);
        assert_eq!(p.max_output_tokens, None);
        assert_eq!(p.max_request_tokens, None);
        assert_eq!(p.max_schema_retries, Some(10));
        assert_eq!(p.max_request_retries, 10);
        assert_eq!(p.request_retry_delay, Duration::from_millis(500));
    }

    #[test]
    fn all_seven_builders_set_their_fields() {
        let system = TicketSystem::new()
            .max_steps(10)
            .max_input_tokens(1000)
            .max_output_tokens(500)
            .max_request_tokens(256)
            .max_schema_retries(3)
            .max_request_retries(5)
            .request_retry_delay(Duration::from_secs(2));
        let p = system.policies();
        assert_eq!(p.max_steps, Some(10));
        assert_eq!(p.max_input_tokens, Some(1000));
        assert_eq!(p.max_output_tokens, Some(500));
        assert_eq!(p.max_request_tokens, Some(256));
        assert_eq!(p.max_schema_retries, Some(3));
        assert_eq!(p.max_request_retries, 5);
        assert_eq!(p.request_retry_delay, Duration::from_secs(2));
    }

    #[test]
    fn default_max_schema_retries_constant_is_ten() {
        assert_eq!(Policies::DEFAULT_MAX_SCHEMA_RETRIES, 10);
    }

    #[test]
    fn default_max_request_retries_constant_is_ten() {
        assert_eq!(Policies::DEFAULT_MAX_REQUEST_RETRIES, 10);
    }

    #[test]
    fn default_request_retry_delay_constant_is_500ms() {
        assert_eq!(
            Policies::DEFAULT_REQUEST_RETRY_DELAY,
            Duration::from_millis(500)
        );
    }

    #[test]
    fn policy_conform_trait_is_implemented_for_ticket_system() {
        let system = TicketSystem::new().max_steps(7);
        let p = <TicketSystem as PolicyConform>::policies(&system);
        assert_eq!(p.max_steps, Some(7));
    }

    #[test]
    fn record_step_increments_counter() {
        let mut system = TicketSystem::default();
        system.record_step("alice");
        system.record_step("bob");
        assert_eq!(system.steps(), 2);
    }

    #[test]
    fn record_request_splits_token_usage_into_individual_counters() {
        let mut system = TicketSystem::default();
        let usage = TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..TokenUsage::default()
        };
        system.record_request("alice", &usage);
        assert_eq!(system.requests(), 1);
        assert_eq!(system.input_tokens(), 10);
        assert_eq!(system.output_tokens(), 5);
    }

    #[test]
    fn record_error_uses_agent_name_as_comment_author() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "broken");
        let err = ProviderError::ConnectionFailed {
            message: "boom".into(),
        };
        system.record_error("alice", &key, &err);
        let comment = system
            .get(&key)
            .unwrap()
            .comments
            .iter()
            .find(|c| c.author == "alice")
            .expect("alice comment");
        assert!(comment.body.starts_with("error:"));
    }

    #[test]
    fn policy_violated_is_false_when_no_limits_set() {
        let mut system = TicketSystem::default();
        system.record_step("alice");
        assert!(!system.policy_violated());
    }

    #[test]
    fn policy_violated_when_max_steps_reached() {
        let mut system = TicketSystem::new().max_steps(2);
        system.record_step("alice");
        assert!(!system.policy_violated());
        system.record_step("alice");
        assert!(system.policy_violated());
    }

    #[test]
    fn policy_violated_when_max_input_tokens_reached() {
        let mut system = TicketSystem::new().max_input_tokens(100);
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 0,
            ..TokenUsage::default()
        };
        system.record_request("alice", &usage);
        assert!(system.policy_violated());
    }

    #[test]
    fn policy_violated_when_max_output_tokens_reached() {
        let mut system = TicketSystem::new().max_output_tokens(100);
        let usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 150,
            ..TokenUsage::default()
        };
        system.record_request("alice", &usage);
        assert!(system.policy_violated());
    }

    #[test]
    fn is_interrupted_reflects_interrupt_signal() {
        let signal = Arc::new(AtomicBool::new(false));
        let system = TicketSystem::new().interrupt_signal(Arc::clone(&signal));
        assert!(!system.is_interrupted());
        signal.store(true, Ordering::Relaxed);
        assert!(system.is_interrupted());
    }
}
