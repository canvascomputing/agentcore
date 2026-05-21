//! Ticket queue and run orchestration. `TicketSystem` owns the shared
//! ticket store, the registered agents, the active policies, the
//! interrupt signal, and the run-time [`Stats`] object.
//! `bind_agent` stamps the ticket Arc, policies, stats, and signal onto
//! each agent at add time; `run` / `finish` then drive the bound
//! agents.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::task::JoinHandle;

use crate::providers::{AsUserMessage, ContentBlock, Message};

use super::agent::Agent;
use super::policy::Policies;
use super::r#loop::run_main_loop;
use super::stats::{Stats, TicketStats};

/// A ticket. Caller-settable fields: `task`, `labels`, `schema`.
/// System-managed fields (`key`, `status`, `reporter`, `created_at`,
/// `result`) are stamped at insertion time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ticket {
    pub task: serde_json::Value,
    pub labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<crate::schemas::Schema>,
    pub(crate) key: String,
    pub(crate) status: Status,
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
    pub(crate) result: Option<serde_json::Value>,
    /// Back-reference to another ticket, or `None` when the ticket
    /// has no parent. Caller-settable via [`Ticket::parent`].
    pub(crate) parent: Option<String>,
    /// Append-only transcript of the messages the agent loop sent to
    /// the provider for this ticket, plus a leading `system` entry for
    /// the system prompt and synthetic `system` entries marking
    /// compaction boundaries. System-managed: callers cannot push
    /// directly.
    pub(crate) comments: Vec<Comment>,
}

impl Ticket {
    /// New ticket carrying `task` as its body. Use the chainable helpers
    /// (`label`, `labels`, `schema`) to populate caller-settable fields.
    /// System-managed fields are stamped by the ticket system at
    /// insertion time; the placeholders set here are overwritten.
    pub fn new<T: Serialize>(task: T) -> Self {
        let value = serde_json::to_value(task).expect("Ticket::new: value must serialize to JSON");
        Self {
            task: value,
            labels: Vec::new(),
            schema: None,
            key: String::new(),
            status: Status::Todo,
            reporter: String::new(),
            created_at: 0,
            started_at: None,
            finished_at: None,
            failed_at: None,
            result: None,
            parent: None,
            comments: Vec::new(),
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

    /// Record a back-reference to another ticket. The meaning is
    /// caller-defined: `handover_ticket` uses it to chain a
    /// child to the ticket that handed off, but any code building a
    /// ticket may set it to express a parent relationship.
    pub fn parent(mut self, key: impl Into<String>) -> Self {
        self.parent = Some(key.into());
        self
    }

    // ---- read-only accessors for system-managed fields ----

    pub fn key(&self) -> &str {
        &self.key
    }

    /// Current status as a lowercase string: `"todo"`, `"in_progress"`,
    /// `"done"`, or `"failed"`. Transitions are driven internally by the
    /// agent loop as it claims, finishes, and fails tickets.
    pub fn status(&self) -> &'static str {
        self.status.as_str()
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

    /// Elapsed duration from creation to terminal status (Done or
    /// Failed), `None` while the ticket has not yet reached one. Mirrors
    /// the naming convention of [`Stats::run_duration`](crate::Stats::run_duration).
    pub fn elapsed(&self) -> Option<Duration> {
        let terminal = self.finished_at.or(self.failed_at)?;
        Some(Duration::from_millis(
            terminal.saturating_sub(self.created_at),
        ))
    }

    pub fn result(&self) -> Option<&serde_json::Value> {
        self.result.as_ref()
    }

    /// Result payload rendered as a String, or `None` when the ticket has
    /// no recorded result. Convenience for callers that want a flat
    /// string view of the result regardless of the underlying JSON shape.
    pub fn result_string(&self) -> Option<String> {
        self.result.as_ref().map(|v| match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    }

    /// Result payload deserialized into `R`, or `None` when the ticket
    /// has no recorded result. Panics when the recorded `Value` does not
    /// deserialize into `R`: schema-bound tickets validate the result
    /// before the framework marks them done, so a mismatch here means
    /// the validator and the type drifted apart.
    pub fn result_as<R>(&self) -> Option<R>
    where
        R: serde::de::DeserializeOwned,
    {
        self.result.as_ref().map(|v| {
            serde_json::from_value(v.clone()).expect(
                "ticket result does not match requested type — validator and type are out of sync",
            )
        })
    }

    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l == label)
    }

    /// Back-reference set via [`Self::parent`], or `None` when no
    /// parent was recorded.
    pub fn parent_key(&self) -> Option<&str> {
        self.parent.as_deref()
    }

    /// Transcript of messages the agent loop sent to the provider for
    /// this ticket, in order. Starts with a leading `system` entry
    /// carrying the system prompt; everything after is `user` /
    /// `assistant` turns.
    pub fn comments(&self) -> &[Comment] {
        &self.comments
    }

    /// True when the model owes a turn: the transcript is empty, the
    /// last comment is user-side, or the model's last reply still has
    /// unresolved tool calls.
    pub fn is_waiting_for_response(&self) -> bool {
        let Some(c) = self.comments.last() else {
            return true;
        };
        if c.author != "assistant" {
            return true;
        }
        c.content
            .iter()
            .any(|x| matches!(x, CommentContent::ToolUse { .. }))
    }

    /// Reduce the transcript to just `summary_text`: every non-system
    /// comment is dropped and a single `user` comment carrying
    /// `summary_text` is appended. System-author comments (the system
    /// prompt) survive unchanged. Used by the loop after a successful
    /// compaction.
    pub(crate) fn summarize(&mut self, summary_text: String) {
        self.comments.retain(|c| c.author == "system");
        self.comments.push(Comment::user_text(summary_text));
    }

}

impl crate::persistence::Persist for Ticket {
    type Key = String;

    fn save(&self, dir: &Path) -> io::Result<()> {
        let path = dir
            .join("tickets")
            .join(&self.key)
            .join(format!("ticket.{}.json", now_millis()));
        let body = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
        crate::persistence::write_atomic(&path, &body)
    }

    fn load(dir: &Path, key: &Self::Key) -> io::Result<Self> {
        let ticket_dir = dir.join("tickets").join(key);
        let path = crate::persistence::latest_path(&ticket_dir).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("no ticket file for {key}"))
        })?;
        let bytes = std::fs::read(&path)?;
        serde_json::from_slice::<Ticket>(&bytes).map_err(io::Error::other)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Status {
    Todo,
    InProgress,
    Done,
    Failed,
}

impl Status {
    /// Lowercase wire form: `"todo"`, `"in_progress"`, `"done"`, `"failed"`.
    /// Single source of truth for the string rendering used by
    /// [`Ticket::status`] and the `tickets.jsonl` event log.
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Todo => "todo",
            Status::InProgress => "in_progress",
            Status::Done => "done",
            Status::Failed => "failed",
        }
    }
}

/// One entry in a ticket's transcript.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Comment {
    /// Role of the originating message: `"user"` or `"assistant"`. The
    /// agent loop also writes `"system"` entries for the system prompt
    /// and for compaction boundaries; those are filtered when
    /// projecting comments back into `Message` values for the provider.
    pub author: String,
    pub content: Vec<CommentContent>,
    /// Millis since epoch.
    pub created_at: u64,
}

/// Ticket-side mirror of [`ContentBlock`]. Keeps the public ticket
/// surface free of provider types while still recording every payload
/// shape the agent loop sends.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CommentContent {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        id: String,
        output: String,
        succeeded: bool,
        /// Absolute path of the offloaded full payload when the inline
        /// `output` carries only a preview. `None` when the full output
        /// fit inline.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<PathBuf>,
    },
}

impl Comment {
    /// Build a `"user"` comment from the provider blocks the loop sent.
    /// `paths` maps `tool_use_id → absolute path` for tool results whose
    /// full output was offloaded to disk; empty when nothing was offloaded.
    pub(crate) fn user(blocks: &[ContentBlock], paths: &HashMap<String, PathBuf>) -> Self {
        Self {
            author: "user".into(),
            content: to_comment_content(blocks, paths),
            created_at: now_millis(),
        }
    }

    /// Build a `"user"` comment carrying a single text payload.
    pub(crate) fn user_text(text: impl Into<String>) -> Self {
        Self {
            author: "user".into(),
            content: vec![CommentContent::Text(text.into())],
            created_at: now_millis(),
        }
    }

    /// Build an `"assistant"` comment from the model's reply content.
    /// Assistant content never carries tool-result blocks, so no paths
    /// map is needed.
    pub(crate) fn assistant(blocks: &[ContentBlock]) -> Self {
        Self {
            author: "assistant".into(),
            content: to_comment_content(blocks, &HashMap::new()),
            created_at: now_millis(),
        }
    }

    /// Build a `"system"` comment carrying a single text payload. Used
    /// for the leading system-prompt entry and compaction boundaries.
    pub(crate) fn system_text(text: impl Into<String>) -> Self {
        Self {
            author: "system".into(),
            content: vec![CommentContent::Text(text.into())],
            created_at: now_millis(),
        }
    }
}

/// Project a slice of comments into the provider's `Message` values.
/// Skips `system`-author comments: the system prompt is passed via
/// `request.system_prompt`, and compaction-boundary comments are
/// audit markers only.
pub(crate) fn to_messages(comments: &[Comment]) -> Vec<Message> {
    comments.iter().filter_map(comment_to_message).collect()
}

fn to_comment_content(
    blocks: &[ContentBlock],
    paths: &HashMap<String, PathBuf>,
) -> Vec<CommentContent> {
    blocks
        .iter()
        .map(|b| content_block_to_comment(b, paths))
        .collect()
}

fn content_block_to_comment(b: &ContentBlock, paths: &HashMap<String, PathBuf>) -> CommentContent {
    match b {
        ContentBlock::Text { text } => CommentContent::Text(text.clone()),
        ContentBlock::ToolUse { id, name, input } => CommentContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            succeeded,
        } => CommentContent::ToolResult {
            id: tool_use_id.clone(),
            output: content.clone(),
            succeeded: *succeeded,
            path: paths.get(tool_use_id).cloned(),
        },
    }
}

fn to_content_blocks(content: &[CommentContent]) -> Vec<ContentBlock> {
    content
        .iter()
        .map(|c| match c {
            CommentContent::Text(text) => ContentBlock::Text { text: text.clone() },
            CommentContent::ToolUse { id, name, input } => ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            CommentContent::ToolResult {
                id,
                output,
                succeeded,
                path: _,
            } => ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content: output.clone(),
                succeeded: *succeeded,
            },
        })
        .collect()
}

fn comment_to_message(c: &Comment) -> Option<Message> {
    let content = to_content_blocks(&c.content);
    match c.author.as_str() {
        "user" => Some(Message::User { content }),
        "assistant" => Some(Message::Assistant { content }),
        _ => None,
    }
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
/// agents, the policies, the interrupt signal, the run stats, and the
/// background task driving the agent loop. Always lives behind
/// `Arc<TicketSystem>` — `new()` returns `Arc<Self>` so each bound
/// `Agent` can hold a `Weak<TicketSystem>` without creating an Arc
/// cycle through the system's `Vec<Agent>`.
pub struct TicketSystem {
    weak_self: Weak<TicketSystem>,
    pub(crate) tickets: Mutex<HashMap<String, Ticket>>,
    agents: Mutex<Vec<Agent>>,
    policies: Mutex<Policies>,
    pub(crate) interrupt_signal: Mutex<Arc<AtomicBool>>,
    pub(crate) stats: Stats,
    dir: Mutex<PathBuf>,
    tickets_log_lock: Mutex<()>,
    join_handle: Mutex<Option<JoinHandle<()>>>,
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
            dir: Mutex::new(PathBuf::from(".agentwerk")),
            tickets_log_lock: Mutex::new(()),
            join_handle: Mutex::new(None),
        })
    }

    /// Open or create a ticket system rooted at `tickets_dir`. Loads
    /// the newest `ticket.<ts>.json` per key under `<tickets_dir>/tickets/`
    /// into the in-memory store and seeds `Stats` from
    /// `<tickets_dir>/stats.json` (or, when that file is missing or
    /// malformed, by deriving from the loaded tickets) so success rate
    /// and counters stay continuous across restarts.
    ///
    /// Pointing this and `Knowledge::load` at the same dir co-locates
    /// knowledge pages with `results.jsonl` and `tickets.jsonl`.
    ///
    /// `InProgress` tickets keep their status and their transcript; the
    /// loop's resume path (`agents/loop.rs`) picks them back up under
    /// the agent whose name is already in the ticket's `labels`.
    ///
    /// Caller contracts:
    /// - Tickets dirs deleted by hand break the `TICKET-N` counter:
    ///   the next inserted ticket may collide with an existing key.
    /// - Agent names must stay stable across restarts; the loop
    ///   matches `InProgress` tickets by name via the ticket's labels.
    pub fn load(tickets_dir: impl Into<PathBuf>) -> io::Result<Arc<Self>> {
        let tickets_dir = tickets_dir.into();
        std::fs::create_dir_all(tickets_dir.join("tickets"))?;

        let mut tickets = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(tickets_dir.join("tickets")) {
            for entry in entries.flatten() {
                let key_dir = entry.path();
                if !key_dir.is_dir() {
                    continue;
                }
                let Some(path) = crate::persistence::latest_path(&key_dir) else {
                    continue;
                };
                let Ok(bytes) = std::fs::read(&path) else {
                    continue;
                };
                let Ok(ticket) = serde_json::from_slice::<Ticket>(&bytes) else {
                    continue;
                };
                tickets.insert(ticket.key.clone(), ticket);
            }
        }

        let stats = Stats::load(&tickets_dir).unwrap_or_else(|_| Stats::derive(&tickets));

        Ok(Arc::new_cyclic(|weak| Self {
            weak_self: weak.clone(),
            tickets: Mutex::new(tickets),
            agents: Mutex::new(Vec::new()),
            policies: Mutex::new(Policies::default()),
            interrupt_signal: Mutex::new(Arc::new(AtomicBool::new(false))),
            stats,
            dir: Mutex::new(tickets_dir),
            tickets_log_lock: Mutex::new(()),
            join_handle: Mutex::new(None),
        }))
    }

    /// Run-time counters. Read after `run` / `finish` returns.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    // ---- policy builders ----

    pub fn max_steps(&self, n: u32) -> &Self {
        self.policies.lock().unwrap().max_steps = Some(n);
        self
    }

    pub fn max_input_tokens(&self, n: u64) -> &Self {
        self.policies.lock().unwrap().max_input_tokens = Some(n);
        self
    }

    pub fn max_output_tokens(&self, n: u64) -> &Self {
        self.policies.lock().unwrap().max_output_tokens = Some(n);
        self
    }

    pub fn max_request_tokens(&self, n: u32) -> &Self {
        self.policies.lock().unwrap().max_request_tokens = Some(n);
        self
    }

    pub fn max_schema_retries(&self, n: u32) -> &Self {
        self.policies.lock().unwrap().max_schema_retries = Some(n);
        self
    }

    pub fn max_request_retries(&self, n: u32) -> &Self {
        self.policies.lock().unwrap().max_request_retries = n;
        self
    }

    pub fn request_retry_delay(&self, d: Duration) -> &Self {
        self.policies.lock().unwrap().request_retry_delay = d;
        self
    }

    /// Maximum elapsed duration `finish` will wait before tripping
    /// the interrupt signal and returning. Hitting the cap is a
    /// graceful stop, not a `PolicyViolated` event.
    pub fn max_time(&self, d: Duration) -> &Self {
        self.policies.lock().unwrap().max_time = Some(d);
        self
    }

    /// Flip the cancel signal on the first SIGINT. Spawns a background
    /// tokio task that listens for ctrl-c once and exits. Callers that
    /// need escalation (e.g. force-exit on a second press) install
    /// their own listener and call [`Self::cancel`] from it.
    pub fn cancel_on_ctrl_c(&self) -> &Self {
        let signal = Arc::clone(&self.interrupt_signal.lock().unwrap());
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            signal.store(true, Ordering::Relaxed);
        });
        self
    }

    /// Override the directory under which the system writes
    /// `results.jsonl`, `tickets.jsonl`, and per-ticket
    /// `tickets/<key>/ticket.<ts>.json` files. Defaults to `./.agentwerk`.
    /// Knowledge co-locates with these files when `Knowledge::open`
    /// points at the same directory.
    pub fn dir(&self, dir: impl Into<PathBuf>) -> &Self {
        *self.dir.lock().unwrap() = dir.into();
        self
    }

    pub(crate) fn dir_value(&self) -> PathBuf {
        self.dir.lock().unwrap().clone()
    }

    // ---- ticket-creation API mirrored on Agent ----

    /// Enqueue a ticket carrying `task` as its body. Returns the new
    /// ticket's key.
    pub fn task<T: Serialize>(&self, task: T) -> String {
        self.dispatch(Ticket::new(task))
    }

    /// Enqueue a ticket carrying `task`, attached to `label` for Path B
    /// routing. Returns the new ticket's key.
    pub fn task_labeled<T: Serialize>(&self, task: T, label: impl Into<String>) -> String {
        self.dispatch(Ticket::new(task).label(label))
    }

    /// Enqueue a fully-built `Ticket`. System-managed fields (key,
    /// reporter, created_at, status, result) are overwritten. To pin the
    /// ticket to a specific agent, label it with the agent's name.
    /// Compose schema and label via `Ticket::new(...).schema(...).label(...)`.
    /// Returns the inserted ticket's key.
    pub fn ticket(&self, ticket: Ticket) -> String {
        self.dispatch(ticket)
    }

    /// Append a user-side text comment to an existing ticket. The
    /// agent loop's wait-for-input branch picks it up on the next
    /// iteration; the model sees it as the next `user` message in
    /// the conversation. Use this to drive multi-turn chats on one
    /// ticket instead of creating a new ticket per turn.
    pub fn comment(&self, key: &str, content: impl Into<String>) -> &Self {
        self.add_comment(key, Comment::user_text(content));
        self
    }

    fn dispatch(&self, ticket: Ticket) -> String {
        self.insert(ticket, "user".to_string())
    }

    // ---- inherent ticket-store methods ----

    /// Insert `ticket`, stamping system fields. The ticket is always born
    /// `Todo`; to pin it to a specific agent, label it with the agent's
    /// name. Returns the inserted ticket's key.
    pub(crate) fn insert(&self, mut ticket: Ticket, reporter: String) -> String {
        let mut store = self.tickets.lock().unwrap();
        let id = store.len() + 1;
        ticket.key = format!("TICKET-{id}");
        ticket.created_at = now_millis();
        ticket.reporter = reporter;
        ticket.result = None;
        ticket.status = Status::Todo;
        let key = ticket.key.clone();
        let labels = ticket.labels.clone();
        let reporter = ticket.reporter.clone();
        let task = ticket.task.clone();
        let created_at = ticket.created_at;
        let parent = ticket.parent.clone();
        store.insert(key.clone(), ticket);
        drop(store);
        self.save_ticket(&key);
        TicketStats::record_created(&self.stats);
        for l in &labels {
            let slice = self.stats.stats_for_label(l);
            TicketStats::record_created(&*slice);
        }
        let mut event = serde_json::json!({
            "event": "created",
            "ts": created_at,
            "key": key,
            "reporter": reporter,
            "labels": labels,
            "task": task,
        });
        if let Some(p) = &parent {
            event["parent"] = serde_json::Value::String(p.clone());
        }
        self.append_ticket_event(event);
        key
    }

    /// Write the ticket at `key` to disk. No-op when the ticket is missing.
    fn save_ticket(&self, key: &str) {
        if let Some(t) = self.get(key) {
            use crate::persistence::Persist;
            let _ = t.save(&self.dir_value());
        }
    }

    /// Append one JSON line to `<dir>/tickets.jsonl` and refresh
    /// `<dir>/stats.json` from the current counters. Both writes happen
    /// under the same lock so a concurrent reader sees consistent
    /// observational state. Errors are swallowed: persistence is
    /// best-effort, not load-bearing for run correctness.
    pub(crate) fn append_ticket_event(&self, event: serde_json::Value) {
        use crate::persistence::{Append, Persist, TicketEvents};
        let dir = self.dir_value();
        let _guard = self.tickets_log_lock.lock().unwrap();
        let _ = TicketEvents::append(&dir, &event);
        let _ = self.stats.save(&dir);
    }

    /// Write a tool's full output to `<dir>/tickets/<key>/outputs/<tool_use_id>.txt`.
    /// Returns the path relative to the configured `dir` on success,
    /// `None` when the write fails. The relative form keeps the comment
    /// transcript portable across moves of the tickets dir; join with
    /// [`Self::dir_value`] to recover the on-disk path. Best-effort,
    /// matching the surrounding observational-persistence contract.
    pub(crate) fn write_tool_output(
        &self,
        key: &str,
        tool_use_id: &str,
        content: &str,
    ) -> Option<PathBuf> {
        let rel = crate::persistence::output_path(key, tool_use_id);
        let absolute = self.dir_value().join(&rel);
        crate::persistence::write_atomic(&absolute, content.as_bytes())
            .ok()
            .map(|_| rel)
    }

    /// Returns a clone of the ticket at `key`, if any.
    pub fn get(&self, key: &str) -> Option<Ticket> {
        self.tickets.lock().unwrap().get(key).cloned()
    }

    /// Atomically find a `Todo` ticket matching `predicate`, label it
    /// with `agent_name`, and transition to `InProgress`.
    pub(crate) fn claim<F>(&self, predicate: F, agent_name: &str) -> Option<String>
    where
        F: Fn(&Ticket) -> bool,
    {
        let now = now_millis();
        let (key, prev, durations, labels) = {
            let mut store = self.tickets.lock().unwrap();
            let mut candidates: Vec<&String> = store
                .iter()
                .filter(|(_, t)| predicate(t))
                .map(|(k, _)| k)
                .collect();
            candidates.sort_by_key(|k| {
                let t = &store[k.as_str()];
                (t.created_at, numeric_id(k))
            });
            let key = candidates.into_iter().next()?.clone();
            let ticket = store.get_mut(&key)?;
            if ticket.status != Status::Todo {
                return None;
            }
            if !ticket.labels.iter().any(|l| l == agent_name) {
                ticket.labels.push(agent_name.to_string());
            }
            let prev = ticket.status;
            stamp_transition_timestamps(ticket, Status::InProgress, now);
            ticket.status = Status::InProgress;
            let durations = terminal_durations(ticket);
            let labels = ticket.labels.clone();
            (key, prev, durations, labels)
        };
        self.record_transition(&key, prev, Status::InProgress, now, durations, &labels);
        self.save_ticket(&key);
        Some(key)
    }

    /// Append `comment` to the ticket's transcript. No-op when the
    /// ticket is missing: the loop drops out shortly afterwards on the
    /// same condition.
    pub(crate) fn add_comment(&self, key: &str, comment: Comment) {
        let ticket_copy = {
            let mut store = self.tickets.lock().unwrap();
            let Some(t) = store.get_mut(key) else { return };
            t.comments.push(comment);
            t.clone()
        };
        {
            use crate::persistence::Persist;
            let _ = ticket_copy.save(&self.dir_value());
        }
    }

    /// Transition a ticket to `Done`.
    pub(crate) fn set_done(&self, key: &str) -> Result<(), TicketError> {
        self.set_final_status(key, Status::Done)
    }

    /// Transition a ticket to `Failed`.
    pub(crate) fn set_failed(&self, key: &str) -> Result<(), TicketError> {
        self.set_final_status(key, Status::Failed)
    }

    fn set_final_status(&self, key: &str, status: Status) -> Result<(), TicketError> {
        let now = now_millis();
        let (prev, durations, labels) = {
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
        self.record_transition(key, prev, status, now, durations, &labels);
        self.save_ticket(key);
        Ok(())
    }

    /// Fire stats recorders and ticket log after a transition.
    fn record_transition(
        &self,
        key: &str,
        prev: Status,
        next: Status,
        now: u64,
        durations: (Duration, Duration),
        labels: &[String],
    ) {
        fire_transition_recorder(&self.stats, prev, next, now, durations);
        fire_label_transition(&self.stats, next, durations, labels);
        self.log_transition(key, prev, next, now, durations, labels);
    }

    /// Append a `started` / `done` / `failed` line to `tickets.jsonl` if
    /// `prev → next` is observable. No-op when prev == next or when the
    /// transition is not one we surface.
    fn log_transition(
        &self,
        key: &str,
        prev: Status,
        next: Status,
        ts: u64,
        (ticket_duration, work_duration): (Duration, Duration),
        labels: &[String],
    ) {
        if prev == next {
            return;
        }
        if prev == Status::Todo && next == Status::InProgress {
            self.append_ticket_event(serde_json::json!({
                "event": "started",
                "ts": ts,
                "key": key,
                "labels": labels,
            }));
        }
        match next {
            Status::Done | Status::Failed => {
                let event = if next == Status::Done {
                    "done"
                } else {
                    "failed"
                };
                self.append_ticket_event(serde_json::json!({
                    "event": event,
                    "ts": ts,
                    "key": key,
                    "duration_ms": ticket_duration.as_millis() as u64,
                    "work_ms": work_duration.as_millis() as u64,
                }));
            }
            _ => {}
        }
    }

    /// Attach a result payload to the ticket at `key`.
    pub(crate) fn set_result(
        &self,
        key: &str,
        result: serde_json::Value,
    ) -> Result<(), TicketError> {
        let ticket_copy = {
            let mut store = self.tickets.lock().unwrap();
            let ticket = store
                .get_mut(key)
                .ok_or_else(|| TicketError::TicketMissing {
                    key: key.to_string(),
                })?;
            ticket.result = Some(result);
            ticket.clone()
        };
        {
            use crate::persistence::Persist;
            let _ = ticket_copy.save(&self.dir_value());
        }
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
        agent.ensure_knowledge_bound();
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

impl TicketSystem {
    /// Bind `agent` to this system: drain any tickets it queued in its
    /// default system into this one and push a clone onto this system's
    /// agents list. Returns the wired agent for chaining (`.task(...)`
    /// etc.).
    ///
    /// May be called before or after `run()` / `finish()`. When called
    /// after `run()`, the new agent starts polling for tickets within
    /// roughly one `IDLE_POLL_INTERVAL` (~100 ms).
    pub fn agent(&self, mut agent: Agent) -> Agent {
        self.bind_agent(&mut agent);
        agent
    }

    /// Bind `n` agents built by `build`. `build(i)` receives the worker
    /// index (0-based) so the caller can suffix names or pick
    /// per-worker resources without writing the loop themselves.
    pub fn pool<F>(&self, n: usize, build: F) -> &Self
    where
        F: Fn(usize) -> Agent,
    {
        for i in 0..n {
            let mut agent = build(i);
            self.bind_agent(&mut agent);
        }
        self
    }

    /// Start the agent loop on a background tokio task. Tickets queued
    /// afterwards are picked up within ~`IDLE_POLL_INTERVAL`. Pair with
    /// [`Self::finish`] to wait for the queue to empty, or with
    /// [`Self::stop`] to cancel and join.
    pub fn start(&self) -> &Self {
        let signal = Arc::clone(&self.interrupt_signal.lock().unwrap());
        // Reset so a system can be re-started after a previous finish
        // left the flag set.
        signal.store(false, Ordering::Relaxed);
        let supervisor = self
            .weak_self
            .upgrade()
            .expect("TicketSystem dropped during start");
        let join = tokio::spawn(async move {
            run_main_loop(&supervisor).await;
            supervisor.stats.mark_finished(now_millis());
        });
        *self.join_handle.lock().unwrap() = Some(join);
        self
    }

    /// Process every queued ticket, then return. Starts a run if none
    /// is in flight; otherwise watches the in-flight one. Polls every
    /// 20 ms; exits when `pending_count == 0`, a policy trips, or
    /// `max_time` elapses. Returns `&self` so callers can chain
    /// [`Self::last_result`], [`Self::all_results`], or
    /// [`Self::tickets`] without rebinding.
    pub async fn finish(&self) -> &Self {
        if self.join_handle.lock().unwrap().is_none() {
            self.start();
        }
        let started = Instant::now();
        let policies = self.policies();
        let signal = Arc::clone(&self.interrupt_signal.lock().unwrap());
        loop {
            tokio::time::sleep(Duration::from_millis(20)).await;
            if signal.load(Ordering::Relaxed) {
                break;
            }
            if policy_violated(&policies, &self.stats) {
                signal.store(true, Ordering::Relaxed);
                break;
            }
            if let Some(limit) = policies.max_time {
                if started.elapsed() >= limit {
                    signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
            if pending_count(self) == 0 {
                signal.store(true, Ordering::Relaxed);
                break;
            }
        }
        self.take_join_handle().await;
        self.stats.mark_finished(now_millis());
        self
    }

    /// Flip the cancel signal. Sync, so it composes with ctrl-c
    /// handlers, drop guards, and other sync callers. The background
    /// task notices on its next poll and exits gracefully. Pair with
    /// [`Self::stop`] from async code if you also want to wait for the
    /// task to exit.
    pub fn cancel(&self) {
        self.interrupt_signal
            .lock()
            .unwrap()
            .store(true, Ordering::Relaxed);
    }

    /// Cancel and wait for the background task to exit. The run
    /// finishes its in-flight ticket; queued tickets are abandoned.
    /// No-op when no run is in flight.
    pub async fn stop(&self) {
        self.cancel();
        self.take_join_handle().await;
    }

    async fn take_join_handle(&self) {
        let handle = self.join_handle.lock().unwrap().take();
        if let Some(h) = handle {
            let _ = h.await;
        }
    }

    /// True once cancellation has been requested. The background task
    /// finishes its in-flight ticket and exits shortly after.
    pub fn is_cancelled(&self) -> bool {
        self.interrupt_signal
            .lock()
            .unwrap()
            .load(Ordering::Relaxed)
    }

    /// Most recent finished ticket's result rendered as a String.
    pub fn last_result(&self) -> Option<String> {
        self.all_results().into_iter().next_back()
    }

    /// Every finished ticket's result rendered as a String, in creation
    /// order.
    pub fn all_results(&self) -> Vec<String> {
        self.filter(|t| t.status == Status::Done && t.result.is_some())
            .iter()
            .filter_map(Ticket::result_string)
            .collect()
    }
}

/// Whether the run-wide policies have been exceeded by the current
/// stats reading. Used by the `finish` watcher and by the per-agent
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

pub(crate) fn pending_count(ticket_system: &TicketSystem) -> usize {
    ticket_system
        .tickets
        .lock()
        .unwrap()
        .values()
        .filter(|t| match t.status {
            Status::Todo => true,
            Status::InProgress => t.is_waiting_for_response(),
            _ => false,
        })
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
    let ticket_duration = ticket.elapsed().unwrap_or_default();
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
/// `started_at` stays zero so `elapsed()` reads `None` on a slice.
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

pub(crate) fn now_millis() -> u64 {
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

    /// Build a `TicketSystem` rooted at a fresh `TempDir` so the default
    /// `.agentwerk` directory never lands in the source tree during tests.
    /// Hold the returned `TempDir` for the test's lifetime.
    fn test_system() -> (Arc<TicketSystem>, crate::test_util::TempDir) {
        let dir = crate::test_util::TempDir::new().unwrap();
        let built = TicketSystem::new();
        built.dir(dir.path().to_path_buf());
        (built, dir)
    }

    fn attach_done_result(sys: &TicketSystem, key: &str, result: &str) {
        sys.set_result(key, serde_json::Value::String(result.into()))
            .unwrap();
        sys.set_done(key).unwrap();
    }

    #[test]
    fn task_creates_ticket_with_user_reporter() {
        let (sys, _tmp) = test_system();
        sys.task("hello");
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.task, serde_json::Value::String("hello".into()));
        assert_eq!(t.reporter(), "user");
        assert_eq!(t.status, Status::Todo);
    }

    #[test]
    fn task_labeled_attaches_label_and_leaves_status_todo() {
        let (sys, _tmp) = test_system();
        sys.task_labeled("hello", "research");
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.labels, vec!["research".to_string()]);
        assert_eq!(t.status, Status::Todo);
    }

    #[test]
    fn create_with_named_label_is_born_todo_and_carries_label() {
        let (sys, _tmp) = test_system();
        sys.ticket(Ticket::new("specific work for alice").label("alice"));
        let t = sys.get("TICKET-1").unwrap();
        assert!(t.has_label("alice"));
        assert_eq!(t.status, Status::Todo);
    }

    #[test]
    fn create_with_label_and_schema_is_stored_verbatim() {
        let (sys, _tmp) = test_system();
        let schema = crate::schemas::Schema::parse(serde_json::json!({"type": "string"})).unwrap();
        sys.ticket(Ticket::new("x").label("urgent").schema(schema));
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.labels, vec!["urgent".to_string()]);
        assert!(t.schema.is_some());
    }

    #[test]
    fn ticket_system_handle_is_shared_between_caller_and_added_agent() {
        let (sys, _tmp) = test_system();
        let alice = sys.agent(Agent::new().name("alice"));
        // Alice's task lands in the same queue.
        alice.task("from alice");
        sys.task("from system");
        let all_keys: Vec<String> = sys
            .filter(|t| t.status == Status::Todo)
            .iter()
            .map(|t| t.key().to_string())
            .collect();
        assert_eq!(all_keys.len(), 2);
    }

    #[test]
    fn agent_must_be_bound_before_task() {
        let alice = Agent::new().name("alice");
        let (sys, _tmp) = test_system();
        let alice = sys.agent(alice);
        // Bound: task() works, lands in the shared queue.
        alice.task("first");
        alice.task("second");
        assert_eq!(sys.count(|t| t.status == Status::Todo), 2);
    }

    #[test]
    #[should_panic(expected = "Agent::task requires a bound TicketSystem")]
    fn unbound_agent_task_panics() {
        let alice = Agent::new().name("alice");
        alice.task("never lands");
    }

    #[test]
    fn search_matches_string_task_case_insensitively() {
        let (sys, _tmp) = test_system();
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
        let (sys, _tmp) = test_system();
        sys.task("hi");
        sys.set_result("TICKET-1", serde_json::Value::String("answer".into()))
            .unwrap();
        let stored = sys.get("TICKET-1").unwrap();
        assert_eq!(
            stored.result(),
            Some(&serde_json::Value::String("answer".into()))
        );
        assert_eq!(stored.result_string().as_deref(), Some("answer"));
    }

    #[test]
    fn first_returns_none_on_empty_system() {
        let (sys, _tmp) = test_system();
        assert!(sys.first().is_none());
        assert!(sys.tickets().is_empty());
    }

    #[test]
    fn first_returns_earliest_ticket_by_creation() {
        let (sys, _tmp) = test_system();
        sys.task("first");
        sys.task("second");
        sys.task("third");
        let first = sys.first().unwrap();
        assert_eq!(first.key(), "TICKET-1");
        assert_eq!(first.task, serde_json::Value::String("first".into()));
    }

    #[test]
    fn tickets_returns_all_in_creation_order() {
        let (sys, _tmp) = test_system();
        sys.task("a");
        sys.task("b");
        sys.task("c");
        let all = sys.tickets();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].key(), "TICKET-1");
        assert_eq!(all[1].key(), "TICKET-2");
        assert_eq!(all[2].key(), "TICKET-3");
    }

    #[test]
    fn all_results_returns_done_payloads_in_creation_order() {
        let (sys, _tmp) = test_system();
        sys.task("a");
        sys.task("b");
        sys.task("c");
        attach_done_result(&sys, "TICKET-1", "first");
        attach_done_result(&sys, "TICKET-3", "third");
        assert_eq!(sys.all_results(), vec!["first", "third"]);
    }

    #[test]
    fn last_result_returns_last_done_payload() {
        let (sys, _tmp) = test_system();
        sys.task("a");
        sys.task("b");
        attach_done_result(&sys, "TICKET-2", "second");
        attach_done_result(&sys, "TICKET-1", "first");
        assert_eq!(sys.last_result().as_deref(), Some("second"));
    }

    #[test]
    fn all_results_orders_by_creation_regardless_of_done_order() {
        let (sys, _tmp) = test_system();
        sys.task("a");
        sys.task("b");
        sys.task("c");
        attach_done_result(&sys, "TICKET-3", "third");
        attach_done_result(&sys, "TICKET-1", "first");
        attach_done_result(&sys, "TICKET-2", "second");
        assert_eq!(sys.all_results(), vec!["first", "second", "third"]);
    }

    #[test]
    fn results_are_empty_when_nothing_done() {
        let (sys, _tmp) = test_system();
        sys.task("pending");
        assert!(sys.last_result().is_none());
        assert!(sys.all_results().is_empty());
    }

    #[test]
    fn done_and_failed_filter_by_status() {
        let (sys, _tmp) = test_system();
        sys.task("ok");
        sys.task("oops");
        sys.task("pending");
        sys.claim(|t| t.key() == "TICKET-1", "agent");
        sys.set_done("TICKET-1").unwrap();
        sys.set_failed("TICKET-2").unwrap();
        let done = sys.filter(|t| t.status == Status::Done);
        let failed = sys.filter(|t| t.status == Status::Failed);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].key(), "TICKET-1");
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].key(), "TICKET-2");
    }

    #[test]
    fn ticket_status_transitions_record_stats() {
        let (sys, _tmp) = test_system();
        sys.task("a");
        sys.task("b");
        sys.task("c");
        // Created 3 tickets.
        assert_eq!(sys.stats().tickets_created(), 3);
        sys.claim(|t| t.key() == "TICKET-1", "agent");
        sys.set_done("TICKET-1").unwrap();
        sys.claim(|t| t.key() == "TICKET-2", "agent");
        sys.set_failed("TICKET-2").unwrap();
        assert_eq!(sys.stats().tickets_done(), 1);
        assert_eq!(sys.stats().tickets_failed(), 1);
    }

    #[test]
    fn stats_for_label_counts_creation_per_label() {
        let (sys, _tmp) = test_system();
        sys.ticket(Ticket::new("a").labels(["scan", "high"]));
        sys.ticket(Ticket::new("b").label("scan"));
        sys.ticket(Ticket::new("c"));
        let stats = sys.stats();
        assert_eq!(stats.tickets_created(), 3);
        assert_eq!(stats.stats_for_label("scan").tickets_created(), 2);
        assert_eq!(stats.stats_for_label("high").tickets_created(), 1);
        assert_eq!(stats.stats_for_label("never-used").tickets_created(), 0);
    }

    #[test]
    fn stats_for_label_counts_terminal_transitions_per_label() {
        let (sys, _tmp) = test_system();
        sys.ticket(Ticket::new("a").labels(["scan", "high"]));
        sys.ticket(Ticket::new("b").label("scan"));
        sys.claim(|t| t.key() == "TICKET-1", "agent");
        sys.set_done("TICKET-1").unwrap();
        sys.claim(|t| t.key() == "TICKET-2", "agent");
        sys.set_failed("TICKET-2").unwrap();
        let stats = sys.stats();
        let scan = stats.stats_for_label("scan");
        let high = stats.stats_for_label("high");
        assert_eq!(scan.tickets_done(), 1);
        assert_eq!(scan.tickets_failed(), 1);
        assert_eq!(scan.tickets_success_rate(), Some(0.5));
        assert_eq!(high.tickets_done(), 1);
        assert_eq!(high.tickets_failed(), 0);
        assert_eq!(high.tickets_success_rate(), Some(1.0));
    }

    #[test]
    fn stats_for_label_set_failed_path_records_per_label() {
        let (sys, _tmp) = test_system();
        sys.ticket(Ticket::new("a").label("scan"));
        sys.set_failed("TICKET-1").unwrap();
        assert_eq!(sys.stats().stats_for_label("scan").tickets_failed(), 1);
    }

    #[test]
    fn stats_for_label_unaffected_by_no_label_ticket() {
        let (sys, _tmp) = test_system();
        sys.ticket(Ticket::new("a"));
        sys.claim(|t| t.key() == "TICKET-1", "agent");
        sys.set_done("TICKET-1").unwrap();
        assert_eq!(sys.stats().tickets_done(), 1);
        assert_eq!(sys.stats().stats_for_label("scan").tickets_done(), 0);
        assert_eq!(sys.stats().stats_for_label("scan").tickets_created(), 0);
    }

    fn read_tickets_log(dir: &std::path::Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(dir.join("tickets.jsonl"))
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn workspace_emits_created_started_done_in_order() {
        let (sys, dir) = test_system();
        sys.task("hello");
        sys.claim(|t| t.key() == "TICKET-1", "agent");
        sys.set_done("TICKET-1").unwrap();
        let lines = read_tickets_log(dir.path());
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["event"], "created");
        assert_eq!(lines[0]["key"], "TICKET-1");
        assert_eq!(lines[0]["reporter"], "user");
        assert_eq!(lines[0]["task"], "hello");
        assert_eq!(lines[1]["event"], "started");
        assert_eq!(lines[1]["key"], "TICKET-1");
        assert_eq!(lines[2]["event"], "done");
        assert_eq!(lines[2]["key"], "TICKET-1");
        assert!(lines[2]["duration_ms"].is_u64());
        assert!(lines[2]["work_ms"].is_u64());
    }

    #[test]
    fn workspace_emits_failed_event_on_set_failed() {
        let (sys, dir) = test_system();
        sys.task("hello");
        sys.set_failed("TICKET-1").unwrap();
        let lines = read_tickets_log(dir.path());
        // created + failed (no started since Todo→Failed via set_failed)
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "created");
        assert_eq!(lines[1]["event"], "failed");
        assert_eq!(lines[1]["key"], "TICKET-1");
    }

    #[test]
    fn workspace_created_event_carries_labels_when_pinned() {
        let (sys, dir) = test_system();
        sys.ticket(Ticket::new("specific").label("alice"));
        let lines = read_tickets_log(dir.path());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["event"], "created");
        assert_eq!(lines[0]["labels"], serde_json::json!(["alice"]));
    }

    #[test]
    fn workspace_logs_one_line_per_lifecycle_step_for_multiple_tickets() {
        let (sys, dir) = test_system();
        sys.task("a");
        sys.task("b");
        sys.claim(|t| t.key() == "TICKET-1", "agent");
        sys.set_done("TICKET-1").unwrap();
        sys.claim(|t| t.key() == "TICKET-2", "agent");
        sys.set_failed("TICKET-2").unwrap();
        let lines = read_tickets_log(dir.path());
        // 2 created + 2 started + 1 done + 1 failed
        assert_eq!(lines.len(), 6);
    }

    // ---- claim / set_done / set_failed unit tests ----

    #[test]
    fn claim_transitions_todo_to_in_progress_and_adds_label() {
        let (sys, _tmp) = test_system();
        sys.task("hello");
        let key = sys.claim(|t| t.status == Status::Todo, "alice").unwrap();
        assert_eq!(key, "TICKET-1");
        let t = sys.get(&key).unwrap();
        assert_eq!(t.status, Status::InProgress);
        assert!(t.has_label("alice"));
        assert!(t.started_at().is_some());
    }

    #[test]
    fn claim_returns_none_when_no_ticket_matches() {
        let (sys, _tmp) = test_system();
        sys.task("hello");
        assert!(sys.claim(|t| t.has_label("nonexistent"), "alice").is_none());
    }

    #[test]
    fn second_claim_of_same_ticket_returns_none() {
        let (sys, _tmp) = test_system();
        sys.task("hello");
        let first = sys.claim(|t| t.key() == "TICKET-1", "alice");
        assert!(first.is_some());
        // Second claim: ticket is now InProgress, not Todo.
        let second = sys.claim(|t| t.key() == "TICKET-1", "bob");
        assert!(second.is_none());
    }

    #[test]
    fn claim_picks_earliest_eligible_ticket() {
        let (sys, _tmp) = test_system();
        sys.task("a");
        sys.task("b");
        sys.task("c");
        let key = sys.claim(|t| t.status == Status::Todo, "alice").unwrap();
        assert_eq!(key, "TICKET-1");
    }

    #[test]
    fn claim_emits_started_event_in_workspace_log() {
        let (sys, dir) = test_system();
        sys.task("hello");
        sys.claim(|t| t.status == Status::Todo, "alice");
        let lines = read_tickets_log(dir.path());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "created");
        assert_eq!(lines[1]["event"], "started");
        assert_eq!(lines[1]["key"], "TICKET-1");
    }

    #[test]
    fn set_done_transitions_to_done() {
        let (sys, _tmp) = test_system();
        sys.task("hello");
        sys.claim(|t| t.status == Status::Todo, "alice");
        sys.set_done("TICKET-1").unwrap();
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.status, Status::Done);
        assert!(t.finished_at().is_some());
    }

    #[test]
    fn set_failed_transitions_to_failed() {
        let (sys, _tmp) = test_system();
        sys.task("hello");
        sys.claim(|t| t.status == Status::Todo, "alice");
        sys.set_failed("TICKET-1").unwrap();
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.status, Status::Failed);
        assert!(t.failed_at().is_some());
    }

    #[test]
    fn ticket_parent_builder_round_trips() {
        let (sys, _tmp) = test_system();
        sys.ticket(Ticket::new("child body").parent("TICKET-1"));
        let stored = sys.get("TICKET-1").unwrap();
        assert_eq!(stored.parent_key(), Some("TICKET-1"));
    }

    #[test]
    fn write_tool_output_returns_relative_path_and_writes_absolute() {
        let (sys, dir) = test_system();
        sys.task("seed");
        let rel = sys
            .write_tool_output("TICKET-1", "call-1", "the full content")
            .expect("write succeeds when dir exists");
        let expected_rel: PathBuf = ["tickets", "TICKET-1", "outputs", "call-1.txt"]
            .iter()
            .collect();
        assert_eq!(rel, expected_rel);
        let body = std::fs::read_to_string(dir.path().join(&rel)).unwrap();
        assert_eq!(body, "the full content");
    }

    #[test]
    fn write_tool_output_creates_outputs_subdir_lazily() {
        let (sys, dir) = test_system();
        sys.task("seed");
        let outputs = dir.path().join("tickets").join("TICKET-1").join("outputs");
        assert!(!outputs.exists());
        sys.write_tool_output("TICKET-1", "call-1", "payload")
            .unwrap();
        assert!(outputs.is_dir());
    }

    #[test]
    fn parent_field_renders_in_created_event() {
        let (sys, dir) = test_system();
        sys.task("first");
        sys.ticket(Ticket::new("child").parent("TICKET-1"));
        let lines = read_tickets_log(dir.path());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "created");
        assert!(lines[0].get("parent").is_none());
        assert_eq!(lines[1]["event"], "created");
        assert_eq!(lines[1]["parent"], "TICKET-1");
    }

    // ---- resumption: TicketSystem::load ----

    #[test]
    fn load_creates_tickets_dir_when_missing() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let sys = TicketSystem::load(dir.path()).unwrap();
        assert!(sys.tickets.lock().unwrap().is_empty());
        assert!(dir.path().join("tickets").is_dir());
    }

    #[test]
    fn load_restores_done_ticket_with_result_and_comments() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let original = TicketSystem::new();
        original.dir(dir.path().to_path_buf());
        original.task("seed work");
        original
            .set_result("TICKET-1", serde_json::json!({"ok": true}))
            .unwrap();
        original.set_done("TICKET-1").unwrap();
        drop(original);

        let resumed = TicketSystem::load(dir.path()).unwrap();
        let t = resumed.get("TICKET-1").unwrap();
        assert_eq!(t.status, Status::Done);
        assert_eq!(t.result(), Some(&serde_json::json!({"ok": true})));
        assert_eq!(t.task, serde_json::Value::String("seed work".into()));
    }

    #[test]
    fn load_restores_in_progress_transcript() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let original = TicketSystem::new();
        original.dir(dir.path().to_path_buf());
        original.task("mid flight");
        original
            .claim(|t| t.status == Status::Todo, "alice")
            .unwrap();
        drop(original);

        let resumed = TicketSystem::load(dir.path()).unwrap();
        let t = resumed.get("TICKET-1").unwrap();
        assert_eq!(t.status, Status::InProgress);
        assert!(t.has_label("alice"));
    }

    #[test]
    fn load_derives_stats_from_ticket_files_when_stats_file_missing() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let original = TicketSystem::new();
        original.dir(dir.path().to_path_buf());
        original.task_labeled("a", "scan");
        original.task_labeled("b", "scan");
        original.task_labeled("c", "scan");
        original
            .set_result("TICKET-1", serde_json::Value::Null)
            .unwrap();
        original.set_done("TICKET-1").unwrap();
        original.set_failed("TICKET-2").unwrap();
        drop(original);

        std::fs::remove_file(dir.path().join("stats.json")).unwrap();

        let resumed = TicketSystem::load(dir.path()).unwrap();
        let s = resumed.stats();
        assert_eq!(s.tickets_created(), 3);
        assert_eq!(s.tickets_done(), 1);
        assert_eq!(s.tickets_failed(), 1);
        let scan = s.stats_for_label("scan");
        assert_eq!(scan.tickets_created(), 3);
        assert_eq!(scan.tickets_done(), 1);
        assert_eq!(scan.tickets_failed(), 1);
    }

    #[test]
    fn load_skips_malformed_ticket_file() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let original = TicketSystem::new();
        original.dir(dir.path().to_path_buf());
        original.task("valid");
        drop(original);

        let broken_dir = dir.path().join("tickets").join("TICKET-99");
        std::fs::create_dir_all(&broken_dir).unwrap();
        std::fs::write(broken_dir.join("ticket.123.json"), "not json").unwrap();

        let resumed = TicketSystem::load(dir.path()).unwrap();
        assert!(resumed.get("TICKET-1").is_some());
        assert!(resumed.get("TICKET-99").is_none());
    }

    #[test]
    fn load_picks_latest_ticket_file_per_key() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let key_dir = dir.path().join("tickets").join("TICKET-1");
        std::fs::create_dir_all(&key_dir).unwrap();
        let older = serde_json::json!({
            "task": "old body", "labels": [], "key": "TICKET-1",
            "status": "Todo", "reporter": "user", "created_at": 100,
            "started_at": null, "finished_at": null, "failed_at": null,
            "result": null, "parent": null, "comments": []
        });
        let newer = serde_json::json!({
            "task": "new body", "labels": [], "key": "TICKET-1",
            "status": "Todo", "reporter": "user", "created_at": 200,
            "started_at": null, "finished_at": null, "failed_at": null,
            "result": null, "parent": null, "comments": []
        });
        std::fs::write(
            key_dir.join("ticket.100.json"),
            serde_json::to_string(&older).unwrap(),
        )
        .unwrap();
        std::fs::write(
            key_dir.join("ticket.200.json"),
            serde_json::to_string(&newer).unwrap(),
        )
        .unwrap();

        let sys = TicketSystem::load(dir.path()).unwrap();
        let t = sys.get("TICKET-1").unwrap();
        assert_eq!(t.task, serde_json::Value::String("new body".into()));
        assert_eq!(t.created_at(), 200);
    }

    #[test]
    fn ticket_lifecycle_event_writes_stats_file() {
        let (sys, dir) = test_system();
        sys.task("seed");
        sys.claim(|t| t.status == Status::Todo, "alice").unwrap();
        sys.set_result("TICKET-1", serde_json::Value::Null).unwrap();
        sys.set_done("TICKET-1").unwrap();

        let bytes = std::fs::read(dir.path().join("stats.json")).expect("stats file written");
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["tickets_created"], 1);
        assert_eq!(body["tickets_done"], 1);
    }

    #[test]
    fn load_prefers_stats_file_over_derivation() {
        let dir = crate::test_util::TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("tickets")).unwrap();
        let body = serde_json::json!({ "steps": 42, "requests": 7 });
        std::fs::write(
            dir.path().join("stats.json"),
            serde_json::to_vec(&body).unwrap(),
        )
        .unwrap();

        let sys = TicketSystem::load(dir.path()).unwrap();
        assert_eq!(sys.stats().steps(), 42);
        assert_eq!(sys.stats().requests(), 7);
    }

    #[test]
    fn load_falls_back_to_derivation_when_stats_file_malformed() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let original = TicketSystem::new();
        original.dir(dir.path().to_path_buf());
        original.task("seed");
        original
            .set_result("TICKET-1", serde_json::Value::Null)
            .unwrap();
        original.set_done("TICKET-1").unwrap();
        drop(original);

        std::fs::write(dir.path().join("stats.json"), "not json").unwrap();

        let sys = TicketSystem::load(dir.path()).unwrap();
        assert_eq!(sys.stats().tickets_created(), 1);
        assert_eq!(sys.stats().tickets_done(), 1);
    }

    #[test]
    fn ticket_with_json_schema_round_trips_through_load() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let original = TicketSystem::new();
        original.dir(dir.path().to_path_buf());
        let schema_doc = serde_json::json!({
            "type": "object",
            "properties": { "n": { "type": "integer" } },
            "required": ["n"],
        });
        let schema = crate::schemas::Schema::parse(schema_doc.clone()).unwrap();
        original.ticket(Ticket::new("counted").schema(schema));
        drop(original);

        let resumed = TicketSystem::load(dir.path()).unwrap();
        let t = resumed.get("TICKET-1").unwrap();
        let restored = t.schema.expect("JSON schema must restore");
        assert!(restored.validate(&serde_json::json!({"n": 3})).is_ok());
        assert!(restored.validate(&serde_json::json!({})).is_err());
    }
}
