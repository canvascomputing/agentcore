//! In-process work inbox that feeds a running agent with late-arriving input (user messages, peer messages, task notifications).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::Notify;

use crate::output::Outcome;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum WorkPriority {
    Next = 0,
    Later = 1,
}

#[derive(Debug, Clone)]
pub(crate) struct Task {
    pub(crate) content: String,
    pub(crate) priority: WorkPriority,
    pub(crate) source: TaskSource,
    pub(crate) agent_name: Option<String>,
}

impl Task {
    /// A task with no agent_name is visible to all agents.
    /// A targeted task is only visible to the named agent.
    fn is_visible_to(&self, agent_name: Option<&str>) -> bool {
        match (&self.agent_name, agent_name) {
            (None, _) => true,
            (Some(target), Some(name)) => target == name,
            (Some(_), None) => false,
        }
    }

    /// Render as the text body of a `Message::user(...)` injected into the
    /// recipient's next step. Peer messages get a header so the LLM sees who
    /// sent them; other sources deliver content verbatim.
    pub(crate) fn as_user_message(&self) -> String {
        match &self.source {
            TaskSource::PeerMessage { from, summary } => {
                let header = match summary {
                    Some(s) => format!("[message from {from}: {s}]"),
                    None => format!("[message from {from}]"),
                };
                format!("{header}\n{}", self.content)
            }
            _ => self.content.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum TaskSource {
    UserInput,
    TaskNotification,
    PeerMessage {
        from: String,
        summary: Option<String>,
    },
}

/// Snapshot of a tracked background spawn: `None` while the child is still
/// running, `Some` once it terminates. The triple carries the verdict, the
/// raw response text, and the validated structured value if a contract was
/// set on the child.
pub(crate) type SpawnState = Option<(Outcome, String, Option<Value>)>;

/// Thread-safe priority inbox of pending tasks. Also tracks the live and
/// settled state of background sub-agents launched via `agent_tool`.
pub(crate) struct Work {
    inner: Arc<Mutex<VecDeque<Task>>>,
    spawns: Mutex<HashMap<String, SpawnState>>,
    notify: Notify,
}

impl Work {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
            spawns: Mutex::new(HashMap::new()),
            notify: Notify::new(),
        }
    }

    pub(crate) fn add(&self, task: Task) {
        self.inner.lock().unwrap().push_back(task);
    }

    pub(crate) fn add_notification(&self, item_id: &str, summary: &str) {
        self.add(Task {
            content: format!("Item {item_id} completed: {summary}"),
            priority: WorkPriority::Later,
            source: TaskSource::TaskNotification,
            agent_name: None,
        });
    }

    /// Record a background sub-agent as launched. Called before `tokio::spawn`
    /// so the id is observable before the model receives the start
    /// confirmation.
    pub(crate) fn spawned(&self, id: &str) {
        self.spawns
            .lock()
            .unwrap()
            .entry(id.to_string())
            .or_insert(None);
    }

    /// Record a background sub-agent's terminal verdict and wake every
    /// blocked poll loop.
    pub(crate) fn settled(
        &self,
        id: &str,
        outcome: Outcome,
        text: String,
        structured: Option<Value>,
    ) {
        self.spawns
            .lock()
            .unwrap()
            .insert(id.to_string(), Some((outcome, text, structured)));
        self.notify.notify_waiters();
    }

    /// Snapshot of a tracked spawn. `None` for unknown ids, `Some(None)` for
    /// running, `Some(Some(_))` once settled.
    pub(crate) fn spawn_state(&self, id: &str) -> Option<SpawnState> {
        self.spawns.lock().unwrap().get(id).cloned()
    }

    /// Resolves on the next `settled` call. Used by the blocking poll path.
    pub(crate) fn notified(&self) -> tokio::sync::futures::Notified<'_> {
        self.notify.notified()
    }

    /// Pop the highest-priority task visible to the given agent that also
    /// satisfies `pred`. Ties break by insertion order. Tasks failing the
    /// predicate are skipped (not removed).
    pub(crate) fn take_if<F>(&self, agent_name: Option<&str>, pred: F) -> Option<Task>
    where
        F: Fn(&Task) -> bool,
    {
        let mut pending = self.inner.lock().unwrap();
        let idx = pending
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_visible_to(agent_name) && pred(c))
            .min_by_key(|(i, c)| (c.priority, *i))?
            .0;
        pending.remove(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(target: Option<&str>, priority: WorkPriority) -> Task {
        Task {
            content: "x".into(),
            priority,
            source: TaskSource::UserInput,
            agent_name: target.map(|s| s.into()),
        }
    }

    #[test]
    fn is_visible_to_broadcast_visible_to_any_agent() {
        let t = task(None, WorkPriority::Next);
        assert!(t.is_visible_to(Some("alice")));
        assert!(t.is_visible_to(Some("bob")));
        assert!(t.is_visible_to(None));
    }

    #[test]
    fn is_visible_to_targeted_visible_only_to_named() {
        let t = task(Some("alice"), WorkPriority::Next);
        assert!(t.is_visible_to(Some("alice")));
        assert!(!t.is_visible_to(Some("bob")));
    }

    #[test]
    fn is_visible_to_targeted_invisible_to_none_reader() {
        let t = task(Some("alice"), WorkPriority::Next);
        assert!(!t.is_visible_to(None));
    }

    #[test]
    fn take_if_returns_none_when_empty() {
        let w = Work::new();
        assert!(w.take_if(Some("alice"), |_| true).is_none());
    }

    #[test]
    fn take_if_skips_items_with_later_priority() {
        let w = Work::new();
        w.add(task(Some("alice"), WorkPriority::Later));

        // Predicate rejects Later → nothing returned, item still in inbox.
        let pred = |t: &Task| t.priority != WorkPriority::Later;
        assert!(w.take_if(Some("alice"), pred).is_none());

        // Without the filter it takes the item.
        assert!(w.take_if(Some("alice"), |_| true).is_some());
    }

    #[test]
    fn take_if_prefers_higher_priority_among_visible_items() {
        let w = Work::new();
        w.add(task(Some("alice"), WorkPriority::Later));
        w.add(task(Some("alice"), WorkPriority::Next));

        let first = w.take_if(Some("alice"), |_| true).unwrap();
        assert_eq!(first.priority, WorkPriority::Next);
        let second = w.take_if(Some("alice"), |_| true).unwrap();
        assert_eq!(second.priority, WorkPriority::Later);
    }

    #[test]
    fn as_user_message_plain_source_is_content_only() {
        let t = Task {
            content: "hello".into(),
            priority: WorkPriority::Next,
            source: TaskSource::UserInput,
            agent_name: None,
        };
        assert_eq!(t.as_user_message(), "hello");
    }

    #[test]
    fn as_user_message_peer_message_prepends_header() {
        let t = Task {
            content: "ping".into(),
            priority: WorkPriority::Next,
            source: TaskSource::PeerMessage {
                from: "alice".into(),
                summary: Some("greeting".into()),
            },
            agent_name: Some("bob".into()),
        };
        assert_eq!(t.as_user_message(), "[message from alice: greeting]\nping");
    }

    #[test]
    fn spawn_state_unknown_id_is_none() {
        let w = Work::new();
        assert!(w.spawn_state("ghost").is_none());
    }

    #[test]
    fn spawn_state_running_then_settled() {
        let w = Work::new();
        w.spawned("t1");
        assert!(matches!(w.spawn_state("t1"), Some(None)));

        w.settled("t1", Outcome::Completed, "done".into(), None);
        let state = w.spawn_state("t1").unwrap().unwrap();
        assert!(matches!(state.0, Outcome::Completed));
        assert_eq!(state.1, "done");
    }

    #[tokio::test]
    async fn settle_wakes_notified_waiters() {
        let w = std::sync::Arc::new(Work::new());
        w.spawned("t1");

        let waiter = w.clone();
        let task = tokio::spawn(async move {
            waiter.notified().await;
        });

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        w.settled("t1", Outcome::Completed, "done".into(), None);

        // notified() resolves when settled() fires.
        tokio::time::timeout(std::time::Duration::from_millis(200), task)
            .await
            .expect("notified waiter must wake")
            .unwrap();
    }

    #[test]
    fn as_user_message_peer_message_without_summary() {
        let t = Task {
            content: "ping".into(),
            priority: WorkPriority::Next,
            source: TaskSource::PeerMessage {
                from: "alice".into(),
                summary: None,
            },
            agent_name: Some("bob".into()),
        };
        assert_eq!(t.as_user_message(), "[message from alice]\nping");
    }
}
