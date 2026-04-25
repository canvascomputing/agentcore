//! In-process command queue that feeds a running agent with late-arriving input (user messages, peer messages, task notifications).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum QueuePriority {
    Next = 0,
    Later = 1,
}

#[derive(Debug, Clone)]
pub(crate) struct Work {
    pub(crate) content: String,
    pub(crate) priority: QueuePriority,
    pub(crate) source: CommandSource,
    pub(crate) agent_name: Option<String>,
}

impl Work {
    /// A command with no agent_name is visible to all agents.
    /// A targeted command is only visible to the named agent.
    fn is_visible_to(&self, agent_name: Option<&str>) -> bool {
        match (&self.agent_name, agent_name) {
            (None, _) => true,
            (Some(target), Some(name)) => target == name,
            (Some(_), None) => false,
        }
    }

    /// Render as the text body of a `Message::user(...)` injected into the
    /// recipient's next turn. Peer messages get a header so the LLM sees who
    /// sent them; other sources deliver content verbatim.
    pub(crate) fn as_user_message(&self) -> String {
        match &self.source {
            CommandSource::PeerMessage { from, summary } => {
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
pub(crate) enum CommandSource {
    UserInput,
    TaskNotification,
    PeerMessage {
        from: String,
        summary: Option<String>,
    },
}

/// Thread-safe priority queue for commands.
pub(crate) struct IncomingWork {
    inner: Arc<Mutex<VecDeque<Work>>>,
}

impl IncomingWork {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub(crate) fn add(&self, work: Work) {
        self.inner.lock().unwrap().push_back(work);
    }

    pub(crate) fn add_notification(&self, task_id: &str, summary: &str) {
        self.add(Work {
            content: format!("Task {task_id} completed: {summary}"),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification,
            agent_name: None,
        });
    }

    /// Dequeue the highest-priority command visible to the given agent that also
    /// satisfies `pred`. Ties break by insertion order. Commands failing the
    /// predicate are skipped (not removed).
    pub(crate) fn dequeue_if<F>(&self, agent_name: Option<&str>, pred: F) -> Option<Work>
    where
        F: Fn(&Work) -> bool,
    {
        let mut queue = self.inner.lock().unwrap();
        let idx = queue
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_visible_to(agent_name) && pred(c))
            .min_by_key(|(i, c)| (c.priority, *i))?
            .0;
        queue.remove(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(target: Option<&str>, priority: QueuePriority) -> Work {
        Work {
            content: "x".into(),
            priority,
            source: CommandSource::UserInput,
            agent_name: target.map(|s| s.into()),
        }
    }

    #[test]
    fn is_visible_to_broadcast_visible_to_any_agent() {
        let c = cmd(None, QueuePriority::Next);
        assert!(c.is_visible_to(Some("alice")));
        assert!(c.is_visible_to(Some("bob")));
        assert!(c.is_visible_to(None));
    }

    #[test]
    fn is_visible_to_targeted_visible_only_to_named() {
        let c = cmd(Some("alice"), QueuePriority::Next);
        assert!(c.is_visible_to(Some("alice")));
        assert!(!c.is_visible_to(Some("bob")));
    }

    #[test]
    fn is_visible_to_targeted_invisible_to_none_reader() {
        let c = cmd(Some("alice"), QueuePriority::Next);
        assert!(!c.is_visible_to(None));
    }

    #[test]
    fn dequeue_if_returns_none_when_empty() {
        let q = IncomingWork::new();
        assert!(q.dequeue_if(Some("alice"), |_| true).is_none());
    }

    #[test]
    fn dequeue_if_skips_items_with_later_priority() {
        let q = IncomingWork::new();
        q.add(cmd(Some("alice"), QueuePriority::Later));

        // Predicate rejects Later → nothing returned, item still in queue.
        let pred = |c: &Work| c.priority != QueuePriority::Later;
        assert!(q.dequeue_if(Some("alice"), pred).is_none());

        // Without the filter it dequeues.
        assert!(q.dequeue_if(Some("alice"), |_| true).is_some());
    }

    #[test]
    fn dequeue_if_prefers_higher_priority_among_visible_items() {
        let q = IncomingWork::new();
        q.add(cmd(Some("alice"), QueuePriority::Later));
        q.add(cmd(Some("alice"), QueuePriority::Next));

        let first = q.dequeue_if(Some("alice"), |_| true).unwrap();
        assert_eq!(first.priority, QueuePriority::Next);
        let second = q.dequeue_if(Some("alice"), |_| true).unwrap();
        assert_eq!(second.priority, QueuePriority::Later);
    }

    #[test]
    fn as_user_message_plain_source_is_content_only() {
        let cmd = Work {
            content: "hello".into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: None,
        };
        assert_eq!(cmd.as_user_message(), "hello");
    }

    #[test]
    fn as_user_message_peer_message_prepends_header() {
        let cmd = Work {
            content: "ping".into(),
            priority: QueuePriority::Next,
            source: CommandSource::PeerMessage {
                from: "alice".into(),
                summary: Some("greeting".into()),
            },
            agent_name: Some("bob".into()),
        };
        assert_eq!(
            cmd.as_user_message(),
            "[message from alice: greeting]\nping"
        );
    }

    #[test]
    fn as_user_message_peer_message_without_summary() {
        let cmd = Work {
            content: "ping".into(),
            priority: QueuePriority::Next,
            source: CommandSource::PeerMessage {
                from: "alice".into(),
                summary: None,
            },
            agent_name: Some("bob".into()),
        };
        assert_eq!(cmd.as_user_message(), "[message from alice]\nping");
    }
}
