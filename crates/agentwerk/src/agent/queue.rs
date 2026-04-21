//! In-process command queue that feeds a running agent with late-arriving input (user messages, peer messages, task notifications).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)] // All variants are part of the priority API; some used only in tests today.
pub(crate) enum QueuePriority {
    Now = 0,
    Next = 1,
    Later = 2,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields read during dequeue; `source` reserved for future routing.
pub(crate) struct QueuedCommand {
    pub(crate) content: String,
    pub(crate) priority: QueuePriority,
    pub(crate) source: CommandSource,
    pub(crate) agent_name: Option<String>,
}

impl QueuedCommand {
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
#[allow(dead_code)] // UserInput used in tests; others used by in-crate tools.
pub(crate) enum CommandSource {
    UserInput,
    TaskNotification {
        task_id: String,
    },
    PeerMessage {
        from: String,
        summary: Option<String>,
    },
}

/// Thread-safe priority queue for commands.
pub(crate) struct CommandQueue {
    inner: Arc<Mutex<VecDeque<QueuedCommand>>>,
}

impl CommandQueue {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub(crate) fn enqueue(&self, command: QueuedCommand) {
        self.inner.lock().unwrap().push_back(command);
    }

    pub(crate) fn enqueue_notification(&self, task_id: &str, summary: &str) {
        self.enqueue(QueuedCommand {
            content: format!("Task {task_id} completed: {summary}"),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification {
                task_id: task_id.to_string(),
            },
            agent_name: None,
        });
    }

    /// Dequeue the highest-priority command visible to the given agent that also
    /// satisfies `pred`. Commands failing the predicate are skipped (not removed).
    pub(crate) fn dequeue_if<F>(&self, agent_name: Option<&str>, pred: F) -> Option<QueuedCommand>
    where
        F: Fn(&QueuedCommand) -> bool,
    {
        let mut queue = self.inner.lock().unwrap();
        let mut best: Option<(usize, QueuePriority)> = None;

        for (i, cmd) in queue.iter().enumerate() {
            if !cmd.is_visible_to(agent_name) {
                continue;
            }
            if !pred(cmd) {
                continue;
            }
            if best.as_ref().is_some_and(|(_, p)| *p <= cmd.priority) {
                continue;
            }

            best = Some((i, cmd.priority.clone()));
            if cmd.priority == QueuePriority::Now {
                break;
            }
        }

        best.and_then(|(i, _)| queue.remove(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(target: Option<&str>, priority: QueuePriority) -> QueuedCommand {
        QueuedCommand {
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
        let q = CommandQueue::new();
        assert!(q.dequeue_if(Some("alice"), |_| true).is_none());
    }

    #[test]
    fn dequeue_if_skips_items_with_later_priority() {
        let q = CommandQueue::new();
        q.enqueue(cmd(Some("alice"), QueuePriority::Later));

        // Predicate rejects Later → nothing returned, item still in queue.
        let pred = |c: &QueuedCommand| c.priority != QueuePriority::Later;
        assert!(q.dequeue_if(Some("alice"), pred).is_none());

        // Without the filter it dequeues.
        assert!(q.dequeue_if(Some("alice"), |_| true).is_some());
    }

    #[test]
    fn dequeue_if_prefers_higher_priority_among_visible_items() {
        let q = CommandQueue::new();
        q.enqueue(cmd(Some("alice"), QueuePriority::Later));
        q.enqueue(cmd(Some("alice"), QueuePriority::Now));
        q.enqueue(cmd(Some("alice"), QueuePriority::Next));

        let first = q.dequeue_if(Some("alice"), |_| true).unwrap();
        assert_eq!(first.priority, QueuePriority::Now);
        let second = q.dequeue_if(Some("alice"), |_| true).unwrap();
        assert_eq!(second.priority, QueuePriority::Next);
    }

    #[test]
    fn as_user_message_plain_source_is_content_only() {
        let cmd = QueuedCommand {
            content: "hello".into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: None,
        };
        assert_eq!(cmd.as_user_message(), "hello");
    }

    #[test]
    fn as_user_message_peer_message_prepends_header() {
        let cmd = QueuedCommand {
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
        let cmd = QueuedCommand {
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
