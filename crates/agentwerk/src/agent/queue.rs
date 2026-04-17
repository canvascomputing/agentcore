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
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // UserInput used in tests; TaskNotification used by enqueue_notification.
pub(crate) enum CommandSource {
    UserInput,
    TaskNotification { task_id: String },
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
