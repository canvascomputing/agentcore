use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueuePriority {
    Now = 0,
    Next = 1,
    Later = 2,
}

#[derive(Debug, Clone)]
pub struct QueuedCommand {
    pub content: String,
    pub priority: QueuePriority,
    pub source: CommandSource,
    pub agent_name: Option<String>,
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
pub enum CommandSource {
    UserInput,
    TaskNotification { task_id: String },
    System,
}

/// Thread-safe priority queue for commands.
pub struct CommandQueue {
    inner: Arc<Mutex<VecDeque<QueuedCommand>>>,
    notify: Arc<tokio::sync::Notify>,
}

impl CommandQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub fn enqueue(&self, command: QueuedCommand) {
        self.inner.lock().unwrap().push_back(command);
        self.notify.notify_one();
    }

    pub fn enqueue_notification(&self, task_id: &str, summary: &str) {
        self.enqueue(QueuedCommand {
            content: format!("Task {task_id} completed: {summary}"),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification {
                task_id: task_id.to_string(),
            },
            agent_name: None,
        });
    }

    pub fn dequeue(&self, agent_name: Option<&str>) -> Option<QueuedCommand> {
        let mut queue = self.inner.lock().unwrap();
        let mut best: Option<(usize, QueuePriority)> = None;

        for (i, cmd) in queue.iter().enumerate() {
            if !cmd.is_visible_to(agent_name) {
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

    pub async fn wait_and_dequeue(&self, agent_name: Option<&str>) -> QueuedCommand {
        loop {
            if let Some(cmd) = self.dequeue(agent_name) {
                return cmd;
            }
            self.notify.notified().await;
        }
    }
}
