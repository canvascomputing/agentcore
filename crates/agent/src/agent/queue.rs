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
    pub agent_id: Option<String>,
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
            agent_id: None,
        });
    }

    pub fn dequeue(&self, agent_id: Option<&str>) -> Option<QueuedCommand> {
        let mut queue = self.inner.lock().unwrap();
        let mut best_idx = None;
        let mut best_priority = None;

        for (i, cmd) in queue.iter().enumerate() {
            let matches = match (&cmd.agent_id, agent_id) {
                (None, _) => true,
                (Some(cmd_id), Some(filter_id)) => cmd_id == filter_id,
                (Some(_), None) => false,
            };
            if matches {
                if best_priority.is_none() || cmd.priority < *best_priority.as_ref().unwrap() {
                    best_idx = Some(i);
                    best_priority = Some(cmd.priority.clone());
                    if cmd.priority == QueuePriority::Now {
                        break;
                    }
                }
            }
        }

        best_idx.and_then(|i| queue.remove(i))
    }

    pub async fn wait_and_dequeue(&self, agent_id: Option<&str>) -> QueuedCommand {
        loop {
            if let Some(cmd) = self.dequeue(agent_id) {
                return cmd;
            }
            self.notify.notified().await;
        }
    }
}
