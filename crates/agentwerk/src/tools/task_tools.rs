//! Task-management tools (create, update, list, claim, …) that let an agent coordinate work with peers through the durable `TaskStore`.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::error::Result;
use crate::persistence::error::PersistenceError;
use crate::persistence::task::{TaskStatus, TaskStore, TaskUpdate};
use crate::tools::error::ToolError;
use crate::tools::tool::{ToolLike, ToolContext, ToolResult};

/// Routes a task-store failure: in-band problems (not found, already
/// completed, blocked) become `ToolResult::Error` so the model can retry;
/// structural failures (lock contention, I/O) become `ToolError`.
fn route(err: PersistenceError) -> Result<ToolResult> {
    match err {
        PersistenceError::TaskNotFound(id) => Ok(ToolResult::error(format!("Task {id} not found"))),
        PersistenceError::TaskAlreadyCompleted(id) => {
            Ok(ToolResult::error(format!("Task {id} already completed")))
        }
        PersistenceError::TaskBlocked {
            task_id,
            blocker_id,
        } => Ok(ToolResult::error(format!(
            "Task {task_id} blocked by unfinished task {blocker_id}"
        ))),
        err @ PersistenceError::LockFailed { .. } | err @ PersistenceError::IoFailed(_) => {
            Err(ToolError::new("task", err.to_string()).into())
        }
    }
}

/// Persistent task management backed by a directory on disk. Lets an agent
/// create, update, list, and claim tasks; tasks survive process restarts and
/// can be shared across peers pointing at the same directory.
pub struct TaskTool {
    store: Arc<Mutex<TaskStore>>,
}

impl TaskTool {
    /// A new task tool storing entries under `data_directory/tasks`.
    pub fn new(data_directory: &Path) -> Self {
        Self {
            store: Arc::new(Mutex::new(TaskStore::new(data_directory, "tasks"))),
        }
    }
}

impl ToolLike for TaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Manage persistent tasks: create, update, list, or get by ID."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "list", "get", "delete", "claim", "add_dependency"],
                    "description": "The action to perform"
                },
                "id": { "type": "string", "description": "Task ID (for get/update/delete/claim)" },
                "subject": { "type": "string", "description": "Task title (for create/update)" },
                "description": { "type": "string", "description": "Task details (for create)" },
                "status": {
                    "type": "string",
                    "enum": ["Pending", "InProgress", "Completed"],
                    "description": "New status (for update)"
                },
                "agent_name": { "type": "string", "description": "Agent claiming the task (for claim)" },
                "from": { "type": "string", "description": "Blocking task ID (for add_dependency)" },
                "to": { "type": "string", "description": "Blocked task ID (for add_dependency)" }
            },
            "required": ["action"]
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn call<'a>(
        &'a self,
        input: Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let action = input["action"].as_str().unwrap_or("");
            match action {
                "create" => {
                    let subject = input["subject"].as_str().unwrap_or("");
                    let description = input["description"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().create(subject, description) {
                        Ok(task) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&task).unwrap(),
                        )),
                        Err(e) => route(e),
                    }
                }
                "update" => {
                    let id = input["id"].as_str().unwrap_or("");
                    let status = input["status"].as_str().and_then(|s| match s {
                        "Pending" => Some(TaskStatus::Pending),
                        "InProgress" => Some(TaskStatus::InProgress),
                        "Completed" => Some(TaskStatus::Completed),
                        _ => None,
                    });
                    let subject = input["subject"].as_str().map(|s| s.to_string());
                    match self.store.lock().unwrap().update(
                        id,
                        TaskUpdate {
                            status,
                            subject,
                            ..Default::default()
                        },
                    ) {
                        Ok(task) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&task).unwrap(),
                        )),
                        Err(e) => route(e),
                    }
                }
                "list" => match self.store.lock().unwrap().list() {
                    Ok(tasks) => Ok(ToolResult::success(
                        serde_json::to_string_pretty(&tasks).unwrap(),
                    )),
                    Err(e) => route(e),
                },
                "get" => {
                    let id = input["id"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().get(id) {
                        Ok(Some(t)) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&t).unwrap(),
                        )),
                        Ok(None) => Ok(ToolResult::error(format!("Task {id} not found"))),
                        Err(e) => route(e),
                    }
                }
                "delete" => {
                    let id = input["id"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().delete(id) {
                        Ok(()) => Ok(ToolResult::success(format!("Task {id} deleted"))),
                        Err(e) => route(e),
                    }
                }
                "claim" => {
                    let id = input["id"].as_str().unwrap_or("");
                    let agent_name = input["agent_name"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().claim(id, agent_name) {
                        Ok(task) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&task).unwrap(),
                        )),
                        Err(e) => route(e),
                    }
                }
                "add_dependency" => {
                    let from = input["from"].as_str().unwrap_or("");
                    let to = input["to"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().add_dependency(from, to) {
                        Ok(()) => Ok(ToolResult::success(format!(
                            "Dependency added: {from} blocks {to}"
                        ))),
                        Err(e) => route(e),
                    }
                }
                _ => Ok(ToolResult::error(format!("Unknown action: {action}"))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_ctx() -> ToolContext {
        ToolContext::new(PathBuf::from("."))
    }

    fn test_tool() -> TaskTool {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.keep();
        TaskTool::new(&path)
    }

    #[tokio::test]
    async fn create_returns_id() {
        let tool = test_tool();
        let result = tool
            .call(serde_json::json!({"action": "create", "subject": "Do stuff", "description": "Details"}), &test_ctx())
            .await
            .unwrap();

        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        let parsed: Value = serde_json::from_str(content).unwrap();
        assert_eq!(parsed["id"], "1");
        assert_eq!(parsed["subject"], "Do stuff");
    }

    #[tokio::test]
    async fn list_returns_all() {
        let tool = test_tool();
        tool.call(
            serde_json::json!({"action": "create", "subject": "A", "description": ""}),
            &test_ctx(),
        )
        .await
        .unwrap();
        tool.call(
            serde_json::json!({"action": "create", "subject": "B", "description": ""}),
            &test_ctx(),
        )
        .await
        .unwrap();

        let result = tool
            .call(serde_json::json!({"action": "list"}), &test_ctx())
            .await
            .unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        let parsed: Vec<Value> = serde_json::from_str(content).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[tokio::test]
    async fn get_returns_details() {
        let tool = test_tool();
        tool.call(
            serde_json::json!({"action": "create", "subject": "My task", "description": "desc"}),
            &test_ctx(),
        )
        .await
        .unwrap();

        let result = tool
            .call(serde_json::json!({"action": "get", "id": "1"}), &test_ctx())
            .await
            .unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        let parsed: Value = serde_json::from_str(content).unwrap();
        assert_eq!(parsed["subject"], "My task");
    }

    #[tokio::test]
    async fn update_changes_status() {
        let tool = test_tool();
        tool.call(
            serde_json::json!({"action": "create", "subject": "Task", "description": ""}),
            &test_ctx(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                serde_json::json!({"action": "update", "id": "1", "status": "InProgress"}),
                &test_ctx(),
            )
            .await
            .unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        let parsed: Value = serde_json::from_str(content).unwrap();
        assert_eq!(parsed["status"], "InProgress");

        let result = tool
            .call(serde_json::json!({"action": "get", "id": "1"}), &test_ctx())
            .await
            .unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        let parsed: Value = serde_json::from_str(content).unwrap();
        assert_eq!(parsed["status"], "InProgress");
    }

    #[tokio::test]
    async fn unknown_action_errors() {
        let tool = test_tool();
        let result = tool
            .call(serde_json::json!({"action": "foobar"}), &test_ctx())
            .await
            .unwrap();
        assert!(matches!(result, ToolResult::Error(_)));
    }
}
