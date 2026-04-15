use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::error::Result;
use crate::persistence::task::{TaskStatus, TaskStore, TaskUpdate};
use crate::tools::tool::{Tool, ToolContext, ToolResult};

/// Persistent task management. The agent can create, update, list, and get tasks.
pub struct TaskTool {
    store: Arc<Mutex<TaskStore>>,
}

impl TaskTool {
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::open(&cwd)
    }

    pub fn open(data_dir: &Path) -> Self {
        Self {
            store: Arc::new(Mutex::new(TaskStore::open(data_dir, "tasks"))),
        }
    }
}

impl Tool for TaskTool {
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
                    let task = self.store.lock().unwrap().create(subject, description)?;
                    Ok(ToolResult::success(serde_json::to_string_pretty(&task)?))
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
                    let task = self.store.lock().unwrap().update(
                        id,
                        TaskUpdate {
                            status,
                            subject,
                            ..Default::default()
                        },
                    )?;
                    Ok(ToolResult::success(serde_json::to_string_pretty(&task)?))
                }
                "list" => {
                    let tasks = self.store.lock().unwrap().list()?;
                    Ok(ToolResult::success(serde_json::to_string_pretty(&tasks)?))
                }
                "get" => {
                    let id = input["id"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().get(id)? {
                        Some(t) => Ok(ToolResult::success(serde_json::to_string_pretty(&t)?)),
                        None => Ok(ToolResult::error(format!("Task {id} not found"))),
                    }
                }
                "delete" => {
                    let id = input["id"].as_str().unwrap_or("");
                    self.store.lock().unwrap().delete(id)?;
                    Ok(ToolResult::success(format!("Task {id} deleted")))
                }
                "claim" => {
                    let id = input["id"].as_str().unwrap_or("");
                    let agent_name = input["agent_name"].as_str().unwrap_or("");
                    let task = self.store.lock().unwrap().claim(id, agent_name)?;
                    Ok(ToolResult::success(serde_json::to_string_pretty(&task)?))
                }
                "add_dependency" => {
                    let from = input["from"].as_str().unwrap_or("");
                    let to = input["to"].as_str().unwrap_or("");
                    self.store.lock().unwrap().add_dependency(from, to)?;
                    Ok(ToolResult::success(format!("Dependency added: {from} blocks {to}")))
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
        TaskTool::open(&path)
    }

    #[tokio::test]
    async fn create_returns_id() {
        let tool = test_tool();
        let result = tool
            .call(serde_json::json!({"action": "create", "subject": "Do stuff", "description": "Details"}), &test_ctx())
            .await
            .unwrap();

        assert!(!result.is_error);
        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["id"], "1");
        assert_eq!(parsed["subject"], "Do stuff");
    }

    #[tokio::test]
    async fn list_returns_all() {
        let tool = test_tool();
        tool.call(serde_json::json!({"action": "create", "subject": "A", "description": ""}), &test_ctx()).await.unwrap();
        tool.call(serde_json::json!({"action": "create", "subject": "B", "description": ""}), &test_ctx()).await.unwrap();

        let result = tool.call(serde_json::json!({"action": "list"}), &test_ctx()).await.unwrap();
        let parsed: Vec<Value> = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[tokio::test]
    async fn get_returns_details() {
        let tool = test_tool();
        tool.call(serde_json::json!({"action": "create", "subject": "My task", "description": "desc"}), &test_ctx()).await.unwrap();

        let result = tool.call(serde_json::json!({"action": "get", "id": "1"}), &test_ctx()).await.unwrap();
        assert!(!result.is_error);
        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["subject"], "My task");
    }

    #[tokio::test]
    async fn update_changes_status() {
        let tool = test_tool();
        tool.call(serde_json::json!({"action": "create", "subject": "Task", "description": ""}), &test_ctx()).await.unwrap();

        let result = tool.call(serde_json::json!({"action": "update", "id": "1", "status": "InProgress"}), &test_ctx()).await.unwrap();
        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["status"], "InProgress");

        let result = tool.call(serde_json::json!({"action": "get", "id": "1"}), &test_ctx()).await.unwrap();
        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["status"], "InProgress");
    }

    #[tokio::test]
    async fn unknown_action_errors() {
        let tool = test_tool();
        let result = tool.call(serde_json::json!({"action": "foobar"}), &test_ctx()).await.unwrap();
        assert!(result.is_error);
    }
}
