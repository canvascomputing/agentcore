//! Todo-list tool (create, update, list, claim items, …) that lets an agent coordinate work with peers through the durable `TodoList`.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

use crate::tools::tool_file::ToolFile;

use crate::error::Result;
use crate::persistence::error::PersistenceError;
use crate::persistence::todo::{TodoItemStatus, TodoItemUpdate, TodoList};
use crate::tools::error::ToolError;
use crate::tools::tool::{ToolContext, ToolLike, ToolResult};

/// Routes a store failure: in-band problems (not found, already completed,
/// blocked) become `ToolResult::Error` so the model can retry; structural
/// failures (lock contention, I/O) become `ToolError`.
fn route(err: PersistenceError) -> Result<ToolResult> {
    match err {
        PersistenceError::TodoItemNotFound(id) => {
            Ok(ToolResult::error(format!("Item {id} not found")))
        }
        PersistenceError::TodoItemAlreadyCompleted(id) => {
            Ok(ToolResult::error(format!("Item {id} already completed")))
        }
        PersistenceError::TodoItemBlocked {
            item_id,
            blocker_id,
        } => Ok(ToolResult::error(format!(
            "Item {item_id} blocked by unfinished item {blocker_id}"
        ))),
        err @ PersistenceError::LockFailed { .. } | err @ PersistenceError::IoFailed(_) => {
            Err(ToolError::ExecutionFailed {
                tool_name: "todo_list_tool".into(),
                message: err.to_string(),
            }
            .into())
        }
    }
}

/// A persistent todo list backed by a directory on disk. Lets an agent
/// create, update, list, and claim items; items survive process restarts and
/// can be shared across peers pointing at the same directory.
pub struct TodoListTool {
    store: Arc<Mutex<TodoList>>,
}

impl TodoListTool {
    /// A new todo-list tool storing entries under `data_dir/items`.
    pub fn new(data_dir: &Path) -> Self {
        Self {
            store: Arc::new(Mutex::new(TodoList::new(data_dir, "items"))),
        }
    }
}

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("todo_list.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for TodoListTool {
    fn name(&self) -> &str {
        &tool_file().name
    }

    fn description(&self) -> &str {
        description()
    }

    fn input_schema(&self) -> Value {
        tool_file().input_schema.clone()
    }

    fn is_read_only(&self) -> bool {
        tool_file().read_only
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
                        Ok(item) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&item).unwrap(),
                        )),
                        Err(e) => route(e),
                    }
                }
                "update" => {
                    let id = input["id"].as_str().unwrap_or("");
                    let status = input["status"].as_str().and_then(|s| match s {
                        "Pending" => Some(TodoItemStatus::Pending),
                        "InProgress" => Some(TodoItemStatus::InProgress),
                        "Completed" => Some(TodoItemStatus::Completed),
                        _ => None,
                    });
                    let subject = input["subject"].as_str().map(|s| s.to_string());
                    match self.store.lock().unwrap().update(
                        id,
                        TodoItemUpdate {
                            status,
                            subject,
                            ..Default::default()
                        },
                    ) {
                        Ok(item) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&item).unwrap(),
                        )),
                        Err(e) => route(e),
                    }
                }
                "list" => match self.store.lock().unwrap().list() {
                    Ok(items) => Ok(ToolResult::success(
                        serde_json::to_string_pretty(&items).unwrap(),
                    )),
                    Err(e) => route(e),
                },
                "get" => {
                    let id = input["id"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().get(id) {
                        Ok(Some(t)) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&t).unwrap(),
                        )),
                        Ok(None) => Ok(ToolResult::error(format!("Item {id} not found"))),
                        Err(e) => route(e),
                    }
                }
                "delete" => {
                    let id = input["id"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().delete(id) {
                        Ok(()) => Ok(ToolResult::success(format!("Item {id} deleted"))),
                        Err(e) => route(e),
                    }
                }
                "claim" => {
                    let id = input["id"].as_str().unwrap_or("");
                    let agent_name = input["agent_name"].as_str().unwrap_or("");
                    match self.store.lock().unwrap().claim(id, agent_name) {
                        Ok(item) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&item).unwrap(),
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

    fn test_tool() -> TodoListTool {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.keep();
        TodoListTool::new(&path)
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
            serde_json::json!({"action": "create", "subject": "My item", "description": "desc"}),
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
        assert_eq!(parsed["subject"], "My item");
    }

    #[tokio::test]
    async fn update_changes_status() {
        let tool = test_tool();
        tool.call(
            serde_json::json!({"action": "create", "subject": "Item", "description": ""}),
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
