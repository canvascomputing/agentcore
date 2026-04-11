use std::sync::{Arc, Mutex};

use crate::task::{TaskStatus, TaskStore, TaskUpdate};
use crate::tool::{Tool, ToolBuilder, ToolResult};

/// Create a `task_create` tool that captures the given TaskStore.
pub fn task_create_tool(store: Arc<Mutex<TaskStore>>) -> impl Tool {
    ToolBuilder::new("task_create", "Create a new task with a subject and description.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "subject": { "type": "string", "description": "Brief title for the task" },
                "description": { "type": "string", "description": "What needs to be done" }
            },
            "required": ["subject", "description"]
        }))
        .handler(move |input, _ctx| {
            let store = store.clone();
            Box::pin(async move {
                let subject = input["subject"].as_str().unwrap_or("");
                let description = input["description"].as_str().unwrap_or("");
                let task = store.lock().unwrap().create(subject, description)?;
                Ok(ToolResult {
                    content: serde_json::to_string_pretty(&task)?,
                    is_error: false,
                })
            })
        })
        .build()
}

/// Create a `task_list` tool that captures the given TaskStore.
pub fn task_list_tool(store: Arc<Mutex<TaskStore>>) -> impl Tool {
    ToolBuilder::new("task_list", "List all tasks.")
        .schema(serde_json::json!({ "type": "object", "properties": {} }))
        .read_only(true)
        .handler(move |_input, _ctx| {
            let store = store.clone();
            Box::pin(async move {
                let tasks = store.lock().unwrap().list()?;
                Ok(ToolResult {
                    content: serde_json::to_string_pretty(&tasks)?,
                    is_error: false,
                })
            })
        })
        .build()
}

/// Create a `task_update` tool that captures the given TaskStore.
pub fn task_update_tool(store: Arc<Mutex<TaskStore>>) -> impl Tool {
    ToolBuilder::new(
        "task_update",
        "Update a task's status or other fields.",
    )
    .schema(serde_json::json!({
        "type": "object",
        "properties": {
            "id": { "type": "string", "description": "Task ID" },
            "status": { "type": "string", "description": "New status: Pending, InProgress, or Completed" },
            "subject": { "type": "string", "description": "New subject (optional)" }
        },
        "required": ["id"]
    }))
    .handler(move |input, _ctx| {
        let store = store.clone();
        Box::pin(async move {
            let id = input["id"].as_str().unwrap_or("");
            let status = input["status"].as_str().and_then(|s| match s {
                "Pending" => Some(TaskStatus::Pending),
                "InProgress" => Some(TaskStatus::InProgress),
                "Completed" => Some(TaskStatus::Completed),
                _ => None,
            });
            let subject = input["subject"].as_str().map(|s| s.to_string());
            let task = store.lock().unwrap().update(
                id,
                TaskUpdate {
                    status,
                    subject,
                    ..Default::default()
                },
            )?;
            Ok(ToolResult {
                content: serde_json::to_string_pretty(&task)?,
                is_error: false,
            })
        })
    })
    .build()
}

/// Create a `task_get` tool that captures the given TaskStore.
pub fn task_get_tool(store: Arc<Mutex<TaskStore>>) -> impl Tool {
    ToolBuilder::new("task_get", "Get a task by ID.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "Task ID" }
            },
            "required": ["id"]
        }))
        .read_only(true)
        .handler(move |input, _ctx| {
            let store = store.clone();
            Box::pin(async move {
                let id = input["id"].as_str().unwrap_or("");
                match store.lock().unwrap().get(id)? {
                    Some(t) => Ok(ToolResult {
                        content: serde_json::to_string_pretty(&t)?,
                        is_error: false,
                    }),
                    None => Ok(ToolResult {
                        content: format!("Task {id} not found"),
                        is_error: true,
                    }),
                }
            })
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolContext;
    use std::path::PathBuf;

    fn test_ctx() -> ToolContext {
        ToolContext::new(PathBuf::from("."))
    }

    fn test_store() -> Arc<Mutex<TaskStore>> {
        let tmp = tempfile::tempdir().unwrap();
        // Keep the tempdir so it lives for the test duration
        let path = tmp.keep();
        Arc::new(Mutex::new(TaskStore::open(&path, "test")))
    }

    #[tokio::test]
    async fn task_create_tool_returns_id() {
        let store = test_store();
        let tool = task_create_tool(store);
        let ctx = test_ctx();

        let result = tool
            .call(
                serde_json::json!({"subject": "Do stuff", "description": "Details"}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!result.is_error);
        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["id"], "1");
        assert_eq!(parsed["subject"], "Do stuff");
    }

    #[tokio::test]
    async fn task_list_tool_returns_json() {
        let store = test_store();
        store.lock().unwrap().create("A", "").unwrap();
        store.lock().unwrap().create("B", "").unwrap();

        let tool = task_list_tool(store);
        let ctx = test_ctx();

        let result = tool.call(serde_json::json!({}), &ctx).await.unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[tokio::test]
    async fn task_get_tool_returns_details() {
        let store = test_store();
        store.lock().unwrap().create("My task", "desc").unwrap();

        let tool = task_get_tool(store);
        let ctx = test_ctx();

        let result = tool
            .call(serde_json::json!({"id": "1"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["subject"], "My task");
    }

    #[tokio::test]
    async fn task_update_tool_changes_status() {
        let store = test_store();
        store.lock().unwrap().create("Task", "").unwrap();

        let update = task_update_tool(store.clone());
        let get = task_get_tool(store);
        let ctx = test_ctx();

        let result = update
            .call(serde_json::json!({"id": "1", "status": "InProgress"}), &ctx)
            .await
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["status"], "InProgress");

        // Verify via get
        let result = get
            .call(serde_json::json!({"id": "1"}), &ctx)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["status"], "InProgress");
    }
}
