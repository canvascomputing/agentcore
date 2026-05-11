//! Atomic finisher + spawner: validate the agent's `result`, finish the
//! current ticket through the shared `write_result` helper, then insert
//! a child ticket pinned to `to` with the current ticket recorded as
//! its `parent`. Sister tool to `WriteResultTool` — both finish the
//! current ticket; this one also chains a follow-up.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use crate::agents::tickets::Ticket;
use crate::providers::ProviderResult;
use crate::schemas::Schema;

use super::super::tool::{ToolContext, ToolLike, ToolResult};
use super::super::tool_file::ToolFile;
use super::{resolve_current_key, write_result};

pub struct WriteHandoverTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("write_handover.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for WriteHandoverTool {
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
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let Some(ticket_system) = ctx.ticket_system_handle().cloned() else {
                return Ok(ToolResult::error(
                    "Ticket system unavailable in this context",
                ));
            };

            let to = match input.get("to").and_then(|v| v.as_str()) {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return Ok(ToolResult::error("Missing required parameter: to")),
            };
            let task = match input.get("task") {
                Some(Value::String(s)) if !s.is_empty() => Value::String(s.clone()),
                Some(Value::String(_)) => {
                    return Ok(ToolResult::error("`task` must not be an empty string"))
                }
                Some(Value::Null) | None => {
                    return Ok(ToolResult::error("Missing required parameter: task"))
                }
                Some(_) => return Ok(ToolResult::error("`task` must be a string")),
            };
            let result = match input.get("result") {
                Some(Value::String(s)) if !s.is_empty() => Value::String(s.clone()),
                Some(Value::String(_)) => {
                    return Ok(ToolResult::error("`result` must not be an empty string"))
                }
                Some(Value::Null) | None => {
                    return Ok(ToolResult::error("Missing required parameter: result"))
                }
                Some(_) => {
                    return Ok(ToolResult::error(
                        "`result` must be a string — pass plain prose, not numbers or arrays. For structured results use `write_result_tool` instead.",
                    ))
                }
            };
            let child_schema: Option<Schema> = match input.get("schema") {
                Some(doc) if !doc.is_null() => match Schema::parse(doc.clone()) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        return Ok(ToolResult::error(format!(
                            "Cannot hand off: supplied `schema` is invalid: {e}"
                        )));
                    }
                },
                _ => None,
            };

            let parent_key = match resolve_current_key(&ticket_system, ctx) {
                Ok(k) => k,
                Err(e) => return Ok(e),
            };

            // Parent-side finish runs first. The result is already
            // validated to be a non-empty string above; schema-bound
            // parents whose schema rejects strings will be caught by
            // the helper's schema validation and abort before any
            // child is created.
            let parent_outcome = write_result(&ticket_system, ctx, &parent_key, result);
            if !matches!(parent_outcome, ToolResult::Success(_)) {
                return Ok(parent_outcome);
            }

            let reporter = ctx
                .agent_name_str()
                .expect("agent_name on ToolContext")
                .to_string();
            let mut child = Ticket::new(task).label(&to).parent(&parent_key);
            if let Some(schema) = child_schema {
                child = child.schema(schema);
            }
            let child_key = ticket_system.insert(child, reporter);

            Ok(ToolResult::success(format!(
                "Ticket {parent_key} marked done; handed off to {child_key} (to: {to})"
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::agents::tickets::{Status, Ticket, TicketSystem};
    use crate::schemas::Schema;

    fn ctx_with(ticket_system: Arc<TicketSystem>, agent: &str, dir: PathBuf) -> ToolContext {
        ToolContext::new(dir)
            .ticket_system(ticket_system)
            .agent_name(agent.to_string())
    }

    fn one_ticket(agent: &str) -> (Arc<TicketSystem>, String) {
        let sys = TicketSystem::new();
        sys.insert(Ticket::new("parent body").label(agent), "tester".into());
        let key = sys
            .claim(|t| t.status == Status::Todo, agent)
            .expect("claim must succeed");
        (sys, key)
    }

    #[tokio::test]
    async fn happy_path_finishes_parent_creates_child_with_parent_link() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, parent_key) = one_ticket("alice");
        let sys = Arc::clone(&sys).dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());

        let outcome = WriteHandoverTool
            .call(
                serde_json::json!({
                    "to": "bob",
                    "task": "continue with X",
                    "result": "summary of alice's work"
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Success(_)));

        let parent = sys.get(&parent_key).unwrap();
        assert_eq!(parent.status, Status::Done);
        assert_eq!(parent.result_string().as_deref(), Some("summary of alice's work"));

        let child = sys.get("TICKET-2").unwrap();
        assert_eq!(child.status, Status::Todo);
        assert_eq!(child.parent_key(), Some(parent_key.as_str()));
        assert_eq!(child.labels, vec!["bob".to_string()]);
        assert_eq!(child.reporter(), "alice");
    }

    #[tokio::test]
    async fn appends_one_ndjson_line_for_parent_result() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, parent_key) = one_ticket("alice");
        let sys = Arc::clone(&sys).dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());

        WriteHandoverTool
            .call(
                serde_json::json!({"to": "bob", "task": "next", "result": "done part 1"}),
                &ctx,
            )
            .await
            .unwrap();

        let log = std::fs::read_to_string(dir.path().join("results.jsonl")).unwrap();
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 1, "only the parent finish writes a result line");
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed["ticket"], parent_key.as_str());
        assert_eq!(parsed["agent"], "alice");
        assert_eq!(parsed["result"], "done part 1");
    }

    #[tokio::test]
    async fn schema_violation_aborts_atomically() {
        // Parent demands a string of at least 50 characters; we pass
        // a short string, which violates the schema but passes the
        // type check, so we exercise the schema-validation abort path.
        let dir = crate::test_util::TempDir::new().unwrap();
        let sys = TicketSystem::new().dir(dir.path().to_path_buf());
        let schema = Schema::parse(serde_json::json!({
            "type": "string",
            "minLength": 50
        }))
        .unwrap();
        sys.insert(
            Ticket::new("strict parent").schema(schema).label("alice"),
            "tester".into(),
        );
        let parent_key = sys
            .claim(|t| t.status == Status::Todo, "alice")
            .expect("claim must succeed");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());

        let outcome = WriteHandoverTool
            .call(
                serde_json::json!({"to": "bob", "task": "next", "result": "too short"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::SchemaError(_)));

        let parent = sys.get(&parent_key).unwrap();
        assert_eq!(parent.status, Status::InProgress);
        assert!(parent.result().is_none());
        assert!(sys.get("TICKET-2").is_none(), "no child created on schema failure");
        assert!(!dir.path().join("results.jsonl").exists());
    }

    #[tokio::test]
    async fn malformed_schema_aborts_atomically() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, parent_key) = one_ticket("alice");
        let sys = Arc::clone(&sys).dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());

        let outcome = WriteHandoverTool
            .call(
                serde_json::json!({
                    "to": "bob",
                    "task": "next",
                    "result": "ok",
                    "schema": {"type": "not_a_real_type"}
                }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));

        let parent = sys.get(&parent_key).unwrap();
        assert_eq!(parent.status, Status::InProgress);
        assert!(sys.get("TICKET-2").is_none());
        assert!(!dir.path().join("results.jsonl").exists());
    }

    #[tokio::test]
    async fn optional_schema_attached_to_child() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, _key) = one_ticket("alice");
        let sys = Arc::clone(&sys).dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());

        WriteHandoverTool
            .call(
                serde_json::json!({
                    "to": "bob",
                    "task": "produce a report",
                    "result": "alice's findings",
                    "schema": {"type": "object", "required": ["title"]}
                }),
                &ctx,
            )
            .await
            .unwrap();

        let child = sys.get("TICKET-2").unwrap();
        assert!(child.schema.is_some());
    }

    #[tokio::test]
    async fn rejects_missing_to() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, _key) = one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteHandoverTool
            .call(serde_json::json!({"task": "x", "result": "y"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn rejects_empty_to() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, _key) = one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteHandoverTool
            .call(
                serde_json::json!({"to": "  ", "task": "x", "result": "y"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn rejects_missing_task() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, _key) = one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteHandoverTool
            .call(serde_json::json!({"to": "bob", "result": "y"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn rejects_missing_result() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, _key) = one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteHandoverTool
            .call(serde_json::json!({"to": "bob", "task": "x"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn rejects_null_or_empty_result() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, _key) = one_ticket("alice");
        let sys = Arc::clone(&sys).dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        for body in [
            serde_json::json!({"to": "bob", "task": "x", "result": null}),
            serde_json::json!({"to": "bob", "task": "x", "result": ""}),
        ] {
            let outcome = WriteHandoverTool.call(body, &ctx).await.unwrap();
            assert!(matches!(outcome, ToolResult::Error(_)));
        }
    }

    #[tokio::test]
    async fn rejects_non_string_result() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, parent_key) = one_ticket("alice");
        let sys = Arc::clone(&sys).dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        for non_string in [
            serde_json::json!({"to": "bob", "task": "next", "result": 42}),
            serde_json::json!({"to": "bob", "task": "next", "result": [1, 2, 3]}),
            serde_json::json!({"to": "bob", "task": "next", "result": {"k": "v"}}),
        ] {
            let outcome = WriteHandoverTool.call(non_string, &ctx).await.unwrap();
            assert!(matches!(outcome, ToolResult::Error(_)));
        }
        assert_eq!(sys.get(&parent_key).unwrap().status, Status::InProgress);
        assert!(!dir.path().join("results.jsonl").exists());
    }

    #[tokio::test]
    async fn rejects_non_string_task() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let (sys, _key) = one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteHandoverTool
            .call(
                serde_json::json!({"to": "bob", "task": 42, "result": "ok"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn errors_when_no_current_ticket() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let sys = TicketSystem::new();
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteHandoverTool
            .call(
                serde_json::json!({"to": "bob", "task": "x", "result": "y"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }
}
