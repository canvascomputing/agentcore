//! Ticket tools — give an agent a call surface for reading and mutating
//! the surrounding `TicketSystem`. Three tools share one dispatch
//! helper: `ReadTicketsTool` (read-only), `WriteTicketsTool` (mutating),
//! `ManageTicketsTool` (both).

use serde_json::Value;

use crate::agents::tickets::{
    insert_ticket, tickets_edit, tickets_get, tickets_search, tickets_set_result,
    tickets_update_status, Status, Ticket, TicketError, TicketSystem,
};
use crate::schemas::{format_violations, Schema};

use super::tool::{ToolContext, ToolResult};

mod manage_tickets;
mod read_tickets;
mod write_tickets;

pub use manage_tickets::ManageTicketsTool;
pub use read_tickets::ReadTicketsTool;
pub use write_tickets::WriteTicketsTool;

/// Action sets each tool exposes. Keeps the dispatch logic in one place
/// and lets each tool reject actions outside its allow-list with a
/// uniform error message.
pub(super) const READ_ACTIONS: &[&str] = &["get", "list", "search"];
pub(super) const WRITE_ACTIONS: &[&str] = &["create", "edit", "done"];

pub(super) fn dispatch(input: Value, ctx: &ToolContext, allowed: &[&str]) -> ToolResult {
    let action = match input["action"].as_str() {
        Some(a) => a,
        None => return ToolResult::error("Missing required parameter: action"),
    };
    if !allowed.contains(&action) {
        return ToolResult::error(format!(
            "Action `{action}` is not supported by this tool. Allowed: {}",
            allowed.join(", ")
        ));
    }
    let Some(ticket_system) = ctx.ticket_system_handle().cloned() else {
        return ToolResult::error("Ticket system unavailable in this context");
    };

    match action {
        "get" => action_get(&ticket_system, &input, ctx),
        "list" => action_list(&ticket_system, &input),
        "search" => action_search(&ticket_system, &input),
        "create" => action_create(&ticket_system, &input, ctx),
        "edit" => action_edit(&ticket_system, &input, ctx),
        "done" => action_done(&ticket_system, &input, ctx),
        other => ToolResult::error(format!("Unknown action `{other}`")),
    }
}

fn resolve_key(
    ticket_system: &TicketSystem,
    input: &Value,
    ctx: &ToolContext,
) -> Result<String, ToolResult> {
    if let Some(k) = input["key"].as_str() {
        return Ok(k.to_string());
    }
    let agent_name = ctx.agent_name_str().ok_or_else(|| {
        ToolResult::error("Missing `key` and no agent_name set on this tool context")
    })?;
    match ticket_system.find(|t| t.is_in_progress() && t.is_assigned_to(agent_name)) {
        Some(t) => Ok(t.key().to_string()),
        None => Err(ToolResult::error(
            "Missing `key` and no current ticket assigned to this agent",
        )),
    }
}

fn ticket_error_message(err: TicketError) -> String {
    err.to_string()
}

fn render_ticket(t: &Ticket) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n", t.key()));
    out.push_str(&format!("- status: {}\n", status_label(t.status())));
    out.push_str(&format!("- reporter: {}\n", t.reporter()));
    out.push_str(&format!(
        "- assignee: {}\n",
        t.assignee().unwrap_or("(none)")
    ));
    let labels_label = if t.labels.is_empty() {
        "(none)".to_string()
    } else {
        t.labels.join(", ")
    };
    out.push_str(&format!("- labels: {labels_label}\n"));
    out.push('\n');
    match &t.task {
        serde_json::Value::String(s) => {
            out.push_str(s);
            out.push('\n');
        }
        other => {
            out.push_str("```json\n");
            out.push_str(&serde_json::to_string_pretty(other).unwrap_or_default());
            out.push_str("\n```\n");
        }
    }
    out.push_str("\n## Result\n");
    out.push_str(t.result().unwrap_or("(no result)"));
    out.push('\n');
    out
}

fn status_label(s: Status) -> &'static str {
    match s {
        Status::Todo => "Todo",
        Status::InProgress => "InProgress",
        Status::Done => "Done",
        Status::Failed => "Failed",
    }
}

fn parse_status_for_list(s: &str) -> Result<Status, ToolResult> {
    match s {
        "Todo" => Ok(Status::Todo),
        "InProgress" => Ok(Status::InProgress),
        "Done" => Ok(Status::Done),
        "Failed" => Ok(Status::Failed),
        other => Err(ToolResult::error(format!(
            "Invalid status `{other}`. Expected one of Todo, InProgress, Done, Failed"
        ))),
    }
}

fn truncate_for_preview(s: &str, max: usize) -> String {
    let one_line = s.lines().next().unwrap_or("");
    if one_line.chars().count() <= max {
        one_line.to_string()
    } else {
        let cut: String = one_line.chars().take(max).collect();
        format!("{cut}…")
    }
}

fn render_summary_list(
    tickets: &[(&str, &str, Status, Option<&str>, &[String])],
) -> String {
    let mut out = String::new();
    for (key, task_preview, status, assignee, labels) in tickets {
        let labels_label = if labels.is_empty() {
            String::new()
        } else {
            format!("[{}] ", labels.join(","))
        };
        out.push_str(&format!(
            "- {key} [{status}] {labels_label}{assignee_label} — {task_preview}\n",
            status = status_label(*status),
            assignee_label = match assignee {
                Some(a) => format!("@{a}"),
                None => "(unassigned)".to_string(),
            },
        ));
    }
    out
}

fn task_preview(task: &serde_json::Value) -> String {
    let raw = match task {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    truncate_for_preview(&raw, 80)
}

fn action_get(ticket_system: &TicketSystem, input: &Value, ctx: &ToolContext) -> ToolResult {
    let key = match resolve_key(ticket_system, input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };
    match tickets_get(ticket_system, &key) {
        Some(t) => ToolResult::success(render_ticket(&t)),
        None => ToolResult::error(format!("Ticket {key} not found")),
    }
}

fn action_list(ticket_system: &TicketSystem, input: &Value) -> ToolResult {
    let assignee = input["assignee"].as_str().map(String::from);
    let status = input["status"].as_str().map(parse_status_for_list);
    let status = match status {
        Some(Ok(s)) => Some(s),
        Some(Err(e)) => return e,
        None => None,
    };

    let pool: Vec<Ticket> = ticket_system.filter(|t| {
        let status_ok = match status {
            Some(s) => t.status() == s,
            None => true,
        };
        let assignee_ok = match assignee.as_deref() {
            Some(a) => t.is_assigned_to(a),
            None => true,
        };
        status_ok && assignee_ok
    });

    if pool.is_empty() {
        return ToolResult::success("(no matching tickets)".to_string());
    }
    let previews: Vec<String> = pool.iter().take(50).map(|t| task_preview(&t.task)).collect();
    let rows: Vec<(&str, &str, Status, Option<&str>, &[String])> = pool
        .iter()
        .take(50)
        .zip(previews.iter())
        .map(|(t, p)| (t.key(), p.as_str(), t.status(), t.assignee(), t.labels.as_slice()))
        .collect();
    ToolResult::success(render_summary_list(&rows))
}

fn action_search(ticket_system: &TicketSystem, input: &Value) -> ToolResult {
    let query = match input["query"].as_str() {
        Some(q) => q,
        None => return ToolResult::error("Missing required parameter: query"),
    };
    let hits = tickets_search(ticket_system, query);
    if hits.is_empty() {
        return ToolResult::success("(no matching tickets)".to_string());
    }
    let previews: Vec<String> = hits.iter().take(50).map(|t| task_preview(&t.task)).collect();
    let rows: Vec<(&str, &str, Status, Option<&str>, &[String])> = hits
        .iter()
        .take(50)
        .zip(previews.iter())
        .map(|(t, p)| (t.key(), p.as_str(), t.status(), t.assignee(), t.labels.as_slice()))
        .collect();
    ToolResult::success(render_summary_list(&rows))
}

fn action_create(ticket_system: &TicketSystem, input: &Value, ctx: &ToolContext) -> ToolResult {
    let task = match input.get("task") {
        Some(v) => v.clone(),
        None => return ToolResult::error("Missing required parameter: task"),
    };

    let labels: Vec<String> = match input.get("labels") {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(_) => return ToolResult::error("`labels` must be an array of strings"),
    };

    let schema = match input.get("schema") {
        Some(doc) if !doc.is_null() => match Schema::parse(doc.clone()) {
            Ok(s) => Some(s),
            Err(e) => {
                return ToolResult::error(format!(
                    "Cannot create: supplied `schema` is invalid: {e}"
                ));
            }
        },
        _ => None,
    };

    let assignee = input
        .get("assignee")
        .and_then(|v| v.as_str())
        .map(String::from);

    let mut ticket = Ticket::new(task).labels(labels);
    if let Some(schema) = schema {
        ticket = ticket.schema(schema);
    }
    if let Some(who) = assignee {
        ticket = ticket.assign_to(who);
    }

    let reporter = ctx
        .agent_name_str()
        .expect("agent_name on ToolContext")
        .to_string();
    let key = insert_ticket(ticket_system, ticket, reporter);
    ToolResult::success(format!("Created ticket {key}"))
}

fn action_edit(ticket_system: &TicketSystem, input: &Value, ctx: &ToolContext) -> ToolResult {
    let key = match resolve_key(ticket_system, input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };

    let new_task = input.get("task").cloned();
    let new_labels: Option<Vec<String>> = match input.get("labels") {
        Some(Value::Array(arr)) => Some(
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
        ),
        Some(Value::Null) | None => None,
        Some(_) => return ToolResult::error("`labels` must be an array of strings"),
    };
    let new_schema: Option<Option<Schema>> = match input.get("schema") {
        Some(Value::Null) => Some(None),
        Some(doc) => match Schema::parse(doc.clone()) {
            Ok(s) => Some(Some(s)),
            Err(e) => {
                return ToolResult::error(format!(
                    "Cannot edit {key}: supplied `schema` is invalid: {e}"
                ));
            }
        },
        None => None,
    };

    if new_task.is_none() && new_labels.is_none() && new_schema.is_none() {
        return ToolResult::error("Edit needs at least one of `task`, `labels`, or `schema`");
    }

    match tickets_edit(ticket_system, &key, new_task, new_labels, new_schema) {
        Ok(()) => ToolResult::success(format!("Edited ticket {key}")),
        Err(e) => ToolResult::error(ticket_error_message(e)),
    }
}

fn action_done(ticket_system: &TicketSystem, input: &Value, ctx: &ToolContext) -> ToolResult {
    let key = match resolve_key(ticket_system, input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };

    let result = input
        .get("result")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let schema = tickets_get(ticket_system, &key).and_then(|t| t.schema.clone());

    if let Some(schema) = schema.as_ref() {
        let parsed: serde_json::Value = match serde_json::from_str(&result) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::schema_error(format!(
                    "Result is not valid JSON: {e}"
                ));
            }
        };
        if let Err(violations) = schema.validate(&parsed) {
            return ToolResult::schema_error(format_violations(&violations));
        }
    }

    if let Err(e) = tickets_set_result(ticket_system, &key, result) {
        return ToolResult::error(ticket_error_message(e));
    }
    match tickets_update_status(ticket_system, &key, Status::Done) {
        Ok(()) => ToolResult::success(format!("Ticket {key} marked done")),
        Err(e) => ToolResult::error(ticket_error_message(e)),
    }
}


#[cfg(test)]
mod tests {
    use super::super::tool::ToolLike;
    use super::*;
    use crate::agents::tickets::{
        insert_ticket, tickets_assign_to, tickets_force_status, tickets_get,
        tickets_update_status, TicketSystem,
    };
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Build a context for a tool test, optionally with a "current
    /// ticket" already InProgress and assigned to `agent`.
    fn ctx_with(
        ticket_system: Arc<TicketSystem>,
        agent: &str,
    ) -> ToolContext {
        ToolContext::new(PathBuf::from("/tmp"))
            .ticket_system(ticket_system)
            .agent_name(agent.to_string())
    }

    /// Insert one Todo ticket, force it to InProgress, assign to `agent`,
    /// so `sys.find(...)` resolves it as the current ticket for `agent`.
    fn shared_with_one_ticket(agent: &str) -> (Arc<TicketSystem>, String) {
        let sys = TicketSystem::new();
        let key = insert_ticket(&sys, Ticket::new("body"), "tester".into());
        tickets_force_status(&sys, &key, Status::InProgress).unwrap();
        tickets_assign_to(&sys, &key, agent).unwrap();
        (sys, key)
    }

    async fn call(
        tool: &dyn ToolLike,
        input: serde_json::Value,
        ctx: &ToolContext,
    ) -> ToolResult {
        tool.call(input, ctx).await.unwrap()
    }

    fn unwrap_text(result: &ToolResult) -> &str {
        let (ToolResult::Success(s) | ToolResult::Error(s) | ToolResult::SchemaError(s)) =
            result;
        s
    }

    #[tokio::test]
    async fn read_get_defaults_key_to_current_ticket() {
        let (sys, key) = shared_with_one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "get"}),
            &ctx,
        )
        .await;
        let text = unwrap_text(&result);
        assert!(text.contains(&key), "expected key in output: {text}");
        assert!(text.contains("body"));
    }

    #[tokio::test]
    async fn read_list_filters_by_status() {
        let sys = TicketSystem::new();
        insert_ticket(&sys, Ticket::new("a"), "tester".into());
        insert_ticket(&sys, Ticket::new("b"), "tester".into());
        tickets_update_status(&sys, "TICKET-1", Status::InProgress).unwrap();

        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "list", "status": "InProgress"}),
            &ctx,
        )
        .await;
        let text = unwrap_text(&result);
        assert!(text.contains("TICKET-1"));
        assert!(!text.contains("TICKET-2"));
    }

    #[tokio::test]
    async fn write_create_stamps_reporter_from_agent_name() {
        let sys = TicketSystem::new();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "create", "task": "new ticket"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, "TICKET-1").unwrap();
        assert_eq!(t.task, serde_json::Value::String("new ticket".into()));
        assert_eq!(t.reporter(), "alice");
    }

    #[tokio::test]
    async fn write_create_with_labels_attaches_them() {
        let sys = TicketSystem::new();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "create",
                "task": "new",
                "labels": ["research"]
            }),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, "TICKET-1").unwrap();
        assert_eq!(t.labels, vec!["research".to_string()]);
        assert!(t.assignee().is_none());
        assert_eq!(t.status(), Status::Todo);
    }

    #[tokio::test]
    async fn write_create_with_explicit_assignee_births_inprogress() {
        let sys = TicketSystem::new();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "create",
                "task": "new",
                "assignee": "alice"
            }),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, "TICKET-1").unwrap();
        assert_eq!(t.assignee(), Some("alice"));
        assert_eq!(t.status(), Status::InProgress);
    }

    #[tokio::test]
    async fn write_create_with_schema_field_stores_schema() {
        let sys = TicketSystem::new();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "create",
                "task": "new",
                "schema": {"type": "string"}
            }),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        assert!(tickets_get(&sys, "TICKET-1").unwrap().schema.is_some());
    }

    #[tokio::test]
    async fn write_done_without_schema_sets_result_and_status() {
        let (sys, key) = shared_with_one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "done", "result": "answer text"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result(), Some("answer text"));
    }

    #[tokio::test]
    async fn write_done_with_schema_returns_schema_error_on_mismatch() {
        let sys = TicketSystem::new();
        let schema = Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        }))
        .unwrap();
        let key = insert_ticket(
            &sys,
            Ticket::new("hi").schema(schema),
            "tester".into(),
        );
        tickets_update_status(&sys, &key, Status::InProgress).unwrap();
        tickets_assign_to(&sys, &key, "alice").unwrap();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "done",
                "result": "{\"x\": 7}"
            }),
            &ctx,
        )
        .await;
        let ToolResult::SchemaError(message) = &result else {
            panic!("expected SchemaError, got {result:?}");
        };
        assert!(message.contains("Schema validation failed"));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.status(), Status::InProgress);
        assert!(t.result().is_none());
    }

    #[tokio::test]
    async fn write_done_with_schema_passes_when_result_is_valid_json() {
        let sys = TicketSystem::new();
        let schema = Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        }))
        .unwrap();
        let key = insert_ticket(
            &sys,
            Ticket::new("hi").schema(schema),
            "tester".into(),
        );
        tickets_update_status(&sys, &key, Status::InProgress).unwrap();
        tickets_assign_to(&sys, &key, "alice").unwrap();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "done",
                "result": "{\"x\": \"answer\"}"
            }),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result(), Some("{\"x\": \"answer\"}"));
    }

    #[tokio::test]
    async fn write_edit_updates_task_and_labels() {
        let (sys, key) = shared_with_one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "edit",
                "task": "new body",
                "labels": ["urgent", "review"]
            }),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.task, serde_json::Value::String("new body".into()));
        assert_eq!(t.labels, vec!["urgent".to_string(), "review".to_string()]);
    }

    #[tokio::test]
    async fn write_rejects_unsupported_actions() {
        let (sys, _key) = shared_with_one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        for action in ["transition", "comment", "assign", "attach"] {
            let result = call(
                &WriteTicketsTool,
                serde_json::json!({"action": action}),
                &ctx,
            )
            .await;
            assert!(
                matches!(result, ToolResult::Error(_)),
                "{action}: {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn manage_supports_done_action() {
        let (sys, key) = shared_with_one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = call(
            &ManageTicketsTool,
            serde_json::json!({"action": "done", "result": "fine"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        assert_eq!(tickets_get(&sys, &key).unwrap().status(), Status::Done);
    }
}
