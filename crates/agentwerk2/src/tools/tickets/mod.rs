//! Ticket tools — give an agent a call surface for reading and mutating
//! the surrounding `TicketSystem`. Three tools share one dispatch
//! helper: `ReadTicketsTool` (read-only), `WriteTicketsTool`
//! (mutating), `ManageTicketsTool` (both).

use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::agents::tickets::{Status, TicketError, TicketSystem};

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
pub(super) const WRITE_ACTIONS: &[&str] = &["create", "edit", "comment", "transition", "assign"];

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
    let Some(tickets) = ctx.tickets_handle().cloned() else {
        return ToolResult::error("Ticket system unavailable in this context");
    };

    match action {
        "get" => action_get(&tickets, &input, ctx),
        "list" => action_list(&tickets, &input),
        "search" => action_search(&tickets, &input),
        "create" => action_create(&tickets, &input, ctx),
        "edit" => action_edit(&tickets, &input, ctx),
        "comment" => action_comment(&tickets, &input, ctx),
        "transition" => action_transition(&tickets, &input, ctx),
        "assign" => action_assign(&tickets, &input, ctx),
        other => ToolResult::error(format!("Unknown action `{other}`")),
    }
}

fn resolve_key(input: &Value, ctx: &ToolContext) -> Result<String, ToolResult> {
    if let Some(k) = input["key"].as_str() {
        return Ok(k.to_string());
    }
    match ctx.current_ticket_key() {
        Some(k) => Ok(k.to_string()),
        None => Err(ToolResult::error(
            "Missing `key` and no current ticket bound to this agent",
        )),
    }
}

fn parse_status(s: &str) -> Result<Status, ToolResult> {
    match s {
        "Todo" => Ok(Status::Todo),
        "InProgress" => Ok(Status::InProgress),
        "InReview" => Ok(Status::InReview),
        "Done" => Ok(Status::Done),
        "Failed" => Ok(Status::Failed),
        other => Err(ToolResult::error(format!(
            "Invalid status `{other}`. Expected one of Todo, InProgress, InReview, Done, Failed"
        ))),
    }
}

fn status_label(s: Status) -> &'static str {
    match s {
        Status::Todo => "Todo",
        Status::InProgress => "InProgress",
        Status::InReview => "InReview",
        Status::Done => "Done",
        Status::Failed => "Failed",
    }
}

fn ticket_error_message(err: TicketError) -> String {
    err.to_string()
}

fn render_ticket(sys: &TicketSystem, key: &str) -> Option<String> {
    let t = sys.get(key)?;
    let mut out = String::new();
    out.push_str(&format!("# {} — {}\n", t.key, t.summary));
    out.push_str(&format!("- status: {}\n", status_label(t.status)));
    out.push_str(&format!("- type: {}\n", t.r#type));
    out.push_str(&format!("- reporter: {}\n", t.reporter));
    out.push_str(&format!(
        "- assignee: {}\n",
        t.assignee.as_deref().unwrap_or("(none)")
    ));
    if !t.description.is_empty() {
        out.push('\n');
        out.push_str(&t.description);
        out.push('\n');
    }
    if !t.comments.is_empty() {
        out.push_str("\n## Comments\n");
        for c in &t.comments {
            out.push_str(&format!("- {}: {}\n", c.author, c.body));
        }
    }
    Some(out)
}

fn render_summary_list(tickets: &[(&str, &str, Status, Option<&str>, &str)]) -> String {
    let mut out = String::new();
    for (key, summary, status, assignee, ticket_type) in tickets {
        out.push_str(&format!(
            "- {key} [{status}] [{ttype}] {assignee_label} — {summary}\n",
            status = status_label(*status),
            ttype = ticket_type,
            assignee_label = match assignee {
                Some(a) => format!("@{a}"),
                None => "(unassigned)".to_string(),
            },
        ));
    }
    out
}

fn action_get(tickets: &Arc<Mutex<TicketSystem>>, input: &Value, ctx: &ToolContext) -> ToolResult {
    let key = match resolve_key(input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let sys = tickets.lock().unwrap();
    match render_ticket(&sys, &key) {
        Some(text) => ToolResult::success(text),
        None => ToolResult::error(format!("Ticket {key} not found")),
    }
}

fn action_list(tickets: &Arc<Mutex<TicketSystem>>, input: &Value) -> ToolResult {
    let sys = tickets.lock().unwrap();
    let assignee = input["assignee"].as_str();
    let status = input["status"].as_str().map(parse_status);
    let status = match status {
        Some(Ok(s)) => Some(s),
        Some(Err(e)) => return e,
        None => None,
    };

    let pool: Vec<&_> = match (status, assignee) {
        (Some(s), Some(a)) => sys
            .list_by_status(s)
            .into_iter()
            .filter(|t| t.assignee.as_deref() == Some(a))
            .collect(),
        (Some(s), None) => sys.list_by_status(s),
        (None, Some(a)) => sys.list_by_assignee(a),
        (None, None) => {
            // No filter — list everything sorted by status grouping.
            let mut all: Vec<&_> = Vec::new();
            for s in [
                Status::Todo,
                Status::InProgress,
                Status::InReview,
                Status::Done,
                Status::Failed,
            ] {
                all.extend(sys.list_by_status(s));
            }
            all
        }
    };

    let mut rows = Vec::with_capacity(pool.len().min(50));
    for t in pool.iter().take(50) {
        rows.push((
            t.key.as_str(),
            t.summary.as_str(),
            t.status,
            t.assignee.as_deref(),
            t.r#type.as_str(),
        ));
    }
    if rows.is_empty() {
        return ToolResult::success("(no matching tickets)".to_string());
    }
    ToolResult::success(render_summary_list(&rows))
}

fn action_search(tickets: &Arc<Mutex<TicketSystem>>, input: &Value) -> ToolResult {
    let query = match input["query"].as_str() {
        Some(q) => q,
        None => return ToolResult::error("Missing required parameter: query"),
    };
    let sys = tickets.lock().unwrap();
    let hits = sys.search(query);
    let mut rows = Vec::with_capacity(hits.len().min(50));
    for t in hits.iter().take(50) {
        rows.push((
            t.key.as_str(),
            t.summary.as_str(),
            t.status,
            t.assignee.as_deref(),
            t.r#type.as_str(),
        ));
    }
    if rows.is_empty() {
        return ToolResult::success("(no matching tickets)".to_string());
    }
    ToolResult::success(render_summary_list(&rows))
}

fn action_create(
    tickets: &Arc<Mutex<TicketSystem>>,
    input: &Value,
    ctx: &ToolContext,
) -> ToolResult {
    let summary = match input["summary"].as_str() {
        Some(s) => s.to_string(),
        None => return ToolResult::error("Missing required parameter: summary"),
    };
    let description = input["description"].as_str().unwrap_or("").to_string();
    let ticket_type = input["type"].as_str().unwrap_or("task").to_string();
    let reporter = input["reporter"]
        .as_str()
        .map(|s| s.to_string())
        .or_else(|| ctx.agent_name_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "agent".to_string());

    let mut sys = tickets.lock().unwrap();
    let key = sys
        .create(summary, description, ticket_type, reporter)
        .key
        .clone();
    ToolResult::success(format!("Created ticket {key}"))
}

fn action_edit(tickets: &Arc<Mutex<TicketSystem>>, input: &Value, ctx: &ToolContext) -> ToolResult {
    let key = match resolve_key(input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let new_summary = input["summary"].as_str().map(|s| s.to_string());
    let new_description = input["description"].as_str().map(|s| s.to_string());
    if new_summary.is_none() && new_description.is_none() {
        return ToolResult::error("Edit needs at least one of `summary` or `description`");
    }
    let mut sys = tickets.lock().unwrap();
    match sys.edit_ticket(&key, new_summary, new_description) {
        Ok(()) => ToolResult::success(format!("Edited ticket {key}")),
        Err(e) => ToolResult::error(ticket_error_message(e)),
    }
}

fn action_comment(
    tickets: &Arc<Mutex<TicketSystem>>,
    input: &Value,
    ctx: &ToolContext,
) -> ToolResult {
    let key = match resolve_key(input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let body = match input["body"].as_str() {
        Some(b) => b.to_string(),
        None => return ToolResult::error("Missing required parameter: body"),
    };
    let author = input["author"]
        .as_str()
        .map(|s| s.to_string())
        .or_else(|| ctx.agent_name_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "agent".to_string());

    let mut sys = tickets.lock().unwrap();
    match sys.add_comment(&key, author, body) {
        Ok(()) => ToolResult::success(format!("Commented on {key}")),
        Err(e) => ToolResult::error(ticket_error_message(e)),
    }
}

fn action_transition(
    tickets: &Arc<Mutex<TicketSystem>>,
    input: &Value,
    ctx: &ToolContext,
) -> ToolResult {
    let key = match resolve_key(input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let status = match input["status"].as_str() {
        Some(s) => s,
        None => return ToolResult::error("Missing required parameter: status"),
    };
    let target = match parse_status(status) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mut sys = tickets.lock().unwrap();
    match sys.update_status(&key, target) {
        Ok(()) => ToolResult::success(format!("Transitioned {key} to {}", status_label(target))),
        Err(e) => ToolResult::error(ticket_error_message(e)),
    }
}

fn action_assign(
    tickets: &Arc<Mutex<TicketSystem>>,
    input: &Value,
    ctx: &ToolContext,
) -> ToolResult {
    let key = match resolve_key(input, ctx) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let assignee = match input["assignee"].as_str() {
        Some(a) => a.to_string(),
        None => return ToolResult::error("Missing required parameter: assignee"),
    };
    let mut sys = tickets.lock().unwrap();
    if assignee.is_empty() {
        match sys.clear_assignee(&key) {
            Ok(()) => ToolResult::success(format!("Cleared assignee on {key}")),
            Err(e) => ToolResult::error(ticket_error_message(e)),
        }
    } else {
        match sys.assign_to(&key, assignee.clone()) {
            Ok(()) => ToolResult::success(format!("Assigned {key} to {assignee}")),
            Err(e) => ToolResult::error(ticket_error_message(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tool::ToolLike;
    use super::*;
    use std::path::PathBuf;

    fn ctx_with(
        tickets: Arc<Mutex<TicketSystem>>,
        current: Option<&str>,
        agent: &str,
    ) -> ToolContext {
        let mut ctx = ToolContext::new(PathBuf::from("/tmp"))
            .tickets(tickets)
            .agent_name(agent.to_string());
        if let Some(k) = current {
            ctx = ctx.current_ticket(k.to_string());
        }
        ctx
    }

    fn shared_with_one_ticket() -> (Arc<Mutex<TicketSystem>>, String) {
        let mut sys = TicketSystem::new();
        let key = sys.create("title", "body", "task", "tester").key.clone();
        (Arc::new(Mutex::new(sys)), key)
    }

    async fn call(tool: &dyn ToolLike, input: serde_json::Value, ctx: &ToolContext) -> ToolResult {
        tool.call(input, ctx).await.unwrap()
    }

    fn unwrap_text(result: &ToolResult) -> &str {
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        s
    }

    // ---- ReadTicketsTool ----

    #[tokio::test]
    async fn read_get_defaults_key_to_current_ticket() {
        let (shared, key) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");
        let result = call(&ReadTicketsTool, serde_json::json!({"action": "get"}), &ctx).await;
        let text = unwrap_text(&result);
        assert!(text.contains(&key), "expected key in output: {text}");
        assert!(text.contains("title"));
    }

    #[tokio::test]
    async fn read_get_with_explicit_key() {
        let (shared, key) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), None, "alice");
        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "get", "key": key}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
    }

    #[tokio::test]
    async fn read_list_filters_by_status() {
        let mut sys = TicketSystem::new();
        let in_progress = sys.create("a", "", "task", "tester").key.clone();
        let _todo = sys.create("b", "", "task", "tester").key.clone();
        sys.update_status(&in_progress, Status::InProgress).unwrap();
        let shared = Arc::new(Mutex::new(sys));

        let ctx = ctx_with(Arc::clone(&shared), None, "alice");
        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "list", "status": "InProgress"}),
            &ctx,
        )
        .await;
        let text = unwrap_text(&result);
        assert!(text.contains(&in_progress));
        assert!(!text.contains("- TICKET-2"));
    }

    #[tokio::test]
    async fn read_list_filters_by_assignee() {
        let mut sys = TicketSystem::new();
        let mine = sys.create("mine", "", "task", "tester").key.clone();
        let theirs = sys.create("theirs", "", "task", "tester").key.clone();
        sys.assign_to(&mine, "alice").unwrap();
        sys.assign_to(&theirs, "bob").unwrap();
        let shared = Arc::new(Mutex::new(sys));

        let ctx = ctx_with(Arc::clone(&shared), None, "alice");
        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "list", "assignee": "alice"}),
            &ctx,
        )
        .await;
        let text = unwrap_text(&result);
        assert!(text.contains(&mine));
        assert!(!text.contains(&theirs));
    }

    #[tokio::test]
    async fn read_search_matches_summary_case_insensitively() {
        let mut sys = TicketSystem::new();
        let _ = sys.create("Fix Login", "", "task", "tester");
        let _ = sys.create("Other", "secret keyword", "task", "tester");
        let shared = Arc::new(Mutex::new(sys));

        let ctx = ctx_with(Arc::clone(&shared), None, "alice");
        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "search", "query": "LOGIN"}),
            &ctx,
        )
        .await;
        let text = unwrap_text(&result);
        assert!(text.contains("Fix Login"));
        assert!(!text.contains("Other"));

        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "search", "query": "keyword"}),
            &ctx,
        )
        .await;
        let text = unwrap_text(&result);
        assert!(text.contains("Other"));
    }

    #[tokio::test]
    async fn read_rejects_write_action() {
        let (shared, _) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), None, "alice");
        let result = call(
            &ReadTicketsTool,
            serde_json::json!({"action": "transition", "status": "Done"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Error(_)));
    }

    // ---- WriteTicketsTool ----

    #[tokio::test]
    async fn write_create_defaults_reporter_to_agent() {
        let shared: Arc<Mutex<TicketSystem>> = Arc::new(Mutex::new(TicketSystem::new()));
        let ctx = ctx_with(Arc::clone(&shared), None, "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "create",
                "summary": "new ticket",
            }),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let sys = shared.lock().unwrap();
        let ticket = sys.get("TICKET-1").unwrap();
        assert_eq!(ticket.summary, "new ticket");
        assert_eq!(ticket.reporter, "alice");
        assert_eq!(ticket.r#type, "task");
    }

    #[tokio::test]
    async fn write_edit_updates_summary_and_description() {
        let (shared, key) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({
                "action": "edit",
                "summary": "new summary",
                "description": "new body",
            }),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let sys = shared.lock().unwrap();
        let t = sys.get(&key).unwrap();
        assert_eq!(t.summary, "new summary");
        assert_eq!(t.description, "new body");
    }

    #[tokio::test]
    async fn write_comment_defaults_author_to_agent() {
        let (shared, key) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "comment", "body": "hi"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        let sys = shared.lock().unwrap();
        let comments = &sys.get(&key).unwrap().comments;
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].author, "alice");
        assert_eq!(comments[0].body, "hi");
    }

    #[tokio::test]
    async fn write_transition_in_progress_to_in_review_succeeds() {
        let (shared, key) = shared_with_one_ticket();
        shared
            .lock()
            .unwrap()
            .update_status(&key, Status::InProgress)
            .unwrap();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "transition", "status": "InReview"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        assert_eq!(
            shared.lock().unwrap().get(&key).unwrap().status,
            Status::InReview
        );
    }

    #[tokio::test]
    async fn write_transition_todo_to_done_returns_validator_error() {
        let (shared, key) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "transition", "status": "Done"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Error(_)));
        assert_eq!(
            shared.lock().unwrap().get(&key).unwrap().status,
            Status::Todo
        );
    }

    #[tokio::test]
    async fn write_assign_sets_and_clears_assignee() {
        let (shared, key) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");

        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "assign", "assignee": "bob"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        assert_eq!(
            shared
                .lock()
                .unwrap()
                .get(&key)
                .unwrap()
                .assignee
                .as_deref(),
            Some("bob"),
        );

        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "assign", "assignee": ""}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Success(_)));
        assert!(shared.lock().unwrap().get(&key).unwrap().assignee.is_none());
    }

    #[tokio::test]
    async fn write_rejects_read_action() {
        let (shared, key) = shared_with_one_ticket();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");
        let result = call(
            &WriteTicketsTool,
            serde_json::json!({"action": "get"}),
            &ctx,
        )
        .await;
        assert!(matches!(result, ToolResult::Error(_)));
    }

    // ---- ManageTicketsTool ----

    #[tokio::test]
    async fn manage_supports_list_and_transition_in_one_tool() {
        let (shared, key) = shared_with_one_ticket();
        shared
            .lock()
            .unwrap()
            .update_status(&key, Status::InProgress)
            .unwrap();
        let ctx = ctx_with(Arc::clone(&shared), Some(&key), "alice");

        let listed = call(
            &ManageTicketsTool,
            serde_json::json!({"action": "list", "status": "InProgress"}),
            &ctx,
        )
        .await;
        let text = unwrap_text(&listed);
        assert!(text.contains(&key));

        let done = call(
            &ManageTicketsTool,
            serde_json::json!({"action": "transition", "status": "Done"}),
            &ctx,
        )
        .await;
        assert!(matches!(done, ToolResult::Success(_)));
        assert_eq!(
            shared.lock().unwrap().get(&key).unwrap().status,
            Status::Done
        );
    }
}
