//! Zero-argument-friendly tool for marking the agent's current ticket
//! `Done`. Auto-registered on every `Agent`, so the done call is always
//! reachable. Optionally records a `result` and validates it against the
//! ticket's `schema`.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use crate::providers::ProviderResult;

use super::super::tool::{ToolContext, ToolLike, ToolResult};
use super::super::tool_file::ToolFile;
use super::{mark_done, resolve_current_key};

pub struct MarkTicketDoneTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("mark_ticket_done.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for MarkTicketDoneTool {
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
            let key = match resolve_current_key(&ticket_system, ctx) {
                Ok(k) => k,
                Err(e) => return Ok(e),
            };
            let result = input
                .get("result")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(mark_done(&ticket_system, &key, result))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::agents::tickets::{
        insert_ticket, tickets_assign_to, tickets_force_status, tickets_get, tickets_update_status,
        Status, Ticket, TicketSystem,
    };
    use crate::schemas::Schema;

    fn ctx_with(ticket_system: Arc<TicketSystem>, agent: &str) -> ToolContext {
        ToolContext::new(PathBuf::from("/tmp"))
            .ticket_system(ticket_system)
            .agent_name(agent.to_string())
    }

    fn shared_with_one_ticket(agent: &str) -> (Arc<TicketSystem>, String) {
        let sys = TicketSystem::new();
        let key = insert_ticket(&sys, Ticket::new("body"), "tester".into());
        tickets_force_status(&sys, &key, Status::InProgress).unwrap();
        tickets_assign_to(&sys, &key, agent).unwrap();
        (sys, key)
    }

    #[tokio::test]
    async fn marks_current_ticket_done_with_result() {
        let (sys, key) = shared_with_one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = MarkTicketDoneTool
            .call(serde_json::json!({"result": "answer text"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result(), Some("answer text"));
    }

    #[tokio::test]
    async fn marks_current_ticket_done_with_no_result() {
        let (sys, key) = shared_with_one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = MarkTicketDoneTool
            .call(serde_json::json!({}), &ctx)
            .await
            .unwrap();
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result(), Some(""));
    }

    #[tokio::test]
    async fn errors_when_no_current_ticket() {
        let sys = TicketSystem::new();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = MarkTicketDoneTool
            .call(serde_json::json!({}), &ctx)
            .await
            .unwrap();
        let ToolResult::Error(message) = &result else {
            panic!("expected Error, got {result:?}");
        };
        assert!(message.contains("no current ticket"));
    }

    #[tokio::test]
    async fn errors_when_ticket_system_unavailable() {
        let ctx = ToolContext::new(PathBuf::from("/tmp")).agent_name("alice".into());
        let result = MarkTicketDoneTool
            .call(serde_json::json!({}), &ctx)
            .await
            .unwrap();
        let ToolResult::Error(message) = &result else {
            panic!("expected Error, got {result:?}");
        };
        assert!(message.contains("Ticket system unavailable"));
    }

    #[tokio::test]
    async fn returns_schema_error_on_mismatch() {
        let sys = TicketSystem::new();
        let schema = Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        }))
        .unwrap();
        let key = insert_ticket(&sys, Ticket::new("hi").schema(schema), "tester".into());
        tickets_update_status(&sys, &key, Status::InProgress).unwrap();
        tickets_assign_to(&sys, &key, "alice").unwrap();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = MarkTicketDoneTool
            .call(serde_json::json!({"result": "{\"x\": 7}"}), &ctx)
            .await
            .unwrap();
        let ToolResult::SchemaError(message) = &result else {
            panic!("expected SchemaError, got {result:?}");
        };
        assert!(message.contains("Schema validation failed"));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.status(), Status::InProgress);
        assert!(t.result().is_none());
    }

    #[tokio::test]
    async fn passes_schema_when_result_is_valid_json() {
        let sys = TicketSystem::new();
        let schema = Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        }))
        .unwrap();
        let key = insert_ticket(&sys, Ticket::new("hi").schema(schema), "tester".into());
        tickets_update_status(&sys, &key, Status::InProgress).unwrap();
        tickets_assign_to(&sys, &key, "alice").unwrap();
        let ctx = ctx_with(Arc::clone(&sys), "alice");
        let result = MarkTicketDoneTool
            .call(serde_json::json!({"result": "{\"x\": \"answer\"}"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(result, ToolResult::Success(_)));
        let t = tickets_get(&sys, &key).unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result(), Some("{\"x\": \"answer\"}"));
    }
}
