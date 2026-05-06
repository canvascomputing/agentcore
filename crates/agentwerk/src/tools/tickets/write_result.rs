//! Single-purpose tool for finishing a ticket. Validates the agent's
//! result against the ticket's schema (when set), appends an NDJSON
//! line to `<results_dir>/results.jsonl`, attaches the `TicketResult`
//! to the ticket, and transitions the ticket to `Done`.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

static RESULTS_WRITE_LOCK: Mutex<()> = Mutex::new(());

pub(super) fn results_write_lock() -> &'static Mutex<()> {
    &RESULTS_WRITE_LOCK
}

use crate::providers::ProviderResult;

use super::super::tool::{ToolContext, ToolLike, ToolResult};
use super::super::tool_file::ToolFile;
use super::{resolve_current_key, write_result};

pub struct WriteResultTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("write_result.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for WriteResultTool {
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
            let result = match input.get("result") {
                Some(v) => v.clone(),
                None => return Ok(ToolResult::error("Missing required parameter: result")),
            };
            Ok(write_result(&ticket_system, ctx, &key, result))
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

    fn ctx_with(ticket_system: Arc<TicketSystem>, agent: &str, working_dir: PathBuf) -> ToolContext {
        ToolContext::new(working_dir)
            .ticket_system(ticket_system)
            .agent_name(agent.to_string())
    }

    fn one_ticket(agent: &str) -> (Arc<TicketSystem>, String) {
        let sys = TicketSystem::new();
        let key = sys.insert(Ticket::new("body"), "tester".into());
        sys.force_status(&key, Status::InProgress).unwrap();
        sys.set_assignee(&key, agent).unwrap();
        (sys, key)
    }

    #[tokio::test]
    async fn writes_string_result_and_marks_done() {
        let dir = tempfile::tempdir().unwrap();
        let (sys, key) = one_ticket("alice");
        let sys = Arc::clone(&sys).results_dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": "the answer"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Success(_)));
        let t = sys.get(&key).unwrap();
        assert_eq!(t.status(), Status::Done);
        let attached = t.result().unwrap();
        assert_eq!(attached.agent, "alice");
        assert_eq!(attached.ticket, key);
        assert_eq!(attached.result, serde_json::Value::String("the answer".into()));

        let log = std::fs::read_to_string(dir.path().join("results.jsonl")).unwrap();
        let line = log.trim_end();
        let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(parsed["agent"], "alice");
        assert_eq!(parsed["ticket"], key.as_str());
        assert_eq!(parsed["result"], "the answer");
    }

    #[tokio::test]
    async fn rejects_empty_string_when_no_schema() {
        let dir = tempfile::tempdir().unwrap();
        let (sys, key) = one_ticket("alice");
        let sys = Arc::clone(&sys).results_dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": ""}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
        let t = sys.get(&key).unwrap();
        assert_eq!(t.status(), Status::InProgress);
        assert!(!dir.path().join("results.jsonl").exists());
    }

    #[tokio::test]
    async fn accepts_structured_value_when_no_schema() {
        let dir = tempfile::tempdir().unwrap();
        let (sys, key) = one_ticket("alice");
        let sys = Arc::clone(&sys).results_dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": {"x": 1, "y": [2, 3]}}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Success(_)));
        let t = sys.get(&key).unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result().unwrap().result["x"], 1);

        // The saved `result` field is a JSON object, not an escaped
        // string of JSON.
        let log = std::fs::read_to_string(dir.path().join("results.jsonl")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(log.trim_end()).unwrap();
        assert!(parsed["result"].is_object(), "expected raw object, got {parsed}");
        assert_eq!(parsed["result"]["x"], 1);
    }

    #[tokio::test]
    async fn rejects_null_result_when_no_schema() {
        let dir = tempfile::tempdir().unwrap();
        let (sys, _key) = one_ticket("alice");
        let sys = Arc::clone(&sys).results_dir(dir.path().to_path_buf());
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": null}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn validates_against_schema() {
        let dir = tempfile::tempdir().unwrap();
        let sys = TicketSystem::new().results_dir(dir.path().to_path_buf());
        let schema = Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        }))
        .unwrap();
        let key = sys.insert(Ticket::new("hi").schema(schema), "tester".into());
        sys.force_status(&key, Status::InProgress).unwrap();
        sys.set_assignee(&key, "alice").unwrap();
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());

        // wrong shape
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": {"x": 7}}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::SchemaError(_)));
        let t = sys.get(&key).unwrap();
        assert_eq!(t.status(), Status::InProgress);

        // valid shape
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": {"x": "ok"}}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Success(_)));
        let t = sys.get(&key).unwrap();
        assert_eq!(t.status(), Status::Done);
        assert_eq!(t.result().unwrap().result["x"], "ok");
    }

    #[tokio::test]
    async fn falls_back_to_working_dir_when_results_dir_unset() {
        let dir = tempfile::tempdir().unwrap();
        let (sys, _key) = one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": "x"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Success(_)));
        assert!(dir.path().join("results.jsonl").exists());
    }

    #[tokio::test]
    async fn errors_when_no_current_ticket() {
        let dir = tempfile::tempdir().unwrap();
        let sys = TicketSystem::new();
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteResultTool
            .call(serde_json::json!({"result": "x"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn errors_when_result_missing() {
        let dir = tempfile::tempdir().unwrap();
        let (sys, _key) = one_ticket("alice");
        let ctx = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        let outcome = WriteResultTool
            .call(serde_json::json!({}), &ctx)
            .await
            .unwrap();
        assert!(matches!(outcome, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn appends_one_line_per_completed_ticket() {
        let dir = tempfile::tempdir().unwrap();
        let sys = TicketSystem::new().results_dir(dir.path().to_path_buf());

        let key1 = sys.insert(Ticket::new("a"), "tester".into());
        sys.force_status(&key1, Status::InProgress).unwrap();
        sys.set_assignee(&key1, "alice").unwrap();
        let ctx_alice = ctx_with(Arc::clone(&sys), "alice", dir.path().to_path_buf());
        WriteResultTool
            .call(serde_json::json!({"result": "from alice"}), &ctx_alice)
            .await
            .unwrap();

        let key2 = sys.insert(Ticket::new("b"), "tester".into());
        sys.force_status(&key2, Status::InProgress).unwrap();
        sys.set_assignee(&key2, "bob").unwrap();
        let ctx_bob = ctx_with(Arc::clone(&sys), "bob", dir.path().to_path_buf());
        WriteResultTool
            .call(serde_json::json!({"result": "from bob"}), &ctx_bob)
            .await
            .unwrap();

        let log = std::fs::read_to_string(dir.path().join("results.jsonl")).unwrap();
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(first["agent"], "alice");
        assert_eq!(first["ticket"], key1.as_str());
        assert_eq!(first["result"], "from alice");
        assert_eq!(second["agent"], "bob");
        assert_eq!(second["ticket"], key2.as_str());
        assert_eq!(second["result"], "from bob");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_writes_produce_one_intact_line_per_ticket() {
        const N: usize = 32;
        let dir = tempfile::tempdir().unwrap();
        let sys = TicketSystem::new().results_dir(dir.path().to_path_buf());

        let mut expected = Vec::with_capacity(N);
        for i in 0..N {
            let agent = format!("agent_{i}");
            let key = sys.insert(Ticket::new(format!("body_{i}")), "tester".into());
            sys.force_status(&key, Status::InProgress).unwrap();
            sys.set_assignee(&key, &agent).unwrap();
            expected.push((agent, key));
        }

        let mut handles = Vec::with_capacity(N);
        for (i, (agent, _)) in expected.iter().enumerate() {
            let sys = Arc::clone(&sys);
            let dir_path = dir.path().to_path_buf();
            let agent = agent.clone();
            handles.push(tokio::spawn(async move {
                let ctx = ctx_with(sys, &agent, dir_path);
                WriteResultTool
                    .call(serde_json::json!({"result": format!("payload_{i}")}), &ctx)
                    .await
                    .unwrap()
            }));
        }
        for h in handles {
            assert!(matches!(h.await.unwrap(), ToolResult::Success(_)));
        }

        let log = std::fs::read_to_string(dir.path().join("results.jsonl")).unwrap();
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), N, "expected {N} lines, got {}", lines.len());

        let mut seen_tickets = std::collections::HashSet::new();
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("corrupt line {line:?}: {e}"));
            let ticket = parsed["ticket"].as_str().unwrap().to_string();
            assert!(seen_tickets.insert(ticket), "duplicate ticket in log");
        }
        let expected_keys: std::collections::HashSet<String> =
            expected.iter().map(|(_, k)| k.clone()).collect();
        assert_eq!(seen_tickets, expected_keys);
    }
}
