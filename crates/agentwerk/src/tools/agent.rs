//! Sub-agent invocation. Auto-registered when an agent has staff; lets a model delegate a subtask to a pre-configured child.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::Value;

use crate::tools::tool_file::ToolFile;

use crate::agent::Agent;
use crate::error::Result;
use crate::tools::error::ToolError;
use crate::tools::tool::{ToolContext, ToolLike, ToolResult};
use crate::util::generate_agent_name;

/// Default identity for ad-hoc sub-agents (when the model doesn't supply one).
const DEFAULT_IDENTITY: &str = "You are a focused helper agent. Answer concisely.";

/// Spawn a sub-agent and return its [`Output`](crate::Output). Auto-registered
/// when an agent calls `.staff(...)`. The sub-agent inherits the
/// caller's provider, model, working directory, event handler, and cancel
/// signal; tools and prompts come from the registered template.
pub struct AgentTool;

/// Tool-control fields. Per-agent config overrides (identity, model, max_*, …)
/// live in the same JSON object and are applied via `Agent::apply_overrides`.
#[derive(Deserialize)]
struct SpawnArgs {
    description: String,
    task: String,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    background: Option<bool>,
}

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("agent.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for AgentTool {
    fn name(&self) -> &str {
        &tool_file().name
    }

    fn description(&self) -> &str {
        description()
    }

    fn is_read_only(&self) -> bool {
        tool_file().read_only
    }

    fn input_schema(&self) -> Value {
        tool_file().input_schema.clone()
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let args: SpawnArgs = match serde_json::from_value(input.clone()) {
                Ok(a) => a,
                Err(e) => return Ok(ToolResult::error(format!("Invalid input: {e}"))),
            };

            let runtime = ctx
                .runtime
                .as_ref()
                .ok_or_else(|| ToolError::ExecutionFailed {
                    tool_name: tool_file().name.clone(),
                    message: "LoopRuntime not available in ToolContext".into(),
                })?
                .clone();
            let caller = ctx
                .caller_spec
                .as_ref()
                .ok_or_else(|| ToolError::ExecutionFailed {
                    tool_name: tool_file().name.clone(),
                    message: "caller LoopSpec not available in ToolContext".into(),
                })?
                .clone();

            // Resolve the base Agent: either a registered sub-agent, or a fresh
            // ad-hoc one seeded with the default identity and max_steps=10. The
            // overrides step below applies every LLM-supplied tuning knob to
            // this base, regardless of path.
            let base = match &args.agent {
                Some(name) => match caller
                    .staff
                    .iter()
                    .find(|a: &&Agent| a.get_name() == name.as_str())
                    .cloned()
                {
                    Some(a) => a,
                    None => return Ok(ToolResult::error(format!("No sub-agent named '{name}'"))),
                },
                None => Agent::new()
                    .name(&args.description)
                    .role(DEFAULT_IDENTITY)
                    .max_steps(10),
            };

            let agent = base.apply_overrides(&input).work(&args.task);

            if args.background.unwrap_or(false) {
                let id = generate_agent_name(&args.description);
                let work = runtime.incoming_work.clone();
                let agent_id = id.clone();
                let caller_for_child = caller.clone();
                tokio::spawn(async move {
                    let summary = match agent.execute_child(&caller_for_child, &runtime).await {
                        Ok(o) => o.response_raw,
                        Err(e) => format!("Failed: {e}"),
                    };
                    if let Some(w) = work {
                        w.add_notification(&agent_id, &summary);
                    }
                });
                Ok(ToolResult::success(format!(
                    "Background agent '{}' started (id: {id})",
                    args.description
                )))
            } else {
                match agent.execute_child(&caller, &runtime).await {
                    Ok(o) => Ok(ToolResult::success(o.response_raw)),
                    Err(e) => Ok(ToolResult::error(format!("Agent error: {e}"))),
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::work::Work;
    use crate::testutil::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn agent_tool_foreground() {
        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .role("Coordinate work.")
            .tool(AgentTool);

        let harness = TestHarness::new(MockProvider::new(vec![
            tool_response(
                "agent_tool",
                "sa1",
                serde_json::json!({
                    "description": "researcher",
                    "task": "Research topic X"
                }),
            ),
            text_response("research findings"),
            text_response("Summary: research findings"),
        ]));

        let output = harness.run_agent(&agent, "Do research").await.unwrap();
        assert_eq!(output.response_raw, "Summary: research findings");
    }

    #[tokio::test]
    async fn agent_tool_background_delivers_notification() {
        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .role("")
            .tool(AgentTool);

        let work = Arc::new(Work::new());

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "agent_tool",
                "sa1",
                serde_json::json!({
                    "description": "bg-worker",
                    "task": "Do work",
                    "background": true
                }),
            ),
            text_response("response-a"),
            text_response("response-b"),
        ]));

        let harness = TestHarness::with_provider_and_work(provider.clone(), work.clone());
        let output = harness
            .run_agent(&agent, "Start background work")
            .await
            .unwrap();
        assert!(!output.response_raw.is_empty());

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let task = work.take_if(None, |_| true);
        assert!(
            task.is_some(),
            "Expected notification from background agent"
        );
        let notification = task.unwrap().content;
        assert!(
            notification.contains("response-") || notification.contains("Failed"),
            "Notification should contain agent result: {notification}"
        );
    }

    #[tokio::test]
    async fn agent_tool_background_with_schema_adds_json() {
        // Background path: the child's `response_raw` is what `add_notification`
        // ships in the work inbox. With the new design, a schema-constrained
        // child's `response_raw` IS the validated JSON text — so the
        // notification must carry the JSON verbatim (modulo the
        // `"Task <id> completed:"` prefix).
        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .role("")
            .tool(AgentTool);

        let work = Arc::new(Work::new());
        let valid_json = r#"{"answer":42}"#;

        // Background spawn means the child's first step races the parent's
        // step 2 for the next mock response. Script both with the same valid
        // JSON so either interleaving succeeds: the child validates and
        // terminates; the parent (no schema) just returns whatever text it
        // got. The notification still carries the child's JSON.
        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "agent_tool",
                "sa1",
                serde_json::json!({
                    "description": "bg-classifier",
                    "task": "Answer.",
                    "identity": "You answer with JSON.",
                    "model": "mock",
                    "background": true,
                    "contract": {
                        "type": "object",
                        "properties": { "answer": { "type": "integer" } },
                        "required": ["answer"]
                    },
                }),
            ),
            text_response(valid_json),
            text_response(valid_json),
        ]));

        let harness = TestHarness::with_provider_and_work(provider.clone(), work.clone());
        let output = harness.run_agent(&agent, "go").await.unwrap();
        assert!(!output.response_raw.is_empty());

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let task = work.take_if(None, |_| true);
        let notification = task
            .expect("background agent must post a notification")
            .content;
        assert!(
            notification.contains(valid_json),
            "notification must carry the validated JSON, got: {notification}"
        );
    }

    #[tokio::test]
    async fn agent_tool_named_sub_agent() {
        let sub = Agent::new()
            .name("specialist")
            .model("mock")
            .role("I am a specialist.");

        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .role("")
            .staff(sub);

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "agent_tool",
                "sa1",
                serde_json::json!({
                    "description": "use specialist",
                    "task": "Do specialized work",
                    "agent": "specialist"
                }),
            ),
            text_response("specialized result"),
            text_response("Got specialized result"),
        ]));

        let harness = TestHarness::with_provider(provider);
        let output = harness
            .run_agent(&agent, "Use the specialist")
            .await
            .unwrap();
        assert_eq!(output.response_raw, "Got specialized result");
    }

    #[tokio::test]
    async fn agent_tool_propagates_max_input_tokens() {
        use crate::event::EventKind;
        use crate::provider::TokenUsage;

        let sub = Agent::new()
            .name("tight-budget")
            .model("mock")
            .role("I do work.")
            .tool(MockTool::new("t", false, "ok"));

        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .role("")
            .staff(sub);

        let mut child_turn = tool_response("t", "c1", serde_json::json!({}));
        child_turn.usage = TokenUsage {
            input_tokens: 5000,
            output_tokens: 0,
            ..Default::default()
        };

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "agent_tool",
                "sa1",
                serde_json::json!({
                    "description": "tight",
                    "task": "Do work",
                    "agent": "tight-budget",
                    "max_input_tokens": 4000,
                }),
            ),
            child_turn,
            text_response("done"),
        ]));

        let harness = TestHarness::with_provider(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let saw = harness.events().all().iter().any(|e| {
            e.agent_name == "tight-budget"
                && matches!(
                    e.kind,
                    EventKind::AgentFinished {
                        outcome: crate::output::Outcome::Failed,
                        ..
                    }
                )
        });
        assert!(
            saw,
            "max_input_tokens override must propagate to the spawned child"
        );
    }

    #[tokio::test]
    async fn agent_tool_propagates_max_output_tokens() {
        use crate::event::EventKind;
        use crate::provider::TokenUsage;

        let sub = Agent::new()
            .name("tight-budget")
            .model("mock")
            .role("I do work.")
            .tool(MockTool::new("t", false, "ok"));

        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .role("")
            .staff(sub);

        let mut child_turn = tool_response("t", "c1", serde_json::json!({}));
        child_turn.usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 5000,
            ..Default::default()
        };

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "agent_tool",
                "sa1",
                serde_json::json!({
                    "description": "tight",
                    "task": "Do work",
                    "agent": "tight-budget",
                    "max_output_tokens": 4000,
                }),
            ),
            child_turn,
            text_response("done"),
        ]));

        let harness = TestHarness::with_provider(provider);
        harness.run_agent(&agent, "go").await.unwrap();

        let saw = harness.events().all().iter().any(|e| {
            e.agent_name == "tight-budget"
                && matches!(
                    e.kind,
                    EventKind::AgentFinished {
                        outcome: crate::output::Outcome::Failed,
                        ..
                    }
                )
        });
        assert!(
            saw,
            "max_output_tokens override must propagate to the spawned child"
        );
    }

    #[tokio::test]
    async fn agent_tool_unknown_agent_errors() {
        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .role("")
            .tool(AgentTool);

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "agent_tool",
                "sa1",
                serde_json::json!({
                    "description": "use unknown",
                    "task": "Do work",
                    "agent": "nonexistent"
                }),
            ),
            text_response("Could not find agent"),
        ]));

        let harness = TestHarness::with_provider(provider);
        let output = harness
            .run_agent(&agent, "Use nonexistent agent")
            .await
            .unwrap();
        assert_eq!(output.response_raw, "Could not find agent");
    }
}
