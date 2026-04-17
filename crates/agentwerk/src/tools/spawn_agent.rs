use std::future::Future;
use std::pin::Pin;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::Agent;
use crate::error::{AgenticError, Result};
use crate::util::generate_agent_name;

use crate::tools::tool::{Tool, ToolContext, ToolResult};

/// Default identity for ad-hoc sub-agents (when the LLM doesn't supply one).
const DEFAULT_IDENTITY: &str = "You are a focused helper agent. Answer concisely.";

/// LLM-facing tool that spawns a sub-agent. Carries no state — every per-call
/// detail (caller's `Runtime`, `AgentSpec`) flows in via `ToolContext`.
pub struct SpawnAgentTool;

/// Tool-control fields. Per-agent config overrides (identity, model, max_*, …)
/// live in the same JSON object and are applied via `Agent::apply_overrides`.
#[derive(Deserialize)]
struct SpawnArgs {
    description: String,
    instruction: String,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    background: Option<bool>,
}

const DESCRIPTION: &str = "\
Spawn a sub-agent to handle a task. Can run in foreground (blocking) or background mode.

# Writing the prompt
Brief the agent like a smart colleague who just walked into the room — it hasn't seen \
this conversation, doesn't know what you've tried, doesn't understand why this matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context that the agent can make judgment calls.

IMPORTANT: Never delegate understanding. Don't write \"based on your findings, do the task.\" \
Write prompts that prove you understood the problem and what specifically needs to happen.

# When NOT to use
- To read a specific file — use read_file instead.
- To search for a pattern — use grep instead.
- For any task a single tool call can accomplish.

# Foreground vs background
- Foreground (default): blocks until the agent completes. Use when you need results before proceeding.
- Background: returns immediately with an agent ID. Use when you have independent work to do in parallel.";

impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short label for the spawned agent — shown in events."
                },
                "instruction": {
                    "type": "string",
                    "description": "The user-level task for the agent to perform."
                },
                "agent": {
                    "type": "string",
                    "description": "Name of a registered sub-agent to use. Omit for an ad-hoc agent."
                },
                "identity": {
                    "type": "string",
                    "description": "System prompt for ad-hoc agents (ignored when 'agent' is set). Defaults to a generic helper identity."
                },
                "model": {
                    "type": "string",
                    "description": "Override the model used for this spawn."
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Cap the agentic loop for this spawn. Ad-hoc agents default to 10."
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Cap output tokens per LLM request for this spawn."
                },
                "max_schema_retries": {
                    "type": "integer",
                    "description": "Override structured-output retry count for this spawn."
                },
                "max_request_retries": {
                    "type": "integer",
                    "description": "Override transient-API retry count for this spawn."
                },
                "request_retry_backoff_ms": {
                    "type": "integer",
                    "description": "Override base backoff (ms) for request retries."
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background (default: false). Returns immediately with an agent id; posts completion to the command queue."
                }
            },
            "required": ["description", "instruction"]
        })
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let args: SpawnArgs =
                serde_json::from_value(input.clone()).map_err(|e| AgenticError::Tool {
                    tool_name: "spawn_agent".into(),
                    message: format!("Invalid input: {e}"),
                })?;

            let runtime = ctx
                .runtime
                .as_ref()
                .ok_or_else(|| AgenticError::Tool {
                    tool_name: "spawn_agent".into(),
                    message: "Runtime not available in ToolContext".into(),
                })?
                .clone();
            let caller = ctx
                .caller_spec
                .as_ref()
                .ok_or_else(|| AgenticError::Tool {
                    tool_name: "spawn_agent".into(),
                    message: "caller AgentSpec not available in ToolContext".into(),
                })?
                .clone();

            // Resolve the base Agent: either a registered sub-agent, or a fresh
            // ad-hoc one seeded with the default identity and max_turns=10. The
            // overrides step below applies every LLM-supplied tuning knob to
            // this base, regardless of path.
            let base = match &args.agent {
                Some(name) => caller
                    .sub_agents
                    .iter()
                    .find(|a| a.name_ref() == Some(name.as_str()))
                    .cloned()
                    .ok_or_else(|| AgenticError::Tool {
                        tool_name: "spawn_agent".into(),
                        message: format!("No sub-agent named '{name}'"),
                    })?,
                None => Agent::new()
                    .name(&args.description)
                    .identity_prompt(DEFAULT_IDENTITY)
                    .max_turns(10),
            };

            let agent = base
                .apply_overrides(&input)
                .instruction_prompt(&args.instruction);

            let description = Some(args.description.clone());

            if args.background.unwrap_or(false) {
                let id = generate_agent_name(&args.description);
                let queue = runtime.command_queue.clone();
                let agent_id = id.clone();
                let parent_model = caller.model.clone();
                let description_for_child = description.clone();
                tokio::spawn(async move {
                    let summary = match agent
                        .run_child(&runtime, &parent_model, description_for_child)
                        .await
                    {
                        Ok(o) => o.response_raw,
                        Err(e) => format!("Failed: {e}"),
                    };
                    if let Some(q) = queue {
                        q.enqueue_notification(&agent_id, &summary);
                    }
                });
                Ok(ToolResult::success(format!(
                    "Background agent '{}' started (id: {id})",
                    args.description
                )))
            } else {
                match agent.run_child(&runtime, &caller.model, description).await {
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
    use crate::agent::queue::CommandQueue;
    use crate::testutil::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn spawn_agent_foreground() {
        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("Coordinate work.")
            .tool(SpawnAgentTool);

        let harness = TestHarness::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "researcher",
                    "instruction": "Research topic X"
                }),
            ),
            text_response("research findings"),
            text_response("Summary: research findings"),
        ]));

        let output = harness.run_agent(&agent, "Do research").await.unwrap();
        assert_eq!(output.response_raw, "Summary: research findings");
    }

    #[tokio::test]
    async fn spawn_agent_background_delivers_notification() {
        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("")
            .tool(SpawnAgentTool);

        let queue = Arc::new(CommandQueue::new());

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "bg-worker",
                    "instruction": "Do work",
                    "background": true
                }),
            ),
            text_response("response-a"),
            text_response("response-b"),
        ]));

        let harness = TestHarness::with_provider_and_queue(provider.clone(), queue.clone());
        let output = harness.run_agent(&agent, "Start background work").await.unwrap();
        assert!(!output.response_raw.is_empty());

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let cmd = queue.dequeue_if(None, |_| true);
        assert!(cmd.is_some(), "Expected notification from background agent");
        let notification = cmd.unwrap().content;
        assert!(
            notification.contains("response-") || notification.contains("Failed"),
            "Notification should contain agent result: {notification}"
        );
    }

    #[tokio::test]
    async fn spawn_agent_named_sub_agent() {
        let sub = Agent::new()
            .name("specialist")
            .model("mock")
            .identity_prompt("I am a specialist.");

        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("")
            .sub_agents([sub]);

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "use specialist",
                    "instruction": "Do specialized work",
                    "agent": "specialist"
                }),
            ),
            text_response("specialized result"),
            text_response("Got specialized result"),
        ]));

        let harness = TestHarness::with_provider(provider);
        let output = harness.run_agent(&agent, "Use the specialist").await.unwrap();
        assert_eq!(output.response_raw, "Got specialized result");
    }

    #[tokio::test]
    async fn spawn_agent_unknown_agent_errors() {
        let agent = Agent::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("")
            .tool(SpawnAgentTool);

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "use unknown",
                    "instruction": "Do work",
                    "agent": "nonexistent"
                }),
            ),
            text_response("Could not find agent"),
        ]));

        let harness = TestHarness::with_provider(provider);
        let output = harness.run_agent(&agent, "Use nonexistent agent").await.unwrap();
        assert_eq!(output.response_raw, "Could not find agent");
    }
}
