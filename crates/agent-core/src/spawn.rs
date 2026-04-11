use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::{Agent, AgentBuilder, AgentOutput, InvocationContext};
use crate::error::{AgenticError, Result};
use crate::message::Usage;
use crate::tool::{Tool, ToolContext, ToolResult};

#[derive(Deserialize)]
struct SpawnAgentInput {
    description: String,
    prompt: String,
    agent: Option<String>,
    model: Option<String>,
    max_turns: Option<u32>,
    background: Option<bool>,
}

/// Tool that spawns sub-agents in foreground or background mode.
pub struct SpawnAgentTool {
    sub_agents: Vec<Arc<dyn Agent>>,
    default_model: String,
}

impl SpawnAgentTool {
    pub fn new() -> Self {
        Self {
            sub_agents: Vec::new(),
            default_model: "claude-sonnet-4-20250514".into(),
        }
    }

    pub fn with_sub_agents(mut self, agents: Vec<Arc<dyn Agent>>) -> Self {
        self.sub_agents = agents;
        self
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    fn find_agent(&self, name: &str) -> Result<Arc<dyn Agent>> {
        self.sub_agents
            .iter()
            .find(|a| a.name() == name)
            .cloned()
            .ok_or_else(|| AgenticError::Tool {
                tool_name: "spawn_agent".into(),
                message: format!("No sub-agent named '{name}'"),
            })
    }

    async fn execute(&self, input: SpawnAgentInput, ctx: InvocationContext) -> Result<AgentOutput> {
        let agent: Arc<dyn Agent> = if let Some(ref name) = input.agent {
            self.find_agent(name)?
        } else {
            AgentBuilder::new()
                .name(&input.description)
                .model(input.model.as_deref().unwrap_or(&self.default_model))
                .system_prompt(&input.prompt)
                .max_turns(input.max_turns.unwrap_or(10))
                .build()?
        };

        let child_ctx = ctx.child(&input.description).with_input(&input.prompt);

        if input.background.unwrap_or(false) {
            let agent_id = child_ctx.agent_id.clone();
            let agent_id_for_msg = agent_id.clone();
            let queue = ctx.command_queue.clone();
            let description = input.description.clone();

            tokio::spawn(async move {
                let result = agent.run(child_ctx).await;
                if let Some(q) = queue {
                    match result {
                        Ok(output) => q.enqueue_notification(&agent_id, &output.content),
                        Err(e) => q.enqueue_notification(&agent_id, &format!("Failed: {e}")),
                    }
                }
            });

            Ok(AgentOutput {
                content: format!(
                    "Background agent '{}' started (id: {agent_id_for_msg})",
                    description
                ),
                ..AgentOutput::empty(Usage::default())
            })
        } else {
            agent.run(child_ctx).await
        }
    }
}

impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Spawn a sub-agent to handle a task. Can run in foreground (blocking) or background mode."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of what the agent should do"
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt/instructions for the agent"
                },
                "agent": {
                    "type": "string",
                    "description": "Name of a registered sub-agent to use (optional)"
                },
                "model": {
                    "type": "string",
                    "description": "Model to use for ad-hoc agents (optional)"
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum turns for the agent (default: 10)"
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background (default: false). Returns immediately with agent ID."
                }
            },
            "required": ["description", "prompt"]
        })
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let spawn_input: SpawnAgentInput =
                serde_json::from_value(input).map_err(|e| AgenticError::Tool {
                    tool_name: "spawn_agent".into(),
                    message: format!("Invalid input: {e}"),
                })?;

            let invocation_ctx = ctx
                .get_extension::<InvocationContext>()
                .ok_or_else(|| AgenticError::Tool {
                    tool_name: "spawn_agent".into(),
                    message: "InvocationContext not available in ToolContext".into(),
                })?
                .clone();

            match self.execute(spawn_input, invocation_ctx).await {
                Ok(output) => Ok(ToolResult {
                    content: output.content,
                    is_error: false,
                }),
                Err(e) => Ok(ToolResult {
                    content: format!("Agent error: {e}"),
                    is_error: true,
                }),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CommandQueue;
    use crate::testutil::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn spawn_agent_foreground() {
        let spawn_tool = SpawnAgentTool::new().with_default_model("mock");

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .system_prompt("Coordinate work.")
            .tool(spawn_tool)
            .build()
            .unwrap();

        // Provider serves: parent spawn call, child response, parent final
        let harness = TestHarness::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "researcher",
                    "prompt": "Research topic X"
                }),
            ),
            text_response("research findings"),
            text_response("Summary: research findings"),
        ]));

        let output = harness.run_agent(agent.as_ref(), "Do research").await.unwrap();
        assert_eq!(output.content, "Summary: research findings");
    }

    #[tokio::test]
    async fn spawn_agent_background_delivers_notification() {
        let spawn_tool = SpawnAgentTool::new().with_default_model("mock");

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .system_prompt("")
            .tool(spawn_tool)
            .build()
            .unwrap();

        let queue = Arc::new(CommandQueue::new());

        // Shared provider: parent and child both consume from this queue.
        // Parent turn 1: spawn background agent → tool call
        // Parent turn 2: final text (after tool result)
        // Child turn: text response (runs concurrently)
        // Order of consumption depends on scheduling, so provide enough for both.
        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "bg-worker",
                    "prompt": "Do work",
                    "background": true
                }),
            ),
            // These two will be consumed by parent and child in arbitrary order
            text_response("response-a"),
            text_response("response-b"),
        ]));

        let harness = TestHarness::new(MockProvider::new(vec![]));
        let mut ctx = harness.build_context("Start background work");
        ctx.provider = provider;
        ctx.command_queue = Some(queue.clone());

        let output = agent.run(ctx).await.unwrap();
        // Parent got one of the text responses
        assert!(!output.content.is_empty());

        // Wait for background task
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that a notification was delivered via the queue
        let cmd = queue.dequeue(None);
        assert!(cmd.is_some(), "Expected notification from background agent");
        let notification = cmd.unwrap().content;
        // Notification contains either the child's response or an error (if it got the other response)
        assert!(notification.contains("response-") || notification.contains("Failed"),
            "Notification should contain agent result: {notification}");
    }

    #[tokio::test]
    async fn spawn_agent_named_sub_agent() {
        let sub = AgentBuilder::new()
            .name("specialist")
            .model("mock")
            .system_prompt("I am a specialist.")
            .build()
            .unwrap();

        let spawn_tool = SpawnAgentTool::new()
            .with_sub_agents(vec![sub])
            .with_default_model("mock");

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .system_prompt("")
            .tool(spawn_tool)
            .build()
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "use specialist",
                    "prompt": "Do specialized work",
                    "agent": "specialist"
                }),
            ),
            // Specialist agent response
            text_response("specialized result"),
            // Orchestrator final
            text_response("Got specialized result"),
        ]));

        let harness = TestHarness::new(MockProvider::new(vec![]));
        let mut ctx = harness.build_context("Use the specialist");
        ctx.provider = provider;

        let output = agent.run(ctx).await.unwrap();
        assert_eq!(output.content, "Got specialized result");
    }

    #[tokio::test]
    async fn spawn_agent_unknown_agent_errors() {
        let spawn_tool = SpawnAgentTool::new();

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .system_prompt("")
            .tool(spawn_tool)
            .build()
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![
            tool_response(
                "spawn_agent",
                "sa1",
                serde_json::json!({
                    "description": "use unknown",
                    "prompt": "Do work",
                    "agent": "nonexistent"
                }),
            ),
            // After error tool result, agent gives final text
            text_response("Could not find agent"),
        ]));

        let harness = TestHarness::new(MockProvider::new(vec![]));
        let mut ctx = harness.build_context("Use nonexistent agent");
        ctx.provider = provider;

        let output = agent.run(ctx).await.unwrap();
        assert_eq!(output.content, "Could not find agent");
    }
}
