use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::{Agent, AgentBuilder, AgentOutput, RuntimeContext};
use crate::error::{AgenticError, Result};
use crate::provider::model::ModelSpec;

use crate::tools::tool::{Tool, ToolContext, ToolResult};

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
    default_model: ModelSpec,
}

impl SpawnAgentTool {
    pub fn new() -> Self {
        Self {
            sub_agents: Vec::new(),
            default_model: ModelSpec::Inherit,
        }
    }

    pub fn sub_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.sub_agents.push(agent);
        self
    }

    /// Set the default model for ad-hoc sub-agents.
    /// Accepts an exact model ID string.
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = ModelSpec::Exact(model.into());
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

    async fn execute(&self, input: SpawnAgentInput, ctx: RuntimeContext) -> Result<AgentOutput> {
        let agent: Arc<dyn Agent> = if let Some(ref name) = input.agent {
            self.find_agent(name)?
        } else {
            let model = input.model.as_deref().unwrap_or(
                match &self.default_model {
                    ModelSpec::Exact(id) => id.as_str(),
                    ModelSpec::Inherit => "inherit",
                },
            );

            let mut builder = AgentBuilder::new()
                .name(&input.description)
                .identity_prompt(&input.prompt)
                .max_turns(input.max_turns.unwrap_or(10));
            if model != "inherit" {
                builder = builder.model(model);
            }
            builder.build()?
        };

        let child_ctx = ctx.child(&input.description).instruction_prompt(&input.prompt);

        if input.background.unwrap_or(false) {
            let agent_id = child_ctx.agent_name.clone();
            let agent_id_for_msg = agent_id.clone();
            let queue = ctx.command_queue.clone();
            let description = input.description.clone();

            tokio::spawn(async move {
                let result = agent.run(child_ctx).await;
                if let Some(q) = queue {
                    match result {
                        Ok(output) => q.enqueue_notification(&agent_id, &output.response_raw),
                        Err(e) => q.enqueue_notification(&agent_id, &format!("Failed: {e}")),
                    }
                }
            });

            Ok(AgentOutput {
                response_raw: format!(
                    "Background agent '{}' started (id: {agent_id_for_msg})",
                    description
                ),
                ..AgentOutput::empty()
            })
        } else {
            agent.run(child_ctx).await
        }
    }
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
                .runtime_context
                .as_ref()
                .ok_or_else(|| AgenticError::Tool {
                    tool_name: "spawn_agent".into(),
                    message: "RuntimeContext not available in ToolContext".into(),
                })?
                .clone();

            match self.execute(spawn_input, invocation_ctx).await {
                Ok(output) => Ok(ToolResult::success(output.response_raw)),
                Err(e) => Ok(ToolResult::error(format!("Agent error: {e}"))),
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
        let spawn_tool = SpawnAgentTool::new().default_model("mock");

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("Coordinate work.")
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
        assert_eq!(output.response_raw, "Summary: research findings");
    }

    #[tokio::test]
    async fn spawn_agent_background_delivers_notification() {
        let spawn_tool = SpawnAgentTool::new().default_model("mock");

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("")
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
        assert!(!output.response_raw.is_empty());

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
            .identity_prompt("I am a specialist.")
            .build()
            .unwrap();

        let spawn_tool = SpawnAgentTool::new()
            .sub_agent(sub)
            .default_model("mock");

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("")
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
        assert_eq!(output.response_raw, "Got specialized result");
    }

    #[tokio::test]
    async fn spawn_agent_unknown_agent_errors() {
        let spawn_tool = SpawnAgentTool::new();

        let agent = AgentBuilder::new()
            .name("orchestrator")
            .model("mock")
            .identity_prompt("")
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
        assert_eq!(output.response_raw, "Could not find agent");
    }
}
