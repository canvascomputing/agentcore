//! Peer-to-peer agent messaging. Routes a message through the shared `CommandQueue` so a running sibling agent picks it up at the next turn boundary.

use std::future::Future;
use std::pin::Pin;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::queue::{CommandSource, QueuePriority, QueuedCommand};
use crate::error::Result;
use crate::tools::error::ToolError;
use crate::tools::tool::{ToolContext, ToolLike, ToolResult};

const NAME: &str = "send_message";

const DESCRIPTION: &str = "\
Send a message to another named agent in the same run-tree. The recipient \
picks the message up automatically on its next turn — there is no inbox to poll.

Use this to coordinate with peers you've spawned or that are running alongside \
you. The recipient sees your agent name as the sender; you do not pass it.

# When NOT to use
- To spawn a new agent — use spawn_agent instead.
- To return a result to your caller — just finish your turn normally.";

/// Deliver a message to a peer agent in the same run-tree. Routes through
/// the shared command queue and is injected into the recipient's next turn.
/// If no agent with the given name is running, the message sits in the queue
/// indefinitely; the caller is responsible for using a correct name.
pub struct SendMessageTool;

#[derive(Deserialize)]
struct SendArgs {
    to: String,
    message: String,
    #[serde(default)]
    summary: Option<String>,
}

impl ToolLike for SendMessageTool {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Name of the recipient agent (must be currently running)."
                },
                "message": {
                    "type": "string",
                    "description": "Body of the message."
                },
                "summary": {
                    "type": "string",
                    "description": "Optional 5-10 word preview shown in the recipient's header."
                }
            },
            "required": ["to", "message"]
        })
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let args: SendArgs = match serde_json::from_value(input) {
                Ok(a) => a,
                Err(e) => return Ok(ToolResult::error(format!("Invalid input: {e}"))),
            };

            let runtime = ctx
                .runtime
                .as_ref()
                .ok_or_else(|| tool_err("LoopRuntime not available in ToolContext"))?;
            let caller = ctx
                .caller_spec
                .as_ref()
                .ok_or_else(|| tool_err("caller LoopSpec not available in ToolContext"))?;
            let queue = runtime
                .command_queue
                .as_ref()
                .ok_or_else(|| tool_err("Command queue not available on LoopRuntime"))?;

            if args.to == caller.name {
                return Ok(ToolResult::error("Cannot send a message to yourself"));
            }

            queue.enqueue(QueuedCommand {
                content: args.message,
                priority: QueuePriority::Next,
                source: CommandSource::PeerMessage {
                    from: caller.name.clone(),
                    summary: args.summary,
                },
                agent_name: Some(args.to.clone()),
            });

            Ok(ToolResult::success(format!("delivered to {}", args.to)))
        })
    }
}

fn tool_err(message: impl Into<String>) -> ToolError {
    ToolError::new(NAME, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::queue::CommandQueue;
    use crate::agent::{Agent, AgentSpec};
    use crate::testutil::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn harness_ctx() -> (ToolContext, Arc<CommandQueue>, Arc<AgentSpec>) {
        let queue = Arc::new(CommandQueue::new());
        let caller = Agent::new()
            .name("alice")
            .model_name("mock")
            .identity_prompt("")
            .provider(Arc::new(MockProvider::text("unused")))
            .command_queue(queue.clone());
        let (spec, runtime) = caller.compile(None);
        let ctx = ToolContext::new(PathBuf::from("."))
            .runtime(Arc::new(runtime))
            .caller_spec(spec.clone());
        (ctx, queue, spec)
    }

    #[tokio::test]
    async fn send_enqueues_targeted_command() {
        let tool = SendMessageTool;
        let (ctx, queue, _) = harness_ctx();

        let input = serde_json::json!({
            "to": "bob",
            "message": "hi",
            "summary": "greeting"
        });
        let out = tool.call(input, &ctx).await.unwrap();
        assert!(matches!(out, ToolResult::Success(_)));

        let cmd = queue
            .dequeue_if(Some("bob"), |_| true)
            .expect("queued for bob");
        assert_eq!(cmd.agent_name.as_deref(), Some("bob"));
        assert_eq!(cmd.content, "hi");
        match cmd.source {
            CommandSource::PeerMessage { from, summary } => {
                assert_eq!(from, "alice");
                assert_eq!(summary.as_deref(), Some("greeting"));
            }
            _ => panic!("expected PeerMessage"),
        }
    }

    #[tokio::test]
    async fn send_to_self_errors() {
        let tool = SendMessageTool;
        let (ctx, _queue, _) = harness_ctx();

        let input = serde_json::json!({ "to": "alice", "message": "hi" });
        let out = tool.call(input, &ctx).await.unwrap();
        assert!(matches!(out, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn sender_is_derived_not_passed() {
        let tool = SendMessageTool;
        let (ctx, queue, _) = harness_ctx();

        let input = serde_json::json!({
            "to": "bob",
            "message": "hi",
            "from": "eve"
        });
        let _ = tool.call(input, &ctx).await.unwrap();

        let cmd = queue.dequeue_if(Some("bob"), |_| true).unwrap();
        match cmd.source {
            CommandSource::PeerMessage { from, .. } => assert_eq!(from, "alice"),
            _ => panic!("expected PeerMessage"),
        }
    }
}
