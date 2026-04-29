//! Peer-to-peer agent messaging. Routes a message through the shared `Work` so a running sibling agent picks it up at the next step boundary.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::work::{Task, TaskSource, WorkPriority};
use crate::error::Result;
use crate::tools::error::ToolError;
use crate::tools::tool::{ToolContext, ToolLike, ToolResult};
use crate::tools::tool_file::ToolFile;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("send_message.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

/// Deliver a message to a peer agent in the same run-tree. Routes through
/// the shared work inbox and is injected into the recipient's next step.
/// If no agent with the given name is running, the message sits in the inbox
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
            let work = runtime
                .incoming_work
                .as_ref()
                .ok_or_else(|| tool_err("Work inbox not available on LoopRuntime"))?;

            if args.to == caller.name {
                return Ok(ToolResult::error("Cannot send a message to yourself"));
            }

            work.add(Task {
                content: args.message,
                priority: WorkPriority::Next,
                source: TaskSource::PeerMessage {
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
    ToolError::ExecutionFailed {
        tool_name: tool_file().name.clone(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::work::Work;
    use crate::agent::{Agent, AgentSpec};
    use crate::testutil::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn harness_ctx() -> (ToolContext, Arc<Work>, Arc<AgentSpec>) {
        let work = Arc::new(Work::new());
        let caller = Agent::new()
            .name("alice")
            .model("mock")
            .role("")
            .provider(Arc::new(MockProvider::text("unused")))
            .incoming_work(work.clone());
        let (spec, runtime) = caller.compile(None);
        let ctx = ToolContext::new(PathBuf::from("."))
            .runtime(Arc::new(runtime))
            .caller_spec(spec.clone());
        (ctx, work, spec)
    }

    #[tokio::test]
    async fn send_adds_targeted_work() {
        let tool = SendMessageTool;
        let (ctx, work, _) = harness_ctx();

        let input = serde_json::json!({
            "to": "bob",
            "message": "hi",
            "summary": "greeting"
        });
        let out = tool.call(input, &ctx).await.unwrap();
        assert!(matches!(out, ToolResult::Success(_)));

        let task = work.take_if(Some("bob"), |_| true).expect("posted for bob");
        assert_eq!(task.agent_name.as_deref(), Some("bob"));
        assert_eq!(task.content, "hi");
        match task.source {
            TaskSource::PeerMessage { from, summary } => {
                assert_eq!(from, "alice");
                assert_eq!(summary.as_deref(), Some("greeting"));
            }
            _ => panic!("expected PeerMessage"),
        }
    }

    #[tokio::test]
    async fn send_to_self_errors() {
        let tool = SendMessageTool;
        let (ctx, _work, _) = harness_ctx();

        let input = serde_json::json!({ "to": "alice", "message": "hi" });
        let out = tool.call(input, &ctx).await.unwrap();
        assert!(matches!(out, ToolResult::Error(_)));
    }

    #[tokio::test]
    async fn sender_is_derived_not_passed() {
        let tool = SendMessageTool;
        let (ctx, work, _) = harness_ctx();

        let input = serde_json::json!({
            "to": "bob",
            "message": "hi",
            "from": "eve"
        });
        let _ = tool.call(input, &ctx).await.unwrap();

        let task = work.take_if(Some("bob"), |_| true).unwrap();
        match task.source {
            TaskSource::PeerMessage { from, .. } => assert_eq!(from, "alice"),
            _ => panic!("expected PeerMessage"),
        }
    }
}
