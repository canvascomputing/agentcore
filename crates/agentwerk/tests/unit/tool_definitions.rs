//! How tool definitions are included in every provider request.
//!
//! Each tool registered via `.tool()` is sent to the provider on every step.
//! The snapshots below show the exact shape of that request, with tools
//! between the system prompt and the message list.

use agentwerk::provider::{ContentBlock, Message, ModelRequest};
use agentwerk::testutil::{text_response, MockProvider, MockTool, TestHarness};
use agentwerk::Agent;

const ONE_TOOL: &str = "\
=== system ===
You are a helper.

=== tools[0] lookup ===
description: A mock tool for testing
schema: {\"properties\":{},\"type\":\"object\"}

=== messages[0] user ===
Find the answer.
";

const TWO_TOOLS: &str = "\
=== system ===
You are a helper.

=== tools[0] search ===
description: A mock tool for testing
schema: {\"properties\":{},\"type\":\"object\"}

=== tools[1] compute ===
description: A mock tool for testing
schema: {\"properties\":{},\"type\":\"object\"}

=== messages[0] user ===
Do both.
";

#[tokio::test]
async fn single_tool_appears_in_request() {
    let provider = MockProvider::new(vec![text_response("done")]);
    let agent = Agent::new()
        .model("mock")
        .role("You are a helper.")
        .behavior("")
        .context("")
        .tool(MockTool::new("lookup", true, "42"));

    let harness = TestHarness::new(provider);
    harness.run_agent(&agent, "Find the answer.").await.unwrap();
    let reqs = harness.provider().requests.lock().unwrap();

    assert_eq!(render(&reqs[0]), ONE_TOOL);
}

#[tokio::test]
async fn all_registered_tools_appear_in_every_request() {
    let provider = MockProvider::new(vec![text_response("done")]);
    let agent = Agent::new()
        .model("mock")
        .role("You are a helper.")
        .behavior("")
        .context("")
        .tool(MockTool::new("search", true, "results"))
        .tool(MockTool::new("compute", false, "42"));

    let harness = TestHarness::new(provider);
    harness.run_agent(&agent, "Do both.").await.unwrap();
    let reqs = harness.provider().requests.lock().unwrap();

    assert_eq!(render(&reqs[0]), TWO_TOOLS);
}

fn render(req: &ModelRequest) -> String {
    let mut out = String::new();
    out.push_str("=== system ===\n");
    out.push_str(&req.system_prompt);
    out.push('\n');
    for (i, def) in req.tools.iter().enumerate() {
        out.push_str(&format!(
            "\n=== tools[{i}] {} ===\ndescription: {}\nschema: {}\n",
            def.name, def.description, def.input_schema
        ));
    }
    for (i, msg) in req.messages.iter().enumerate() {
        let (role, body) = match msg {
            Message::System { content } => ("system", content.clone()),
            Message::User { content } => ("user", render_blocks(content)),
            Message::Assistant { content } => ("assistant", render_blocks(content)),
        };
        out.push_str(&format!("\n=== messages[{i}] {role} ===\n{body}\n"));
    }
    out
}

fn render_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::ToolUse { id, name, input } => {
                format!("[tool_use {id}] {name}({input})")
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let tag = if *is_error { "ERR" } else { "ok" };
                format!("[tool_result {tool_use_id} {tag}] {content}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
