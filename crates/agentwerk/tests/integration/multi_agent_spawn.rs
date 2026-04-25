//! End-to-end: an agent spawns sub-agents through `AgentTool` against a live LLM. Guards that sub-agent invocation wires up correctly through the real provider path.

use super::common;

use std::sync::Arc;

use agentwerk::event::EventKind;
use agentwerk::{Agent, Event};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let event_handler = Arc::new(|event: Event| match &event.kind {
        EventKind::TextChunkReceived { content } => {
            if event.agent_name == "orchestrator" {
                print!("{content}")
            }
        }
        EventKind::ToolCallStarted { tool_name, .. } => {
            eprintln!("\n[{}] tool: {tool_name}", event.agent_name)
        }
        EventKind::AgentStarted { .. } => eprintln!("[{}] started", event.agent_name),
        EventKind::AgentFinished { turns, .. } => {
            eprintln!("[{}] done ({turns} turns)", event.agent_name)
        }
        _ => {}
    });

    let researcher = Agent::new()
        .name("researcher")
        .model(&model)
        .role("You are a research assistant. Answer the given question concisely in 1-2 sentences.")
        .max_turns(1);

    let output = Agent::new()
        .provider(provider)
        .model(&model)
        .name("orchestrator")
        .role(
            "You coordinate research tasks. Use agent with agent: \"researcher\" to delegate questions. \
             Summarize the results. Be concise.",
        )
        .hire(researcher)
        .max_turns(10)
        .event_handler(event_handler)
        .task("What is the capital of France? Use the researcher agent to find out, then tell me.")
        .await?;

    common::print_result(&output);

    assert!(output.statistics.tool_calls >= 1);

    Ok(())
}
