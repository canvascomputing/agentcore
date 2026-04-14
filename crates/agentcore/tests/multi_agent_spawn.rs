mod common;

use std::sync::Arc;

use agentcore::{AgentBuilder, Event, SpawnAgentTool};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let event_handler = Arc::new(|event: Event| match event {
        Event::TextChunk { content, agent_name } => {
            if agent_name == "orchestrator" {
                print!("{content}")
            }
        }
        Event::ToolCallStart { tool_name, agent_name, .. } => {
            eprintln!("\n[{agent_name}] tool: {tool_name}")
        }
        Event::AgentStart { agent_name } => eprintln!("[{agent_name}] started"),
        Event::AgentEnd { agent_name, turns } => eprintln!("[{agent_name}] done ({turns} turns)"),
        _ => {}
    });

    let researcher = AgentBuilder::new()
        .name("researcher")
        .model(&model)
        .identity_prompt("You are a research assistant. Answer the given question concisely in 1-2 sentences.")
        .max_turns(1)
        .build()?;

    let spawn_tool = SpawnAgentTool::new()
        .with_sub_agents(vec![researcher])
        .with_default_model(&model);

    let output = AgentBuilder::new()
        .provider(provider)
        .model(&model)
        .name("orchestrator")
        .identity_prompt(
            "You coordinate research tasks. Use spawn_agent with agent: \"researcher\" to delegate questions. \
             Summarize the results. Be concise.",
        )
        .instruction_prompt("What is the capital of France? Use the researcher agent to find out, then tell me.")
        .tool(spawn_tool)
        .max_turns(10)
        .event_handler(event_handler)
        .run()
        .await?;

    common::print_result(&output);

    assert!(output.statistics.tool_calls >= 1);

    Ok(())
}
