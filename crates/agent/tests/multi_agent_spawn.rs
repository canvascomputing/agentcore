use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use agent::{
    AgentBuilder, AnthropicProvider, CommandQueue, Event, LiteLlmProvider, LlmProvider,
    SpawnAgentTool,
};

fn build_provider() -> Option<(Arc<dyn LlmProvider>, String)> {
    let client = reqwest::Client::new();
    if let Ok(url) = std::env::var("LITELLM_API_URL") {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, client).base_url(url)), model));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let mut p = AnthropicProvider::new(key, client);
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            p = p.base_url(url);
        }
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(p), model));
    }
    if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, client).base_url("http://localhost:4000".into())), model));
    }
    None
}

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some((provider, model)) = build_provider() else { eprintln!("SKIPPED: no provider"); return Ok(()); };

    let researcher = AgentBuilder::new()
        .name("researcher")
        .model(&model)
        .system_prompt("You are a research assistant. Answer the given question concisely in 1-2 sentences.")
        .max_turns(1)
        .build()?;

    let spawn_tool = SpawnAgentTool::new()
        .with_sub_agents(vec![researcher])
        .with_default_model(&model);

    let output = AgentBuilder::new()
        .name("orchestrator")
        .model(&model)
        .system_prompt(
            "You coordinate research tasks. Use spawn_agent with agent: \"researcher\" to delegate questions. \
             Summarize the results. Be concise.",
        )
        .max_turns(5)
        .tool(spawn_tool)
        .provider(provider)
        .prompt("What is the capital of France? Use the researcher agent to find out, then tell me.")
        .command_queue(Arc::new(CommandQueue::new()))
        .event_handler(Arc::new(|event| match event {
            Event::TextChunk { content, agent_name } => {
                if agent_name == "orchestrator" { print!("{content}") }
            }
            Event::ToolCallStart { tool_name, agent_name, .. } => eprintln!("\n[{agent_name}] tool: {tool_name}"),
            Event::AgentStart { agent_name } => eprintln!("[{agent_name}] started"),
            Event::AgentEnd { agent_name, turns } => eprintln!("[{agent_name}] done ({turns} turns)"),
            _ => {}
        }))
        .cancel_signal(Arc::new(AtomicBool::new(false)))
        .run()
        .await?;

    println!("\n\n--- Output ---");
    println!("{}", output.response_raw);
    println!("\n--- Cost ---");
    println!("Cost: ${:.4}", output.statistics.costs);

    Ok(())
}
