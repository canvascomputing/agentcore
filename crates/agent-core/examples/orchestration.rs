use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent_core::{
    AgenticError, AgentBuilder, AnthropicProvider, CommandQueue, CostTracker, Event, HttpTransport,
    InvocationContext, SpawnAgentTool, generate_agent_id,
};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("Set ANTHROPIC_API_KEY environment variable");

    let transport: HttpTransport = Box::new(|url, headers, body| {
        let url = url.to_string();
        let headers: Vec<(String, String)> = headers
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Box::pin(async move {
            let client = reqwest::Client::new();
            let mut req = client.post(&url).json(&body);
            for (key, value) in &headers {
                req = req.header(key.as_str(), value.as_str());
            }
            let resp = req
                .send()
                .await
                .map_err(|e| AgenticError::Other(e.to_string()))?;
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| AgenticError::Other(e.to_string()))?;
            Ok(json)
        })
    });

    let provider = Arc::new(AnthropicProvider::new(api_key, transport));

    // Build a specialist sub-agent
    let researcher = AgentBuilder::new()
        .name("researcher")
        .model("claude-haiku-4-5-20251001")
        .system_prompt(
            "You are a research assistant. Answer the given question concisely in 1-2 sentences.",
        )
        .max_turns(1)
        .build()?;

    // Build orchestrator with spawn_agent tool
    let spawn_tool = SpawnAgentTool::new()
        .with_sub_agents(vec![researcher])
        .with_default_model("claude-haiku-4-5-20251001");

    let orchestrator = AgentBuilder::new()
        .name("orchestrator")
        .model("claude-sonnet-4-20250514")
        .system_prompt(
            "You coordinate research tasks. Use spawn_agent with agent: \"researcher\" to delegate questions. \
             Summarize the results. Be concise.",
        )
        .max_turns(5)
        .tool(spawn_tool)
        .build()?;

    let cost_tracker = CostTracker::new();
    let queue = Arc::new(CommandQueue::new());

    let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match event {
        Event::Text { text, agent } => {
            if agent == "orchestrator" {
                print!("{text}");
            }
        }
        Event::ToolStart { tool, agent, .. } => eprintln!("\n[{agent}] tool: {tool}"),
        Event::AgentStart { agent } => eprintln!("[{agent}] started"),
        Event::AgentEnd { agent, turns } => eprintln!("[{agent}] done ({turns} turns)"),
        _ => {}
    });

    let ctx = InvocationContext {
        input: "What is the capital of France? Use the researcher agent to find out, then tell me."
            .into(),
        state: HashMap::new(),
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        provider,
        cost_tracker: cost_tracker.clone(),
        on_event,
        cancelled: Arc::new(AtomicBool::new(false)),
        session_store: None,
        command_queue: Some(queue),
        agent_id: generate_agent_id("orchestrator"),
    };

    let output = orchestrator.run(ctx).await?;

    println!("\n\n--- Output ---");
    println!("{}", output.content);
    println!("\n--- Cost ---");
    println!("{}", cost_tracker.summary());

    Ok(())
}
