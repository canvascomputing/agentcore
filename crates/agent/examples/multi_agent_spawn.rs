use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent::{
    AgentBuilder, AgenticError, AnthropicProvider, CommandQueue, Event, HttpTransport,
    InvocationContext, LiteLlmProvider, LlmProvider, SpawnAgentTool, generate_agent_name,
};

fn build_transport() -> HttpTransport {
    Box::new(|url, headers, body| {
        let url = url.to_string();
        let headers: Vec<(String, String)> = headers
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Box::pin(async move {
            let client = reqwest::Client::new();
            let mut req = client.post(&url).json(&body);
            for (k, v) in &headers {
                req = req.header(k.as_str(), v.as_str());
            }
            let resp = req.send().await.map_err(|e| AgenticError::Other(e.to_string()))?;
            resp.json().await.map_err(|e| AgenticError::Other(e.to_string()))
        })
    })
}

fn build_provider() -> (Arc<dyn LlmProvider>, String) {
    let transport = build_transport();
    if let Ok(url) = std::env::var("LITELLM_API_URL") {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return (Arc::new(LiteLlmProvider::new(key, transport).base_url(url)), model);
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let mut p = AnthropicProvider::new(key, transport);
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            p = p.base_url(url);
        }
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return (Arc::new(p), model);
    }
    if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return (Arc::new(LiteLlmProvider::new(key, transport).base_url("http://localhost:4000".into())), model);
    }
    let supported = ["ANTHROPIC_API_KEY", "LITELLM_API_URL"];
    eprintln!("Error: Set {}", supported.join(" or "));
    std::process::exit(1);
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = build_provider();

    let researcher = AgentBuilder::new()
        .name("researcher")
        .model(&model)
        .system_prompt(
            "You are a research assistant. Answer the given question concisely in 1-2 sentences.",
        )
        .max_turns(1)
        .build()?;

    let spawn_tool = SpawnAgentTool::new()
        .with_sub_agents(vec![researcher])
        .with_default_model(&model);

    let orchestrator = AgentBuilder::new()
        .name("orchestrator")
        .model(&model)
        .system_prompt(
            "You coordinate research tasks. Use spawn_agent with agent: \"researcher\" to delegate questions. \
             Summarize the results. Be concise.",
        )
        .max_turns(5)
        .tool(spawn_tool)
        .build()?;

    let queue = Arc::new(CommandQueue::new());

    let event_handler: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match event {
        Event::TextChunk { content: text, agent_name } => {
            if agent_name == "orchestrator" {
                print!("{text}");
            }
        }
        Event::ToolCallStart { tool_name: tool, agent_name, .. } => eprintln!("\n[{agent_name}] tool: {tool}"),
        Event::AgentStart { agent_name: agent } => eprintln!("[{agent}] started"),
        Event::AgentEnd { agent_name: agent, turns } => eprintln!("[{agent}] done ({turns} turns)"),
        _ => {}
    });

    let ctx = InvocationContext {
        prompt: "What is the capital of France? Use the researcher agent to find out, then tell me."
            .into(),
        template_variables: HashMap::new(),
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        provider,
        event_handler,
        cancel_signal: Arc::new(AtomicBool::new(false)),
        session_store: None,
        command_queue: Some(queue),
        agent_name: generate_agent_name("orchestrator"),
    };

    let output = orchestrator.run(ctx).await?;

    println!("\n\n--- Output ---");
    println!("{}", output.response_raw);
    println!("\n--- Cost ---");
    println!("Cost: ${:.4}", output.statistics.costs);

    Ok(())
}
