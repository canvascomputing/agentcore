use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent::{
    AgentBuilder, AgenticError, AnthropicProvider, CostTracker, Event, HttpTransport,
    InvocationContext, LiteLlmProvider, LlmProvider, ToolBuilder, ToolResult, generate_agent_id,
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

    let echo_tool = ToolBuilder::new("echo", "Echoes the input text back")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"}
            },
            "required": ["text"]
        }))
        .read_only(true)
        .handler(|input, _ctx| {
            Box::pin(async move {
                let text = input["text"].as_str().unwrap_or("").to_string();
                Ok(ToolResult {
                    content: format!("Echo: {text}"),
                    is_error: false,
                })
            })
        })
        .build();

    let agent = AgentBuilder::new()
        .name("assistant")
        .model(&model)
        .system_prompt("You are a helpful assistant. You have an echo tool available. Be concise.")
        .max_turns(5)
        .tool(echo_tool)
        .build()?;

    let cost_tracker = CostTracker::new();

    let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match event {
        Event::TextChunk { content: text, .. } => print!("{text}"),
        Event::ToolCallStart { tool_name: tool, .. } => eprintln!("\n[tool] {tool}"),
        Event::ToolCallEnd { output: result, is_error, .. } => {
            if is_error {
                eprintln!("[error] {result}");
            } else {
                eprintln!("[result] {result}");
            }
        }
        Event::AgentEnd { turns, .. } => eprintln!("\n[done in {turns} turn(s)]"),
        _ => {}
    });

    let ctx = InvocationContext {
        input: "Use the echo tool to echo 'Hello from agent!' and then say goodbye.".into(),
        state: HashMap::new(),
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        provider,
        cost_tracker: cost_tracker.clone(),
        on_event,
        cancelled: Arc::new(AtomicBool::new(false)),
        session_store: None,
        command_queue: None,
        agent_id: generate_agent_id("assistant"),
    };

    let output = agent.run(ctx).await?;

    println!("\n\n--- Output ---");
    println!("{}", output.content);
    println!("\n--- Cost ---");
    println!("{}", cost_tracker.summary());

    Ok(())
}
