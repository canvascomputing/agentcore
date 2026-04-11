use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent_core::{
    AgenticError, AgentBuilder, AnthropicProvider, CostTracker, Event, HttpTransport,
    InvocationContext, ToolBuilder, ToolResult, generate_agent_id,
};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("Set ANTHROPIC_API_KEY environment variable");

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

    // Build a simple echo tool
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
        .model("claude-sonnet-4-20250514")
        .system_prompt("You are a helpful assistant. You have an echo tool available. Be concise.")
        .max_turns(5)
        .tool(echo_tool)
        .build()?;

    let cost_tracker = CostTracker::new();

    let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match event {
        Event::Text { text, .. } => print!("{text}"),
        Event::ToolStart { tool, .. } => eprintln!("\n[tool] {tool}"),
        Event::ToolEnd { result, is_error, .. } => {
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
        input: "Use the echo tool to echo 'Hello from agent-core!' and then say goodbye.".into(),
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
