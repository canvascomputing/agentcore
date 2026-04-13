use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use agent::{
    AgentBuilder, AnthropicProvider, Event, LiteLlmProvider, LlmProvider, ToolBuilder, ToolResult,
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
                Ok(ToolResult { content: format!("Echo: {text}"), is_error: false })
            })
        })
        .build();

    let output = AgentBuilder::new()
        .name("assistant")
        .model(&model)
        .system_prompt("You are a helpful assistant. You have an echo tool available. Be concise.")
        .max_turns(5)
        .tool(echo_tool)
        .provider(provider)
        .prompt("Use the echo tool to echo 'Hello from agent!' and then say goodbye.")
        .event_handler(Arc::new(|event| match event {
            Event::TextChunk { content, .. } => print!("{content}"),
            Event::ToolCallStart { tool_name, .. } => eprintln!("\n[tool] {tool_name}"),
            Event::ToolCallEnd { output, is_error, .. } => {
                if is_error { eprintln!("[error] {output}") } else { eprintln!("[result] {output}") }
            }
            Event::AgentEnd { turns, .. } => eprintln!("\n[done in {turns} turn(s)]"),
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
