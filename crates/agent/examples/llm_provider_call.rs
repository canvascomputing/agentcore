use std::sync::Arc;

use agent::{
    AgenticError, AnthropicProvider, CompletionRequest, ContentBlock, CostTracker, HttpTransport,
    LiteLlmProvider, LlmProvider, Message,
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

    let request = CompletionRequest {
        model: model.clone(),
        system_prompt: "You are a helpful assistant. Be concise.".into(),
        messages: vec![Message::User {
            content: vec![ContentBlock::Text {
                text: "Say hello in one sentence.".into(),
            }],
        }],
        tools: vec![],
        max_tokens: 256,
        tool_choice: None,
    };

    println!("Sending request...");
    let response = provider.complete(request).await?;

    for block in &response.content {
        if let ContentBlock::Text { text } = block {
            println!("Response: {text}");
        }
    }
    println!("Model: {}", response.model);
    println!(
        "Usage: {} input, {} output tokens",
        response.usage.input_tokens, response.usage.output_tokens
    );

    let tracker = CostTracker::new();
    tracker.record_usage(&response.model, &response.usage);
    println!("\n{}", tracker.summary());

    Ok(())
}
