use agent_core::{
    AgenticError, AnthropicProvider, CompletionRequest, ContentBlock, CostTracker, LlmProvider,
    Message, HttpTransport,
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

    let provider = AnthropicProvider::new(api_key, transport);

    let request = CompletionRequest {
        model: "claude-sonnet-4-20250514".into(),
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

    println!("Sending request to Anthropic API...");
    let response = provider.complete(request).await?;

    // Print response
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

    // Demonstrate cost tracking
    let tracker = CostTracker::new();
    tracker.record_usage(&response.model, &response.usage);
    println!("\n{}", tracker.summary());

    Ok(())
}
