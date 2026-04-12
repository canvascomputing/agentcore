use std::sync::Arc;

use agent::{
    AgentBuilder, AgenticError, AnthropicProvider, HttpTransport, InvocationContext,
    LiteLlmProvider, LlmProvider, MistralProvider, ReadFileTool, GrepTool, ListDirectoryTool,
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

fn build_provider() -> Option<(Arc<dyn LlmProvider>, String)> {
    let transport = build_transport();
    if let Ok(url) = std::env::var("LITELLM_API_URL") {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, transport).base_url(url)), model));
    }
    if let Some(key) = std::env::var("MISTRAL_API_KEY").ok().filter(|k| !k.is_empty()) {
        let model = std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-medium-2508".into());
        return Some((Arc::new(MistralProvider::new(key, transport)), model));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(AnthropicProvider::new(key, transport)), model));
    }
    if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, transport).base_url("http://localhost:4000".into())), model));
    }
    None
}

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some((provider, model)) = build_provider() else { eprintln!("SKIPPED: no provider"); return Ok(()); };

    let agent = AgentBuilder::new()
        .name("reviewer")
        .model(&model)
        .system_prompt(
            "List the top-level directory, read one file, and summarize what this project is. Be concise.",
        )
        .tool(ReadFileTool)
        .tool(GrepTool)
        .tool(ListDirectoryTool)
        .max_turns(10)
        .build()?;

    let mut ctx = InvocationContext::new(provider);
    ctx.prompt = "What is this project?".into();

    let output = agent.run(ctx).await?;
    assert!(!output.response_raw.is_empty());
    assert!(output.statistics.tool_calls >= 1);

    Ok(())
}
