#![allow(dead_code)]

use std::sync::Arc;

use agentcore::{
    AgentOutput, AnthropicProvider, LiteLlmProvider, LlmProvider, MistralProvider,
};

pub fn build_provider() -> (Arc<dyn LlmProvider>, String) {
    let client = reqwest::Client::new();

    if let Ok(url) = std::env::var("LITELLM_API_URL") {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return (Arc::new(LiteLlmProvider::new(key, client).base_url(url)), model);
    }

    if let Some(key) = std::env::var("MISTRAL_API_KEY").ok().filter(|k| !k.is_empty()) {
        let model = std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-medium-2508".into());
        return (Arc::new(MistralProvider::new(key, client)), model);
    }

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let mut p = AnthropicProvider::new(key, client);
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            p = p.base_url(url);
        }
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return (Arc::new(p), model);
    }

    if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL")
            .expect("LITELLM_MODEL must be set when using localhost:4000 proxy");
        return (
            Arc::new(LiteLlmProvider::new(key, client).base_url("http://localhost:4000".into())),
            model,
        );
    }

    panic!("No LLM provider configured. Set LITELLM_API_URL, MISTRAL_API_KEY, or ANTHROPIC_API_KEY.");
}

pub fn print_result(output: &AgentOutput) {
    let json = serde_json::json!({
        "response": output.response.clone().unwrap_or_else(|| serde_json::Value::String(output.response_raw.clone())),
        "turns": output.statistics.turns,
        "tool_calls": output.statistics.tool_calls,
        "tokens_in": output.statistics.input_tokens,
        "tokens_out": output.statistics.output_tokens,
    });
    eprintln!("{}", serde_json::to_string_pretty(&json).unwrap());
}
