#![allow(dead_code)]

use std::sync::Arc;

use agentwerk::{AgentOutput, Provider, provider_from_env};

pub fn build_provider() -> (Arc<dyn Provider>, String) {
    provider_from_env().expect("LLM provider required for integration tests")
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
