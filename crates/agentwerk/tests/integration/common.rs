//! Shared setup for integration tests: provider construction from env, event handler, and JSON output helpers.

#![allow(dead_code)]

use std::sync::Arc;

use agentwerk::{provider, Output, Provider};

pub fn build_provider() -> (Arc<dyn Provider>, String) {
    let provider = provider::from_env().expect("LLM provider required for integration tests");
    let model = provider::model_from_env().expect("model name required for integration tests");
    (provider, model)
}

pub fn print_result(output: &Output) {
    let json = serde_json::json!({
        "response": output.response.clone().unwrap_or_else(|| serde_json::Value::String(output.response_raw.clone())),
        "steps": output.statistics.steps,
        "tool_calls": output.statistics.tool_calls,
        "tokens_in": output.statistics.input_tokens,
        "tokens_out": output.statistics.output_tokens,
    });
    eprintln!("{}", serde_json::to_string_pretty(&json).unwrap());
}
