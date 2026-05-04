//! Shared setup for integration tests: provider construction from env
//! and a JSON result printer.

#![allow(dead_code)]

use std::sync::Arc;

use agentwerk::providers::{from_env, model_from_env, Provider};
use agentwerk::Stats;

pub fn build_provider() -> (Arc<dyn Provider>, String) {
    let provider = from_env().expect("LLM provider required for integration tests");
    let model = model_from_env().expect("model name required for integration tests");
    (provider, model)
}

pub fn print_result(response: &str, stats: &Stats) {
    let json = serde_json::json!({
        "response": response,
        "steps": stats.steps(),
        "tool_calls": stats.tool_calls(),
        "tokens_in": stats.input_tokens(),
        "tokens_out": stats.output_tokens(),
    });
    eprintln!("{}", serde_json::to_string_pretty(&json).unwrap());
}
