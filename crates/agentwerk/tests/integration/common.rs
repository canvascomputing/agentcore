//! Shared setup for integration tests: provider construction from env
//! and a JSON result printer.

#![allow(dead_code)]

use std::sync::Arc;

use agentwerk::providers::{model_from_env, provider_from_env, Provider};
use agentwerk::{Stats, TicketResult};

pub fn build_provider() -> (Arc<dyn Provider>, String) {
    let provider = provider_from_env().expect("LLM provider required for integration tests");
    let model = model_from_env().expect("model name required for integration tests");
    (provider, model)
}

pub fn print_result(results: &[TicketResult], stats: &Stats) {
    let response = last_result_string(results);
    let json = serde_json::json!({
        "response": response,
        "steps": stats.steps(),
        "tool_calls": stats.tool_calls(),
        "tokens_in": stats.input_tokens(),
        "tokens_out": stats.output_tokens(),
    });
    eprintln!("{}", serde_json::to_string_pretty(&json).unwrap());
}

pub fn last_result_string(results: &[TicketResult]) -> String {
    results
        .last()
        .map(|r| r.result_string())
        .unwrap_or_default()
}
