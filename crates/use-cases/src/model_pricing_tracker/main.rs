//! Model Pricing Tracker: fetches current model pricing from provider websites.
//!
//! Usage: model-pricing-tracker [--output <path>]
//!
//! Spawns a pricing researcher agent that fetches live pricing from provider
//! web pages, then outputs structured JSON with per-model pricing.
//!
//! Environment:
//!   ANTHROPIC_API_KEY   (or other provider env vars)

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agentwerk::{Agent, AgentEvent, AgentEventKind, WebFetchTool};

const PRICING_RESEARCHER_PROMPT: &str = "\
You are a pricing researcher. Fetch current model pricing from provider websites.\n\n\
Fetch these URLs using the web_fetch tool:\n\
1. https://platform.claude.com/docs/en/about-claude/pricing\n\
2. https://mistral.ai/pricing#api\n\n\
For each model, extract:\n\
- Model API identifier (e.g. claude-sonnet-4-20250514)\n\
- Input cost per million tokens (USD)\n\
- Output cost per million tokens (USD)\n\n\
If a page doesn't render (JS-only), report that.\n\n\
Output ONLY a list like:\n\
  provider: model-id input=$X.XX output=$X.XX\n\
One line per model. Nothing else.";

const ORCHESTRATOR_PROMPT: &str = "\
You coordinate pricing research. You MUST use the spawn_agent tool with the 'agent' parameter.\n\n\
Step 1: Call spawn_agent with {\"agent\": \"pricing_researcher\", \"description\": \"fetch pricing\", \
\"instruction\": \"Fetch current model pricing from all provider websites\"}.\n\n\
Step 2: After it returns, produce your structured output listing every model found. \
Include provider, model_id, input_per_million, output_per_million for each.\n\n\
CRITICAL: Always set the 'agent' field to 'pricing_researcher'. \
Do NOT add explanation. Just spawn the agent, then output the structured result.";

fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "provider": { "type": "string", "description": "Provider name (anthropic, mistral)" },
                        "model_id": { "type": "string", "description": "Model API identifier" },
                        "input_per_million": { "type": "number", "description": "Input cost per million tokens in USD" },
                        "output_per_million": { "type": "number", "description": "Output cost per million tokens in USD" }
                    },
                    "required": ["provider", "model_id", "input_per_million", "output_per_million"]
                }
            }
        },
        "required": ["models"]
    })
}

#[tokio::main]
async fn main() {
    let output_path = parse_args();

    eprintln!("Model Pricing Tracker\n");

    let pricing_researcher = Agent::new()
        .name("pricing_researcher")
        .identity_prompt(PRICING_RESEARCHER_PROMPT)
        .tool(WebFetchTool)
        .max_turns(10);

    let output = match Agent::new()
        .name("pricing_tracker")
        .provider_from_env()
        .expect("LLM provider required")
        .identity_prompt(ORCHESTRATOR_PROMPT)
        .sub_agents([pricing_researcher])
        .output_schema(output_schema())
        .max_turns(10)
        .instruction_prompt("Gather current model pricing from all supported providers.")
        .event_handler(Arc::new(|event| log_event(&event)))
        .cancel_signal(setup_cancel_signal())
        .run()
        .await
    {
        Ok(out) => out,
        Err(e) => {
            eprintln!("\nError: {e}");
            std::process::exit(1);
        }
    };

    let json = match &output.response {
        Some(data) => serde_json::to_string_pretty(data).unwrap_or_default(),
        None => output.response_raw.clone(),
    };

    match std::fs::write(&output_path, &json) {
        Ok(_) => eprintln!("\nWrote {output_path}"),
        Err(e) => eprintln!("Failed to write {output_path}: {e}"),
    }
    println!("{json}");
    eprintln!(
        "Tokens: {} in, {} out",
        output.statistics.input_tokens, output.statistics.output_tokens
    );
}

fn log_event(event: &AgentEvent) {
    match &event.kind {
        AgentEventKind::RequestStart { model, .. } => {
            eprintln!("[{}] requesting {model}...", event.agent_name);
        }
        AgentEventKind::ToolCallStart {
            tool_name, input, ..
        } if tool_name != "StructuredOutput" => {
            eprintln!(
                "[{}] {tool_name}: {}",
                event.agent_name,
                tool_call_detail(tool_name, input)
            );
        }
        AgentEventKind::ToolCallError {
            tool_name, error, ..
        } => {
            eprintln!("[error] {tool_name}: {error}");
        }
        _ => {}
    }
}

fn tool_call_detail(tool_name: &str, input: &serde_json::Value) -> String {
    let key = match tool_name {
        "web_fetch" => "url",
        "spawn_agent" => {
            return input["agent"]
                .as_str()
                .or(input["description"].as_str())
                .unwrap_or("")
                .into()
        }
        _ => return serde_json::to_string(input).unwrap_or_default(),
    };
    input[key].as_str().unwrap_or("").into()
}

fn parse_args() -> String {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: model-pricing-tracker [--output <path>]");
        eprintln!();
        eprintln!("Fetches current model pricing from provider websites.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --output <path>  Output file (default: pricing.json)");
        eprintln!();
        eprintln!("Environment:");
        eprintln!("  ANTHROPIC_API_KEY   (or other provider env vars)");
        std::process::exit(0);
    }

    for i in 0..args.len() {
        if (args[i] == "--output" || args[i] == "-o") && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }

    "pricing.json".into()
}

fn setup_cancel_signal() -> Arc<AtomicBool> {
    let signal = Arc::new(AtomicBool::new(false));
    let handle = signal.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        handle.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    signal
}
