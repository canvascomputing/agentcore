//! Deep Research: a researcher gathers evidence, then a report writer synthesizes a decision.
//!
//! Usage: deep-research <QUESTION>
//!
//! Example: deep-research "Should we use Rust or Go for our backend?"
//!
//! Environment:
//!   BRAVE_API_KEY       Required for web search
//!   ANTHROPIC_API_KEY   (or other provider env vars)

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use agentwerk::{
    Agent, AgenticError, Event, EventKind, ToolBuilder, ToolResult,
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let question = parse_question();
    let brave_key = check_required_env();
    let (provider, model) = use_cases::provider_from_env().expect("LLM provider required");

    eprintln!("Question: {question}\n");

    let researchers: Vec<_> = (1..=3)
        .map(|i| {
            Agent::new()
                .name(format!("researcher_{i}"))
                .model(&model)
                .identity_prompt(RESEARCHER_PROMPT)
                .tool(brave_search_tool(brave_key.clone()))
                .max_turns(3)
        })
        .collect();

    let output = match Agent::new()
        .name("report_writer")
        .model(&model)
        .identity_prompt(REPORT_WRITER_PROMPT)
        .sub_agents(researchers)
        .output_schema(output_schema())
        .max_turns(10)
        .provider(provider)
        .instruction_prompt(question)
        .event_handler(Arc::new(|event| log_event(&event)))
        .cancel_signal(setup_cancel_signal())
        .run()
        .await
    {
        Ok(output) => output,
        Err(e) => {
            eprintln!("\nError: {e}");
            std::process::exit(1);
        }
    };

    eprintln!();
    match &output.response {
        Some(data) => println!("\n{}\n", format_title_first(data)),
        None => println!("\n{}\n", output.response_raw),
    }
    eprintln!("Tokens: {} in, {} out", output.statistics.input_tokens, output.statistics.output_tokens);
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const RESEARCHER_PROMPT: &str =
    "You are a thorough researcher. Given a question, search the web 1-2 times \
     for evidence from both sides — arguments in favor AND against. \
     Include pros, cons, costs, and real-world experiences. \
     Produce a factual report with sources.";

const REPORT_WRITER_PROMPT: &str =
    "You are a decision analyst. Given a question:\n\
     1. Spawn all three researchers in parallel: 'researcher_1', 'researcher_2', 'researcher_3'\n\
     2. Review and aggregate all research reports\n\
     3. Produce your final recommendation as structured output\n\n\
     IMPORTANT: Call spawn_agent three times in your FIRST response — one call per researcher, \
     all in the same message. Do NOT wait for one to finish before spawning the next.\n\n\
     Your output must be plain text only — no markdown, no bullet points, no special formatting. \
     The research field must be under 500 characters.";

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

fn format_title_first(data: &serde_json::Value) -> String {
    let Some(obj) = data.as_object() else {
        return serde_json::to_string_pretty(data).unwrap_or_default();
    };

    // serde_json::Map is sorted alphabetically, so we format manually
    let mut entries: Vec<(&str, &serde_json::Value)> = Vec::new();
    if let Some(title) = obj.get("title") {
        entries.push(("title", title));
    }
    for (k, v) in obj {
        if k != "title" {
            entries.push((k, v));
        }
    }

    let fields: Vec<String> = entries
        .iter()
        .map(|(k, v)| format!("  \"{k}\": {}", serde_json::to_string_pretty(v).unwrap_or_default()))
        .collect();

    format!("{{\n{}\n}}", fields.join(",\n"))
}

fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "title": { "type": "string", "description": "Short title summarizing the question" },
            "research": { "type": "string", "description": "Plain text summary of findings and recommendation, max 500 characters" }
        },
        "required": ["title", "research"]
    })
}

// ---------------------------------------------------------------------------
// Brave Search tool
// ---------------------------------------------------------------------------

fn brave_search_tool(api_key: String) -> impl agentwerk::Tool {
    ToolBuilder::new("brave_search", "Search the web. Returns titles, URLs, and descriptions.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "count": { "type": "integer", "description": "Results count (1-20, default: 5)" }
            },
            "required": ["query"]
        }))
        .read_only(true)
        .handler(move |input, _ctx| {
            let api_key = api_key.clone();
            Box::pin(async move { brave_search(&api_key, &input).await })
        })
        .build()
}

async fn brave_search(api_key: &str, input: &serde_json::Value) -> agentwerk::Result<ToolResult> {
    let query = input["query"].as_str().unwrap_or("").trim();
    let count = input["count"].as_u64().unwrap_or(5).min(20);

    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
        urlencode(query), count,
    );

    let response = reqwest::Client::new()
        .get(&url)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| AgenticError::Other(format!("Brave search failed: {e}")))?;

    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| AgenticError::Other(format!("Failed to parse response: {e}")))?;

    let Some(results) = json["web"]["results"].as_array() else {
        return Ok(ToolResult::success("No results found."));
    };

    let text = results
        .iter()
        .map(|r| {
            format!(
                "## {}\n{}\n{}\n",
                r["title"].as_str().unwrap_or(""),
                r["url"].as_str().unwrap_or(""),
                r["description"].as_str().unwrap_or(""),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(ToolResult::success(text))
}

fn urlencode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '&' => "%26".to_string(),
            '?' => "%3F".to_string(),
            '#' => "%23".to_string(),
            '+' => "%2B".to_string(),
            '=' => "%3D".to_string(),
            _ if c.is_ascii_alphanumeric() || "-_.~".contains(c) => c.to_string(),
            _ => format!("%{:02X}", c as u32),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Event handler
// ---------------------------------------------------------------------------

fn log_event(event: &Event) {
    match &event.kind {
        EventKind::RequestStart { model } => {
            eprintln!("[{}] requesting {model}...", event.agent_name);
        }
        EventKind::ToolCallStart { tool_name, input, .. } if tool_name != "StructuredOutput" => {
            let detail = tool_call_summary(tool_name, input);
            eprintln!("[{}] {tool_name}: {detail}", event.agent_name);
        }
        EventKind::ToolCallEnd { tool_name, output, is_error: true, .. } => {
            eprintln!("[error] {tool_name}: {output}");
        }
        _ => {}
    }
}

fn tool_call_summary(tool_name: &str, input: &serde_json::Value) -> String {
    match tool_name {
        "brave_search" => {
            let q = input["query"].as_str().unwrap_or("");
            if q.len() > 50 { format!("{}…", &q[..50]) } else { q.into() }
        }
        "spawn_agent" => input["agent"].as_str().or(input["description"].as_str()).unwrap_or("").into(),
        _ => serde_json::to_string(input).unwrap_or_default(),
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_question() -> String {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: deep-research <QUESTION>");
        eprintln!();
        eprintln!("Example: deep-research \"Should we use Rust or Go for our backend?\"");
        eprintln!();
        eprintln!("Environment:");
        eprintln!("  BRAVE_API_KEY       Required for web search");
        eprintln!("  ANTHROPIC_API_KEY   (or other provider env vars)");
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    args[1..].join(" ")
}

fn check_required_env() -> String {
    let brave_key = std::env::var("BRAVE_API_KEY").unwrap_or_default();
    if brave_key.is_empty() {
        eprintln!("Error: missing environment variable: BRAVE_API_KEY");
        std::process::exit(1);
    }
    brave_key
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
