//! Deep Research: spawns 2 sub-agents to research pro/con angles, then synthesizes a decision.
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

use agent::{
    AgentBuilder, AgenticError, Event, InvocationContext, SpawnAgentTool, ToolBuilder, ToolResult,
};

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const PRO_RESEARCHER_PROMPT: &str =
    "You are a researcher finding ARGUMENTS IN FAVOR of the proposal. \
     Use brave_search to find supporting evidence, success stories, and benefits. \
     Search 2-3 times with different queries. \
     Produce a concise report with sources.";

const CON_RESEARCHER_PROMPT: &str =
    "You are a researcher finding ARGUMENTS AGAINST the proposal. \
     Use brave_search to find risks, failures, hidden costs, and drawbacks. \
     Search 2-3 times with different queries. \
     Produce a concise report with sources.";

const ORCHESTRATOR_PROMPT: &str =
    "You are a decision analyst. Given a question, you:\n\
     1. Spawn 'pro_researcher' with a prompt to find arguments in favor\n\
     2. Spawn 'con_researcher' with a prompt to find arguments against\n\
     3. Synthesize both reports into a structured recommendation\n\n\
     Use spawn_agent with agent: \"pro_researcher\" and agent: \"con_researcher\".\n\
     After receiving both reports, provide your final analysis as structured output.";

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "recommendation": { "type": "string", "description": "Clear recommendation: yes, no, or conditional" },
            "confidence": { "type": "string", "description": "high, medium, or low" },
            "pro_summary": { "type": "string", "description": "Key arguments in favor" },
            "con_summary": { "type": "string", "description": "Key arguments against" },
            "key_factors": {
                "type": "array",
                "items": { "type": "string" },
                "description": "The most important factors in the decision"
            }
        },
        "required": ["recommendation", "confidence", "pro_summary", "con_summary", "key_factors"]
    })
}

// ---------------------------------------------------------------------------
// Brave Search tool
// ---------------------------------------------------------------------------

fn brave_search_tool(api_key: String) -> impl agent::Tool {
    ToolBuilder::new("brave_search", "Search the web using Brave Search. Returns titles, URLs, and descriptions.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "count": { "type": "integer", "description": "Number of results (1-20, default: 5)" }
            },
            "required": ["query"]
        }))
        .read_only(true)
        .handler(move |input, _ctx| {
            let api_key = api_key.clone();
            Box::pin(async move {
                let query = input["query"].as_str().unwrap_or("").to_string();
                let count = input["count"].as_u64().unwrap_or(5).min(20);

                let url = format!(
                    "https://api.search.brave.com/res/v1/web/search?q={}&count={}&extra_snippets=true",
                    urlencode(&query),
                    count,
                );

                let client = reqwest::Client::new();
                let resp = client
                    .get(&url)
                    .header("X-Subscription-Token", &api_key)
                    .header("Accept", "application/json")
                    .send()
                    .await
                    .map_err(|e| AgenticError::Other(format!("Brave search failed: {e}")))?;

                let json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| AgenticError::Other(format!("Failed to parse response: {e}")))?;

                let results = json["web"]["results"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .map(|r| {
                                let title = r["title"].as_str().unwrap_or("");
                                let url = r["url"].as_str().unwrap_or("");
                                let desc = r["description"].as_str().unwrap_or("");
                                let extra = r["extra_snippets"]
                                    .as_array()
                                    .map(|s| s.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(" "))
                                    .unwrap_or_default();
                                if extra.is_empty() {
                                    format!("## {title}\n{url}\n{desc}\n")
                                } else {
                                    format!("## {title}\n{url}\n{desc}\n{extra}\n")
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_else(|| "No results found.".into());

                Ok(ToolResult { content: results, is_error: false })
            })
        })
        .build()
}

fn urlencode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '&' => "%26".to_string(),
            '=' => "%3D".to_string(),
            '?' => "%3F".to_string(),
            '#' => "%23".to_string(),
            '+' => "%2B".to_string(),
            _ if c.is_ascii_alphanumeric() || "-_.~".contains(c) => c.to_string(),
            _ => format!("%{:02X}", c as u32),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
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

    let question = args[1..].join(" ");

    let mut missing = Vec::new();
    let brave_key = std::env::var("BRAVE_API_KEY").ok();
    if brave_key.is_none() {
        missing.push("BRAVE_API_KEY");
    }
    let has_provider = std::env::var("ANTHROPIC_API_KEY").is_ok()
        || std::env::var("MISTRAL_API_KEY").is_ok()
        || std::env::var("LITELLM_API_URL").is_ok()
        || std::net::TcpStream::connect("127.0.0.1:4000").is_ok();
    if !has_provider {
        missing.push("ANTHROPIC_API_KEY (or MISTRAL_API_KEY or LITELLM_API_URL)");
    }
    if !missing.is_empty() {
        eprintln!("Error: missing environment variables:");
        for var in &missing {
            eprintln!("  {var}");
        }
        std::process::exit(1);
    }
    let brave_key = brave_key.unwrap();

    let (provider, model) = use_cases::auto_detect_provider();

    eprintln!("Question: {question}\n");

    // Sub-agents
    let pro_researcher = AgentBuilder::new()
        .name("pro_researcher")
        .model(&model)
        .system_prompt(PRO_RESEARCHER_PROMPT)
        .tool(brave_search_tool(brave_key.clone()))
        .max_turns(8)
        .build()
        .expect("Failed to build pro_researcher");

    let con_researcher = AgentBuilder::new()
        .name("con_researcher")
        .model(&model)
        .system_prompt(CON_RESEARCHER_PROMPT)
        .tool(brave_search_tool(brave_key))
        .max_turns(8)
        .build()
        .expect("Failed to build con_researcher");

    // Orchestrator
    let spawn_tool = SpawnAgentTool::new()
        .with_sub_agents(vec![pro_researcher, con_researcher])
        .with_default_model(&model);

    let orchestrator = AgentBuilder::new()
        .name("orchestrator")
        .model(&model)
        .system_prompt(ORCHESTRATOR_PROMPT)
        .tool(spawn_tool)
        .output_schema(output_schema())
        .max_turns(10)
        .build()
        .expect("Failed to build orchestrator");

    // Events
    let event_handler: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match &event {
        Event::AgentStart { agent_name } => eprintln!("[{agent_name}] started"),
        Event::AgentEnd { agent_name, turns } => eprintln!("[{agent_name}] done ({turns} turns)"),
        Event::ToolCallStart { agent_name, tool_name, input, .. } => {
            let summary = if tool_name == "brave_search" {
                input["query"].as_str().unwrap_or("").to_string()
            } else if tool_name == "spawn_agent" {
                input["agent"].as_str()
                    .or(input["description"].as_str())
                    .unwrap_or("")
                    .to_string()
            } else {
                serde_json::to_string(input).unwrap_or_default()
            };
            eprintln!("[{agent_name}] {tool_name}: {summary}");
        }
        Event::TextChunk { agent_name, content } if agent_name == "orchestrator" => print!("{content}"),
        _ => {}
    });

    // Cancellation
    let cancel_signal = Arc::new(AtomicBool::new(false));
    {
        let c = cancel_signal.clone();
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            c.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }

    // Run
    let mut ctx = InvocationContext::new(provider);
    ctx.prompt = question;
    ctx.event_handler = event_handler;
    ctx.cancel_signal = cancel_signal;

    match orchestrator.run(ctx).await {
        Ok(output) => {
            if let Some(decision) = &output.response {
                println!("\n\n━━━ Decision ━━━\n");
                println!("Recommendation: {}", decision["recommendation"].as_str().unwrap_or(""));
                println!("Confidence: {}", decision["confidence"].as_str().unwrap_or(""));
                println!("\n── Pro ──\n{}", decision["pro_summary"].as_str().unwrap_or(""));
                println!("\n── Con ──\n{}", decision["con_summary"].as_str().unwrap_or(""));
                if let Some(factors) = decision["key_factors"].as_array() {
                    println!("\n── Key Factors ──");
                    for f in factors {
                        println!("  • {}", f.as_str().unwrap_or(""));
                    }
                }
            } else {
                println!("\n{}", output.response_raw);
            }
            eprintln!("\nCost: ${:.4}", output.statistics.costs);
        }
        Err(e) => {
            eprintln!("\nError: {e}");
            std::process::exit(1);
        }
    }
}
