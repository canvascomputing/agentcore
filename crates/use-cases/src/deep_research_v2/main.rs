//! Deep Research.
//!
//! Two phases run against separate `TicketSystem`s:
//!   1. Three `researcher` agents drain three research tickets in
//!      parallel via Path B label pickup. Each researcher calls
//!      `brave_search` and finishes its ticket by calling
//!      `write_result_tool` with the findings string as `result`.
//!   2. The driver assembles those findings into a single
//!      schema-checked ticket and hands it to the `report_writer`
//!      agent. The report writer calls `write_result_tool` with a JSON
//!      value the framework validates against the ticket's schema.
//!
//! Usage: deep-research-v2 <QUESTION>
//!
//! Environment:
//!   BRAVE_API_KEY       Required for web search
//!   ANTHROPIC_API_KEY   (or other provider env vars)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agentwerk::providers::{model_from_env, provider_from_env, ProviderResult};
use agentwerk::tools::ManageTicketsTool;
use agentwerk::{Agent, Event, EventKind, Runnable, Schema, TicketSystem, Tool, ToolResult};

const RESEARCHER_ROLE: &str = include_str!("prompts/researcher.role.md");
const REPORT_WRITER_ROLE: &str = include_str!("prompts/report-writer.role.md");

#[tokio::main]
async fn main() {
    let question = parse_question();
    let brave_key = check_required_env();

    eprintln!("Question: {question}\n");

    let provider = provider_from_env().expect("LLM provider required");
    let model = model_from_env().expect("model name required");
    let signal = setup_interrupt_signal();
    let event_handler: Arc<dyn Fn(Event) + Send + Sync> =
        Arc::new(|event: Event| log_event(&event));

    // ---- Phase 1: parallel researchers ------------------------------
    let tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&signal))
        .max_steps(30);

    let mut research_keys: Vec<String> = Vec::new();
    for i in 1..=3 {
        let body = format!(
            "Research perspective {i}\n\nQuestion: {question}\n\nProduce evidence and \
             sources for one perspective on this question. Focus on a different angle \
             than perspectives 1..3 — the report writer will compare all three."
        );
        tickets.task_labeled(body, "research");
        research_keys.push(format!("TICKET-{i}"));
    }

    let researchers: Vec<Agent> = (1..=3)
        .map(|i| {
            Agent::new()
                .name(format!("researcher_{i}"))
                .provider(Arc::clone(&provider))
                .model(&model)
                .role(RESEARCHER_ROLE)
                .label("research")
                .tool(brave_search_tool(brave_key.clone()))
                .tool(ManageTicketsTool)
                .event_handler(Arc::clone(&event_handler))
        })
        .collect();

    for r in researchers {
        tickets.add(r);
    }
    tickets.run_dry().await;

    if signal.load(Ordering::Relaxed) {
        eprintln!("\nCancelled.");
        std::process::exit(130);
    }

    let findings: Vec<String> = research_keys
        .iter()
        .filter_map(|k| {
            let t = tickets.get(k)?;
            let body = t
                .result_string()
                .unwrap_or_else(|| "(no findings)".to_string());
            Some(format!("### {} ({:?})\n{body}", t.key(), t.status()))
        })
        .collect();

    if findings.iter().all(|f| f.lines().count() <= 1) {
        eprintln!("\nNo researcher findings recorded — aborting before the report writer.");
        std::process::exit(1);
    }

    // ---- Phase 2: synthesise ----------------------------------------
    let tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&signal))
        .max_steps(10);

    let final_schema = Schema::parse(serde_json::json!({
        "type": "object",
        "properties": {
            "title":    { "type": "string", "minLength": 1 },
            "research": { "type": "string", "minLength": 1, "maxLength": 500 }
        },
        "required": ["title", "research"],
        "additionalProperties": false
    }))
    .expect("final-report schema is well-formed");

    let final_body = format!(
        "Question:\n{question}\n\n--- Researcher findings ---\n\n{}",
        findings.join("\n\n")
    );
    tickets.task_schema_labeled(final_body, final_schema, "report");

    let report_writer = Agent::new()
        .name("report_writer")
        .provider(Arc::clone(&provider))
        .model(&model)
        .role(REPORT_WRITER_ROLE)
        .label("report")
        .tool(ManageTicketsTool)
        .event_handler(Arc::clone(&event_handler));

    tickets.add(report_writer);
    let results = tickets.run_dry().await;
    let report = results
        .last()
        .map(|r| r.result_string())
        .unwrap_or_default();

    if signal.load(Ordering::Relaxed) {
        eprintln!("\nCancelled.");
        std::process::exit(130);
    }

    if report.is_empty() {
        let status = tickets.first().map(|t| t.status());
        eprintln!("\nReport writer left the ticket in {status:?}; expected Done with a result.");
        std::process::exit(1);
    }
    let parsed: serde_json::Value = match serde_json::from_str(&report) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("\nReport writer's result is not valid JSON: {e}");
            std::process::exit(1);
        }
    };

    println!("\n{}\n", format_title_first(&parsed));
    let stats = tickets.stats();
    eprintln!("Duration:  {:?}", stats.run_duration().unwrap_or_default());
    eprintln!("Work time: {:?}", stats.total_work_duration());
    eprintln!(
        "Tickets:   {} done, {} failed ({:.0}%)",
        stats.tickets_done(),
        stats.tickets_failed(),
        stats.success_rate().map(|r| r * 100.0).unwrap_or(0.0),
    );
    eprintln!(
        "Avg time:  {:?}",
        stats.avg_ticket_duration().unwrap_or_default()
    );
    eprintln!(
        "Tokens:    {} in, {} out",
        stats.input_tokens(),
        stats.output_tokens(),
    );
    eprintln!(
        "Activity:  {} requests · {} tool calls · {} errors",
        stats.requests(),
        stats.tool_calls(),
        stats.errors(),
    );
}

// ---- helpers -------------------------------------------------------

fn brave_search_tool(api_key: String) -> Tool {
    Tool::new(
        "brave_search",
        "Search the web. Returns titles, URLs, and descriptions.",
    )
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
}

async fn brave_search(api_key: &str, input: &serde_json::Value) -> ProviderResult<ToolResult> {
    let query = input["query"].as_str().unwrap_or("").trim();
    let count = input["count"].as_u64().unwrap_or(5).min(20);

    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
        urlencode(query),
        count,
    );

    let response = match reqwest::Client::new()
        .get(&url)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Ok(ToolResult::error(format!("Brave search failed: {e}"))),
    };

    let json: serde_json::Value = match response.json().await {
        Ok(j) => j,
        Err(e) => return Ok(ToolResult::error(format!("Failed to parse response: {e}"))),
    };

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

fn log_event(event: &Event) {
    match &event.kind {
        EventKind::TicketStarted { key } => {
            eprintln!("[{}] started {key}", event.agent_name);
        }
        EventKind::RequestStarted { model } => {
            eprintln!("[{}] requesting {model}…", event.agent_name);
        }
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => {
            eprintln!(
                "[{}] {tool_name}: {}",
                event.agent_name,
                tool_call_summary(tool_name, input)
            );
        }
        EventKind::ToolCallFailed {
            tool_name,
            message,
            kind,
            ..
        } => {
            eprintln!("[{}] ✗ {tool_name} ({kind:?}): {message}", event.agent_name);
        }
        EventKind::PolicyViolated { kind, limit } => {
            eprintln!(
                "[{}] policy violated: {kind:?} limit={limit}",
                event.agent_name
            );
        }
        EventKind::TicketDone { key } => {
            eprintln!("[{}] done {key}", event.agent_name);
        }
        EventKind::TicketFailed { key } => {
            eprintln!("[{}] failed {key}", event.agent_name);
        }
        _ => {}
    }
}

fn tool_call_summary(tool_name: &str, input: &serde_json::Value) -> String {
    match tool_name {
        "brave_search" => {
            let q = input["query"].as_str().unwrap_or("");
            if q.chars().count() > 50 {
                let cut: String = q.chars().take(50).collect();
                format!("{cut}…")
            } else {
                q.into()
            }
        }
        "manage_tickets_tool" => input["action"].as_str().unwrap_or("?").into(),
        "write_result_tool" => {
            let result = match &input["result"] {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if result.chars().count() > 50 {
                let cut: String = result.chars().take(50).collect();
                format!("done: {cut}…")
            } else {
                format!("done: {result}")
            }
        }
        _ => serde_json::to_string(input).unwrap_or_default(),
    }
}

fn format_title_first(data: &serde_json::Value) -> String {
    let Some(obj) = data.as_object() else {
        return serde_json::to_string_pretty(data).unwrap_or_default();
    };
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
        .map(|(k, v)| {
            format!(
                "  \"{k}\": {}",
                serde_json::to_string_pretty(v).unwrap_or_default()
            )
        })
        .collect();
    format!("{{\n{}\n}}", fields.join(",\n"))
}

fn parse_question() -> String {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: deep-research-v2 <QUESTION>");
        eprintln!();
        eprintln!("Example: deep-research-v2 \"Should we use Rust or Go for our backend?\"");
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

fn setup_interrupt_signal() -> Arc<AtomicBool> {
    let signal = Arc::new(AtomicBool::new(false));
    let handle = signal.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        handle.store(true, Ordering::Relaxed);
    });
    signal
}
