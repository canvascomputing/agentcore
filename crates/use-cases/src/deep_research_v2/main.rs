//! Deep Research, ported to agentwerk2.
//!
//! Two phases run against separate `TicketSystem`s:
//!   1. Three `researcher` agents drain three `research_subquestion`
//!      tickets in parallel via Path B pickup. Each researcher calls
//!      `brave_search`, drops findings into a ticket comment, and
//!      transitions the ticket to `Done`.
//!   2. The driver assembles those findings into a single
//!      `final_report` ticket, hands it to the `report_writer` agent,
//!      which uses `manage_tickets_tool` `attach` to publish a
//!      schema-validated structured answer.
//!
//! v1's synchronous sub-agent invocation (`.staff_more`) doesn't
//! exist in v2 yet, so cross-agent flow happens through the ticket
//! queue instead. Phase 1 / Phase 2 are sequential because the
//! report writer needs the researchers' output before it starts.
//!
//! Usage: deep-research-v2 <QUESTION>
//!
//! Environment:
//!   BRAVE_API_KEY       Required for web search
//!   ANTHROPIC_API_KEY   (or other provider env vars)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agentwerk2::agents::agent::Agent;
use agentwerk2::agents::r#loop::Runnable;
use agentwerk2::agents::tickets::{Status, TicketSystem};
use agentwerk2::event::EventKind;
use agentwerk2::providers::{from_env, model_from_env, ProviderResult};
use agentwerk2::tools::{ManageTicketsTool, Tool, ToolResult};
use agentwerk2::Event;

const RESEARCHER_ROLE: &str =
    include_str!("prompts/researcher.role.md");
const RESEARCHER_BEHAVIOR: &str =
    include_str!("prompts/researcher.behavior.md");
const REPORT_WRITER_ROLE: &str =
    include_str!("prompts/report-writer.role.md");
const REPORT_WRITER_BEHAVIOR: &str =
    include_str!("prompts/report-writer.behavior.md");

#[tokio::main]
async fn main() {
    let question = parse_question();
    let brave_key = check_required_env();

    eprintln!("Question: {question}\n");

    let provider = from_env().expect("LLM provider required");
    let model = model_from_env().expect("model name required");
    let signal = setup_interrupt_signal();
    let event_handler: Arc<dyn Fn(Event) + Send + Sync> =
        Arc::new(|event: Event| log_event(&event));

    // ---- Phase 1: parallel researchers ------------------------------
    let mut tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&signal))
        .max_steps(30);
    let research_keys: Vec<String> = (1..=3)
        .map(|i| {
            let summary = format!("Research perspective {i}");
            let description = format!(
                "Question: {question}\n\nProduce evidence and sources for one perspective \
                 on this question. Focus on a different angle than perspectives 1..3 — \
                 the report writer will compare all three."
            );
            tickets
                .create(summary, description, "research_subquestion", "user")
                .key
                .clone()
        })
        .collect();

    let researchers: Vec<Agent> = (1..=3)
        .map(|i| {
            Agent::new()
                .name(format!("researcher_{i}"))
                .provider(Arc::clone(&provider))
                .model(&model)
                .role(RESEARCHER_ROLE)
                .behavior(RESEARCHER_BEHAVIOR)
                .ticket_type("research_subquestion")
                .tool(brave_search_tool(brave_key.clone()))
                .tool(ManageTicketsTool)
                .event_handler(Arc::clone(&event_handler))
        })
        .collect();

    let tickets = tickets.assign_all(researchers).run_until_empty().await;

    if signal.load(Ordering::Relaxed) {
        eprintln!("\nCancelled.");
        std::process::exit(130);
    }

    let findings: Vec<String> = research_keys
        .iter()
        .filter_map(|k| {
            let t = tickets.get(k)?;
            let body = t
                .comments
                .iter()
                .map(|c| format!("{}: {}", c.author, c.body))
                .collect::<Vec<_>>()
                .join("\n\n");
            Some(format!("### {} ({:?})\n{body}", t.key, t.status))
        })
        .collect();

    if findings.iter().all(|f| f.lines().count() <= 1) {
        eprintln!(
            "\nNo researcher findings recorded — aborting before the report writer."
        );
        std::process::exit(1);
    }

    // ---- Phase 2: synthesise ----------------------------------------
    let mut tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&signal))
        .max_steps(10);
    let final_summary = "Synthesise the final answer".to_string();
    let final_description = format!(
        "Question:\n{question}\n\n--- Researcher findings ---\n\n{}",
        findings.join("\n\n")
    );
    let final_key = tickets
        .create(final_summary, final_description, "final_report", "user")
        .key
        .clone();

    let report_writer = Agent::new()
        .name("report_writer")
        .provider(Arc::clone(&provider))
        .model(&model)
        .role(REPORT_WRITER_ROLE)
        .behavior(REPORT_WRITER_BEHAVIOR)
        .ticket_type("final_report")
        .tool(ManageTicketsTool)
        .event_handler(Arc::clone(&event_handler));

    let tickets = tickets.assign(report_writer).run_until_empty().await;

    if signal.load(Ordering::Relaxed) {
        eprintln!("\nCancelled.");
        std::process::exit(130);
    }

    let final_ticket = match tickets.get(&final_key) {
        Some(t) => t,
        None => {
            eprintln!("\nFinal ticket vanished — bug.");
            std::process::exit(1);
        }
    };
    if final_ticket.status != Status::Done {
        eprintln!(
            "\nReport writer left the ticket in {:?}; expected Done.",
            final_ticket.status
        );
        std::process::exit(1);
    }
    let attachment = match final_ticket.attachments.last() {
        Some(a) => a,
        None => {
            eprintln!(
                "\nReport writer marked the ticket Done without attaching the answer."
            );
            std::process::exit(1);
        }
    };

    println!("\n{}\n", format_title_first(&attachment.content));
    eprintln!(
        "Tokens: {} in, {} out · {} steps · {} requests",
        tickets.input_tokens(),
        tickets.output_tokens(),
        tickets.steps(),
        tickets.requests(),
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
        EventKind::TicketClaimed { key } => {
            eprintln!("[{}] claimed {key}", event.agent_name);
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
            eprintln!("[{}] policy violated: {kind:?} limit={limit}", event.agent_name);
        }
        EventKind::TicketFinished { key } => {
            eprintln!("[{}] finished {key}", event.agent_name);
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
        "manage_tickets_tool" => {
            let action = input["action"].as_str().unwrap_or("?");
            match action {
                "transition" => format!(
                    "transition → {}",
                    input["status"].as_str().unwrap_or("?")
                ),
                "attach" => format!(
                    "attach {}",
                    input["filename"].as_str().unwrap_or("(no filename)")
                ),
                "comment" => "comment".into(),
                other => other.into(),
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

