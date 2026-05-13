//! Deep Research with handover chain.
//!
//! One `TicketSystem` holds the whole pipeline. The driver enqueues a
//! single starter ticket pinned to `researcher_1`. Each researcher
//! calls `brave_search`, reads its parent ticket via
//! `read_tickets_tool` to build on prior findings, and hands off via
//! `write_handover_tool` to the next agent. The final researcher
//! attaches the report schema to its handover so the report writer's
//! result is validated by the framework. The report writer finishes
//! the chain with `write_result_tool`.
//!
//! Usage: deep-research <QUESTION>
//!
//! Environment:
//!   BRAVE_API_KEY       Required for web search
//!   ANTHROPIC_API_KEY   (or other provider env vars)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agentwerk::providers::{provider_from_env, ProviderResult};
use agentwerk::tools::{ReadTicketsTool, WriteHandoverTool};
use agentwerk::{Agent, Event, EventKind, Schema, Ticket, TicketSystem, Tool, ToolResult};

const RESEARCHER_1_ROLE: &str = include_str!("prompts/researcher_1.role.md");
const RESEARCHER_2_ROLE: &str = include_str!("prompts/researcher_2.role.md");
const REPORT_WRITER_ROLE: &str = include_str!("prompts/report-writer.role.md");

#[tokio::main]
async fn main() {
    let question = parse_question();
    let brave_key = check_required_env();

    eprintln!("Question: {question}\n");

    let provider = provider_from_env().expect("LLM provider required");
    let signal = setup_interrupt_signal();
    let event_handler: Arc<dyn Fn(Event) + Send + Sync> =
        Arc::new(|event: Event| log_event(&event));

    let schema_json_pretty = serde_json::to_string_pretty(&final_report_schema_value()).unwrap();

    let workdir = prepare_workdir();

    let tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&signal))
        .dir(workdir.clone());

    let researcher_1 = Agent::empty()
        .name("researcher_1")
        .provider(Arc::clone(&provider))
        .model_from_env()
        .role(RESEARCHER_1_ROLE)
        .label("researcher_1")
        .tool(brave_search_tool(brave_key.clone()))
        .tool(ReadTicketsTool)
        .tool(WriteHandoverTool)
        .event_handler(Arc::clone(&event_handler));

    let researcher_2 = Agent::empty()
        .name("researcher_2")
        .provider(Arc::clone(&provider))
        .model_from_env()
        .role(RESEARCHER_2_ROLE)
        .label("researcher_2")
        .template_variable("schema_json", schema_json_pretty.clone())
        .tool(brave_search_tool(brave_key.clone()))
        .tool(ReadTicketsTool)
        .tool(WriteHandoverTool)
        .event_handler(Arc::clone(&event_handler));

    let report_writer = Agent::new()
        .name("report_writer")
        .provider(Arc::clone(&provider))
        .model_from_env()
        .role(REPORT_WRITER_ROLE)
        .label("report")
        .tool(ReadTicketsTool)
        .event_handler(Arc::clone(&event_handler));

    tickets.agent(researcher_1);
    tickets.agent(researcher_2);
    tickets.agent(report_writer);

    let starter = format!(
        "Question: {question}\n\nKick off the research chain. You are researcher_1; pick \
         one angle and produce evidence with sources. The next two researchers will \
         extend the coverage."
    );
    // The schema-bound starter forces researcher_1 down the
    // `write_handover_tool` path: a text-only reply leaves no result
    // attached, and the loop's terminal-reply path then transitions
    // the ticket to `Failed` rather than silently `Done`.
    let starter_schema = Schema::parse(serde_json::json!({
        "type": "string",
        "minLength": 100
    }))
    .expect("starter schema is well-formed");
    tickets.ticket(
        Ticket::new(starter)
            .schema(starter_schema)
            .label("researcher_1"),
    );

    // Drive the run manually instead of via `run_dry`. The chain's
    // handover step has a brief window between marking the parent
    // `Done` and inserting the child, during which the queue is
    // empty; `run_dry` would race against that window and exit
    // prematurely. Polling for the report ticket directly, with a
    // grace period when the queue settles, is race-free.
    let running = tickets.run();
    let outcome = wait_for_outcome(&tickets, &signal).await;
    running.stop();
    running.join().await;

    print_chain_summary(&tickets);
    print_stats(&tickets);
    print_research_outcome(&tickets, &outcome);

    match outcome {
        Outcome::Report(_) => {}
        Outcome::Cancelled => std::process::exit(130),
        Outcome::Stalled => std::process::exit(1),
    }
}

fn print_research_outcome(tickets: &TicketSystem, outcome: &Outcome) {
    eprintln!("\n══════════════════════════════════════════════════════════");
    match outcome {
        Outcome::Report(ticket) => {
            let report_value = ticket.result().expect("done ticket carries a result");
            let title = report_value["title"].as_str().unwrap_or("(no title)");
            let research = report_value["research"].as_str().unwrap_or("(no body)");
            eprintln!(" REPORT");
            eprintln!("══════════════════════════════════════════════════════════\n");
            println!("## {title}\n\n{research}\n");
        }
        Outcome::Cancelled | Outcome::Stalled => {
            let label = match outcome {
                Outcome::Cancelled => {
                    "PARTIAL RESEARCH — run cancelled before report writer finished"
                }
                Outcome::Stalled => {
                    "PARTIAL RESEARCH — chain stalled before report writer finished"
                }
                Outcome::Report(_) => unreachable!(),
            };
            eprintln!(" {label}");
            eprintln!("══════════════════════════════════════════════════════════\n");
            let all = tickets.tickets();
            let researcher_findings: Vec<_> = all
                .iter()
                .filter(|t| t.status() == "done")
                .filter(|t| !t.has_label("report"))
                .filter_map(|t| t.result_string().map(|r| (t.key().to_string(), r)))
                .collect();
            if researcher_findings.is_empty() {
                eprintln!("(no researcher produced findings)");
            } else {
                for (key, findings) in researcher_findings {
                    println!("### {key}\n\n{findings}\n");
                }
            }
        }
    }
}

enum Outcome {
    Report(agentwerk::Ticket),
    Cancelled,
    Stalled,
}

async fn wait_for_outcome(tickets: &TicketSystem, signal: &Arc<AtomicBool>) -> Outcome {
    use std::time::Duration;

    let report_ticket = || tickets.find(|t| t.has_label("report") && t.status() == "done");
    let pending = || tickets.count(|t| matches!(t.status(), "todo" | "in_progress"));

    loop {
        if signal.load(Ordering::Relaxed) {
            return Outcome::Cancelled;
        }
        if let Some(ticket) = report_ticket() {
            return Outcome::Report(ticket);
        }
        if pending() == 0 {
            // Queue is empty — but a handover may be mid-flight,
            // between parent-Done and child-Insert. Give it a beat
            // before declaring the chain stalled.
            tokio::time::sleep(Duration::from_millis(250)).await;
            if signal.load(Ordering::Relaxed) {
                return Outcome::Cancelled;
            }
            if let Some(ticket) = report_ticket() {
                return Outcome::Report(ticket);
            }
            if pending() == 0 {
                return Outcome::Stalled;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn print_chain_summary(tickets: &TicketSystem) {
    eprintln!("\nChain summary:");
    let all = tickets.tickets();
    if all.is_empty() {
        eprintln!("  (no tickets)");
        return;
    }
    for t in &all {
        let parent = t
            .parent_key()
            .map(|p| format!(" ⟵ {p}"))
            .unwrap_or_default();
        let labels = if t.labels.is_empty() {
            String::new()
        } else {
            format!(" [{}]", t.labels.join(","))
        };
        let preview = t
            .result_string()
            .map(|s| truncate(&s, 100))
            .unwrap_or_else(|| "(no result)".into());
        eprintln!(
            "  {key} {status}{labels}{parent}\n      → {preview}",
            key = t.key(),
            status = t.status(),
        );
    }
}

fn print_stats(tickets: &TicketSystem) {
    let stats = tickets.stats();
    eprintln!("\nStats:");
    eprintln!(
        "  Duration : {:?}",
        stats.run_duration().unwrap_or_default()
    );
    eprintln!("  Work time: {:?}", stats.work_duration());
    eprintln!(
        "  Tickets  : {} done, {} failed ({:.0}%)",
        stats.tickets_done(),
        stats.tickets_failed(),
        stats
            .tickets_success_rate()
            .map(|r| r * 100.0)
            .unwrap_or(0.0),
    );
    eprintln!(
        "  Avg time : {:?}",
        stats.avg_ticket_duration().unwrap_or_default()
    );
    eprintln!(
        "  Tokens   : {} in, {} out",
        stats.input_tokens(),
        stats.output_tokens(),
    );
    eprintln!(
        "  Activity : {} requests · {} tool calls · {} errors",
        stats.requests(),
        stats.tool_calls(),
        stats.errors(),
    );
}

fn prepare_workdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("agentwerk-deep-research");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create deep-research workdir");
    dir
}

fn final_report_schema_value() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "title":    { "type": "string", "minLength": 1 },
            "research": { "type": "string", "minLength": 1 }
        },
        "required": ["title", "research"],
        "additionalProperties": false
    })
}

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
    let count = input["count"].as_u64().unwrap_or(5).min(20).to_string();

    let response = match reqwest::Client::new()
        .get("https://api.search.brave.com/res/v1/web/search")
        .query(&[("q", query), ("count", &count)])
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

fn log_event(event: &Event) {
    let agent = &event.agent_name;
    match &event.kind {
        EventKind::TicketStarted { key } => {
            eprintln!("\n┌─ [{agent}] picked up {key}");
        }
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => {
            for line in format_tool_call(tool_name, input) {
                eprintln!("│  {line}");
            }
        }
        EventKind::ToolCallFailed {
            tool_name,
            message,
            kind,
            ..
        } => {
            eprintln!("│  ✗ {tool_name} ({kind:?}): {message}");
        }
        EventKind::SchemaRetried {
            attempt,
            max_attempts,
            message,
        } => {
            eprintln!(
                "│  ↻ retry {attempt}/{max_attempts}: {}",
                truncate(message, 110)
            );
        }
        EventKind::PolicyViolated { kind, limit } => {
            eprintln!("│  ⚠ policy: {kind:?} limit={limit}");
        }
        EventKind::TicketDone { key } => {
            eprintln!("└─ ✓ finished {key}");
        }
        EventKind::TicketFailed { key } => {
            eprintln!("└─ ✗ failed {key}");
        }
        _ => {}
    }
}

fn format_tool_call(tool_name: &str, input: &serde_json::Value) -> Vec<String> {
    match tool_name {
        "brave_search" => vec![format!(
            "🔎 search: {}",
            truncate(input["query"].as_str().unwrap_or(""), 70),
        )],
        "read_tickets_tool" => {
            let action = input["action"].as_str().unwrap_or("?");
            let key = input.get("key").and_then(|v| v.as_str()).unwrap_or("");
            let suffix = if key.is_empty() {
                String::new()
            } else {
                format!(" {key}")
            };
            vec![format!("📖 read tickets {action}{suffix}")]
        }
        "write_handover_tool" => {
            let to = input["to"].as_str().unwrap_or("?");
            let task = preview_value(input.get("task"), 70);
            let result = preview_value(input.get("result"), 70);
            let schema_note = if input.get("schema").is_some() && !input["schema"].is_null() {
                " (+schema)"
            } else {
                ""
            };
            vec![
                format!("📤 handoff → {to}{schema_note}"),
                format!("      · task    : {task}"),
                format!("      · findings: {result}"),
            ]
        }
        "write_result_tool" => {
            let result = preview_value(input.get("result"), 80);
            vec![format!("✅ final result: {result}")]
        }
        _ => vec![format!(
            "{tool_name}: {}",
            serde_json::to_string(input).unwrap_or_default()
        )],
    }
}

fn preview_value(value: Option<&serde_json::Value>, max: usize) -> String {
    let raw = match value {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    truncate(&raw, max)
}

fn truncate(s: &str, max: usize) -> String {
    let one_line: String = s
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");
    if one_line.chars().count() <= max {
        return one_line;
    }
    let cut: String = one_line.chars().take(max).collect();
    format!("{cut}…")
}

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

fn setup_interrupt_signal() -> Arc<AtomicBool> {
    let signal = Arc::new(AtomicBool::new(false));
    let handle = signal.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        handle.store(true, Ordering::Relaxed);
    });
    signal
}
