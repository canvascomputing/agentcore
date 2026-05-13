//! Interactive terminal chat. One `TicketSystem` + `Agent` + `Knowledge`
//! lives for the whole session; each input line enqueues a ticket and
//! runs `finish` once. The agent has `knowledge(...)` bound, so durable
//! facts the model writes via `knowledge_tool` survive across turns and
//! across process restarts (the store lives at `./.agentwerk/`).
//! The model's response streams to stdout via
//! `EventKind::TextChunkReceived`, and the agent finishes its ticket
//! via the auto-registered `write_result_tool`. Slash commands:
//! `/exit` quits, `/knowledge` prints the current knowledge index,
//! `/clear` resets it. Ctrl-C at the prompt exits with code 130;
//! Ctrl-C during a turn cancels that turn (a second Ctrl-C while the
//! cancel is still draining force-quits with exit code 130).
//!
//! Every exit path goes through `std::process::exit` rather than a
//! plain `return`: the stdin reader runs on a tokio blocking thread
//! parked in `read(2)`, which the runtime can't cancel on shutdown.
//! Exiting the process directly bypasses the runtime drop and avoids
//! a hang on outstanding blocking tasks.

use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agentwerk::event::{Event, EventKind};
use agentwerk::tools::{GlobTool, GrepTool, ListDirectoryTool, ReadFileTool};
use agentwerk::{Agent, Knowledge, TicketSystem};

const ROLE: &str = include_str!("prompts/repl.role.md");

#[tokio::main]
async fn main() {
    let style = Style::detect();
    eprintln!(
        "{}agentwerk REPL: /exit /knowledge /clear, Ctrl-C to cancel.{}",
        style.dim, style.reset,
    );

    // Optional first positional arg: synthetic context-window size in
    // **tokens**, used only to drive a REPL-side usage line and warning
    // when the reported `input_tokens` crosses 70% of it. The library's
    // own `ContextCompacted` event is unaffected and still fires only
    // at the real model window.
    let test_window: Option<u64> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    if let Some(w) = test_window {
        let threshold = w.saturating_mul(7) / 10;
        eprintln!(
            "{}test context window: {w} tokens (synthetic warning at {threshold} tokens used){}",
            style.dim, style.reset,
        );
    }

    let role = ROLE.trim();

    let user_prompt = format!("\n{}you ›{} ", style.user, style.reset);

    let event_style = style.clone();
    // `midstream` tracks whether the last byte written was streamed
    // model text (no trailing newline). Stderr event lines consult it
    // to break out of the stream exactly once, instead of every
    // `eprintln!` doubling up newlines.
    let midstream = Arc::new(AtomicBool::new(false));
    let handler_midstream = Arc::clone(&midstream);
    let handler: Arc<dyn Fn(Event) + Send + Sync> =
        Arc::new(move |e: Event| print_event(&e, &event_style, test_window, &handler_midstream));

    let tickets = TicketSystem::new();
    tickets.max_steps(40);

    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let knowledge_dir = cwd.join(".agentwerk");
    let knowledge = Knowledge::open(&knowledge_dir).expect("open knowledge store");

    let _agent = tickets.agent(
        Agent::new()
            .name("orchestrator")
            .from_env()
            .role(role)
            .dir(&cwd)
            .tool(GlobTool)
            .tool(GrepTool)
            .tool(ListDirectoryTool)
            .tool(ReadFileTool)
            .event_handler(handler)
            .knowledge(&knowledge),
    );

    let mut prev_steps: u64 = 0;
    let mut prev_requests: u64 = 0;
    let mut prev_tool_calls: u64 = 0;
    let mut prev_input: u64 = 0;
    let mut prev_output: u64 = 0;

    loop {
        let line = tokio::select! {
            line = read_line(&user_prompt) => line,
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n{}^C{}", style.dim, style.reset);
                std::process::exit(130);
            }
        };
        let Some(line) = line else {
            std::process::exit(0)
        };
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/quit" {
            std::process::exit(0);
        }
        if line == "/knowledge" {
            let idx = knowledge.index();
            let rendered = if idx.is_empty() {
                "(knowledge empty)".into()
            } else {
                idx
            };
            eprintln!("{}{rendered}{}", style.dim, style.reset);
            continue;
        }
        if line == "/clear" {
            knowledge.clear().ok();
            eprintln!("{}knowledge cleared{}", style.dim, style.reset);
            continue;
        }

        announce_assistant(&style);
        // "agent › " left stdout mid-line; mark so the first event
        // breaks out before its own content.
        midstream.store(true, Ordering::Relaxed);
        tickets.task(line);

        let run_fut = tickets.finish();
        tokio::pin!(run_fut);
        let cancelled = tokio::select! {
            _ = &mut run_fut => false,
            _ = tokio::signal::ctrl_c() => {
                tickets.cancel();
                if midstream.swap(false, Ordering::Relaxed) {
                    eprintln!();
                }
                eprintln!("{}cancelling…{}", style.dim, style.reset);
                tokio::select! {
                    _ = &mut run_fut => {}
                    _ = tokio::signal::ctrl_c() => std::process::exit(130),
                }
                true
            }
        };

        let stats = tickets.stats();
        let outcome = match tickets.tickets().last().map(|t| t.status()) {
            Some("done") => "completed",
            Some("failed") => "failed",
            _ if cancelled => "cancelled",
            _ => "incomplete",
        };

        let steps = stats.steps().saturating_sub(prev_steps);
        let requests = stats.requests().saturating_sub(prev_requests);
        let tool_calls = stats.tool_calls().saturating_sub(prev_tool_calls);
        let input = stats.input_tokens().saturating_sub(prev_input);
        let output = stats.output_tokens().saturating_sub(prev_output);
        prev_steps = stats.steps();
        prev_requests = stats.requests();
        prev_tool_calls = stats.tool_calls();
        prev_input = stats.input_tokens();
        prev_output = stats.output_tokens();

        if midstream.swap(false, Ordering::Relaxed) {
            eprintln!();
        }
        eprintln!(
            "{}{outcome} · {steps} steps · {requests} requests · {tool_calls} tools · {input} in / {output} out{}",
            style.dim, style.reset,
        );
    }
}

fn announce_assistant(style: &Style) {
    print!("\n{}agent ›{} ", style.agent, style.reset);
    let _ = io::stdout().flush();
}

fn print_event(event: &Event, style: &Style, test_window: Option<u64>, midstream: &AtomicBool) {
    // Emit a single leading newline only when streamed model text just
    // landed on stdout without a trailing newline; subsequent events
    // print directly on their own line.
    let break_stream = || {
        if midstream.swap(false, Ordering::Relaxed) {
            eprintln!();
        }
    };
    match &event.kind {
        EventKind::TextChunkReceived { content } => {
            print!("{content}");
            let _ = io::stdout().flush();
            midstream.store(true, Ordering::Relaxed);
        }
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => {
            break_stream();
            let arg = input["pattern"]
                .as_str()
                .or_else(|| input["path"].as_str())
                .or_else(|| input["query"].as_str())
                .unwrap_or("");
            if arg.is_empty() {
                eprintln!("{}· {tool_name}{}", style.dim, style.reset);
            } else {
                eprintln!("{}· {tool_name}({arg}){}", style.dim, style.reset);
            }
        }
        EventKind::ToolCallFailed {
            tool_name, message, ..
        } => {
            break_stream();
            eprintln!("{}✗ {tool_name}: {message}{}", style.red, style.reset);
        }
        EventKind::RequestFinished { usage, .. } => {
            if let Some(window) = test_window {
                break_stream();
                let used = usage.input_tokens;
                let remaining = window.saturating_sub(used);
                let threshold = window.saturating_mul(7) / 10;
                let (marker, color) = if used >= threshold {
                    ("⚠", style.red)
                } else {
                    ("·", style.dim)
                };
                eprintln!(
                    "{color}{marker} {used} / {window} tokens used ({remaining} left, warn at {threshold}){reset}",
                    reset = style.reset,
                );
            }
        }
        EventKind::ContextCompacted {
            tokens,
            threshold,
            reason,
            ..
        } => {
            break_stream();
            eprintln!(
                "{}⚠ compact {tokens}/{threshold} ({reason:?}){}",
                style.red, style.reset,
            );
        }
        EventKind::RequestFailed { message, .. } => {
            break_stream();
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!("{}✗ request failed: {short}{}", style.red, style.reset);
        }
        EventKind::RequestRetried {
            attempt,
            max_attempts,
            message,
            ..
        } => {
            break_stream();
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!(
                "{}↻ retry {attempt}/{max_attempts}: {short}{}",
                style.dim, style.reset,
            );
        }
        EventKind::SchemaRetried {
            attempt,
            max_attempts,
            message,
        } => {
            break_stream();
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!(
                "{}↻ schema retry {attempt}/{max_attempts}: {short}{}",
                style.dim, style.reset,
            );
        }
        EventKind::PolicyViolated { kind, limit } => {
            break_stream();
            eprintln!(
                "{}✗ policy {kind:?} (limit {limit}){}",
                style.red, style.reset,
            );
        }
        _ => {}
    }
}

async fn read_line(prompt: &str) -> Option<String> {
    print!("{prompt}");
    io::stdout().flush().ok()?;
    tokio::task::spawn_blocking(|| io::stdin().lines().next()?.ok().map(|s| s.trim().into()))
        .await
        .ok()
        .flatten()
}

#[derive(Clone)]
struct Style {
    dim: &'static str,
    user: &'static str,
    agent: &'static str,
    red: &'static str,
    reset: &'static str,
}

impl Style {
    fn detect() -> Self {
        if io::stdout().is_terminal() && io::stderr().is_terminal() {
            Self {
                dim: "\x1b[2m",
                user: "\x1b[1;33m",
                agent: "\x1b[1;36m",
                red: "\x1b[31m",
                reset: "\x1b[0m",
            }
        } else {
            Self {
                dim: "",
                user: "",
                agent: "",
                red: "",
                reset: "",
            }
        }
    }
}
