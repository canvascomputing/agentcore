//! Interactive terminal chat. One `TicketSystem` + `Agent` + `Memory`
//! lives for the whole session; each input line enqueues a ticket and
//! runs `run_dry` once. The agent has `memory(...)` bound, so durable
//! facts the model writes via `memory_tool` survive across turns and
//! across process restarts (the file lives at `./.agentwerk-memory/`).
//! The model's response streams to stdout via
//! `EventKind::TextChunkReceived`, and the agent finishes its ticket
//! via the auto-registered `write_result_tool`. Slash commands:
//! `/exit` quits, `/memory` prints the current memory contents,
//! `/clear` resets them. Ctrl-C at the prompt exits with code 130;
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

use agentwerk::providers::{model_from_env, provider_from_env};
use agentwerk::tools::{GlobTool, GrepTool, ListDirectoryTool, ReadFileTool};
use agentwerk::{Agent, Event, EventKind, Memory, Status, TicketSystem};

const ROLE: &str = include_str!("prompts/repl.role.md");

#[tokio::main]
async fn main() {
    let style = Style::detect();
    eprintln!(
        "{}agentwerk REPL — /exit /history /clear, Ctrl-C to cancel.{}",
        style.dim, style.reset,
    );

    let provider = provider_from_env().expect("LLM provider required");
    let model = model_from_env().expect("model name required");
    let role = ROLE.trim();

    let user_prompt = format!("\n{}you ›{} ", style.user, style.reset);

    let cancel = Arc::new(AtomicBool::new(false));
    let event_style = style.clone();
    let handler: Arc<dyn Fn(Event) + Send + Sync> =
        Arc::new(move |e: Event| print_event(&e, &event_style));

    let tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&cancel))
        .max_steps(40);

    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let memory_dir = cwd.join(".agentwerk");
    let memory = Memory::open(&memory_dir).expect("open memory store");

    let _agent = tickets.add(
        Agent::new()
            .name("orchestrator")
            .provider(Arc::clone(&provider))
            .model(&model)
            .role(role)
            .working_dir(&cwd)
            .tool(GlobTool)
            .tool(GrepTool)
            .tool(ListDirectoryTool)
            .tool(ReadFileTool)
            .event_handler(handler)
            .memory(&memory),
    );

    let mut prev_steps: u64 = 0;
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
        if line == "/memory" {
            let entries = memory.entries();
            let rendered = if entries.is_empty() {
                "(memory empty)".to_string()
            } else {
                entries.join("\n\n")
            };
            eprintln!("{}{}{}", style.dim, rendered, style.reset);
            continue;
        }
        if line == "/clear" {
            memory.rewrite(Vec::new()).ok();
            eprintln!("{}memory cleared{}", style.dim, style.reset);
            continue;
        }

        announce_assistant(&style);
        cancel.store(false, Ordering::Relaxed);
        tickets.task(line);

        let run_fut = tickets.run_dry();
        tokio::pin!(run_fut);
        let cancelled = tokio::select! {
            _ = &mut run_fut => false,
            _ = tokio::signal::ctrl_c() => {
                cancel.store(true, Ordering::Relaxed);
                eprintln!("\n{}cancelling…{}", style.dim, style.reset);
                tokio::select! {
                    _ = &mut run_fut => {}
                    _ = tokio::signal::ctrl_c() => std::process::exit(130),
                }
                true
            }
        };

        let stats = tickets.stats();
        let outcome = match tickets.tickets().last().map(|t| t.status()) {
            Some(Status::Done) => "completed",
            Some(Status::Failed) => "failed",
            _ if cancelled => "cancelled",
            _ => "incomplete",
        };

        let steps = stats.steps().saturating_sub(prev_steps);
        let input = stats.input_tokens().saturating_sub(prev_input);
        let output = stats.output_tokens().saturating_sub(prev_output);
        prev_steps = stats.steps();
        prev_input = stats.input_tokens();
        prev_output = stats.output_tokens();

        if cancelled {
            println!();
        }
        eprintln!(
            "\n{}— {} · {} steps · {} in / {} out{}",
            style.dim, outcome, steps, input, output, style.reset,
        );
    }
}

fn announce_assistant(style: &Style) {
    print!("\n{}agent ›{} ", style.agent, style.reset);
    let _ = io::stdout().flush();
}

fn print_event(event: &Event, style: &Style) {
    match &event.kind {
        EventKind::TextChunkReceived { content } => {
            print!("{content}");
            let _ = io::stdout().flush();
        }
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => {
            let arg = input["pattern"]
                .as_str()
                .or_else(|| input["path"].as_str())
                .or_else(|| input["query"].as_str())
                .unwrap_or("");
            if arg.is_empty() {
                eprintln!("\n{}· {tool_name}{}", style.dim, style.reset);
            } else {
                eprintln!("\n{}· {tool_name}({arg}){}", style.dim, style.reset);
            }
        }
        EventKind::ToolCallFailed {
            tool_name, message, ..
        } => eprintln!("\n{}✗ {tool_name}: {message}{}", style.red, style.reset),
        EventKind::RequestFailed { message, .. } => {
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!("\n{}✗ request failed: {short}{}", style.red, style.reset);
        }
        EventKind::RequestRetried {
            attempt,
            max_attempts,
            message,
            ..
        } => {
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!(
                "\n{}↻ retry {attempt}/{max_attempts}: {short}{}",
                style.dim, style.reset,
            );
        }
        EventKind::SchemaRetried {
            attempt,
            max_attempts,
            message,
        } => {
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!(
                "\n{}↻ schema retry {attempt}/{max_attempts}: {short}{}",
                style.dim, style.reset,
            );
        }
        EventKind::PolicyViolated { kind, limit } => {
            eprintln!(
                "\n{}✗ policy {kind:?} (limit {limit}){}",
                style.red, style.reset
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
