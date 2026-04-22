//! Interactive terminal chat with an agent. Demonstrates `Agent::spawn` + `AgentHandle` for back-and-forth conversation against a live LLM.

use std::io::{self, IsTerminal, Write};
use std::sync::Arc;

use agentwerk::{
    Agent, Error, Event, EventKind, GlobTool, GrepTool, ListDirectoryTool, Output, ReadFileTool,
    Status,
};
use tokio::sync::Notify;

const IDENTITY: &str = "You are a local-repo search assistant. Help the user explore this \
     repository using glob, grep, list_directory, and read_file. Cite file:line when \
     referencing code. Do not invent facts: if unsure, search first.";

#[tokio::main]
async fn main() {
    let style = Style::detect();
    eprintln!(
        "{}agentwerk REPL — /exit to quit, Ctrl-C to cancel.{}",
        style.dim, style.reset,
    );

    let user_prompt = format!("\n{}you ›{} ", style.user, style.reset);
    let Some(first) = read_line(&user_prompt).await else {
        return;
    };
    if first.is_empty() || first == "/exit" || first == "/quit" {
        return;
    }

    let idle = Arc::new(Notify::new());
    let handler_idle = idle.clone();
    let handler_style = style.clone();
    announce_assistant(&style);
    let (running, output) = Agent::new()
        .name("orchestrator")
        .provider_from_env()
        .expect("LLM provider required")
        .model_from_env()
        .expect("model name required")
        .identity_prompt(IDENTITY)
        .instruction_prompt(first)
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(ListDirectoryTool)
        .tool(ReadFileTool)
        .event_handler(Arc::new(move |e: Event| {
            print_event(&e, &handler_idle, &handler_style)
        }))
        .spawn();

    tokio::pin!(output);
    let mut early_result: Option<Result<Output, Error>> = None;

    'session: loop {
        tokio::select! {
            _ = idle.notified() => {}
            res = &mut output => {
                early_result = Some(res);
                break 'session;
            }
            _ = tokio::signal::ctrl_c() => break 'session,
        }
        let line = tokio::select! {
            line = read_line(&user_prompt) => line,
            _ = tokio::signal::ctrl_c() => { eprintln!("{}^C{}", style.dim, style.reset); None }
        };
        let Some(line) = line else { break };
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/quit" {
            break;
        }
        announce_assistant(&style);
        running.send(line);
    }

    running.cancel();
    let result = match early_result {
        Some(r) => r,
        None => (&mut output).await,
    };
    match result {
        Ok(o) => eprintln!(
            "\n{}— {} · {} turns · {} in / {} out{}",
            style.dim,
            status_label(&o.status),
            o.statistics.turns,
            o.statistics.input_tokens,
            o.statistics.output_tokens,
            style.reset,
        ),
        Err(e) => {
            let msg = e.to_string();
            let short = msg.split_once(':').map(|(h, _)| h).unwrap_or(&msg);
            eprintln!("{}error:{} {short}", style.red, style.reset);
        }
    }
    std::process::exit(0);
}

fn announce_assistant(style: &Style) {
    print!("\n{}agent ›{} ", style.agent, style.reset);
    let _ = io::stdout().flush();
}

fn status_label(status: &Status) -> &'static str {
    match status {
        Status::Completed => "completed",
        Status::Cancelled => "cancelled",
        Status::TurnLimitReached { .. } => "turn limit reached",
        Status::InputBudgetExhausted { .. } => "input budget exhausted",
        Status::OutputBudgetExhausted { .. } => "output budget exhausted",
        Status::HaltRequested => "halted",
    }
}

fn print_event(event: &Event, idle: &Arc<Notify>, style: &Style) {
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
                .unwrap_or("");
            if arg.is_empty() {
                eprintln!("\n{}· {tool_name}{}", style.dim, style.reset);
            } else {
                eprintln!("\n{}· {tool_name}({arg}){}", style.dim, style.reset);
            }
        }
        EventKind::ToolCallError {
            tool_name, error, ..
        } => eprintln!("\n{}✗ {tool_name}: {error}{}", style.red, style.reset),
        EventKind::RequestRetried {
            attempt,
            max_attempts,
            error,
        } => {
            let short = error.split_once(':').map(|(h, _)| h).unwrap_or(error);
            eprintln!(
                "\n{}↻ retry {attempt}/{max_attempts}: {short}{}",
                style.yellow, style.reset,
            );
        }
        EventKind::RequestError { error } => {
            let short = error.split_once(':').map(|(h, _)| h).unwrap_or(error);
            eprintln!("\n{}✗ request failed: {short}{}", style.red, style.reset);
        }
        EventKind::AgentPaused | EventKind::AgentFinished { .. } => {
            idle.notify_one();
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
    yellow: &'static str,
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
                yellow: "\x1b[33m",
                red: "\x1b[31m",
                reset: "\x1b[0m",
            }
        } else {
            Self {
                dim: "",
                user: "",
                agent: "",
                yellow: "",
                red: "",
                reset: "",
            }
        }
    }
}
