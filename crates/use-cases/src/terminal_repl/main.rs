//! Interactive terminal chat with an agent. Demonstrates `Agent::spawn` + `AgentHandle` for back-and-forth conversation against a live LLM.

use std::io::{self, Write};
use std::sync::Arc;

use agentwerk::{
    Agent, AgentOutput, AgenticError, Event, EventKind, GlobTool, GrepTool, ListDirectoryTool,
    ReadFileTool,
};
use tokio::sync::Notify;

const IDENTITY: &str = "You are a local-repo search assistant. Help the user explore this \
     repository using glob, grep, list_directory, and read_file. Cite file:line when \
     referencing code. Do not invent facts: if unsure, search first.";

#[tokio::main]
async fn main() {
    eprintln!("agentwerk REPL: /exit to quit, Ctrl-C to cancel.\n");
    let Some(first) = read_line("> ").await else {
        return;
    };
    if first.is_empty() || first == "/exit" || first == "/quit" {
        return;
    }

    let idle = Arc::new(Notify::new());
    let handler_idle = idle.clone();
    let (running, output) = Agent::new()
        .name("orchestrator")
        .provider_from_env()
        .expect("LLM provider required")
        .identity_prompt(IDENTITY)
        .instruction_prompt(first)
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(ListDirectoryTool)
        .tool(ReadFileTool)
        .event_handler(Arc::new(move |e: Event| print_event(&e, &handler_idle)))
        .spawn();

    tokio::pin!(output);
    let mut early_result: Option<Result<AgentOutput, AgenticError>> = None;

    'session: loop {
        tokio::select! {
            _ = idle.notified() => {}
            // Loop exited on its own (success or error) — don't wait for an
            // idle notification that will never come.
            res = &mut output => {
                early_result = Some(res);
                break 'session;
            }
            _ = tokio::signal::ctrl_c() => break 'session,
        }
        let line = tokio::select! {
            line = read_line("> ") => line,
            _ = tokio::signal::ctrl_c() => { eprintln!("^C"); None }
        };
        let Some(line) = line else { break };
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/quit" {
            break;
        }
        running.send(line);
    }

    running.cancel();
    let result = match early_result {
        Some(r) => r,
        None => (&mut output).await,
    };
    match result {
        Ok(o) => eprintln!(
            "[ended: {:?}, turns={}, tokens={}in/{}out]",
            o.status, o.statistics.turns, o.statistics.input_tokens, o.statistics.output_tokens,
        ),
        Err(e) => {
            let msg = e.to_string();
            let short = msg.split_once(':').map(|(h, _)| h).unwrap_or(&msg);
            eprintln!("[error: {short}]");
        }
    }
    std::process::exit(0);
}

fn print_event(event: &Event, idle: &Arc<Notify>) {
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
            eprintln!("· {tool_name}({arg})");
        }
        EventKind::ToolCallError {
            tool_name, error, ..
        } => eprintln!("✗ {tool_name}: {error}"),
        EventKind::RequestRetried {
            attempt,
            max_attempts,
            error,
        } => {
            let short = error.split_once(':').map(|(h, _)| h).unwrap_or(error);
            eprintln!("↻ retry {attempt}/{max_attempts}: {short}");
        }
        EventKind::RequestError { error } => {
            let short = error.split_once(':').map(|(h, _)| h).unwrap_or(error);
            eprintln!("✗ request failed: {short}");
        }
        EventKind::AgentPaused | EventKind::AgentFinished { .. } => {
            println!();
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
