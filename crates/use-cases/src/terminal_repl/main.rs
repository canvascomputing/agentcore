//! Interactive terminal chat with an agent. Demonstrates `Agent::spawn` + `AgentHandle` for back-and-forth conversation against a live LLM.

use std::io::{self, Write};
use std::sync::Arc;

use agentwerk::{
    Agent, AgentEvent, AgentEventKind, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
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
        .event_handler(Arc::new(move |e: AgentEvent| {
            print_event(&e, &handler_idle)
        }))
        .spawn();

    loop {
        tokio::select! {
            _ = idle.notified() => {}
            _ = tokio::signal::ctrl_c() => break,
        }
        if running.is_stopped() {
            break;
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
    if let Ok(o) = output.await {
        eprintln!(
            "[ended: {:?}, turns={}, tokens={}in/{}out]",
            o.status, o.statistics.turns, o.statistics.input_tokens, o.statistics.output_tokens,
        );
    }
    std::process::exit(0);
}

fn print_event(event: &AgentEvent, idle: &Arc<Notify>) {
    match &event.kind {
        AgentEventKind::ResponseTextChunk { content } => {
            print!("{content}");
            let _ = io::stdout().flush();
        }
        AgentEventKind::ToolCallStart {
            tool_name, input, ..
        } => {
            let arg = input["pattern"]
                .as_str()
                .or_else(|| input["path"].as_str())
                .unwrap_or("");
            eprintln!("· {tool_name}({arg})");
        }
        AgentEventKind::ToolCallError {
            tool_name, error, ..
        } => eprintln!("✗ {tool_name}: {error}"),
        AgentEventKind::AgentIdle | AgentEventKind::AgentEnd { .. } => {
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
