//! Integration test: Agent-driven task management with persistence.
//!
//! Exercises the full stack: agent loop → tool calls → TaskStore/SessionStore → disk.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use agent::{
    AgentBuilder, AnthropicProvider, Event, LiteLlmProvider, LlmProvider, SessionStore, TaskStore,
    task_create_tool, task_list_tool, task_update_tool,
};

fn build_provider() -> Option<(Arc<dyn LlmProvider>, String)> {
    let client = reqwest::Client::new();
    if let Ok(url) = std::env::var("LITELLM_API_URL") {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, client).base_url(url)), model));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let mut p = AnthropicProvider::new(key, client);
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            p = p.base_url(url);
        }
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(p), model));
    }
    if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, client).base_url("http://localhost:4000".into())), model));
    }
    None
}

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some((provider, model)) = build_provider() else { eprintln!("SKIPPED: no provider"); return Ok(()); };

    let tmp = tempfile::tempdir()?;
    let base = tmp.path();

    let task_store = Arc::new(Mutex::new(TaskStore::open(base, "integration-test")));
    let session_store = SessionStore::new(base, "test-session");

    let output = AgentBuilder::new()
        .name("planner")
        .model(&model)
        .system_prompt("You are a project planner. Use the task tools to manage work items. Be concise.")
        .max_turns(10)
        .tool(task_create_tool(task_store.clone()))
        .tool(task_update_tool(task_store.clone()))
        .tool(task_list_tool(task_store.clone()))
        .provider(provider)
        .prompt(
            "Create two tasks: 'Design API' and 'Write tests'. \
             Then mark 'Design API' as Completed. \
             Finally list all tasks and summarize their status.",
        )
        .session_store(Arc::new(Mutex::new(session_store)))
        .event_handler(Arc::new(|event| match &event {
            Event::TextChunk { content, .. } => print!("{content}"),
            Event::ToolCallStart { tool_name, .. } => eprintln!("\n[tool] {tool_name}"),
            Event::ToolCallEnd { tool_name, output, is_error, .. } => {
                if *is_error {
                    eprintln!("[error] {tool_name}: {output}");
                } else {
                    eprintln!("[result] {}", &output[..output.len().min(120)]);
                }
            }
            Event::AgentEnd { turns, .. } => eprintln!("\n[done in {turns} turn(s)]"),
            _ => {}
        }))
        .cancel_signal(Arc::new(AtomicBool::new(false)))
        .run()
        .await?;

    println!("\n\n--- Verification ---");

    let verify_store = TaskStore::open(base, "integration-test");
    let tasks = verify_store.list()?;
    println!("Tasks on disk: {}", tasks.len());
    for task in &tasks {
        println!("  #{} [{:?}] {}", task.id, task.status, task.subject);
    }

    let entries = SessionStore::load(base, "test-session")?;
    println!("Transcript entries: {}", entries.len());

    println!("\n--- Cost ---");
    println!("Cost: ${:.4}", output.statistics.costs);

    Ok(())
}
