//! Integration test: Agent-driven task management with persistence.
//!
//! Exercises the full stack: agent loop → tool calls → TaskStore/SessionStore → disk.
//! Requires ANTHROPIC_API_KEY.
//!
//! Run: cargo run -p agent-core --example minimal_persistence

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use agent_core::{
    AgentBuilder, AgenticError, AnthropicProvider, CostTracker, Event, HttpTransport,
    InvocationContext, SessionStore, TaskStore, generate_agent_id, task_create_tool,
    task_list_tool, task_update_tool,
};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("Set ANTHROPIC_API_KEY environment variable");

    let transport: HttpTransport = Box::new(|url, headers, body| {
        let url = url.to_string();
        let headers: Vec<(String, String)> = headers
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Box::pin(async move {
            let client = reqwest::Client::new();
            let mut req = client.post(&url).json(&body);
            for (key, value) in &headers {
                req = req.header(key.as_str(), value.as_str());
            }
            let resp = req
                .send()
                .await
                .map_err(|e| AgenticError::Other(e.to_string()))?;
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| AgenticError::Other(e.to_string()))?;
            Ok(json)
        })
    });

    let provider = Arc::new(AnthropicProvider::new(api_key, transport));

    // --- Set up persistence in a temp directory ---
    let tmp = tempfile::tempdir()?;
    let base = tmp.path();

    let task_store = Arc::new(Mutex::new(TaskStore::open(base, "integration-test")));
    let session_store = SessionStore::new(base, "test-session");

    // --- Build agent with task tools ---
    let agent = AgentBuilder::new()
        .name("planner")
        .model("claude-sonnet-4-20250514")
        .system_prompt(
            "You are a project planner. Use the task tools to manage work items. Be concise.",
        )
        .max_turns(10)
        .tool(task_create_tool(task_store.clone()))
        .tool(task_update_tool(task_store.clone()))
        .tool(task_list_tool(task_store.clone()))
        .build()?;

    let cost_tracker = CostTracker::new();

    let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match &event {
        Event::Text { text, .. } => print!("{text}"),
        Event::ToolStart { tool, .. } => eprintln!("\n[tool] {tool}"),
        Event::ToolEnd { tool, result, is_error, .. } => {
            if *is_error {
                eprintln!("[error] {tool}: {result}");
            } else {
                eprintln!("[result] {}", &result[..result.len().min(120)]);
            }
        }
        Event::AgentEnd { turns, .. } => eprintln!("\n[done in {turns} turn(s)]"),
        _ => {}
    });

    let ctx = InvocationContext {
        input: "Create two tasks: 'Design API' and 'Write tests'. \
                Then mark 'Design API' as Completed. \
                Finally list all tasks and summarize their status."
            .into(),
        state: HashMap::new(),
        working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        provider,
        cost_tracker: cost_tracker.clone(),
        on_event,
        cancelled: Arc::new(AtomicBool::new(false)),
        session_store: Some(Arc::new(Mutex::new(session_store))),
        command_queue: None,
        agent_id: generate_agent_id("planner"),
    };

    // --- Run ---
    let _output = agent.run(ctx).await?;

    // --- Verify persistence ---
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
    println!("{}", cost_tracker.summary());

    Ok(())
}
