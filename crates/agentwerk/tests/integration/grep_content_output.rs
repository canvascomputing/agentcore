//! End-to-end: a real LLM agent is asked to locate a unique string
//! buried deep inside a long line — past column 100. The role does NOT
//! hint at grep or content mode. Proves the agent can find the match
//! and that `grep_tool` reports the correct column position.

use std::fs;
use std::sync::{Arc, Mutex};

use super::common;

use agentwerk::event::EventKind;
use agentwerk::tools::{GlobTool, GrepTool, ListDirectoryTool, ReadFileTool};
use agentwerk::{default_logger, Agent, Event, TicketSystem};

const NEEDLE: &str = "XYZZY_PLUGH_42";

#[derive(Clone)]
struct CapturedCall {
    name: String,
    input: serde_json::Value,
    output: Option<String>,
}

#[tokio::test]
async fn finds_string_buried_deep_in_line() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let dir = tempfile::tempdir()?;
    let root = dir.path();
    fs::create_dir_all(root.join("src"))?;

    // Decoy files — none containing the needle.
    fs::write(root.join("src/main.rs"), "fn main() { run(); }\n")?;
    fs::write(root.join("src/server.rs"), "pub fn run() { loop {} }\n")?;

    // Bury the needle past column 100 inside a long line.
    let filler = "x".repeat(120);
    let target_line = format!("const DATA: &str = \"{filler}{NEEDLE}\";");
    fs::write(
        root.join("src/config.rs"),
        format!("// config\n{target_line}\n"),
    )?;

    // Derive the expected column from what we just wrote.
    let expected_col = target_line.find(NEEDLE).unwrap() + 1;
    assert!(
        expected_col > 100,
        "test setup: needle should be past column 100"
    );

    let calls: Arc<Mutex<Vec<CapturedCall>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = Arc::clone(&calls);
    let logger = default_logger();
    let event_handler = Arc::new(move |e: Event| {
        match &e.kind {
            EventKind::ToolCallStarted {
                tool_name, input, ..
            } => {
                collected.lock().unwrap().push(CapturedCall {
                    name: tool_name.clone(),
                    input: input.clone(),
                    output: None,
                });
            }
            EventKind::ToolCallFinished {
                tool_name, output, ..
            } => {
                let mut g = collected.lock().unwrap();
                if let Some(slot) = g
                    .iter_mut()
                    .rev()
                    .find(|c| &c.name == tool_name && c.output.is_none())
                {
                    slot.output = Some(output.clone());
                }
            }
            _ => {}
        }
        logger(e);
    });

    let tickets = TicketSystem::new().max_steps(10);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .working_dir(root)
        .role(
            "Investigate the working directory and answer the user's question. \
             Use the available tools — pick whichever one fits. \
             When you have the answer, settle the ticket via \
             `write_result_tool`.",
        )
        .tool(GrepTool)
        .tool(GlobTool)
        .tool(ListDirectoryTool)
        .tool(ReadFileTool)
        .event_handler(event_handler);
    tickets.agent(agent);
    tickets.task(format!(
        "Which source file contains the string `{NEEDLE}`? \
         Answer with the file path.",
    ));

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    let recorded = calls.lock().unwrap().clone();

    // The agent must have called grep_tool with the needle.
    let grep_call = recorded
        .iter()
        .find(|c| {
            c.name == "grep_tool"
                && c.input
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .is_some_and(|p| p.contains(NEEDLE))
        })
        .unwrap_or_else(|| {
            panic!(
                "agent should call `grep_tool` with `{NEEDLE}` in pattern; \
                 instead called: {:?}",
                recorded
                    .iter()
                    .map(|c| (&c.name, &c.input))
                    .collect::<Vec<_>>()
            )
        });

    let output = grep_call
        .output
        .as_deref()
        .expect("grep_tool call should have produced output");

    // Grep must find the needle in config.rs, not in any decoy.
    assert!(
        output.contains("config.rs"),
        "grep should find the needle in config.rs; got: {output:?}"
    );
    assert!(
        !output.contains("main.rs") && !output.contains("server.rs"),
        "grep should not match decoy files; got: {output:?}"
    );

    // If content mode was used, the column must be present.
    if grep_call
        .input
        .get("output_mode")
        .and_then(|v| v.as_str())
        .is_some_and(|m| m == "content")
    {
        let marker = format!(":2:{expected_col}: ");
        assert!(
            output.contains(&marker),
            "content output should include {marker}; got: {output:?}"
        );
    }

    // The agent's final answer should name config.rs.
    let answer = common::last_result_string(&results);
    assert!(
        answer.contains("config.rs"),
        "agent should report config.rs; got: {answer:?}"
    );

    Ok(())
}
