//! End-to-end: a real LLM is asked to find every `lib.rs` anywhere in a
//! nested project tree. The role does NOT name `glob_tool` and does NOT
//! describe its argument shape. Proves the tool's *description* is good
//! enough for a model to (1) pick filename pattern matching, and (2)
//! produce a recursive `**/lib.rs` pattern instead of a flat `*.rs` —
//! which would miss every nested file.

use std::fs;
use std::sync::{Arc, Mutex};

use super::common;

use agentwerk::event::EventKind;
use agentwerk::tools::{GlobTool, GrepTool, ListDirectoryTool, ReadFileTool};
use agentwerk::{default_logger, Agent, Event, Runnable, TicketSystem};

#[derive(Clone)]
struct CapturedCall {
    name: String,
    input: serde_json::Value,
    output: Option<String>,
}

#[tokio::test]
async fn finds_every_lib_rs_in_nested_tree() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let (provider, model) = common::build_provider();

    let dir = tempfile::tempdir()?;
    let root = dir.path();

    // Two crates with `lib.rs` at different depths, plus look-alike files
    // (`main.rs`, `mod.rs`, `lib.md`) and one `lib.rs` two levels deep.
    fs::create_dir_all(root.join("crate_a/src"))?;
    fs::create_dir_all(root.join("crate_a/tests"))?;
    fs::create_dir_all(root.join("crate_b/src/internal"))?;
    fs::write(root.join("crate_a/src/lib.rs"), "// crate_a lib\n")?;
    fs::write(root.join("crate_a/src/main.rs"), "fn main() {}\n")?;
    fs::write(
        root.join("crate_a/tests/test_basic.rs"),
        "#[test] fn smoke() {}\n",
    )?;
    fs::write(root.join("crate_b/src/lib.rs"), "// crate_b lib\n")?;
    fs::write(root.join("crate_b/src/internal/mod.rs"), "// internal\n")?;
    fs::write(
        root.join("crate_b/src/internal/lib.rs"),
        "// nested lib in crate_b\n",
    )?;
    fs::write(root.join("README.md"), "# project\n")?;
    fs::write(root.join("Cargo.toml"), "[workspace]\n")?;
    fs::write(root.join("notes_lib.md"), "# notes\n")?; // distractor

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
             Use the available tools — pick whichever one fits the question. \
             When you have the answer, settle the ticket via \
             `write_result_tool`.",
        )
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(ListDirectoryTool)
        .tool(ReadFileTool)
        .event_handler(event_handler);
    tickets.add(agent);
    tickets.task(
        "Find every `lib.rs` file anywhere in the project tree, including nested directories.",
    );

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    let recorded = calls.lock().unwrap().clone();

    // The model must call glob_tool with a recursive pattern (`**`) — a
    // flat `*.rs` would miss every nested file. Either `**/lib.rs` (tight)
    // or `**/*.rs` (broad, then filter) is acceptable; both prove the
    // model picked up `**` semantics from the description.
    let glob_call = recorded
        .iter()
        .find(|c| {
            c.name == "glob_tool"
                && c.input
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .is_some_and(|p| p.contains("**") && p.contains(".rs"))
        })
        .unwrap_or_else(|| {
            panic!(
                "model should call `glob_tool` with a recursive (`**`) pattern \
                 over Rust files; instead called: {:?}",
                recorded
                    .iter()
                    .map(|c| (&c.name, &c.input))
                    .collect::<Vec<_>>()
            )
        });

    // The tool's output must contain every nested `lib.rs` — proves the
    // model's `**` pattern actually exercised the recursion through two
    // levels of nesting (`crate_b/src/internal/lib.rs`).
    let output = glob_call
        .output
        .as_deref()
        .expect("glob_tool call should have produced output");
    for expected in [
        "crate_a/src/lib.rs",
        "crate_b/src/lib.rs",
        "crate_b/src/internal/lib.rs",
    ] {
        assert!(
            output.contains(expected),
            "glob_tool output should reach nested `{expected}`; got: {output:?}"
        );
    }
    let _ = results;

    Ok(())
}
