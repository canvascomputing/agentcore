//! End-to-end: a real LLM uses `EditFileTool` to swap one substring in a
//! pre-populated file while leaving the rest untouched. We assert on the
//! final file contents, not on the model's text response.

use std::fs;

use super::common;

use agentwerk::tools::EditFileTool;
use agentwerk::{Agent, TicketSystem};

const ORIGINAL: &str = "setting=old_value\nother=keep_me\n";

#[tokio::test]
async fn replaces_substring_in_place() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let dir = tempfile::tempdir()?;
    let root = dir.path();
    let path = root.join("config.txt");
    fs::write(&path, ORIGINAL)?;

    let tickets = TicketSystem::new().max_steps(10);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .working_dir(root)
        .role(
            "Step 1: call `edit_file_tool` to perform an exact substring \
             replacement in the existing file. Do not rewrite the whole file. \
             Step 2: immediately call `write_result_tool` to settle the \
             ticket. Do not write any prose — your only output must be tool \
             calls.",
        )
        .tool(EditFileTool);
    tickets.agent(agent);
    tickets.task(
        "In `config.txt`, change the substring `old_value` to `new_value`. \
         Leave the rest of the file untouched.",
    );

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    assert!(
        tickets.stats().tool_calls() >= 1,
        "agent must call at least one tool"
    );

    let content = fs::read_to_string(&path)?;
    assert!(
        content.contains("setting=new_value"),
        "expected `setting=new_value`; got:\n{content}"
    );
    assert!(
        content.contains("other=keep_me"),
        "untouched line `other=keep_me` was lost; got:\n{content}"
    );
    assert!(
        !content.contains("old_value"),
        "`old_value` should have been replaced; got:\n{content}"
    );

    Ok(())
}
