//! End-to-end: a real LLM uses `WriteFileTool` to create a file containing
//! a fresh random token, and we verify the file landed on disk with the
//! expected contents. No JSON schema — the disk state is the assertion.

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::common;

use agentwerk::tools::WriteFileTool;
use agentwerk::{Agent, Runnable, TicketSystem};

#[tokio::test]
async fn creates_file_with_token() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let token = ten_digit_token();
    let dir = tempfile::tempdir()?;
    let root = dir.path();

    let tickets = TicketSystem::new().max_steps(10);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .working_dir(root)
        .role(
            "Step 1: call `write_file_tool` to create exactly the file the user \
             asks for, with exactly the content they specify (and nothing else). \
             Step 2: immediately call `write_result_tool` to settle the \
             ticket. Do not write any prose — your only output must be tool \
             calls.",
        )
        .tool(WriteFileTool);
    tickets.add(agent);
    tickets.task(format!(
        "Create a file named `report.md` in the working directory containing \
         exactly the line `token={token}`."
    ));

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    assert!(
        tickets.stats().tool_calls() >= 1,
        "agent must call at least one tool"
    );

    let path = root.join("report.md");
    assert!(
        path.exists(),
        "expected `report.md` to be created at {path:?}"
    );
    let content = fs::read_to_string(&path)?;
    assert!(
        content.contains(&format!("token={token}")),
        "report.md does not contain `token={token}`; got:\n{content}"
    );

    Ok(())
}

fn ten_digit_token() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    1_000_000_000 + (nanos.wrapping_mul(2_654_435_761) % 9_000_000_000)
}
