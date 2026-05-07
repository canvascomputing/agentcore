//! End-to-end: a real LLM combines `GlobTool` and `ReadFileTool` to
//! explore a directory.

use super::common;

use agentwerk::tools::{GlobTool, ManageTicketsTool, ReadFileTool};
use agentwerk::{Agent, TicketSystem};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let tickets = TicketSystem::new().max_steps(10);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .role(
            "Explore the repository to answer the task. When you have an answer, \
             settle the ticket via `manage_tickets_tool` with `action: \"done\"` \
             and `result` set to your answer.",
        )
        .tool(ReadFileTool)
        .tool(GlobTool)
        .tool(ManageTicketsTool);
    tickets.agent(agent);
    tickets.task("Find all Rust source files and describe what this project does.");

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    assert!(tickets.stats().tool_calls() >= 1);

    Ok(())
}
