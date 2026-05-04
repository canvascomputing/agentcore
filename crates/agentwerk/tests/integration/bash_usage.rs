//! End-to-end: a real LLM uses `BashTool` to run commands and settles
//! its ticket with a JSON result validated against the ticket schema.

use super::common;

use agentwerk::tools::{BashTool, ManageTicketsTool};
use agentwerk::{Agent, Runnable, Schema, TicketSystem};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let schema = Schema::parse(serde_json::json!({
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Files in the directory"
            },
            "line_count": {
                "type": "integer",
                "description": "Number of lines in Cargo.toml"
            }
        },
        "required": ["files", "line_count"]
    }))?;

    let ls = BashTool::new("ls", "ls*").read_only(true);
    let cat = BashTool::new("cat", "cat *").read_only(true);
    let wc = BashTool::new("wc", "wc *").read_only(true);

    let tickets = TicketSystem::new().max_steps(10);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .role(
            "You have three shell tools (ls, cat, wc) and the ticket-management tool. \
             No other tools are available. Settle the ticket via `manage_tickets_tool` \
             with `action: \"done\"` and `result` set to a JSON string matching the \
             ticket's schema.",
        )
        .tool(ls)
        .tool(cat)
        .tool(wc)
        .tool(ManageTicketsTool);
    tickets.add(agent);
    tickets.task_schema(
        "List the files in the current directory, read the Cargo.toml file, \
         and count its lines. Report the result.",
        schema,
    );

    let result = tickets.run_dry().await;
    common::print_result(&result, tickets.stats());

    let json: serde_json::Value = serde_json::from_str(&result)?;
    assert!(json["line_count"].as_u64().unwrap_or(0) > 1);
    assert!(json["files"].as_array().map_or(0, |a| a.len()) > 1);

    Ok(())
}
