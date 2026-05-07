//! End-to-end: a real LLM drives three pattern-restricted `BashTool`
//! commands (`ls`, `cat`, `wc`) and settles its ticket with a
//! JSON result validated against the ticket schema.

use super::common;

use agentwerk::tools::BashTool;
use agentwerk::{Agent, Schema, TicketSystem};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let schema = Schema::parse(serde_json::json!({
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Files in the working directory"
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
            "Step 1: call `ls`, `cat Cargo.toml`, and `wc -l Cargo.toml` to \
             gather the file list and Cargo.toml line count. \
             Step 2: immediately call `write_result_tool` with `result` \
             set to a JSON object in exactly this shape: \
             {\"files\": [\"<filename>\", ...], \"line_count\": <integer>}. \
             Pass the result as a JSON value, not a JSON-encoded string. \
             Never prose, never a sentence.",
        )
        .tool(ls)
        .tool(cat)
        .tool(wc);
    tickets.add(agent);
    tickets.task_schema(
        "List the files in the current directory, read the Cargo.toml file, \
         and count its lines. Report the result.",
        schema,
    );

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    let response = common::last_result_string(&results);
    let json: serde_json::Value = serde_json::from_str(&response)?;
    assert!(json["line_count"].as_u64().unwrap_or(0) > 1);
    assert!(json["files"].as_array().map_or(0, |a| a.len()) > 1);

    Ok(())
}
