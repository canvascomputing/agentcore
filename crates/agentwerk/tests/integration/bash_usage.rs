use super::common;

use agentwerk::{Agent, BashTool};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let output_schema = serde_json::json!({
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
    });

    let ls = BashTool::new("ls", "ls*").read_only(true);
    let cat = BashTool::new("cat", "cat *").read_only(true);
    let wc = BashTool::new("wc", "wc *").read_only(true);

    let output = Agent::new()
        .provider(provider)
        .model(&model)
        .identity_prompt(
            "You have three shell tools: ls, cat, and wc. \
             No other tools are available. Use them to accomplish the task.",
        )
        .instruction_prompt(
            "List the files in the current directory, read the Cargo.toml file, \
             and count its lines. Report the result.",
        )
        .tool(ls)
        .tool(cat)
        .tool(wc)
        .output_schema(output_schema)
        .max_turns(10)
        .run()
        .await?;

    common::print_result(&output);

    let json = output.response.as_ref().expect("Expected structured JSON response");
    assert!(json["line_count"].as_u64().unwrap_or(0) > 1);
    assert!(json["files"].as_array().map_or(0, |a| a.len()) > 1);

    Ok(())
}
