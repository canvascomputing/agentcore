//! End-to-end: a real LLM uses `ListDirectoryTool` to enumerate a tempdir
//! with known files and subdirectories, and reports the split via a
//! schema-validated JSON object.

use std::fs;

use super::common;

use agentwerk::tools::ListDirectoryTool;
use agentwerk::{Agent, Runnable, Schema, TicketSystem};

#[tokio::test]
async fn separates_files_and_directories() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let dir = tempfile::tempdir()?;
    let root = dir.path();

    for name in ["alpha.txt", "beta.txt", "gamma.txt"] {
        fs::write(root.join(name), "x\n")?;
    }
    for name in ["logs", "cache"] {
        fs::create_dir(root.join(name))?;
    }

    let schema = Schema::parse(serde_json::json!({
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Basenames of regular files at the top level."
            },
            "directories": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Basenames of subdirectories at the top level."
            }
        },
        "required": ["files", "directories"]
    }))?;

    let tickets = TicketSystem::new().max_steps(10);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .working_dir(root)
        .role(
            "Step 1: call `list_directory_tool` with `path: \".\"` to see the \
             working directory's top level. \
             Step 2: immediately call `write_result_tool` with `result` set \
             to a JSON object in exactly this shape: \
             {\"files\": [\"<basename>\", ...], \"directories\": [\"<basename>\", ...]}. \
             Pass the result as a JSON value, not a JSON-encoded string. \
             Never prose, never a bullet list, never a sentence. Do not output \
             any text outside of tool calls.",
        )
        .tool(ListDirectoryTool);
    tickets.add(agent);
    tickets.task_schema(
        "List the top-level entries in the working directory, separating files from directories.",
        schema,
    );

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    assert!(
        tickets.stats().tool_calls() >= 1,
        "agent must call at least one tool"
    );

    let response = common::last_result_string(&results);
    let json: serde_json::Value = serde_json::from_str(&response)?;
    let mut files = sorted_basenames(&json["files"]);
    let mut dirs = sorted_basenames(&json["directories"]);
    files.sort();
    dirs.sort();
    assert_eq!(
        files,
        vec![
            "alpha.txt".to_string(),
            "beta.txt".to_string(),
            "gamma.txt".to_string()
        ],
        "model reported wrong file set"
    );
    assert_eq!(
        dirs,
        vec!["cache".to_string(), "logs".to_string()],
        "model reported wrong directory set"
    );

    Ok(())
}

fn sorted_basenames(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .expect("expected JSON array")
        .iter()
        .filter_map(|v| v.as_str())
        .map(|s| {
            std::path::Path::new(s)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| s.to_string())
        })
        .collect()
}
