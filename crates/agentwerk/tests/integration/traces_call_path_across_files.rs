//! End-to-end: a real LLM combines grep + read_file across multiple files
//! to reconstruct a 3-hop call chain starting from a named entry function.
//! Decoy functions share names with real callees so the model must read
//! source, not just match identifiers, to resolve the chain.

use std::fs;

use super::common;

use agentwerk::tools::{GlobTool, GrepTool, ListDirectoryTool, ReadFileTool};
use agentwerk::{Agent, Runnable, Schema, TicketSystem};

#[tokio::test]
async fn traces_three_hop_call_path() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let dir = tempfile::tempdir()?;
    let root = dir.path();
    fs::create_dir_all(root.join("src"))?;

    fs::write(
        root.join("src/lib.rs"),
        "pub mod request;\n\
         pub mod handler;\n\
         pub mod decoy_a;\n\
         pub mod decoy_b;\n\
         \n\
         pub fn entry() { request::dispatch_request(); }\n",
    )?;
    fs::write(
        root.join("src/request.rs"),
        "use crate::handler;\n\
         \n\
         pub fn dispatch_request() { handler::serve_payload(); }\n",
    )?;
    fs::write(
        root.join("src/handler.rs"),
        "pub fn serve_payload() { println!(\"ok\"); }\n",
    )?;
    fs::write(
        root.join("src/decoy_a.rs"),
        "pub fn dispatch_request() { /* unrelated stub, never called from entry */ }\n",
    )?;
    fs::write(
        root.join("src/decoy_b.rs"),
        "pub fn handler() {}\n\
         pub fn entry_helper() {}\n",
    )?;
    fs::write(
        root.join("README.md"),
        "# project\n\nNotes about the entry point and the handler module.\n",
    )?;

    let schema = Schema::parse(serde_json::json!({
        "type": "object",
        "properties": {
            "call_path": {
                "type": "array",
                "description": "Ordered call chain from the starting function to the terminal function. Each hop names the function and the path of the file it is defined in.",
                "items": {
                    "type": "object",
                    "properties": {
                        "function": {
                            "type": "string",
                            "description": "Bare function name, e.g. `entry`."
                        },
                        "file": {
                            "type": "string",
                            "description": "Path of the source file that defines the function, relative to the working directory."
                        }
                    },
                    "required": ["function", "file"]
                }
            }
        },
        "required": ["call_path"]
    }))?;

    let tickets = TicketSystem::new().max_steps(15);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .working_dir(root)
        .role(
            "Investigate the working directory to answer the task. Use the \
             available read-only tools as you see fit. Multiple functions in \
             this codebase share names: rely on the source of each function, \
             not just its name, to decide what calls what. When you have the \
             full chain, finish the ticket via `write_result_tool` with the \
             result shaped to match the schema.",
        )
        .tool(GrepTool)
        .tool(GlobTool)
        .tool(ListDirectoryTool)
        .tool(ReadFileTool);
    tickets.add(agent);
    tickets.task_schema(
        "Starting from the function `entry`, follow each function call \
         through the source files and report the full ordered call chain \
         until you reach a function that does not call any other function \
         in this project.",
        schema,
    );

    let results = tickets.run_dry().await;
    common::print_result(&results, tickets.stats());

    assert!(
        tickets.stats().tool_calls() >= 2,
        "tracing a call path requires at least two read-only tool calls"
    );

    let response = common::last_result_string(&results);
    let json: serde_json::Value = serde_json::from_str(&response)?;
    let chain = json["call_path"]
        .as_array()
        .expect("call_path must be an array");

    let hops: Vec<(String, String)> = chain
        .iter()
        .map(|hop| {
            (
                hop["function"].as_str().unwrap_or("").to_string(),
                hop["file"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect();

    assert_eq!(hops.len(), 3, "expected exactly 3 hops, got {hops:?}");
    assert_eq!(hops[0].0, "entry");
    assert!(
        hops[0].1.ends_with("src/lib.rs"),
        "hop 0 file: {:?}",
        hops[0].1
    );
    assert_eq!(hops[1].0, "dispatch_request");
    assert!(
        hops[1].1.ends_with("src/request.rs"),
        "hop 1 must resolve to src/request.rs, not the decoy: {:?}",
        hops[1].1
    );
    assert_eq!(hops[2].0, "serve_payload");
    assert!(
        hops[2].1.ends_with("src/handler.rs"),
        "hop 2 file: {:?}",
        hops[2].1
    );

    Ok(())
}
