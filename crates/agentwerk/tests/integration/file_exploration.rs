//! End-to-end: a real LLM combines `GlobTool` and `ReadFileTool` to explore a directory. Guards the file-discovery loop against a live provider.

use super::common;

use agentwerk::tools::{GlobTool, ReadFileTool};
use agentwerk::Agent;

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let output = Agent::new()
        .provider(provider)
        .model(&model)
        .tool(ReadFileTool)
        .tool(GlobTool)
        .max_steps(10)
        .task("Find all Rust source files and describe what this project does.")
        .await?;

    common::print_result(&output);

    assert!(output.statistics.tool_calls >= 1);

    Ok(())
}
