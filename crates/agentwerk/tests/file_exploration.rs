mod common;

use agentwerk::{AgentBuilder, GlobTool, ReadFileTool};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let output = AgentBuilder::new()
        .provider(provider)
        .model(&model)
        .instruction_prompt("Find all Rust source files and describe what this project does.")
        .tool(ReadFileTool)
        .tool(GlobTool)
        .max_turns(10)
        .run()
        .await?;

    common::print_result(&output);

    assert!(output.statistics.tool_calls >= 1);

    Ok(())
}
