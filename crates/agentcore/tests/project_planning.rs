//! Integration test: agent-driven task management with all TaskTool features.
//!
//! Exercises: create, update, list, get, delete, claim, add_dependency.

mod common;

use agentcore::{AgentBuilder, TaskTool};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let tmp = tempfile::tempdir()?;

    let output = AgentBuilder::new()
        .provider(provider)
        .model(&model)
        .identity_prompt(
            "You are a project planner. Use the task tool to manage work items. \
             The task tool supports these actions: create, update, list, get, delete, claim, add_dependency. \
             Be concise. Always use the task tool, never simulate results.",
        )
        .instruction_prompt(
            "Do the following steps in order:\n\
             1. Create three tasks: 'Design API', 'Write tests', 'Deploy'\n\
             2. Add a dependency: 'Design API' blocks 'Write tests'\n\
             3. Claim 'Design API' as agent 'alice'\n\
             4. Mark 'Design API' as Completed\n\
             5. Delete 'Deploy'\n\
             6. List all remaining tasks\n\
             7. Get details of 'Write tests' by ID\n\
             8. Summarize what you did",
        )
        .tool(TaskTool::open(tmp.path()))
        .max_turns(15)
        .run()
        .await?;

    common::print_result(&output);

    assert!(output.statistics.tool_calls >= 3);

    Ok(())
}
