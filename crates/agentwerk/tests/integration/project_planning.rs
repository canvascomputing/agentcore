//! Integration test: agent-driven todo-list management with all TodoListTool features.
//!
//! Exercises: create, update, list, get, delete, claim, add_dependency.

use super::common;

use agentwerk::tools::TodoListTool;
use agentwerk::Agent;

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let tmp = tempfile::tempdir()?;

    let output = Agent::new()
        .provider(provider)
        .model(&model)
        .role(
            "You are a project planner. Use the todo_list tool to manage work items. \
             The todo_list tool supports these actions: create, update, list, get, delete, claim, add_dependency. \
             Be concise. Always use the todo_list tool, never simulate results.",
        )
        .tool(TodoListTool::new(tmp.path()))
        .max_turns(15)
        .task(
            "Do the following steps in order:\n\
             1. Create three items: 'Design API', 'Write tests', 'Deploy'\n\
             2. Add a dependency: 'Design API' blocks 'Write tests'\n\
             3. Claim 'Design API' as agent 'alice'\n\
             4. Mark 'Design API' as Completed\n\
             5. Delete 'Deploy'\n\
             6. List all remaining items\n\
             7. Get details of 'Write tests' by ID\n\
             8. Summarize what you did",
        )
        .await?;

    common::print_result(&output);

    assert!(output.statistics.tool_calls >= 3);

    Ok(())
}
