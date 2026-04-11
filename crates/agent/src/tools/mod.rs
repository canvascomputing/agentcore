mod bash;
mod edit_file;
mod glob;
mod grep;
mod list_directory;
mod read_file;
pub mod spawn_agent;
pub mod task_tools;
mod tool_search;
mod write_file;

pub use bash::BashTool;
pub use edit_file::EditFileTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use list_directory::ListDirectoryTool;
pub use read_file::ReadFileTool;
pub use spawn_agent::SpawnAgentTool;
pub use task_tools::{task_create_tool, task_get_tool, task_list_tool, task_update_tool};
pub use tool_search::ToolSearchTool;
pub use write_file::WriteFileTool;

use crate::tool::{Tool, Toolset};

/// Built-in toolset providing file operations, search, directory listing,
/// shell execution, and tool discovery.
pub struct BuiltinToolset;

impl Toolset for BuiltinToolset {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(EditFileTool),
            Box::new(GlobTool),
            Box::new(GrepTool),
            Box::new(ListDirectoryTool),
            Box::new(BashTool),
            Box::new(ToolSearchTool),
        ]
    }
}
