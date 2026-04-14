pub mod tool;
mod bash;
mod bash_single;
mod edit_file;
mod glob;
mod grep;
mod list_directory;
mod read_file;
pub mod spawn_agent;
pub mod task_tools;
mod tool_search;
pub(crate) mod util;
mod web_fetch;
mod write_file;

// Re-export tool infrastructure
pub use tool::{
    Tool, ToolBuilder, ToolCall, ToolContext, ToolDefinition, ToolRegistry, ToolResult,
    ToolSearchResult, Toolset, execute_tool_calls,
};

// Re-export built-in tools
pub use bash::BashTool;
pub use bash_single::BashSingleTool;
pub use edit_file::EditFileTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use list_directory::ListDirectoryTool;
pub use read_file::ReadFileTool;
pub use spawn_agent::SpawnAgentTool;
pub use task_tools::TaskTool;
pub use tool_search::ToolSearchTool;
pub use web_fetch::WebFetchTool;
pub use write_file::WriteFileTool;

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
            Box::new(WebFetchTool),
        ]
    }
}
