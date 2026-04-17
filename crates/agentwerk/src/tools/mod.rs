pub mod tool;
mod bash;
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
pub use tool::{Tool, ToolBuilder, ToolContext, ToolResult};
pub(crate) use tool::ToolRegistry;
pub(crate) use tool::{ToolCall, execute_tool_calls};

// Re-export built-in tools
pub use bash::BashTool;
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
