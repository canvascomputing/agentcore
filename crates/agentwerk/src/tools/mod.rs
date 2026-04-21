//! Tool system: the `Toolable` trait and the built-in tools that give agents reach into the filesystem, shell, search, web, sub-agents, and task records.

mod bash;
mod edit_file;
mod glob;
mod grep;
mod list_directory;
mod read_file;
pub mod send_message;
pub mod spawn_agent;
pub mod task_tools;
pub mod tool;
mod tool_search;
pub(crate) mod util;
mod web_fetch;
mod write_file;

// Re-export tool infrastructure
pub use tool::{Tool, ToolContext, ToolResult, Toolable};
pub(crate) use tool::{ToolCall, ToolRegistry};

// Re-export built-in tools
pub use bash::BashTool;
pub use edit_file::EditFileTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use list_directory::ListDirectoryTool;
pub use read_file::ReadFileTool;
pub use send_message::SendMessageTool;
pub use spawn_agent::SpawnAgentTool;
pub use task_tools::TaskTool;
pub use tool_search::ToolSearchTool;
pub use web_fetch::WebFetchTool;
pub use write_file::WriteFileTool;
