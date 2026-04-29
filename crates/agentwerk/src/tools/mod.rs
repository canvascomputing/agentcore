//! Tool system: the `ToolLike` trait, the ad-hoc `Tool` struct, and the built-in tools that give agents reach into the filesystem, shell, search, web, sub-agents, and todo lists.

mod agent;
mod bash;
mod edit_file;
pub(crate) mod error;
mod glob;
mod grep;
mod list_directory;
mod read_file;
mod send_message;
mod todo_list;
mod tool;
mod tool_file;
mod tool_search;
pub(crate) mod util;
mod web_fetch;
mod write_file;

// Re-export tool infrastructure
pub use error::ToolError;
pub use tool::{Tool, ToolContext, ToolDefinition, ToolLike, ToolResult};
pub use tool_file::ToolFile;
pub(crate) use tool::{ToolCall, ToolRegistry};

// Re-export built-in tools
pub use agent::AgentTool;
pub use bash::BashTool;
pub use edit_file::EditFileTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use list_directory::ListDirectoryTool;
pub use read_file::ReadFileTool;
pub use send_message::SendMessageTool;
pub use todo_list::TodoListTool;
pub use tool_search::ToolSearchTool;
pub use web_fetch::WebFetchTool;
pub use write_file::WriteFileTool;
