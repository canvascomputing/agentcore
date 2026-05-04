//! Tool system: the `ToolLike` trait, the ad-hoc `Tool` struct, and the
//! registry the loop consults before each provider call.

mod error;
mod tool;
mod tool_file;
pub(crate) mod util;

mod bash;
mod edit_file;
mod glob;
mod grep;
mod list_directory;
mod read_file;
mod tickets;
mod tool_search;
mod web_fetch;
mod write_file;

pub use error::ToolError;
pub use tool::{Tool, ToolContext, ToolLike, ToolResult};
pub(crate) use tool::{ToolCall, ToolRegistry};
pub use tool_file::ToolFile;

pub use bash::BashTool;
pub use edit_file::EditFileTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use list_directory::ListDirectoryTool;
pub use read_file::ReadFileTool;
pub use tickets::{ManageTicketsTool, MarkTicketDoneTool, ReadTicketsTool, WriteTicketsTool};
pub use tool_search::ToolSearchTool;
pub use web_fetch::WebFetchTool;
pub use write_file::WriteFileTool;
