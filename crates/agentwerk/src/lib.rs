pub mod error;
pub mod provider;
pub mod tools;
pub mod agent;
pub(crate) mod persistence;
pub(crate) mod util;

#[cfg(test)]
pub(crate) mod testutil;

// Errors
pub use error::{AgenticError, Result};

/// Pass to any `max_` builder method to disable the limit.
pub const UNLIMITED: u32 = 0;

// Provider and message types
pub use provider::{
    AnthropicProvider, ContentBlock, LlmProvider, Message, OpenAiProvider, TokenUsage,
};
pub use provider::environment::Environment;

// Tool infrastructure and built-in tools
pub use tools::{
    BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SpawnAgentTool, TaskTool, Tool, ToolBuilder, ToolContext,
    ToolResult, ToolSearchTool, WebFetchTool, WriteFileTool,
};

// Agent
pub use agent::{
    Agent, AgentBuilder, AgentOutput, BehaviorPrompt, Event, Pipeline, Statistics,
};
