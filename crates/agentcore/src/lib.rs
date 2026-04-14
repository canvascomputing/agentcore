pub mod error;
pub mod provider;
pub mod tools;
pub mod agent;
pub(crate) mod persistence;

#[cfg(test)]
pub(crate) mod testutil;

// Errors
pub use error::{AgenticError, Result};

/// Pass to any `max_` builder method to disable the limit.
pub const UNLIMITED: u32 = 0;

// Provider and message types
pub use provider::{
    AnthropicProvider, ContentBlock, LiteLlmProvider, LlmProvider, Message, MistralProvider,
    OpenAiProvider, TokenUsage,
};

// Tool infrastructure and built-in tools
pub use tools::{
    BashGlobTool, BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SpawnAgentTool, TaskTool, Tool, ToolBuilder, ToolContext, ToolRegistry,
    ToolResult, ToolSearchTool, WebFetchTool, WriteFileTool,
};

// Agent
pub use agent::{
    Agent, AgentBuilder, AgentOutput, BehaviorPrompt, Event, OutputSchema, Statistics,
};
