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

// Provider and message types
pub use provider::{
    AnthropicProvider, ContentBlock, LiteLLMProvider, LlmProvider, Message, MistralProvider,
    OpenAiProvider, TokenUsage,
    provider_from_env,
};

// Tool infrastructure and built-in tools
pub use tools::{
    BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SpawnAgentTool, TaskTool, Tool, ToolBuilder, ToolContext,
    ToolResult, ToolSearchTool, WebFetchTool, WriteFileTool,
};

// Agent
pub use agent::{
    Agent, AgentOutput, AgentPool, DEFAULT_BEHAVIOR_PROMPT, Event, EventKind, JobId, PoolStrategy,
    Statistics, Status,
};
