pub mod error;
pub mod provider;
pub mod tools;
pub mod agent;
pub(crate) mod persistence;

#[cfg(test)]
pub(crate) mod testutil;

// Errors
pub use error::{AgenticError, Result};

// Provider and message types
pub use provider::{
    AnthropicProvider, CompletionRequest, ContentBlock, LiteLlmProvider, MistralProvider,
    LlmProvider, Message, ModelResponse, OpenAiProvider,
    StopReason, StreamEvent, TokenUsage, ToolChoice, prewarm_connection,
};

// Tool infrastructure and built-in tools
pub use tools::{
    BashSingleTool, BashTool, BuiltinToolset, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SpawnAgentTool, TaskTool, Tool, ToolBuilder, ToolCall, ToolContext, ToolDefinition, ToolRegistry,
    ToolResult, ToolSearchResult, ToolSearchTool, Toolset, WebFetchTool, WriteFileTool, execute_tool_calls,
};

// Agent
pub use agent::{
    Agent, AgentBuilder, AgentOutput, BehaviorPrompt, CommandQueue, Event,
    OutputSchema, Statistics, validate_value,
};
