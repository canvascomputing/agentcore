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
    BashTool, BuiltinToolset, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SpawnAgentTool, Tool, ToolBuilder, ToolCall, ToolContext, ToolDefinition, ToolRegistry,
    ToolResult, ToolSearchResult, ToolSearchTool, Toolset, WriteFileTool, execute_tool_calls,
    task_create_tool, task_get_tool, task_list_tool, task_update_tool,
};

// Agent
pub use agent::{
    Agent, AgentBuilder, AgentOutput, BehaviorPrompt, Event,
    OutputSchema, Statistics, validate_value,
};
