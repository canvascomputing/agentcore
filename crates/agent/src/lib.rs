pub mod error;
pub mod provider;
pub mod tools;
pub mod agent;
pub mod persistence;

#[cfg(test)]
pub(crate) mod testutil;

// Errors
pub use error::{AgenticError, Result};

// Provider and message types
pub use provider::{
    AnthropicProvider, CompletionRequest, ContentBlock, HttpTransport, LiteLlmProvider, MistralProvider,
    LlmProvider, Message, ModelResponse, StopReason, TokenUsage, ToolChoice,
};

// Tool infrastructure and built-in tools
pub use tools::{
    BashTool, BuiltinToolset, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SpawnAgentTool, Tool, ToolBuilder, ToolCall, ToolContext, ToolDefinition, ToolRegistry,
    ToolResult, ToolSearchResult, ToolSearchTool, Toolset, WriteFileTool, execute_tool_calls,
    task_create_tool, task_get_tool, task_list_tool, task_update_tool,
};

// Prompt construction
pub use agent::prompt::{EnvironmentContext, PromptBuilder, PromptSection};

// Persistence
pub use persistence::{
    EntryType, SessionMetadata, SessionStore, Task, TaskStatus, TaskStore, TaskUpdate,
    TranscriptEntry,
};

// Agent
pub use agent::{
    Agent, AgentBuilder, AgentOutput, CommandQueue, CommandSource, Event, InvocationContext,
    OutputSchema, QueuePriority, QueuedCommand, Statistics, generate_agent_name, validate_value,
};
