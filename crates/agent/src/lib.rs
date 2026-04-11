pub mod error;
pub mod message;
pub mod tool;
pub mod provider;
pub mod cost;
pub mod prompt;
pub mod session;
pub mod task;
pub mod agent;
pub mod tools;

#[cfg(test)]
pub(crate) mod testutil;

// Errors
pub use error::{AgenticError, Result};

// Messages
pub use message::{ContentBlock, Message, ModelResponse, StopReason, Usage};

// Tool traits and infrastructure
pub use tool::{
    Tool, ToolBuilder, ToolCall, ToolContext, ToolDefinition, ToolRegistry, ToolResult,
    ToolSearchResult, Toolset, execute_tool_calls,
};

// Built-in tools
pub use tools::{
    BashTool, BuiltinToolset, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SpawnAgentTool, ToolSearchTool, WriteFileTool,
    task_create_tool, task_get_tool, task_list_tool, task_update_tool,
};

// LLM providers
pub use provider::{
    AnthropicProvider, CompletionRequest, HttpTransport, LiteLlmProvider, LlmProvider, ToolChoice,
};

// Cost tracking
pub use cost::{CostTracker, ModelCosts, ModelUsage};

// Prompt construction
pub use prompt::{EnvironmentContext, PromptBuilder, PromptSection};

// Session persistence
pub use session::{EntryType, SessionMetadata, SessionStore, TranscriptEntry};

// Task persistence
pub use task::{Task, TaskStatus, TaskStore, TaskUpdate};

// Agent
pub use agent::{
    Agent, AgentBuilder, AgentOutput, CommandQueue, CommandSource, Event, InvocationContext,
    OutputSchema, QueuePriority, QueuedCommand, generate_agent_id, validate_value,
};
