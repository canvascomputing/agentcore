pub mod error;
pub mod provider;
pub mod tools;
pub mod agent;
pub(crate) mod persistence;
pub(crate) mod util;

pub mod testutil;

// Errors
pub use error::{AgenticError, Result};

// Provider and message types
pub use provider::{
    AnthropicProvider, CompletionRequest, ContentBlock, LiteLlmProvider, Message, MistralProvider,
    Model, ModelLookup, OpenAiProvider, Provider, ProviderError, TokenUsage,
};

// Tool infrastructure and built-in tools
pub use tools::{
    BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
    SendMessageTool, SpawnAgentTool, TaskTool, Tool, Toolable, ToolContext,
    ToolResult, ToolSearchTool, WebFetchTool, WriteFileTool,
};

// Agent
pub use agent::{
    Agent, AgentEvent, AgentEventKind, AgentJobId, AgentOutput, AgentPool, AgentPoolStrategy,
    AgentStatistics, AgentStatus, CompactReason, DEFAULT_BEHAVIOR_PROMPT, RunningAgent,
};
