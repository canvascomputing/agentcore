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
    AnthropicProvider, CompletionRequest, ContentBlock, LiteLLMProvider, LlmProvider, Message,
    MistralProvider, OpenAiProvider, ProviderError, TokenUsage,
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
    Agent, AgentOutput, AgentPool, CompactReason, DEFAULT_BEHAVIOR_PROMPT, Event, EventKind, JobId,
    PoolStrategy, Statistics, Status,
    compact_threshold, BUFFER_TOKENS, RESERVED_OUTPUT_TOKENS,
};
