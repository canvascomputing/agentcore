//! agentwerk: build LLM-powered agents as small function-call-shaped units, composed from prompts, tools, and a shared execution loop.

pub mod agent;
pub mod error;
pub(crate) mod persistence;
pub mod provider;
pub mod tools;
pub(crate) mod util;

pub mod testutil;

pub use error::{AgenticError, Result};

pub use provider::{
    AnthropicProvider, CompletionRequest, ContentBlock, LiteLlmProvider, Message, MistralProvider,
    Model, ModelLookup, OpenAiProvider, Provider, ProviderError, TokenUsage,
};

pub use tools::{
    BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool, SendMessageTool,
    SpawnAgentTool, TaskTool, Tool, ToolContext, ToolResult, ToolSearchTool, Toolable,
    WebFetchTool, WriteFileTool,
};

pub use agent::{
    Agent, AgentEvent, AgentEventKind, AgentHandle, AgentJobId, AgentOutput, AgentOutputFuture,
    AgentPool, AgentPoolStrategy, AgentStatistics, AgentStatus, CompactReason,
    DEFAULT_BEHAVIOR_PROMPT,
};
