//! agentwerk: build LLM-powered agents as small function-call-shaped units, composed from prompts, tools, and a shared execution loop.

pub mod agent;
pub mod config;
pub mod error;
pub(crate) mod persistence;
pub mod provider;
pub mod tools;
pub(crate) mod util;

pub mod testutil;

pub use config::ConfigError;
pub use error::{Error, Result};
pub use persistence::error::PersistenceError;

pub use provider::{
    AnthropicProvider, CompletionRequest, ContentBlock, LiteLlmProvider, Message, MistralProvider,
    Model, ModelLookup, OpenAiProvider, Provider, ProviderError, RequestErrorKind, TokenUsage,
};

pub use tools::{
    BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool, SendMessageTool,
    SpawnAgentTool, TaskTool, Tool, ToolContext, ToolError, ToolResult, ToolSearchTool, Toolable,
    WebFetchTool, WriteFileTool,
};

pub use agent::{
    Agent, AgentError, AgentHandle, Batch, BatchHandle, BatchOutputStream, CompactReason, Event,
    EventKind, Output, OutputError, OutputFuture, Statistics, Status, DEFAULT_BEHAVIOR_PROMPT,
};
