//! agentwerk: build LLM-powered agents as small function-call-shaped units, composed from prompts, tools, and a shared execution loop.

pub mod agent;
pub mod error;
pub(crate) mod persistence;
pub mod provider;
pub mod tools;
pub(crate) mod util;

pub mod testutil;

pub use error::{Error, Result};

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
    Agent, AgentHandle, Batch, BatchHandle, BatchOutputStream, CompactReason, Event, EventKind,
    Output, OutputFuture, Statistics, Status, DEFAULT_BEHAVIOR_PROMPT,
};
