pub mod error;
pub mod message;
pub mod tool;
pub mod provider;
pub mod cost;
pub mod prompt;
pub mod agent;
pub mod spawn;

#[cfg(test)]
pub(crate) mod testutil;

// Errors
pub use error::{AgenticError, Result};

// Messages
pub use message::{ContentBlock, Message, ModelResponse, StopReason, Usage};

// Tools
pub use tool::{
    Tool, ToolBuilder, ToolCall, ToolContext, ToolDefinition, ToolRegistry, ToolResult,
    ToolSearchResult, Toolset, execute_tool_calls,
};

// LLM providers
pub use provider::{
    AnthropicProvider, CompletionRequest, HttpTransport, LiteLlmProvider, LlmProvider, ToolChoice,
};

// Cost tracking
pub use cost::{CostTracker, ModelCosts, ModelUsage};

// Prompt construction
pub use prompt::{EnvironmentContext, PromptBuilder, PromptSection};

// Agent
pub use agent::{
    Agent, AgentBuilder, AgentOutput, CommandQueue, CommandSource, Event, InvocationContext,
    OutputSchema, QueuePriority, QueuedCommand, generate_agent_id, validate_value,
};

// Orchestration
pub use spawn::SpawnAgentTool;
