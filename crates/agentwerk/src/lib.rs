//! A minimal core for agentic applications: build agents as small,
//! function-call-shaped units.
//!
//! [`Agent`] is the entry point. Build with `Agent::new()`, chain
//! configurations, then call `.run()`. The same agent can be cloned and run
//! again with a new instruction: the static template (tools, sub-agents,
//! behavior prompts) is shared, the per-run fields are not.
//!
//! # Quick start
//!
//! ```
//! use std::sync::Arc;
//! use agentwerk::Agent;
//! use agentwerk::testutil::MockProvider;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let provider = Arc::new(MockProvider::text("done"));
//!
//! let output = Agent::new()
//!     .provider(provider)
//!     .model_name("claude-sonnet-4-20250514")
//!     .instruction_prompt("Find all Rust source files.")
//!     .run()
//!     .await
//!     .unwrap();
//!
//! assert_eq!(output.response_raw, "done");
//! # });
//! ```
//!
//! For a runnable example against a real provider, see the README's Quick Start.
//!
//! # Where to look
//!
//! | Module | Purpose | Headline type |
//! |---|---|---|
//! | [`agent`] | Build and run agents; observe events; validate output | [`Agent`] |
//! | [`provider`] | Anthropic, OpenAI, Mistral, LiteLLM | [`Provider`] |
//! | [`tools`] | Built-in tools and the trait for custom ones | [`Toolable`] |
//! | [`error`] | The single error type every fallible call returns | [`Error`] |
//!
//! # Crate conventions
//!
//! - Every fallible call returns [`Result`] (alias for `Result<T, Error>`).
//! - The loop emits [`Event`]s; observers, not hooks.
//! - Agents are cheap to clone and may be `.run()` again.

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
