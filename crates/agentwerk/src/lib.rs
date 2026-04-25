//! A minimal core for agentic applications: build agents as small,
//! function-call-shaped units.
//!
//! [`Agent`] is the entry point. Build with `Agent::new()`, chain
//! configurations, then call `.work()`. The same agent can be cloned and run
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
//!     .instruction("Find all Rust source files.")
//!     .work()
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
//! | [`tools`] | Built-in tools and the trait for custom ones | [`Tool`] |
//! | [`error`] | The single error type every fallible call returns | [`Error`] |
//!
//! # Crate conventions
//!
//! - Every fallible call returns [`Result`] (alias for `Result<T, Error>`).
//! - The loop emits [`Event`]s; observers, not hooks.
//! - Agents are cheap to clone and may be `.work()` again.

pub mod agent;
pub mod error;
pub mod event;
pub mod output;
pub(crate) mod persistence;
pub mod provider;
pub mod tools;
pub(crate) mod util;
pub mod werk;

pub mod testutil;

pub use error::{Error, Result};

pub use provider::{Model, Provider};

pub use tools::Tool;

pub use agent::Agent;
pub use event::Event;
pub use output::Output;
pub use werk::Werk;
