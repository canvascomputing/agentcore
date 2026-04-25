//! The `Provider` trait every backend implements, plus the request shape callers pass in. The one seam between the agent loop and any particular vendor API.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::error::ProviderResult;
use super::types::{Message, ModelResponse, StreamEvent};
use crate::tools::ToolDefinition;

/// One request to a provider. Built by the agent loop from the agent's
/// configuration and the running conversation; passed to [`Provider::respond`]
/// together with a streaming event callback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Model name (e.g. `claude-sonnet-4-20250514`).
    pub model: String,
    /// System prompt assembled from role, behavior, and environment.
    pub system_prompt: String,
    /// Conversation history including the latest user input.
    pub messages: Vec<Message>,
    /// Tool definitions available to the model this turn.
    pub tools: Vec<ToolDefinition>,
    /// Cap on output tokens for this single request. `None` lets the
    /// provider apply its default.
    pub max_request_tokens: Option<u32>,
    /// Constraint on which tool the model may pick this turn.
    pub tool_choice: Option<ToolChoice>,
}

/// Constraint on which tool the model may pick on a given turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    /// The model picks freely (or replies without a tool call).
    Auto,
    /// Force the model to call this tool.
    Specific { name: String },
}

/// Core provider trait every backend implements. Object-safe via boxed futures.
///
/// Build a provider with one of the vendor constructors
/// ([`AnthropicProvider`](crate::provider::AnthropicProvider),
/// [`OpenAiProvider`](crate::provider::OpenAiProvider),
/// [`MistralProvider`](crate::provider::MistralProvider),
/// [`LiteLlmProvider`](crate::provider::LiteLlmProvider)) or pick one from the
/// environment with [`from_env`](crate::provider::from_env).
///
/// # Examples
///
/// Pick a provider from environment variables and hand it to an agent:
///
/// ```no_run
/// use agentwerk::Agent;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let output = Agent::new()
///     .provider_from_env()
///     .unwrap()
///     .model_from_env()
///     .unwrap()
///     .instruction("Summarize the project.")
///     .work()
///     .await
///     .unwrap();
///
/// println!("{}", output.response_raw);
/// # });
/// ```
pub trait Provider: Send + Sync {
    /// Drive one model turn. Emits incremental [`StreamEvent`]s via the
    /// callback as the model generates, then returns the assembled reply.
    /// Callers that don't care about incremental events pass a no-op closure
    /// and wait for the final [`ModelResponse`].
    fn respond(
        &self,
        request: ModelRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>>;

    /// Warm the TCP+TLS connection pool before the first request.
    ///
    /// Called automatically by the agent loop before the first turn. Sends
    /// a fire-and-forget HEAD request to the provider's base URL so the
    /// TLS handshake (~100-200 ms) overlaps with agent startup, letting
    /// the first real request reuse the already-established connection.
    ///
    /// Default implementation is a no-op: override in providers that own
    /// a `reqwest::Client` and call `prewarm_with` from there.
    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}

pub(crate) const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Build the `reqwest::Client` every built-in provider uses. The
/// whole-request timeout makes a hung endpoint surface as a retryable
/// `ConnectionFailed` instead of blocking indefinitely. The 10-minute default
/// mirrors Anthropic's own reference agent (`claude-code`).
pub(crate) fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("reqwest::Client with timeout should build")
}

/// Fire-and-forget HEAD request to warm the TCP+TLS connection pool. Helper
/// for `Provider::prewarm` overrides — call this with the provider's own
/// `reqwest::Client` and base URL.
pub(crate) async fn prewarm_with(client: &reqwest::Client, base_url: &str) {
    let _ = tokio::time::timeout(Duration::from_secs(10), client.head(base_url).send()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_request_timeout_is_ten_minutes() {
        assert_eq!(DEFAULT_REQUEST_TIMEOUT, Duration::from_secs(600));
    }
}
