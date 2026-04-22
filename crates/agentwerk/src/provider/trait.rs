//! The `Provider` trait every LLM backend implements, plus the request shape callers pass in. The one seam between the agent loop and any particular LLM API.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::error::ProviderResult;
use super::types::{CompletionResponse, Message, StreamEvent};
use crate::tools::tool::ToolDefinition;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_request_tokens: Option<u32>,
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Specific { name: String },
}

/// Core LLM provider trait. Object-safe via boxed futures.
pub trait Provider: Send + Sync {
    /// One-shot completion. Returns the full response after the model stops.
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<CompletionResponse>> + Send + '_>>;

    /// Streaming completion. Emits incremental [`StreamEvent`]s via the
    /// callback as the model generates, then returns the assembled response.
    /// Default implementation falls back to [`Self::complete`] and emits a
    /// single `MessageDone`.
    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<CompletionResponse>> + Send + '_>> {
        Box::pin(async move {
            let response = self.complete(request).await?;
            on_event(StreamEvent::MessageDone);
            Ok(response)
        })
    }

    /// Warm the TCP+TLS connection pool before the first API request.
    ///
    /// Called automatically by the agent loop before the first turn. Sends
    /// a fire-and-forget HEAD request to the provider's base URL so the
    /// TLS handshake (~100-200 ms) overlaps with agent startup, letting
    /// the first real LLM call reuse the already-established connection.
    ///
    /// Default implementation is a no-op — override in providers that own
    /// a `reqwest::Client` and call `prewarm_with` from there.
    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}

/// Default `reqwest::Client` used by every built-in provider when the caller
/// doesn't supply their own via `with_client`. Both timeouts exist so that
/// unreachable or silently-stuck endpoints surface as retryable
/// `ConnectionFailed` errors (and observable `RequestRetried` /
/// `RequestFailed` events) instead of blocking forever with no signal:
///
/// - `connect_timeout` fires when the TCP+TLS handshake can't complete.
/// - `read_timeout` fires when the server accepts the connection but stops
///   sending bytes. Applied per-chunk, so legitimate long generations keep
///   working as long as tokens keep flowing.
pub(crate) fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(3))
        .build()
        .expect("reqwest::Client with timeouts should build")
}

/// Fire-and-forget HEAD request to warm the TCP+TLS connection pool. Helper
/// for `Provider::prewarm` overrides — call this with the provider's own
/// `reqwest::Client` and base URL.
pub(crate) async fn prewarm_with(client: &reqwest::Client, base_url: &str) {
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        client.head(base_url).send(),
    )
    .await;
}
