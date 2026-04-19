use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::error::ProviderResult;
use super::types::{Message, ModelResponse, StreamEvent};
use crate::tools::tool::ToolDefinition;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_output_tokens: Option<u32>,
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Specific { name: String },
}

/// Core LLM provider trait. Object-safe via boxed futures.
pub trait LlmProvider: Send + Sync {
    /// One-shot completion. Returns the full response after the model stops.
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>>;

    /// Streaming completion. Emits incremental [`StreamEvent`]s via the
    /// callback as the model generates, then returns the assembled response.
    /// Default implementation falls back to [`Self::complete`] and emits a
    /// single `MessageDone`.
    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
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
    /// a `reqwest::Client`.
    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}

/// Fire-and-forget HEAD request to warm the TCP+TLS connection pool.
pub(crate) async fn prewarm_connection(client: &reqwest::Client, base_url: &str) {
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        client.head(base_url).send(),
    )
    .await;
}
