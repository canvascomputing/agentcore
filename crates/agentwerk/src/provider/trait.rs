use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::types::{Message, ModelResponse, StreamEvent};
use crate::tools::tool::ToolDefinition;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Specific { name: String },
}

/// Core LLM provider trait. Object-safe via boxed futures.
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>>;

    /// Streaming variant that emits incremental events via callback.
    /// Default implementation falls back to `complete()` and emits `MessageDone`.
    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        Box::pin(async move {
            let response = self.complete(request).await?;
            on_event(StreamEvent::MessageDone);
            Ok(response)
        })
    }

    /// Warm the TCP+TLS connection pool before the first API request.
    ///
    /// Sends a fire-and-forget HEAD request to the provider's base URL.
    /// This overlaps the TLS handshake (~100-200ms) with agent startup,
    /// so the first real LLM call reuses the already-established connection.
    ///
    /// Called automatically by the agent loop before the first turn.
    /// Default implementation is a no-op — override in providers that
    /// own a `reqwest::Client`.
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
