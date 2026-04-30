//! LiteLLM proxy provider. Points the OpenAI-compatible wire format at a local LiteLLM instance so callers can switch backend providers without touching agent code.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use super::error::ProviderResult;
use super::openai::OpenAiProvider;
use super::provider::{ModelRequest, Provider};
use super::types::{ModelResponse, StreamEvent};

/// LiteLLM proxy provider. Delegates to an inner [`OpenAiProvider`] with
/// `cache_tokens = true` so cache-read / cache-creation counts from
/// upstream providers are preserved.
pub struct LiteLlmProvider(OpenAiProvider);

const DEFAULT_BASE_URL: &str = "http://localhost:4000";

impl LiteLlmProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(OpenAiProvider::raw(api_key, DEFAULT_BASE_URL, true))
    }

    pub fn base_url(self, url: impl Into<String>) -> Self {
        Self(self.0.base_url(url))
    }

    pub fn timeout(self, d: Duration) -> Self {
        Self(self.0.timeout(d))
    }

    pub(crate) fn from_env() -> ProviderResult<Self> {
        use super::environment::env_or;
        Ok(Self::new(env_or("LITELLM_API_KEY", ""))
            .base_url(env_or("LITELLM_BASE_URL", DEFAULT_BASE_URL)))
    }
}

impl Provider for LiteLlmProvider {
    fn respond(
        &self,
        request: ModelRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        self.0.respond(request, on_event)
    }

    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.0.prewarm()
    }
}
