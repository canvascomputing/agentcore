//! LiteLLM proxy provider — OpenAI-compatible wire format against a local
//! LiteLLM instance (default `http://localhost:4000`). Enables prompt caching
//! since LiteLLM proxies providers (Anthropic, Bedrock, etc.) that report
//! cache-read and cache-creation token counts.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::error::ProviderResult;
use super::openai::OpenAiProvider;
use super::r#trait::{CompletionRequest, Provider};
use super::types::{ModelResponse, StreamEvent};
use crate::error::Result;

/// LiteLLM proxy provider. Delegates to an inner [`OpenAiProvider`] with
/// `cache_tokens = true` so cache-read / cache-creation counts from
/// upstream providers are preserved.
pub struct LiteLLMProvider(OpenAiProvider);

const DEFAULT_BASE_URL: &str = "http://localhost:4000";
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";

impl LiteLLMProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_client(api_key, reqwest::Client::new())
    }

    pub fn with_client(api_key: impl Into<String>, client: reqwest::Client) -> Self {
        Self(OpenAiProvider::raw(api_key, DEFAULT_BASE_URL, client, true))
    }

    pub fn base_url(self, url: String) -> Self {
        Self(self.0.base_url(url))
    }

    pub(crate) fn from_env() -> Result<(Self, String)> {
        use super::environment::env_or;
        let provider = Self::new(env_or("LITELLM_API_KEY", ""))
            .base_url(env_or("LITELLM_BASE_URL", DEFAULT_BASE_URL));
        let model = env_or("LITELLM_MODEL", DEFAULT_MODEL);
        Ok((provider, model))
    }
}

impl Provider for LiteLLMProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        self.0.complete(request)
    }

    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        self.0.complete_streaming(request, on_event)
    }

    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.0.prewarm()
    }
}
