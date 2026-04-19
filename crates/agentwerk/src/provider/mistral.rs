//! Mistral provider — OpenAI-compatible wire format against api.mistral.ai.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::error::ProviderResult;
use super::openai::OpenAiProvider;
use super::r#trait::{CompletionRequest, Provider};
use super::types::{ModelResponse, StreamEvent};
use crate::error::Result;

/// Mistral LLM provider. Speaks OpenAI's chat-completions API, so it
/// delegates to an inner [`OpenAiProvider`] pointed at `api.mistral.ai`.
pub struct MistralProvider(OpenAiProvider);

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";
const DEFAULT_MODEL: &str = "mistral-medium-2508";

impl MistralProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_client(api_key, reqwest::Client::new())
    }

    pub fn with_client(api_key: impl Into<String>, client: reqwest::Client) -> Self {
        Self(OpenAiProvider::raw(api_key, DEFAULT_BASE_URL, client, false))
    }

    pub fn base_url(self, url: String) -> Self {
        Self(self.0.base_url(url))
    }

    pub(crate) fn from_env() -> Result<(Self, String)> {
        use super::environment::{env_or, env_required};
        let provider = Self::new(env_required("MISTRAL_API_KEY")?)
            .base_url(env_or("MISTRAL_BASE_URL", DEFAULT_BASE_URL));
        let model = env_or("MISTRAL_MODEL", DEFAULT_MODEL);
        Ok((provider, model))
    }
}

impl Provider for MistralProvider {
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
