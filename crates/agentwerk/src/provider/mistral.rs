//! Mistral provider. Thin wrapper that points the OpenAI-compatible wire format at api.mistral.ai.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use super::error::ProviderResult;
use super::openai::OpenAiProvider;
use super::r#trait::{ModelRequest, Provider};
use super::types::{ModelResponse, StreamEvent};
use crate::error::Result;

/// Mistral LLM provider. Speaks OpenAI's chat-completions API, so it
/// delegates to an inner [`OpenAiProvider`] pointed at `api.mistral.ai`.
pub struct MistralProvider(OpenAiProvider);

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";

impl MistralProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(OpenAiProvider::raw(api_key, DEFAULT_BASE_URL, false))
    }

    pub fn base_url(self, url: impl Into<String>) -> Self {
        Self(self.0.base_url(url))
    }

    pub fn timeout(self, d: Duration) -> Self {
        Self(self.0.timeout(d))
    }

    pub(crate) fn from_env() -> Result<Self> {
        use super::environment::{env_or, env_required};
        Ok(Self::new(env_required("MISTRAL_API_KEY")?)
            .base_url(env_or("MISTRAL_BASE_URL", DEFAULT_BASE_URL)))
    }
}

impl MistralProvider {
    pub(crate) fn lookup_context_window_size(id: &str) -> Option<u64> {
        let m = id.to_ascii_lowercase();
        if m.contains("codestral") {
            return Some(256_000);
        }
        if m.contains("mistral-large")
            || m.contains("mistral-medium")
            || m.contains("mistral-small")
        {
            return Some(131_072);
        }
        None
    }
}

impl Provider for MistralProvider {
    fn respond(
        &self,
        request: ModelRequest,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        self.0.respond(request)
    }

    fn respond_streaming(
        &self,
        request: ModelRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
        self.0.respond_streaming(request, on_event)
    }

    fn prewarm(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.0.prewarm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_codestral_returns_256k() {
        let lookup = MistralProvider::lookup_context_window_size;
        assert_eq!(lookup("codestral-latest"), Some(256_000));
    }

    #[test]
    fn lookup_mistral_families_return_131k() {
        let lookup = MistralProvider::lookup_context_window_size;
        assert_eq!(lookup("mistral-large-2411"), Some(131_072));
        assert_eq!(lookup("mistral-medium-2508"), Some(131_072));
        assert_eq!(lookup("mistral-small-latest"), Some(131_072));
    }

    #[test]
    fn lookup_unknown_models_return_none() {
        let lookup = MistralProvider::lookup_context_window_size;
        assert_eq!(lookup("claude-sonnet-4-20250514"), None);
        assert_eq!(lookup("gpt-5"), None);
        assert_eq!(lookup("llama-3-70b"), None);
    }

    #[test]
    fn timeout_delegates_to_inner() {
        let p = MistralProvider::new("test-key").timeout(Duration::from_secs(42));
        assert_eq!(p.0.request_timeout(), Duration::from_secs(42));
    }
}
