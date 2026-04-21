//! The `Provider` trait and the vendor-specific implementations that speak to Anthropic, OpenAI-compatible APIs, Mistral, and LiteLLM.

mod anthropic;
pub mod environment;
mod error;
mod litellm;
mod mistral;
pub(crate) mod model;
mod openai;
pub(crate) mod retry;
pub(crate) mod stream;
mod r#trait;
pub mod types;

pub use anthropic::AnthropicProvider;
pub use environment::from_env;
pub use error::{ProviderError, ProviderResult};
pub use litellm::LiteLlmProvider;
pub use mistral::MistralProvider;
pub use model::{Model, ModelLookup};
pub use openai::OpenAiProvider;
pub use r#trait::{CompletionRequest, Provider, ToolChoice};
pub use types::{ContentBlock, Message, TokenUsage};

pub(crate) fn retry_after_ms_from_headers(resp: &reqwest::Response) -> Option<u64> {
    let value = resp.headers().get("retry-after")?.to_str().ok()?;
    value.parse::<u64>().ok().map(|secs| secs * 1000)
}

/// HTTP error handler shared by every provider. Non-2xx responses are passed
/// to the provider-specific `classify` closure, which maps recognised
/// `(status, body)` signatures to typed [`ProviderError`] variants. Anything
/// `classify` returns `None` for falls through: 429/529 become
/// [`ProviderError::RateLimited`], 5xx become retryable
/// [`ProviderError::UnexpectedStatus`], and other non-2xx codes become
/// terminal [`ProviderError::UnexpectedStatus`].
pub(crate) async fn map_http_errors<F>(
    resp: reqwest::Response,
    classify: F,
) -> ProviderResult<reqwest::Response>
where
    F: FnOnce(u16, &str) -> Option<ProviderError>,
{
    let status = resp.status().as_u16();
    if status < 400 {
        return Ok(resp);
    }
    let retry_after_ms = retry_after_ms_from_headers(&resp);
    let body = resp.text().await.unwrap_or_default();
    if let Some(e) = classify(status, &body) {
        return Err(e);
    }
    Err(fallback_http_error(status, body, retry_after_ms))
}

fn fallback_http_error(status: u16, body: String, retry_after_ms: Option<u64>) -> ProviderError {
    match status {
        429 | 529 => ProviderError::RateLimited {
            message: body,
            status,
            retry_after_ms,
        },
        _ => ProviderError::UnexpectedStatus {
            status,
            message: body,
            retryable: matches!(status, 500 | 502 | 503 | 504),
            retry_after_ms,
        },
    }
}
