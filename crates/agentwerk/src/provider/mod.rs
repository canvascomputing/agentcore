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
pub use environment::{from_env, model_from_env};
pub use error::{ProviderError, ProviderResult, RequestErrorKind};
pub use litellm::LiteLlmProvider;
pub use mistral::MistralProvider;
pub use model::{Model, ModelLookup};
pub use openai::OpenAiProvider;
pub use r#trait::{CompletionRequest, Provider, ToolChoice};
pub use types::{ContentBlock, Message, TokenUsage};

pub(crate) fn request_retry_delay_from_headers(
    resp: &reqwest::Response,
) -> Option<std::time::Duration> {
    let value = resp.headers().get("retry-after")?.to_str().ok()?;
    value
        .parse::<u64>()
        .ok()
        .map(std::time::Duration::from_secs)
}

/// HTTP error handler shared by every provider. Non-2xx responses are passed
/// to the provider-specific `classify` closure, which maps recognised
/// `(status, body)` signatures to typed [`ProviderError`] variants. Anything
/// `classify` returns `None` for falls through: 429/529 become
/// [`ProviderError::RateLimited`], 5xx become retryable
/// [`ProviderError::StatusUnclassified`], and other non-2xx codes become
/// terminal [`ProviderError::StatusUnclassified`].
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
    let request_retry_delay = request_retry_delay_from_headers(&resp);
    let body = resp.text().await.unwrap_or_default();
    if let Some(e) = classify(status, &body) {
        return Err(e);
    }
    Err(fallback_http_error(status, body, request_retry_delay))
}

fn fallback_http_error(
    status: u16,
    body: String,
    request_retry_delay: Option<std::time::Duration>,
) -> ProviderError {
    match status {
        429 | 529 => ProviderError::RateLimited {
            message: body,
            status,
            request_retry_delay,
        },
        _ => ProviderError::StatusUnclassified {
            status,
            message: body,
            retryable: matches!(status, 500 | 502 | 503 | 504),
            request_retry_delay,
        },
    }
}
