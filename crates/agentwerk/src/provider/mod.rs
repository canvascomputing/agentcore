mod anthropic;
pub mod environment;
pub(crate) mod model;
mod openai;
pub(crate) mod retry;
pub(crate) mod stream;
mod r#trait;
pub mod types;

pub use anthropic::AnthropicProvider;
pub use environment::provider_from_env;
pub use openai::{LiteLLMProvider, MistralProvider, OpenAiProvider};
pub use r#trait::LlmProvider;
pub(crate) use r#trait::{CompletionRequest, ToolChoice};
pub use types::{ContentBlock, Message, TokenUsage};

fn parse_retry_after_header(resp: &reqwest::Response) -> Option<u64> {
    let value = resp.headers().get("retry-after")?.to_str().ok()?;
    value.parse::<u64>().ok().map(|secs| secs * 1000)
}

/// Check for HTTP error status and return a structured `AgenticError::Api` if non-2xx.
pub(crate) async fn check_http_error(
    resp: reqwest::Response,
) -> crate::error::Result<reqwest::Response> {
    let status = resp.status().as_u16();
    if status >= 400 {
        let retry_after_ms = parse_retry_after_header(&resp);
        let retryable = matches!(status, 429 | 529 | 500 | 502 | 503 | 504);
        let body_text = resp.text().await.unwrap_or_default();
        return Err(crate::error::AgenticError::Api {
            message: body_text,
            status: Some(status),
            retryable,
            retry_after_ms,
        });
    }
    Ok(resp)
}
