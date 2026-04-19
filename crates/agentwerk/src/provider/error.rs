//! Provider-layer error type.
//!
//! Errors a `LlmProvider` raises before returning a successful
//! [`ModelResponse`](super::types::ModelResponse). Anything that maps to a
//! valid response-with-status belongs on
//! [`ResponseStatus`](super::types::ResponseStatus), not here.

use std::fmt;

/// Failure produced by a provider call.
#[derive(Debug)]
#[non_exhaustive]
pub enum ProviderError {
    /// HTTP 401: invalid, revoked, or missing credentials.
    AuthenticationFailed { provider_message: String },
    /// HTTP 403: authenticated but not allowed to use the resource.
    PermissionDenied { provider_message: String },
    /// HTTP 400/404: unknown model id.
    ModelNotFound { provider_message: String },
    /// HTTP 400 pre-flight: request tokens exceed the model's context window.
    ContextWindowExceeded { provider_message: String },
    /// Provider-side safety filter blocked the request input.
    SafetyFilterTriggered { provider_message: String },
    /// HTTP 429 / 529: retry with backoff, honouring `retry_after_ms` if set.
    RateLimited {
        message: String,
        status: u16,
        retry_after_ms: Option<u64>,
    },
    /// HTTP error with no more specific classification (unclassified 4xx,
    /// generic 5xx). `retryable` is true for standard transient server
    /// errors (500/502/503/504).
    UnexpectedStatus {
        status: u16,
        message: String,
        retryable: bool,
        retry_after_ms: Option<u64>,
    },
    /// Network / TLS / connection failure before any HTTP response.
    ConnectionFailed { reason: String },
    /// The response arrived but its body couldn't be parsed — malformed
    /// JSON, unexpected shape, or a broken SSE frame.
    InvalidResponse { reason: String },
}

impl ProviderError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ProviderError::RateLimited { .. }
                | ProviderError::ConnectionFailed { .. }
                | ProviderError::UnexpectedStatus { retryable: true, .. }
        )
    }

    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            ProviderError::RateLimited { retry_after_ms, .. } => *retry_after_ms,
            ProviderError::UnexpectedStatus { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::AuthenticationFailed { provider_message } => {
                write!(f, "Authentication failed: {provider_message}")
            }
            ProviderError::PermissionDenied { provider_message } => {
                write!(f, "Permission denied: {provider_message}")
            }
            ProviderError::ModelNotFound { provider_message } => {
                write!(f, "Model not found: {provider_message}")
            }
            ProviderError::ContextWindowExceeded { provider_message } => {
                write!(f, "Context window exceeded: {provider_message}")
            }
            ProviderError::SafetyFilterTriggered { provider_message } => {
                write!(f, "Safety filter triggered: {provider_message}")
            }
            ProviderError::RateLimited { message, status, .. } => {
                write!(f, "Rate limited (status {status}): {message}")
            }
            ProviderError::UnexpectedStatus {
                status,
                message,
                retryable,
                ..
            } => {
                write!(
                    f,
                    "HTTP error (status {status}): {message} (retryable: {retryable})"
                )
            }
            ProviderError::ConnectionFailed { reason } => {
                write!(f, "Connection failed: {reason}")
            }
            ProviderError::InvalidResponse { reason } => {
                write!(f, "Invalid response: {reason}")
            }
        }
    }
}

impl std::error::Error for ProviderError {}

/// Result alias for [`LlmProvider`](super::r#trait::LlmProvider) calls.
pub type ProviderResult<T> = std::result::Result<T, ProviderError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limited_is_retryable_and_carries_retry_after() {
        let err = ProviderError::RateLimited {
            message: "slow down".into(),
            status: 429,
            retry_after_ms: Some(500),
        };
        assert!(err.is_retryable());
        assert_eq!(err.retry_after_ms(), Some(500));
    }

    #[test]
    fn connection_failed_is_retryable() {
        let err = ProviderError::ConnectionFailed { reason: "dns".into() };
        assert!(err.is_retryable());
        assert_eq!(err.retry_after_ms(), None);
    }

    #[test]
    fn unexpected_status_honours_retryable_flag() {
        let retryable = ProviderError::UnexpectedStatus {
            status: 503,
            message: "unavailable".into(),
            retryable: true,
            retry_after_ms: None,
        };
        let terminal = ProviderError::UnexpectedStatus {
            status: 418,
            message: "teapot".into(),
            retryable: false,
            retry_after_ms: None,
        };
        assert!(retryable.is_retryable());
        assert!(!terminal.is_retryable());
    }

    #[test]
    fn classified_variants_are_not_retryable() {
        for err in [
            ProviderError::AuthenticationFailed { provider_message: String::new() },
            ProviderError::PermissionDenied { provider_message: String::new() },
            ProviderError::ModelNotFound { provider_message: String::new() },
            ProviderError::ContextWindowExceeded { provider_message: String::new() },
            ProviderError::SafetyFilterTriggered { provider_message: String::new() },
            ProviderError::InvalidResponse { reason: String::new() },
        ] {
            assert!(!err.is_retryable(), "expected terminal: {err:?}");
        }
    }

    #[test]
    fn all_variants_display_non_empty() {
        let variants = [
            ProviderError::AuthenticationFailed { provider_message: "bad key".into() },
            ProviderError::PermissionDenied { provider_message: "nope".into() },
            ProviderError::ModelNotFound { provider_message: "no such model".into() },
            ProviderError::ContextWindowExceeded { provider_message: "too long".into() },
            ProviderError::SafetyFilterTriggered { provider_message: "blocked".into() },
            ProviderError::RateLimited {
                message: "slow".into(),
                status: 429,
                retry_after_ms: Some(1000),
            },
            ProviderError::UnexpectedStatus {
                status: 500,
                message: "boom".into(),
                retryable: true,
                retry_after_ms: None,
            },
            ProviderError::ConnectionFailed { reason: "dns".into() },
            ProviderError::InvalidResponse { reason: "bad json".into() },
        ];
        for v in &variants {
            assert!(!format!("{v}").is_empty(), "empty display: {v:?}");
        }
    }
}
