//! Single error type every fallible API returns, so callers match one `Result` surface instead of a union of provider-, tool-, IO-, and validation-specific errors.

use std::fmt;

use crate::provider::ProviderError;

pub type Result<T> = std::result::Result<T, AgenticError>;

#[derive(Debug)]
pub enum AgenticError {
    Provider(ProviderError),
    Tool { tool_name: String, message: String },
    Io(std::io::Error),
    Json(serde_json::Error),
    Aborted,
    MaxTurnsExceeded(u32),
    ContextOverflow { token_count: u64, limit: u64 },
    SchemaValidation { path: String, message: String },
    SchemaRetryExhausted { retries: u32 },
    NotImplemented(&'static str),
    Other(String),
}

impl AgenticError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, AgenticError::Provider(p) if p.is_retryable())
    }

    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            AgenticError::Provider(p) => p.retry_after_ms(),
            _ => None,
        }
    }
}

impl fmt::Display for AgenticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgenticError::Provider(err) => write!(f, "{err}"),
            AgenticError::Tool { tool_name, message } => {
                write!(f, "Tool error ({tool_name}): {message}")
            }
            AgenticError::Io(err) => write!(f, "IO error: {err}"),
            AgenticError::Json(err) => write!(f, "JSON error: {err}"),
            AgenticError::Aborted => write!(f, "Operation aborted"),
            AgenticError::MaxTurnsExceeded(n) => write!(f, "Maximum turns exceeded: {n}"),
            AgenticError::ContextOverflow { token_count, limit } => {
                write!(
                    f,
                    "Context overflow: {token_count} tokens exceeds limit of {limit}"
                )
            }
            AgenticError::SchemaValidation { path, message } => {
                write!(f, "Schema validation error at {path}: {message}")
            }
            AgenticError::SchemaRetryExhausted { retries } => {
                write!(f, "Schema retry exhausted after {retries} attempts")
            }
            AgenticError::NotImplemented(what) => {
                write!(f, "Not implemented: {what}")
            }
            AgenticError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for AgenticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgenticError::Provider(err) => Some(err),
            AgenticError::Io(err) => Some(err),
            AgenticError::Json(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ProviderError> for AgenticError {
    fn from(err: ProviderError) -> Self {
        AgenticError::Provider(err)
    }
}

impl From<std::io::Error> for AgenticError {
    fn from(err: std::io::Error) -> Self {
        AgenticError::Io(err)
    }
}

impl From<serde_json::Error> for AgenticError {
    fn from(err: serde_json::Error) -> Self {
        AgenticError::Json(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_delegates_to_provider_error() {
        let err = AgenticError::Provider(ProviderError::RateLimited {
            message: "rate limited".into(),
            status: 429,
            retry_after_ms: None,
        });
        let display = format!("{err}");
        assert!(display.contains("429"));
        assert!(display.contains("rate limited"));
    }

    #[test]
    fn retry_delegates_to_provider_error() {
        let retryable = AgenticError::Provider(ProviderError::RateLimited {
            message: String::new(),
            status: 429,
            retry_after_ms: Some(500),
        });
        let terminal = AgenticError::Provider(ProviderError::AuthenticationFailed {
            provider_message: String::new(),
        });
        assert!(retryable.is_retryable());
        assert_eq!(retryable.retry_after_ms(), Some(500));
        assert!(!terminal.is_retryable());
        assert_eq!(terminal.retry_after_ms(), None);
    }

    #[test]
    fn non_provider_errors_are_not_retryable() {
        let err = AgenticError::Aborted;
        assert!(!err.is_retryable());
        assert_eq!(err.retry_after_ms(), None);
    }

    #[test]
    fn from_provider_error() {
        let err: AgenticError = ProviderError::ConnectionFailed {
            reason: "dns".into(),
        }
        .into();
        assert!(matches!(
            err,
            AgenticError::Provider(ProviderError::ConnectionFailed { .. })
        ));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AgenticError = io_err.into();
        assert!(matches!(err, AgenticError::Io(_)));
        assert!(format!("{err}").contains("file not found"));
    }

    #[test]
    fn from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let err: AgenticError = json_err.into();
        assert!(matches!(err, AgenticError::Json(_)));
    }

    #[test]
    fn all_variants_display_non_empty() {
        let variants: Vec<AgenticError> = vec![
            AgenticError::Provider(ProviderError::RateLimited {
                message: "slow".into(),
                status: 429,
                retry_after_ms: None,
            }),
            AgenticError::Tool {
                tool_name: "tool".into(),
                message: "err".into(),
            },
            AgenticError::Io(std::io::Error::new(std::io::ErrorKind::Other, "io")),
            AgenticError::Json(serde_json::from_str::<()>("bad").unwrap_err()),
            AgenticError::Aborted,
            AgenticError::MaxTurnsExceeded(10),
            AgenticError::ContextOverflow {
                token_count: 200_000,
                limit: 100_000,
            },
            AgenticError::SchemaValidation {
                path: "/a".into(),
                message: "bad".into(),
            },
            AgenticError::SchemaRetryExhausted { retries: 3 },
            AgenticError::NotImplemented("context compaction"),
            AgenticError::Other("other".into()),
        ];
        for variant in &variants {
            let display = format!("{variant}");
            assert!(!display.is_empty(), "Empty display for: {variant:?}");
        }
    }
}
