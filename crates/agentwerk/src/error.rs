//! Single error type every fallible API returns, so callers match one `Result` surface instead of a union of provider-, tool-, IO-, and validation-specific errors.

use std::fmt;

use crate::provider::ProviderError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Provider(ProviderError),
    Tool { tool_name: String, message: String },
    Io(std::io::Error),
    Json(serde_json::Error),
    Cancelled,
    MaxTurnsExceeded(u32),
    ContextOverflow { token_count: u64, limit: u64 },
    SchemaValidation { path: String, message: String },
    SchemaRetryExhausted { retries: u32 },
    NotImplemented(&'static str),
    Other(String),
}

impl Error {
    pub fn is_retryable(&self) -> bool {
        matches!(self, Error::Provider(p) if p.is_retryable())
    }

    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            Error::Provider(p) => p.retry_after_ms(),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Provider(err) => write!(f, "{err}"),
            Error::Tool { tool_name, message } => {
                write!(f, "Tool error ({tool_name}): {message}")
            }
            Error::Io(err) => write!(f, "IO error: {err}"),
            Error::Json(err) => write!(f, "JSON error: {err}"),
            Error::Cancelled => write!(f, "Operation cancelled"),
            Error::MaxTurnsExceeded(n) => write!(f, "Maximum turns exceeded: {n}"),
            Error::ContextOverflow { token_count, limit } => {
                write!(
                    f,
                    "Context overflow: {token_count} tokens exceeds limit of {limit}"
                )
            }
            Error::SchemaValidation { path, message } => {
                write!(f, "Schema validation error at {path}: {message}")
            }
            Error::SchemaRetryExhausted { retries } => {
                write!(f, "Schema retry exhausted after {retries} attempts")
            }
            Error::NotImplemented(what) => {
                write!(f, "Not implemented: {what}")
            }
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Provider(err) => Some(err),
            Error::Io(err) => Some(err),
            Error::Json(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ProviderError> for Error {
    fn from(err: ProviderError) -> Self {
        Error::Provider(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Json(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_delegates_to_provider_error() {
        let err = Error::Provider(ProviderError::RateLimited {
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
        let retryable = Error::Provider(ProviderError::RateLimited {
            message: String::new(),
            status: 429,
            retry_after_ms: Some(500),
        });
        let terminal = Error::Provider(ProviderError::AuthenticationFailed {
            provider_message: String::new(),
        });
        assert!(retryable.is_retryable());
        assert_eq!(retryable.retry_after_ms(), Some(500));
        assert!(!terminal.is_retryable());
        assert_eq!(terminal.retry_after_ms(), None);
    }

    #[test]
    fn non_provider_errors_are_not_retryable() {
        let err = Error::Cancelled;
        assert!(!err.is_retryable());
        assert_eq!(err.retry_after_ms(), None);
    }

    #[test]
    fn from_provider_error() {
        let err: Error = ProviderError::ConnectionFailed {
            reason: "dns".into(),
        }
        .into();
        assert!(matches!(
            err,
            Error::Provider(ProviderError::ConnectionFailed { .. })
        ));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
        assert!(format!("{err}").contains("file not found"));
    }

    #[test]
    fn from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn all_variants_display_non_empty() {
        let variants: Vec<Error> = vec![
            Error::Provider(ProviderError::RateLimited {
                message: "slow".into(),
                status: 429,
                retry_after_ms: None,
            }),
            Error::Tool {
                tool_name: "tool".into(),
                message: "err".into(),
            },
            Error::Io(std::io::Error::new(std::io::ErrorKind::Other, "io")),
            Error::Json(serde_json::from_str::<()>("bad").unwrap_err()),
            Error::Cancelled,
            Error::MaxTurnsExceeded(10),
            Error::ContextOverflow {
                token_count: 200_000,
                limit: 100_000,
            },
            Error::SchemaValidation {
                path: "/a".into(),
                message: "bad".into(),
            },
            Error::SchemaRetryExhausted { retries: 3 },
            Error::NotImplemented("context compaction"),
            Error::Other("other".into()),
        ];
        for variant in &variants {
            let display = format!("{variant}");
            assert!(!display.is_empty(), "Empty display for: {variant:?}");
        }
    }
}
