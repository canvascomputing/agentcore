//! Single error type every fallible API returns, so callers match one `Result` surface instead of a union of provider-, tool-, IO-, and validation-specific errors.

use std::fmt;
use std::time::Duration;

use crate::agent::error::AgentError;
use crate::provider::ProviderError;
use crate::tools::ToolError;

pub type Result<T> = std::result::Result<T, Error>;

/// Categorical top-level error. Each variant wraps a domain-specific sub-enum
/// that lives beside the code raising it.
#[derive(Debug)]
pub enum Error {
    /// Provider call and construction failures (pre-response HTTP, transport,
    /// parse, or env-based selection).
    Provider(ProviderError),
    /// Agent run-lifecycle and builder failures: cancellation, missing
    /// configuration, lifecycle misuse.
    Agent(AgentError),
    /// Tool-system failures raised as `Err` (distinct from in-band
    /// `ToolResult::Error` strings that most tool failures use).
    Tool(ToolError),
}

impl Error {
    /// Whether the error should be retried with backoff. Delegates to
    /// [`ProviderError::is_retryable`]; all other categories are terminal.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Error::Provider(p) if p.is_retryable())
    }

    /// Server-suggested retry delay (e.g. `Retry-After`), if present.
    /// Delegates to [`ProviderError::retry_delay`].
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            Error::Provider(p) => p.retry_delay(),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Provider(err) => write!(f, "{err}"),
            Error::Agent(err) => write!(f, "{err}"),
            Error::Tool(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Provider(err) => Some(err),
            Error::Agent(err) => Some(err),
            Error::Tool(err) => Some(err),
        }
    }
}

impl From<ProviderError> for Error {
    fn from(err: ProviderError) -> Self {
        Error::Provider(err)
    }
}

impl From<AgentError> for Error {
    fn from(err: AgentError) -> Self {
        Error::Agent(err)
    }
}

impl From<ToolError> for Error {
    fn from(err: ToolError) -> Self {
        Error::Tool(err)
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
            retry_delay: None,
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
            retry_delay: Some(Duration::from_millis(500)),
        });
        let terminal = Error::Provider(ProviderError::AuthenticationFailed {
            message: String::new(),
        });
        assert!(retryable.is_retryable());
        assert_eq!(
            retryable.retry_delay(),
            Some(Duration::from_millis(500))
        );
        assert!(!terminal.is_retryable());
        assert_eq!(terminal.retry_delay(), None);
    }

    #[test]
    fn non_provider_errors_are_not_retryable() {
        let err = Error::Agent(AgentError::AgentCrashed {
            message: "panic".into(),
        });
        assert!(!err.is_retryable());
        assert_eq!(err.retry_delay(), None);
    }

    #[test]
    fn from_provider_error() {
        let err: Error = ProviderError::ConnectionFailed {
            message: "dns".into(),
        }
        .into();
        assert!(matches!(
            err,
            Error::Provider(ProviderError::ConnectionFailed { .. })
        ));
    }

    #[test]
    fn from_agent_error() {
        let err: Error = AgentError::AgentCrashed {
            message: "panic".into(),
        }
        .into();
        assert!(matches!(err, Error::Agent(AgentError::AgentCrashed { .. })));
    }

    #[test]
    fn from_tool_error() {
        let err: Error = ToolError::new("send_message", "no runtime").into();
        assert!(matches!(err, Error::Tool(_)));
    }

    #[test]
    fn all_variants_display_non_empty() {
        let variants: Vec<Error> = vec![
            Error::Provider(ProviderError::RateLimited {
                message: "slow".into(),
                status: 429,
                retry_delay: None,
            }),
            Error::Provider(ProviderError::ProviderUnrecognized {
                message: "no provider".into(),
            }),
            Error::Agent(AgentError::AgentCrashed {
                message: "panic".into(),
            }),
            Error::Tool(ToolError::new("t", "m")),
            Error::Tool(ToolError::new("task", "lock contention")),
        ];
        for variant in &variants {
            let display = format!("{variant}");
            assert!(!display.is_empty(), "Empty display for: {variant:?}");
        }
    }
}
