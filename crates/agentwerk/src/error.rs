//! Single error type every fallible API returns, so callers match one `Result` surface instead of a union of provider-, tool-, IO-, and validation-specific errors.

use std::fmt;
use std::time::Duration;

use crate::agent::error::AgentError;
use crate::agent::output::OutputError;
use crate::config::ConfigError;
use crate::persistence::error::PersistenceError;
use crate::provider::ProviderError;
use crate::tools::ToolError;

pub type Result<T> = std::result::Result<T, Error>;

/// Categorical top-level error. Each variant wraps a domain-specific sub-enum
/// that lives beside the code raising it.
#[derive(Debug)]
pub enum Error {
    /// Provider call failures (pre-response HTTP, transport, or parse).
    Provider(ProviderError),
    /// Run-lifecycle failures: cancellation, internal stubs, lifecycle misuse.
    Agent(AgentError),
    /// Tool-system failures raised as `Err` (distinct from in-band
    /// `ToolResult::Error` strings that most tool failures use).
    Tool(ToolError),
    /// Structured-output schema validation failures.
    Output(OutputError),
    /// Task store / session store failures.
    Persistence(PersistenceError),
    /// Configuration failures: env vars, builder misconfiguration, unreadable
    /// prompt files.
    Config(ConfigError),
}

impl Error {
    /// Whether the error should be retried with backoff. Delegates to
    /// [`ProviderError::is_retryable`]; all other categories are terminal.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Error::Provider(p) if p.is_retryable())
    }

    /// Server-suggested retry delay (e.g. `Retry-After`), if present.
    /// Delegates to [`ProviderError::request_retry_delay`].
    pub fn request_retry_delay(&self) -> Option<Duration> {
        match self {
            Error::Provider(p) => p.request_retry_delay(),
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
            Error::Output(err) => write!(f, "{err}"),
            Error::Persistence(err) => write!(f, "{err}"),
            Error::Config(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Provider(err) => Some(err),
            Error::Agent(err) => Some(err),
            Error::Tool(err) => Some(err),
            Error::Output(err) => Some(err),
            Error::Persistence(err) => Some(err),
            Error::Config(err) => Some(err),
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

impl From<OutputError> for Error {
    fn from(err: OutputError) -> Self {
        Error::Output(err)
    }
}

impl From<PersistenceError> for Error {
    fn from(err: PersistenceError) -> Self {
        Error::Persistence(err)
    }
}

impl From<ConfigError> for Error {
    fn from(err: ConfigError) -> Self {
        Error::Config(err)
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
            request_retry_delay: None,
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
            request_retry_delay: Some(Duration::from_millis(500)),
        });
        let terminal = Error::Provider(ProviderError::AuthenticationFailed {
            message: String::new(),
        });
        assert!(retryable.is_retryable());
        assert_eq!(
            retryable.request_retry_delay(),
            Some(Duration::from_millis(500))
        );
        assert!(!terminal.is_retryable());
        assert_eq!(terminal.request_retry_delay(), None);
    }

    #[test]
    fn non_provider_errors_are_not_retryable() {
        let err = Error::Agent(AgentError::Cancelled);
        assert!(!err.is_retryable());
        assert_eq!(err.request_retry_delay(), None);
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
        let err: Error = AgentError::Cancelled.into();
        assert!(matches!(err, Error::Agent(AgentError::Cancelled)));
    }

    #[test]
    fn from_tool_error() {
        let err: Error = ToolError::ContextUnavailable {
            tool_name: "send_message".into(),
            message: "no runtime".into(),
        }
        .into();
        assert!(matches!(
            err,
            Error::Tool(ToolError::ContextUnavailable { .. })
        ));
    }

    #[test]
    fn from_output_error() {
        let err: Error = OutputError::SchemaRetryExhausted { retries: 3 }.into();
        assert!(matches!(
            err,
            Error::Output(OutputError::SchemaRetryExhausted { retries: 3 })
        ));
    }

    #[test]
    fn from_persistence_error() {
        let err: Error = PersistenceError::TaskNotFound("t1".into()).into();
        assert!(matches!(
            err,
            Error::Persistence(PersistenceError::TaskNotFound(_))
        ));
    }

    #[test]
    fn from_config_error() {
        let err: Error = ConfigError::ProviderNotConfigured.into();
        assert!(matches!(
            err,
            Error::Config(ConfigError::ProviderNotConfigured)
        ));
    }

    #[test]
    fn all_variants_display_non_empty() {
        let variants: Vec<Error> = vec![
            Error::Provider(ProviderError::RateLimited {
                message: "slow".into(),
                status: 429,
                request_retry_delay: None,
            }),
            Error::Agent(AgentError::Cancelled),
            Error::Agent(AgentError::NotImplemented("something")),
            Error::Agent(AgentError::PolledAfterCompletion),
            Error::Tool(ToolError::ContextUnavailable {
                tool_name: "t".into(),
                message: "m".into(),
            }),
            Error::Tool(ToolError::ArgumentsRejected {
                tool_name: "t".into(),
                message: "m".into(),
            }),
            Error::Output(OutputError::SchemaViolated {
                path: "/a".into(),
                message: "bad".into(),
            }),
            Error::Output(OutputError::SchemaRetryExhausted { retries: 3 }),
            Error::Persistence(PersistenceError::TaskNotFound("t1".into())),
            Error::Persistence(PersistenceError::TaskAlreadyCompleted("t1".into())),
            Error::Persistence(PersistenceError::TaskBlocked {
                task_id: "t1".into(),
                blocker_id: "t0".into(),
            }),
            Error::Persistence(PersistenceError::LockFailed { attempts: 30 }),
            Error::Persistence(PersistenceError::IoFailed(std::io::Error::new(
                std::io::ErrorKind::Other,
                "io",
            ))),
            Error::Config(ConfigError::EnvVarNotSet("FOO")),
            Error::Config(ConfigError::ProviderNotConfigured),
            Error::Config(ConfigError::FileReadFailed {
                path: "/tmp/x".into(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "x"),
            }),
            Error::Config(ConfigError::Invalid("bad".into())),
        ];
        for variant in &variants {
            let display = format!("{variant}");
            assert!(!display.is_empty(), "Empty display for: {variant:?}");
        }
    }
}
