use std::fmt;

pub type Result<T> = std::result::Result<T, AgenticError>;

#[derive(Debug)]
pub enum AgenticError {
    Api {
        message: String,
        status: Option<u16>,
        retryable: bool,
    },
    Tool {
        tool_name: String,
        message: String,
    },
    Io(std::io::Error),
    Json(serde_json::Error),
    Aborted,
    MaxTurnsExceeded(u32),
    BudgetExceeded {
        spent: f64,
        limit: f64,
    },
    ContextOverflow {
        token_count: u64,
        limit: u64,
    },
    SchemaValidation {
        path: String,
        message: String,
    },
    SchemaRetryExhausted {
        retries: u32,
    },
    Other(String),
}

impl fmt::Display for AgenticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgenticError::Api {
                message,
                status,
                retryable,
            } => match status {
                Some(code) => write!(
                    f,
                    "API error (status {code}): {message} (retryable: {retryable})"
                ),
                None => write!(f, "API error: {message} (retryable: {retryable})"),
            },
            AgenticError::Tool { tool_name, message } => {
                write!(f, "Tool error ({tool_name}): {message}")
            }
            AgenticError::Io(err) => write!(f, "IO error: {err}"),
            AgenticError::Json(err) => write!(f, "JSON error: {err}"),
            AgenticError::Aborted => write!(f, "Operation aborted"),
            AgenticError::MaxTurnsExceeded(n) => write!(f, "Maximum turns exceeded: {n}"),
            AgenticError::BudgetExceeded { spent, limit } => {
                write!(f, "Budget exceeded: spent ${spent:.4}, limit ${limit:.4}")
            }
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
            AgenticError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for AgenticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AgenticError::Io(err) => Some(err),
            AgenticError::Json(err) => Some(err),
            _ => None,
        }
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
    fn display_api_error() {
        let err = AgenticError::Api {
            message: "rate limited".into(),
            status: Some(429),
            retryable: true,
        };
        let display = format!("{err}");
        assert!(display.contains("429"));
        assert!(display.contains("rate limited"));
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
    fn budget_exceeded_shows_amounts() {
        let err = AgenticError::BudgetExceeded {
            spent: 5.1234,
            limit: 3.0,
        };
        let display = format!("{err}");
        assert!(display.contains("5.1234"));
        assert!(display.contains("3.0000"));
    }

    #[test]
    fn all_variants_display_non_empty() {
        let variants: Vec<AgenticError> = vec![
            AgenticError::Api {
                message: "msg".into(),
                status: Some(500),
                retryable: false,
            },
            AgenticError::Tool {
                tool_name: "tool".into(),
                message: "err".into(),
            },
            AgenticError::Io(std::io::Error::new(std::io::ErrorKind::Other, "io")),
            AgenticError::Json(serde_json::from_str::<()>("bad").unwrap_err()),
            AgenticError::Aborted,
            AgenticError::MaxTurnsExceeded(10),
            AgenticError::BudgetExceeded {
                spent: 1.0,
                limit: 0.5,
            },
            AgenticError::ContextOverflow {
                token_count: 200_000,
                limit: 100_000,
            },
            AgenticError::SchemaValidation {
                path: "/a".into(),
                message: "bad".into(),
            },
            AgenticError::SchemaRetryExhausted { retries: 3 },
            AgenticError::Other("other".into()),
        ];
        for variant in &variants {
            let display = format!("{variant}");
            assert!(!display.is_empty(), "Empty display for: {variant:?}");
        }
    }
}
