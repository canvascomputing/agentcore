//! Errors raised when the caller has not supplied enough configuration to build or run an agent: missing env vars, missing builder fields, unreadable prompt files.

use std::fmt;
use std::path::PathBuf;

/// Configuration failures surfaced by `from_env`, the agent builder, and
/// `_file` prompt loaders.
#[derive(Debug)]
pub enum ConfigError {
    /// Required environment variable is missing or empty.
    EnvVarNotSet(&'static str),
    /// The agent builder was finalized without a `.provider(...)` call.
    ProviderNotConfigured,
    /// Reading a prompt or schema file from disk failed.
    FileReadFailed {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Free-form configuration failure that does not yet have a typed variant.
    /// Use sparingly; prefer adding a specific variant.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::EnvVarNotSet(name) => {
                write!(f, "{name} environment variable not set")
            }
            ConfigError::ProviderNotConfigured => {
                write!(f, "Agent::run() requires a provider")
            }
            ConfigError::FileReadFailed { path, source } => {
                write!(f, "Failed to read {}: {source}", path.display())
            }
            ConfigError::Invalid(msg) => write!(f, "Configuration invalid: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::FileReadFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}
