//! Errors raised by the internal on-disk stores (task records, session transcripts).

use std::fmt;

/// Failures from the task store and session store.
#[derive(Debug)]
pub enum PersistenceError {
    /// A task id that was expected to exist could not be found.
    TaskNotFound(String),
    /// A task write was attempted on a task already marked completed.
    TaskAlreadyCompleted(String),
    /// A task is blocked by another task that has not yet completed.
    TaskBlocked { task_id: String, blocker_id: String },
    /// Acquiring the on-disk lock failed after the configured retry budget.
    LockFailed { attempts: u32 },
    /// Underlying filesystem I/O failed.
    IoFailed(std::io::Error),
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistenceError::TaskNotFound(id) => write!(f, "Task {id} not found"),
            PersistenceError::TaskAlreadyCompleted(id) => {
                write!(f, "Task {id} already completed")
            }
            PersistenceError::TaskBlocked {
                task_id,
                blocker_id,
            } => write!(f, "Task {task_id} blocked by unfinished task {blocker_id}"),
            PersistenceError::LockFailed { attempts } => {
                write!(f, "Failed to acquire lock after {attempts} attempts")
            }
            PersistenceError::IoFailed(err) => write!(f, "Persistence I/O failed: {err}"),
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PersistenceError::IoFailed(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(err: std::io::Error) -> Self {
        PersistenceError::IoFailed(err)
    }
}

impl From<serde_json::Error> for PersistenceError {
    fn from(err: serde_json::Error) -> Self {
        PersistenceError::IoFailed(std::io::Error::new(std::io::ErrorKind::InvalidData, err))
    }
}

/// Result alias used inside `persistence/`. Converts to the crate-level
/// `Result` via `?` at the call boundary.
pub(crate) type PersistenceResult<T> = std::result::Result<T, PersistenceError>;
