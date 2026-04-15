use std::path::Path;
use std::time::Duration;

use crate::tools::tool::ToolResult;

pub(crate) const DEFAULT_TIMEOUT_MS: u64 = 120_000;
pub(crate) const MAX_TIMEOUT_MS: u64 = 600_000;

/// Execute a command via `sh -c` with a timeout, returning combined stdout/stderr.
pub(crate) async fn run_shell_command(
    command: &str,
    working_directory: &Path,
    timeout_ms: u64,
) -> ToolResult {
    let result = tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(working_directory)
            .output(),
    )
    .await;

    match result {
        Err(_) => ToolResult::error(format!("Command timed out after {timeout_ms}ms")),
        Ok(Err(e)) => ToolResult::error(format!("Failed to execute command: {e}")),
        Ok(Ok(output)) => {
            let mut content = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !stderr.is_empty() {
                content.push_str("\n--- stderr ---\n");
                content.push_str(&stderr);
            }

            let is_error = !output.status.success();
            ToolResult { content, is_error }
        }
    }
}

/// Simple glob matching supporting `*` (any chars) and `?` (single char).
pub(crate) fn glob_match(pattern: &str, text: &str) -> bool {
    glob_match_bytes(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_bytes(pattern: &[u8], text: &[u8]) -> bool {
    if pattern.is_empty() {
        return text.is_empty();
    }

    match pattern[0] {
        b'*' => {
            // Star matches zero chars (skip star) or one char (advance text).
            glob_match_bytes(&pattern[1..], text)
                || (!text.is_empty() && glob_match_bytes(pattern, &text[1..]))
        }
        b'?' if text.is_empty() => false,
        b'?' => glob_match_bytes(&pattern[1..], &text[1..]),
        literal if text.is_empty() || text[0] != literal => false,
        _ => glob_match_bytes(&pattern[1..], &text[1..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        assert!(glob_match("hello", "hello"));
        assert!(!glob_match("hello", "world"));
    }

    #[test]
    fn star_wildcard() {
        assert!(glob_match("git *", "git status"));
        assert!(glob_match("git *", "git log --oneline"));
        assert!(!glob_match("git *", "cargo build"));
        assert!(!glob_match("git *", "git"));
    }

    #[test]
    fn question_mark_wildcard() {
        assert!(glob_match("?.rs", "a.rs"));
        assert!(!glob_match("?.rs", "ab.rs"));
    }

    #[test]
    fn match_all() {
        assert!(glob_match("*", "anything goes"));
        assert!(glob_match("*", ""));
    }
}
