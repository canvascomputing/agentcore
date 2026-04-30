//! Shared helpers for the file and shell tools — glob matching and process invocation.

use std::time::Duration;

use super::tool::{ToolContext, ToolResult};

/// Execute a command via `sh -c`, returning combined stdout/stderr. Bounded by
/// `timeout` and interruptible via the context's cancel signal: a fired
/// cancel drops the pending output future, and `kill_on_drop(true)` cascades
/// SIGKILL to the subprocess so a hanging `python3` / `sleep` doesn't outlive
/// the cancel.
pub(crate) async fn run_shell_command(
    command: &str,
    timeout: Duration,
    ctx: &ToolContext,
) -> ToolResult {
    let output_fut = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&ctx.working_dir)
        .kill_on_drop(true)
        .output();

    let result = tokio::select! {
        biased;
        _ = ctx.wait_for_cancel() => return ToolResult::error("Command cancelled"),
        r = tokio::time::timeout(timeout, output_fut) => r,
    };

    match result {
        Err(_) => ToolResult::error(format!("Command timed out after {}ms", timeout.as_millis())),
        Ok(Err(e)) => ToolResult::error(format!("Failed to execute command: {e}")),
        Ok(Ok(output)) => {
            let mut content = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !stderr.is_empty() {
                content.push_str("\n--- stderr ---\n");
                content.push_str(&stderr);
            }

            if output.status.success() {
                ToolResult::success(content)
            } else {
                ToolResult::error(content)
            }
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
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    #[tokio::test]
    async fn cancel_interrupts_long_running_subprocess() {
        let ctx = ToolContext::new(std::env::current_dir().unwrap());
        let flag = ctx.interrupt_signal.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            flag.store(true, Ordering::Relaxed);
        });

        let started = Instant::now();
        let result = run_shell_command("sleep 30", Duration::from_millis(60_000), &ctx).await;
        let elapsed = started.elapsed();

        let (ToolResult::Success(content) | ToolResult::Error(content) | ToolResult::SchemaError(content)) = &result;
        assert!(
            matches!(result, ToolResult::Error(_)),
            "expected cancelled result"
        );
        assert!(content.contains("cancelled"));
        assert!(
            elapsed < Duration::from_millis(500),
            "cancel should return within 500ms, took {elapsed:?}",
        );
    }

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
