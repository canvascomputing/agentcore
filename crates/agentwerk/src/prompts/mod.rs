//! Default context block, and the `Section` / `PromptBuilder` that
//! composes the role prompt and (caller-supplied) directives.

mod builder;
mod section;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub use builder::{Prompt, PromptBuilder};
pub(crate) use section::Section;

use crate::agents::policy::Policies;
use crate::agents::stats::Stats;

const DEFAULT_CONTEXT_TEMPLATE: &str = include_str!("default.context.md");

const SCHEMA_RETRY_TEMPLATE: &str = include_str!("schema-retry.directive.md");

/// Render the corrective user message sent after a schema-validation
/// failure. Pushed alongside the failing tool's `ToolResult::SchemaError`
/// content block so the model sees both the error and a directive to
/// reply with a corrected JSON value. Ported from
/// `aa725dc^:crates/agentwerk/src/prompts/contract-retry.directive.md`.
pub(crate) fn schema_retry(detail: &str) -> String {
    SCHEMA_RETRY_TEMPLATE.replace("{detail}", detail)
}

/// Build the default context body: a `## Context` markdown block with the
/// working directory, platform, OS version, and date, plus a `… remaining`
/// bullet for each `Policies` budget that is `Some(_)`. Budgets left as
/// `None` (unlimited) stay invisible. Pass empty `Policies::default()` and
/// `Stats::new()` when you only want the static facts.
pub fn default_context(working_dir: &Path, policies: &Policies, stats: &Stats) -> String {
    let working_dir_str = working_dir.display().to_string();
    let platform = std::env::consts::OS;
    let os_version = std::process::Command::new("uname")
        .arg("-r")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let date = format_current_date();
    let mut body = DEFAULT_CONTEXT_TEMPLATE
        .replace("{working_dir}", &working_dir_str)
        .replace("{platform}", platform)
        .replace("{os_version}", &os_version)
        .replace("{date}", &date);
    if let Some(extra) = runtime_budgets(policies, stats) {
        body.push('\n');
        body.push_str(&extra);
    }
    Section::context(body).render()
}

/// Bullets for each configured budget, joined by `\n`. `None` when no
/// budget is set, so the caller can skip the join.
fn runtime_budgets(policies: &Policies, stats: &Stats) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    if let Some(limit) = policies.max_steps {
        let remaining = u64::from(limit).saturating_sub(stats.steps());
        lines.push(format!("- Steps remaining: {remaining}"));
    }
    if let Some(limit) = policies.max_input_tokens {
        let remaining = limit.saturating_sub(stats.input_tokens());
        lines.push(format!("- Input tokens remaining: {remaining}"));
    }
    if let Some(limit) = policies.max_output_tokens {
        let remaining = limit.saturating_sub(stats.output_tokens());
        lines.push(format!("- Output tokens remaining: {remaining}"));
    }
    if let Some(limit) = policies.max_time {
        if let Some(elapsed) = stats.elapsed() {
            let remaining = limit.saturating_sub(elapsed);
            lines.push(format!("- Time remaining: {}s", remaining.as_secs()));
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

/// Today's date as `YYYY-MM-DD`, via the civil-from-days algorithm.
fn format_current_date() -> String {
    let epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = epoch_secs / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::stats::{LoopStats, TicketStats};
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn schema_retry_substitutes_detail_placeholder() {
        let rendered = schema_retry("expected integer at /partial_sum");
        assert!(rendered.contains("expected integer at /partial_sum"));
        assert!(!rendered.contains("{detail}"));
    }

    #[test]
    fn default_context_renders_markdown_block_with_substituted_values() {
        let rendered = default_context(
            &PathBuf::from("/tmp/check"),
            &Policies::default(),
            &Stats::new(),
        );
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines[0], "## Context");
        assert_eq!(lines[1], "");
        assert_eq!(lines[2], "- Working directory: /tmp/check");
        assert!(lines[3].starts_with("- Platform: "));
        assert!(lines[4].starts_with("- OS version: "));
        assert!(lines[5].starts_with("- Date: "));
        assert!(!rendered.contains('{'), "no unsubstituted placeholders");
    }

    #[test]
    fn default_context_lists_each_set_step_and_token_budget() {
        let working_dir = PathBuf::from("/tmp/check");
        let policies = Policies {
            max_steps: Some(10),
            max_input_tokens: Some(100_000),
            max_output_tokens: Some(20_000),
            ..Policies::default()
        };
        let stats = Stats::new();
        stats.record_step();
        stats.record_step();
        stats.record_request(5_000, 8_000);

        let rendered = default_context(&working_dir, &policies, &stats);

        // Visualizes the exact appended block. Static prefix (working dir,
        // platform, OS version, date) is rebuilt with empty policy/stats
        // so the expected literal stays portable across CI hosts.
        let expected = format!(
            "{static_prefix}\n\
             - Steps remaining: 8\n\
             - Input tokens remaining: 95000\n\
             - Output tokens remaining: 12000",
            static_prefix = default_context(&working_dir, &Policies::default(), &Stats::new()),
        );
        assert_eq!(rendered, expected);
    }

    #[test]
    fn default_context_only_shows_configured_budgets() {
        let working_dir = PathBuf::from("/tmp/check");
        let policies = Policies {
            max_steps: Some(5),
            ..Policies::default()
        };
        let stats = Stats::new();
        stats.record_step();

        let rendered = default_context(&working_dir, &policies, &stats);

        let expected = format!(
            "{static_prefix}\n- Steps remaining: 4",
            static_prefix = default_context(&working_dir, &Policies::default(), &Stats::new()),
        );
        assert_eq!(rendered, expected);
        assert!(!rendered.contains("Input tokens"));
        assert!(!rendered.contains("Output tokens"));
        assert!(!rendered.contains("Time remaining"));
    }

    #[test]
    fn default_context_saturates_remaining_at_zero() {
        let working_dir = PathBuf::from("/tmp/check");
        let policies = Policies {
            max_steps: Some(2),
            ..Policies::default()
        };
        let stats = Stats::new();
        for _ in 0..5 {
            stats.record_step();
        }

        let rendered = default_context(&working_dir, &policies, &stats);

        let expected = format!(
            "{static_prefix}\n- Steps remaining: 0",
            static_prefix = default_context(&working_dir, &Policies::default(), &Stats::new()),
        );
        assert_eq!(rendered, expected);
    }

    #[test]
    fn default_context_omits_time_when_run_not_started() {
        let working_dir = PathBuf::from("/tmp/check");
        let policies = Policies {
            max_time: Some(Duration::from_secs(300)),
            ..Policies::default()
        };
        let stats = Stats::new();

        let rendered = default_context(&working_dir, &policies, &stats);

        // No `record_started` call → `Stats::elapsed` is `None`, so the
        // time bullet must not appear.
        let baseline = default_context(&working_dir, &Policies::default(), &Stats::new());
        assert_eq!(rendered, baseline);
        assert!(!rendered.contains("Time remaining"));
    }

    #[test]
    fn default_context_includes_time_bullet_once_started() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let working_dir = PathBuf::from("/tmp/check");
        let policies = Policies {
            max_time: Some(Duration::from_secs(3600)),
            ..Policies::default()
        };
        let stats = Stats::new();
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        stats.record_started(now_ms);

        let rendered = default_context(&working_dir, &policies, &stats);

        // Exact remaining seconds depend on millisecond-level timing
        // (truncating elapsed > 0ms drops one second), so assert the
        // bullet shape and that the value is within the tight live
        // window (3599 or 3600 seconds).
        let baseline = default_context(&working_dir, &Policies::default(), &Stats::new());
        assert!(rendered.starts_with(&baseline));
        let trailing = &rendered[baseline.len()..];
        assert!(
            trailing == "\n- Time remaining: 3600s"
                || trailing == "\n- Time remaining: 3599s",
            "unexpected runtime block: {trailing:?}",
        );
    }
}
