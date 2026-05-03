//! Default context block, and the `Section` / `PromptBuilder` that
//! composes the role prompt and (caller-supplied) directives.

mod builder;
mod section;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub use builder::{Prompt, PromptBuilder};
pub(crate) use section::Section;

const DEFAULT_CONTEXT_TEMPLATE: &str = include_str!("default.context.md");

/// Build the default context body: a `## Context` markdown block with the
/// working directory, platform, OS version, and date. Pass the result to
/// `Agent::context(...)` if you want this block as the agent's first user
/// message.
pub fn default_context(working_dir: &Path) -> String {
    let working_dir = working_dir.display().to_string();
    let platform = std::env::consts::OS;
    let os_version = std::process::Command::new("uname")
        .arg("-r")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let date = format_current_date();
    let body = DEFAULT_CONTEXT_TEMPLATE
        .replace("{working_dir}", &working_dir)
        .replace("{platform}", platform)
        .replace("{os_version}", &os_version)
        .replace("{date}", &date);
    Section::context(body).render()
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
    use std::path::PathBuf;

    #[test]
    fn default_context_renders_markdown_block_with_substituted_values() {
        let rendered = default_context(&PathBuf::from("/tmp/check"));
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines[0], "## Context");
        assert_eq!(lines[1], "");
        assert_eq!(lines[2], "- Working directory: /tmp/check");
        assert!(lines[3].starts_with("- Platform: "));
        assert!(lines[4].starts_with("- OS version: "));
        assert!(lines[5].starts_with("- Date: "));
        assert!(!rendered.contains('{'), "no unsubstituted placeholders");
    }
}
