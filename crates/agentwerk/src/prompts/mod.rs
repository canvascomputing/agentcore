//! Default behavior, runtime context block, structured-output directive, and the typed `Section` + `PromptBuilder` that compose them into a prompt envelope per the canonical Context → Role → Behavior → Tools → Task order.

mod builder;
mod section;

use std::path::Path;

#[allow(unused_imports)] // Prompt is part of the builder's surface; not yet read internally
pub(crate) use builder::{Prompt, PromptBuilder};
pub(crate) use section::Section;

use crate::util::format_current_date;

/// Default behavioral directives appended to the system prompt after the
/// role prompt. Override with `Agent::behavior()`.
pub const DEFAULT_BEHAVIOR: &str = include_str!("default.behavior.md");

const DEFAULT_CONTEXT_TEMPLATE: &str = include_str!("default.context.md");

/// Build the default context prompt: a `## Context` markdown block with the
/// working directory, platform, OS version, and date. Sent as the first user
/// message when `.context(...)` is not set. Override with `Agent::context()`;
/// inspect with `Agent::default_context()`.
///
/// The `## Context` heading is added by the `Section` view-model — the
/// template file holds only the bullet body.
pub(crate) fn default_context(working_dir: &Path) -> String {
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

pub(crate) const MAX_TOKENS_CONTINUATION: &str =
    include_str!("max-tokens-continuation.directive.md");

pub(crate) const STRUCTURED_OUTPUT_INSTRUCTION: &str =
    include_str!("structured-output.directive.md");

const CONTRACT_RETRY_TEMPLATE: &str = include_str!("contract-retry.directive.md");

/// Render the corrective user message sent after a contract violation.
/// `{detail}` placeholder is filled with the validator's specific complaint.
pub(crate) fn contract_retry(detail: &str) -> String {
    CONTRACT_RETRY_TEMPLATE.replace("{detail}", detail)
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
