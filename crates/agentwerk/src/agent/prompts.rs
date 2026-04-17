//! Prompt infrastructure: behavior defaults, context building, and constants.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::provider::types::{ContentBlock, Message};

// ---------------------------------------------------------------------------
// Behavior prompts
// ---------------------------------------------------------------------------

/// Behavioral directives injected into the system prompt of every LLM request.
///
/// All four variants are always present. Each has a sensible default that can be
/// replaced via [`AgentBuilder::behavior_prompt()`]. They appear in the system
/// prompt after the identity prompt, in the order listed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BehaviorPrompt {
    /// How the agent approaches work.
    TaskExecution,
    /// When to use which tool.
    ToolUsage,
    /// Awareness of consequences before acting.
    SafetyConcerns,
    /// How the agent structures its output.
    Communication,
}

impl BehaviorPrompt {
    pub(crate) fn default_content(&self) -> &'static str {
        match self {
            Self::TaskExecution => DEFAULT_TASK_EXECUTION,
            Self::ToolUsage => DEFAULT_TOOL_USAGE,
            Self::SafetyConcerns => DEFAULT_SAFETY_CONCERNS,
            Self::Communication => DEFAULT_COMMUNICATION_STYLE,
        }
    }

    /// All variants in the order they should appear in the system prompt.
    pub(crate) fn all() -> &'static [BehaviorPrompt] {
        &[
            Self::TaskExecution,
            Self::ToolUsage,
            Self::SafetyConcerns,
            Self::Communication,
        ]
    }
}

// ---------------------------------------------------------------------------
// Default behavior content
// ---------------------------------------------------------------------------

pub(crate) const DEFAULT_TASK_EXECUTION: &str = "\
# Task execution
- Do not propose changes to files you have not read. Read first, then modify.
- Do not add features or improvements beyond what was asked.
- Do not create files unless absolutely necessary. Prefer editing existing files.
- If an approach fails, diagnose why before switching tactics.";

pub(crate) const DEFAULT_TOOL_USAGE: &str = "\
# Tool usage
- Do NOT use bash when a dedicated tool exists (read_file over cat, edit_file over sed, grep over rg, glob over find).
- Call multiple tools in a single response. Make independent calls in parallel.
- If tool calls depend on previous results, call them sequentially — do not guess parameters.";

pub(crate) const DEFAULT_SAFETY_CONCERNS: &str = "\
# Safety concerns
- Consider the reversibility and impact of actions before executing them.
- Prefer reversible operations over destructive ones when both achieve the goal.
- If an approach fails, diagnose the root cause before retrying or switching tactics.";

pub(crate) const DEFAULT_COMMUNICATION_STYLE: &str = "\
# Communication
- Be direct. Lead with the answer or action, not the reasoning.
- Keep output concise — omit filler, preamble, and unnecessary transitions.
- Try the simplest approach first.";

// ---------------------------------------------------------------------------
// Structured output constants
// ---------------------------------------------------------------------------

pub(crate) const STRUCTURED_OUTPUT_INSTRUCTION: &str =
    "\n\nIMPORTANT: You must provide your final response using the StructuredOutput tool \
     with the required structured format. After using any other tools needed to complete \
     the task, always call StructuredOutput with your final answer in the specified schema.";

pub(crate) const STRUCTURED_OUTPUT_RETRY: &str =
    "You MUST call the StructuredOutput tool to complete \
     this request. Call this tool now with the required schema.";

pub(crate) const STRUCTURED_OUTPUT_TOOL_DESCRIPTION: &str =
    "Return your final response using the required output schema. \
     Call this tool exactly once at the end to provide the structured result.";

pub(crate) const STRUCTURED_OUTPUT_TOOL_NAME: &str = "StructuredOutput";

// ---------------------------------------------------------------------------
// Template interpolation
// ---------------------------------------------------------------------------

/// Replace {key} placeholders in a template with values from state.
pub(crate) fn interpolate(template: &str, state: &HashMap<String, Value>) -> String {
    let mut result = template.to_string();
    for (key, value) in state {
        let replacement = match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        result = result.replace(&format!("{{{key}}}"), &replacement);
    }
    result
}

// ---------------------------------------------------------------------------
// Environment context
// ---------------------------------------------------------------------------

/// Collect environment information and format it as a tagged string for the context message.
pub(crate) fn collect_environment_context(cwd: &Path) -> String {
    let working_directory = cwd.display();
    let platform = std::env::consts::OS;
    let os_version = std::process::Command::new("uname")
        .arg("-r")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let date = format_current_date();

    format!(
        "<environment>\nWorking directory: {working_directory}\nPlatform: {platform}\nOS version: {os_version}\nDate: {date}\n</environment>"
    )
}

/// Build a user message from pre-formatted parts. Returns `None` if empty.
pub(crate) fn build_message(parts: &[String]) -> Option<Message> {
    if parts.is_empty() {
        return None;
    }
    Some(Message::User {
        content: vec![ContentBlock::Text {
            text: parts.join("\n\n"),
        }],
    })
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert epoch seconds to a date string using the civil-from-days algorithm.
/// http://howardhinnant.github.io/date_algorithms.html
fn format_current_date() -> String {
    let epoch_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
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

    #[test]
    fn behavior_prompt_defaults_non_empty() {
        for kind in BehaviorPrompt::all() {
            assert!(!kind.default_content().is_empty());
        }
    }

    #[test]
    fn environment_context_in_message() {
        let ctx = collect_environment_context(std::path::Path::new("/home/user/project"));
        let parts = vec![ctx];
        let text = extract_message_text(&parts);
        assert!(text.contains("/home/user/project"));
        assert!(text.contains("<environment>"));
    }

    #[test]
    fn no_message_when_empty() {
        assert!(build_message(&[]).is_none());
    }

    #[test]
    fn user_context_injected() {
        let parts = vec![format!("<context>\nGit status: clean\n</context>")];
        let text = extract_message_text(&parts);
        assert!(text.contains("Git status: clean"));
        assert!(text.contains("<context>"));
    }

    fn extract_message_text(parts: &[String]) -> String {
        let msg = build_message(parts).unwrap();
        match &msg {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => text.clone(),
                _ => panic!("Expected text"),
            },
            _ => panic!("Expected user message"),
        }
    }

}
