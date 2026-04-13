//! Prompt infrastructure: behavior defaults, context building, and constants.
//!
//! `ContextBuilder` is internal — callers use `AgentBuilder` methods instead.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::provider::types::{ContentBlock, Message};

// ---------------------------------------------------------------------------
// Behavior prompts
// ---------------------------------------------------------------------------

/// Behavioral directives that govern how an agent operates.
/// Each variant has a default that can be replaced via `AgentBuilder::behavior_prompt()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BehaviorPrompt {
    TaskExecution,
    ToolUsage,
    ActionSafety,
    OutputEfficiency,
}

impl BehaviorPrompt {
    pub fn default_content(&self) -> &'static str {
        match self {
            Self::TaskExecution => DEFAULT_DOING_TASKS,
            Self::ToolUsage => DEFAULT_USING_TOOLS,
            Self::ActionSafety => DEFAULT_ACTIONS_CARE,
            Self::OutputEfficiency => DEFAULT_OUTPUT_EFFICIENCY,
        }
    }

    /// All variants in the order they should appear in the system prompt.
    pub(crate) fn all() -> &'static [BehaviorPrompt] {
        &[
            Self::TaskExecution,
            Self::ToolUsage,
            Self::ActionSafety,
            Self::OutputEfficiency,
        ]
    }
}

// ---------------------------------------------------------------------------
// Default behavior content
// ---------------------------------------------------------------------------

pub const DEFAULT_DOING_TASKS: &str = "\
# Doing tasks
- Do not propose changes to files you have not read. Read first, then modify.
- Do not add features or improvements beyond what was asked.
- Do not create files unless absolutely necessary. Prefer editing existing files.
- If an approach fails, diagnose why before switching tactics.";

pub const DEFAULT_USING_TOOLS: &str = "\
# Using your tools
- Do NOT use bash when a dedicated tool exists (read_file over cat, edit_file over sed, grep over rg, glob over find).
- You can call multiple tools in a single response. Make independent calls in parallel for efficiency.
- If tool calls depend on previous results, make them sequentially — do not guess parameters.";

pub const DEFAULT_ACTIONS_CARE: &str = "\
# Executing actions with care
- Consider the reversibility and blast radius of actions before executing them.
- For destructive or hard-to-reverse operations (deleting files, force-push, dropping data), confirm with the user first.
- If an approach fails, diagnose why before switching tactics. \
Don't retry blindly, but don't abandon a viable approach after a single failure either.";

pub const DEFAULT_OUTPUT_EFFICIENCY: &str = "\
# Output efficiency
- Go straight to the point. Try the simplest approach first.
- Keep text output brief and direct. Lead with the answer or action, not the reasoning.
- Skip filler words, preamble, and unnecessary transitions.
- If you can say it in one sentence, do not use three.";

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

/// Environment information collected once per session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    pub working_directory: String,
    pub platform: String,
    pub os_version: String,
    pub date: String,
}

impl EnvironmentContext {
    pub fn collect(cwd: &Path) -> Self {
        let os_version = std::process::Command::new("uname")
            .arg("-r")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        Self {
            working_directory: cwd.display().to_string(),
            platform: std::env::consts::OS.to_string(),
            os_version,
            date: format_current_date(),
        }
    }
}

// ---------------------------------------------------------------------------
// Context builder (internal)
// ---------------------------------------------------------------------------

/// Builds the context user message (memory, instructions, environment).
/// Not public — callers use `AgentBuilder` methods instead.
#[derive(Clone)]
pub(crate) struct ContextBuilder {
    sections: Vec<(String, String)>,
    user_context_blocks: Vec<String>,
    memory_content: Option<String>,
    instruction_files: Vec<InstructionFile>,
}

impl ContextBuilder {
    pub(crate) fn new() -> Self {
        Self {
            sections: Vec::new(),
            user_context_blocks: Vec::new(),
            memory_content: None,
            instruction_files: Vec::new(),
        }
    }

    pub(crate) fn environment_context(&mut self, env: &EnvironmentContext) -> &mut Self {
        let content = format!(
            "<environment>\nWorking directory: {}\nPlatform: {}\nOS version: {}\nDate: {}\n</environment>",
            env.working_directory, env.platform, env.os_version, env.date
        );
        self.sections.push(("environment".into(), content));
        self
    }

    pub(crate) fn memory(&mut self, memory_dir: &Path) -> Result<&mut Self> {
        let memory_path = memory_dir.join("MEMORY.md");
        if memory_path.exists() {
            let content = std::fs::read_to_string(&memory_path)?;
            if !content.trim().is_empty() {
                self.memory_content = Some(content);
            }
        }
        Ok(self)
    }

    pub(crate) fn user_context(&mut self, context: String) -> &mut Self {
        self.user_context_blocks.push(context);
        self
    }

    pub(crate) fn instruction_files(&mut self, cwd: &Path) -> Result<&mut Self> {
        let ancestor_dirs = ancestors_root_first(cwd);

        for dir in &ancestor_dirs {
            self.try_load_instruction(&dir.join("INSTRUCTIONS.md"), InstructionType::Project);
            self.try_load_instruction(
                &dir.join(".agent").join("INSTRUCTIONS.md"),
                InstructionType::Project,
            );
            self.try_load_instruction(
                &dir.join("INSTRUCTIONS.local.md"),
                InstructionType::Local,
            );
            self.load_rules_directory(&dir.join(".agent").join("rules"));
        }

        if let Ok(home) = std::env::var("HOME") {
            self.try_load_instruction(
                &Path::new(&home).join(".agent").join("INSTRUCTIONS.md"),
                InstructionType::User,
            );
        }

        Ok(self)
    }

    /// Build the context user message (environment + memory + instruction files + user context).
    /// Returns None if no context was added.
    pub(crate) fn build_context_message(&self) -> Option<Message> {
        let mut parts = Vec::new();

        for (_, content) in &self.sections {
            parts.push(content.clone());
        }

        if let Some(ref memory) = self.memory_content {
            parts.push(format!("<memory>\n{memory}\n</memory>"));
        }

        if !self.instruction_files.is_empty() {
            parts.push(self.format_instructions());
        }

        for ctx in &self.user_context_blocks {
            parts.push(format!("<context>\n{ctx}\n</context>"));
        }

        if parts.is_empty() {
            return None;
        }

        Some(Message::User {
            content: vec![ContentBlock::Text {
                text: parts.join("\n\n"),
            }],
        })
    }

    fn try_load_instruction(&mut self, path: &Path, instruction_type: InstructionType) {
        let Ok(content) = std::fs::read_to_string(path) else {
            return;
        };
        if content.trim().is_empty() {
            return;
        }
        self.instruction_files.push(InstructionFile {
            path: path.display().to_string(),
            content,
            instruction_type,
        });
    }

    fn load_rules_directory(&mut self, rules_dir: &Path) {
        let Ok(entries) = std::fs::read_dir(rules_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "md") {
                self.try_load_instruction(&path, InstructionType::Rule);
            }
        }
    }

    fn format_instructions(&self) -> String {
        let mut output = String::from("Instructions are shown below. Adhere to these instructions.\n\n");
        for file in &self.instruction_files {
            let label = match file.instruction_type {
                InstructionType::Project => "project instructions, checked in",
                InstructionType::Local => "private user instructions, not checked in",
                InstructionType::Rule => "project rule",
                InstructionType::User => "user global instructions",
            };
            output.push_str(&format!(
                "Contents of {} ({}):\n\n{}\n\n",
                file.path, label, file.content
            ));
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum InstructionType {
    Project,
    Local,
    Rule,
    User,
}

#[derive(Debug, Clone)]
struct InstructionFile {
    path: String,
    content: String,
    instruction_type: InstructionType,
}

fn ancestors_root_first(cwd: &Path) -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    let mut current = cwd.to_path_buf();
    loop {
        dirs.push(current.clone());
        match current.parent() {
            Some(parent) if parent != current => current = parent.to_path_buf(),
            _ => break,
        }
    }
    dirs.reverse();
    dirs
}

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
        let mut builder = ContextBuilder::new();
        let env = EnvironmentContext {
            working_directory: "/home/user/project".into(),
            platform: "linux".into(),
            os_version: "6.1.0".into(),
            date: "2025-01-15".into(),
        };
        builder.environment_context(&env);

        let text = extract_context_text(&builder);
        assert!(text.contains("/home/user/project"));
        assert!(text.contains("linux"));
    }

    #[test]
    fn instruction_file_discovery_walks_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let child = root.join("a");
        let grandchild = root.join("a").join("b");
        std::fs::create_dir_all(&grandchild).unwrap();

        std::fs::write(root.join("INSTRUCTIONS.md"), "Root instructions").unwrap();
        std::fs::write(child.join("INSTRUCTIONS.md"), "Child instructions").unwrap();

        let mut builder = ContextBuilder::new();
        builder.instruction_files(&grandchild).unwrap();

        let text = extract_context_text(&builder);
        assert!(text.contains("Root instructions"));
        assert!(text.contains("Child instructions"));
    }

    #[test]
    fn agent_rules_directory_discovered() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let rules_dir = root.join(".agent").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();

        std::fs::write(rules_dir.join("rust-conventions.md"), "Use snake_case.").unwrap();
        std::fs::write(rules_dir.join("test-patterns.md"), "Write unit tests.").unwrap();

        let mut builder = ContextBuilder::new();
        builder.instruction_files(root).unwrap();

        let text = extract_context_text(&builder);
        assert!(text.contains("Use snake_case."));
        assert!(text.contains("Write unit tests."));
    }

    #[test]
    fn frontmatter_paths_parsed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let rules_dir = root.join(".agent").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();

        std::fs::write(
            rules_dir.join("api-rules.md"),
            "---\npaths: [\"src/api/**/*.rs\"]\n---\n\nValidate all API inputs.",
        )
        .unwrap();

        let mut builder = ContextBuilder::new();
        builder.instruction_files(root).unwrap();

        let text = extract_context_text(&builder);
        assert!(text.contains("Validate all API inputs."));
    }

    #[test]
    fn no_context_message_when_empty() {
        let builder = ContextBuilder::new();
        assert!(builder.build_context_message().is_none());
    }

    #[test]
    fn user_context_injected() {
        let mut builder = ContextBuilder::new();
        builder.user_context("Git status: clean".into());

        let text = extract_context_text(&builder);
        assert!(text.contains("Git status: clean"));
        assert!(text.contains("<context>"));
    }

    fn extract_context_text(builder: &ContextBuilder) -> String {
        let ctx = builder.build_context_message().unwrap();
        match &ctx {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => text.clone(),
                _ => panic!("Expected text"),
            },
            _ => panic!("Expected user message"),
        }
    }
}
