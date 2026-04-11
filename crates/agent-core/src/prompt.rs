use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::message::{ContentBlock, Message};

/// A named section of the system prompt.
#[derive(Debug, Clone)]
pub struct PromptSection {
    pub name: String,
    pub content: String,
}

/// Type of instruction file, used for labeling in the context message.
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

/// Builds the complete prompt context for an agent turn.
pub struct PromptBuilder {
    base_system_prompt: String,
    sections: Vec<PromptSection>,
    user_context_blocks: Vec<String>,
    memory_content: Option<String>,
    instruction_files: Vec<InstructionFile>,
}

/// Environment information collected once per session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    pub working_directory: String,
    pub platform: String,
    pub os_version: String,
    pub date: String,
}

impl EnvironmentContext {
    /// Collect from the current environment.
    pub fn collect(cwd: &Path) -> Self {
        let os_version = std::process::Command::new("uname")
            .arg("-r")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        let date = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            // Simple date calculation (days since epoch)
            let days = now / 86400;
            // Algorithm from http://howardhinnant.github.io/date_algorithms.html
            let z = days + 719468;
            let era = z / 146097;
            let doe = z - era * 146097;
            let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            let y = yoe + era * 400;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let d = doy - (153 * mp + 2) / 5 + 1;
            let m = if mp < 10 { mp + 3 } else { mp - 9 };
            let y = if m <= 2 { y + 1 } else { y };
            format!("{y:04}-{m:02}-{d:02}")
        };

        Self {
            working_directory: cwd.display().to_string(),
            platform: std::env::consts::OS.to_string(),
            os_version,
            date,
        }
    }
}

impl PromptBuilder {
    pub fn new(base_system_prompt: String) -> Self {
        Self {
            base_system_prompt,
            sections: Vec::new(),
            user_context_blocks: Vec::new(),
            memory_content: None,
            instruction_files: Vec::new(),
        }
    }

    /// Add a named section to the system prompt (appended after base).
    pub fn section(&mut self, name: &str, content: String) -> &mut Self {
        self.sections.push(PromptSection {
            name: name.to_string(),
            content,
        });
        self
    }

    /// Add environment context (cwd, OS, date).
    pub fn environment_context(&mut self, env: &EnvironmentContext) -> &mut Self {
        let content = format!(
            "<environment>\nWorking directory: {}\nPlatform: {}\nOS version: {}\nDate: {}\n</environment>",
            env.working_directory, env.platform, env.os_version, env.date
        );
        self.section("environment", content)
    }

    /// Load and attach memory from a directory (reads MEMORY.md).
    pub fn memory(&mut self, memory_dir: &Path) -> Result<&mut Self> {
        let memory_path = memory_dir.join("MEMORY.md");
        if memory_path.exists() {
            let content = std::fs::read_to_string(&memory_path)?;
            if !content.trim().is_empty() {
                self.memory_content = Some(content);
            }
        }
        Ok(self)
    }

    /// Load instruction files by walking from cwd up to root.
    pub fn instruction_files(&mut self, cwd: &Path) -> Result<&mut Self> {
        let mut dirs = Vec::new();
        let mut current = cwd.to_path_buf();

        loop {
            dirs.push(current.clone());
            match current.parent() {
                Some(parent) if parent != current => current = parent.to_path_buf(),
                _ => break,
            }
        }

        // Process root-first (so cwd files have highest priority / appear last)
        dirs.reverse();
        for dir in &dirs {
            self.try_load_instruction(
                &dir.join("INSTRUCTIONS.md"),
                InstructionType::Project,
            );
            self.try_load_instruction(
                &dir.join(".agent").join("INSTRUCTIONS.md"),
                InstructionType::Project,
            );
            self.try_load_instruction(
                &dir.join("INSTRUCTIONS.local.md"),
                InstructionType::Local,
            );

            let rules_dir = dir.join(".agent").join("rules");
            if let Ok(entries) = std::fs::read_dir(&rules_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "md") {
                        self.try_load_instruction(&path, InstructionType::Rule);
                    }
                }
            }
        }

        // User-level global instructions
        if let Ok(home) = std::env::var("HOME") {
            self.try_load_instruction(
                &Path::new(&home).join(".agent").join("INSTRUCTIONS.md"),
                InstructionType::User,
            );
        }

        Ok(self)
    }

    fn try_load_instruction(&mut self, path: &Path, instruction_type: InstructionType) {
        if let Ok(content) = std::fs::read_to_string(path) {
            if !content.trim().is_empty() {
                self.instruction_files.push(InstructionFile {
                    path: path.display().to_string(),
                    content,
                    instruction_type,
                });
            }
        }
    }

    /// Add arbitrary user context (injected as first user message in <context> tags).
    pub fn user_context(&mut self, context: String) -> &mut Self {
        self.user_context_blocks.push(context);
        self
    }

    /// Build the final system prompt string (all sections concatenated).
    pub fn build_system_prompt(&self) -> String {
        let mut parts = vec![self.base_system_prompt.clone()];
        for section in &self.sections {
            parts.push(section.content.clone());
        }
        parts.join("\n\n")
    }

    /// Build the context user message (memory + instruction files + user context).
    /// Returns None if no context was added.
    pub fn build_context_message(&self) -> Option<Message> {
        let mut parts = Vec::new();

        // Memory
        if let Some(ref memory) = self.memory_content {
            parts.push(format!(
                "<memory>\n{memory}\n</memory>"
            ));
        }

        // Instruction files
        if !self.instruction_files.is_empty() {
            let mut instructions = String::new();
            instructions.push_str("Instructions are shown below. Adhere to these instructions.\n\n");
            for file in &self.instruction_files {
                let label = match file.instruction_type {
                    InstructionType::Project => "project instructions, checked in",
                    InstructionType::Local => "private user instructions, not checked in",
                    InstructionType::Rule => "project rule",
                    InstructionType::User => "user global instructions",
                };
                instructions.push_str(&format!(
                    "Contents of {} ({}):\n\n{}\n\n",
                    file.path, label, file.content
                ));
            }
            parts.push(instructions);
        }

        // User context
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_system_prompt_concatenates_sections() {
        let mut builder = PromptBuilder::new("Base prompt.".into());
        builder.section("rules", "Rule 1.".into());
        builder.section("tools", "Use tools carefully.".into());

        let prompt = builder.build_system_prompt();
        assert!(prompt.contains("Base prompt."));
        assert!(prompt.contains("Rule 1."));
        assert!(prompt.contains("Use tools carefully."));
    }

    #[test]
    fn environment_context_included() {
        let mut builder = PromptBuilder::new("Base.".into());
        let env = EnvironmentContext {
            working_directory: "/home/user/project".into(),
            platform: "linux".into(),
            os_version: "6.1.0".into(),
            date: "2025-01-15".into(),
        };
        builder.environment_context(&env);

        let prompt = builder.build_system_prompt();
        assert!(prompt.contains("/home/user/project"));
        assert!(prompt.contains("linux"));
    }

    #[test]
    fn instruction_file_discovery_walks_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Create a 3-level hierarchy
        let child = root.join("a");
        let grandchild = root.join("a").join("b");
        std::fs::create_dir_all(&grandchild).unwrap();

        std::fs::write(root.join("INSTRUCTIONS.md"), "Root instructions").unwrap();
        std::fs::write(child.join("INSTRUCTIONS.md"), "Child instructions").unwrap();

        let mut builder = PromptBuilder::new("Base.".into());
        builder.instruction_files(&grandchild).unwrap();

        let ctx = builder.build_context_message().unwrap();
        let text = match &ctx {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => text.clone(),
                _ => panic!("Expected text"),
            },
            _ => panic!("Expected user message"),
        };

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

        let mut builder = PromptBuilder::new("Base.".into());
        builder.instruction_files(root).unwrap();

        let ctx = builder.build_context_message().unwrap();
        let text = match &ctx {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => text.clone(),
                _ => panic!("Expected text"),
            },
            _ => panic!("Expected user message"),
        };

        assert!(text.contains("Use snake_case."));
        assert!(text.contains("Write unit tests."));
    }

    #[test]
    fn frontmatter_paths_parsed() {
        // Verify files with YAML frontmatter are still loaded (content after frontmatter)
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let rules_dir = root.join(".agent").join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();

        std::fs::write(
            rules_dir.join("api-rules.md"),
            "---\npaths: [\"src/api/**/*.rs\"]\n---\n\nValidate all API inputs.",
        )
        .unwrap();

        let mut builder = PromptBuilder::new("Base.".into());
        builder.instruction_files(root).unwrap();

        let ctx = builder.build_context_message().unwrap();
        let text = match &ctx {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => text.clone(),
                _ => panic!("Expected text"),
            },
            _ => panic!("Expected user message"),
        };

        assert!(text.contains("Validate all API inputs."));
    }

    #[test]
    fn no_context_message_when_empty() {
        let builder = PromptBuilder::new("Base.".into());
        assert!(builder.build_context_message().is_none());
    }

    #[test]
    fn user_context_injected() {
        let mut builder = PromptBuilder::new("Base.".into());
        builder.user_context("Git status: clean".into());

        let ctx = builder.build_context_message().unwrap();
        let text = match &ctx {
            Message::User { content } => match &content[0] {
                ContentBlock::Text { text } => text.clone(),
                _ => panic!("Expected text"),
            },
            _ => panic!("Expected user message"),
        };

        assert!(text.contains("Git status: clean"));
        assert!(text.contains("<context>"));
    }
}
