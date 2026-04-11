# Prompt Construction

## Overview

The `PromptBuilder` assembles the system prompt and context messages from multiple composable layers: static instructions, environment context, instruction file discovery, persistent memory, and user-provided context. This is one of the most important subsystems for agentic behavior.

## Dependencies

- [Core types](../1-base/types.md): `Message`, `ContentBlock`, `AgenticError`, `Result`

## Files

```
crates/agent-core/src/prompt.rs
```

## Specification

### 3. Prompt Construction (`prompt.rs`)

#### Feature Classification

| Feature | Classification | Rationale |
|---------|---------------|-----------|
| `PromptBuilder::new()`, `section()`, `build_system_prompt()`, `build_context_message()` | **Core** | Essential prompt assembly |
| `environment_context()`, `EnvironmentContext` | **Core** | Agents need to know their runtime environment |
| `user_context()` | **Core** | Injecting arbitrary context |
| `PromptSection` | **Core** | Data type for sections |
| `instruction_files()` — walk cwd→root | **Nice-to-have** | Opinionated directory layout; applications can call `section()` with their own discovery logic |
| `memory()` — load MEMORY.md | **Nice-to-have** | Opinionated file format; applications can load and inject memory via `user_context()` |
| YAML frontmatter parsing for conditional rules | **Nice-to-have** | Complex feature for path-conditional rules; defer to application layer |
| `InstructionType` enum | **Nice-to-have** | Only needed if instruction_files() is implemented |

#### 3.1 Architecture

A well-designed agentic system constructs its system prompt in these layers:

1. **Static instructions** — core identity, behavioral rules, tool usage guidance (cacheable globally)
2. **Dynamic boundary** — `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` marker separates cacheable from session-specific content
3. **Dynamic sections** — environment info, memory, session guidance (per-session)
4. **User context** — project instruction files discovered by traversing from cwd to root (injected as the first user message, wrapped in XML tags)
5. **System context** — date, OS info (appended to system prompt)
6. **Memory** — persistent memory files loaded from a project-specific memory directory
7. **Tool definitions** — sent separately via the API `tools` parameter (not embedded in system prompt text)

#### 3.2 Rust Design

```rust
/// A named section of the system prompt.
#[derive(Debug, Clone)]
pub struct PromptSection {
    pub name: String,
    pub content: String,
}

/// Builds the complete prompt context for an agent turn.
pub struct PromptBuilder {
    base_system_prompt: String,
    sections: Vec<PromptSection>,       // dynamic sections appended to system prompt
    user_context_blocks: Vec<String>,   // prepended as first user message
    memory_content: Option<String>,     // loaded from memory files
}

impl PromptBuilder {
    pub fn new(base_system_prompt: String) -> Self;

    /// Add a named section to the system prompt (appended after base).
    pub fn section(&mut self, name: &str, content: String) -> &mut Self;

    /// Add environment context (cwd, OS, date).
    pub fn environment_context(&mut self, env: &EnvironmentContext) -> &mut Self;

    /// Load and attach memory from a directory (reads MEMORY.md).
    pub fn memory(&mut self, memory_dir: &Path) -> Result<&mut Self>;

    /// Load instruction files by walking from cwd up to root.
    /// Discovers: INSTRUCTIONS.md, .agent/INSTRUCTIONS.md, .agent/rules/*.md
    pub fn instruction_files(&mut self, cwd: &Path) -> Result<&mut Self>;

    /// Add arbitrary user context (injected as first user message in <context> tags).
    pub fn user_context(&mut self, context: String) -> &mut Self;

    /// Build the final system prompt string (all sections concatenated).
    pub fn build_system_prompt(&self) -> String;

    /// Build the context user message (memory + instruction files + user context).
    /// Returns None if no context was added.
    /// This becomes the first message in the conversation, before the user's actual prompt.
    pub fn build_context_message(&self) -> Option<Message>;
}

/// Environment information collected once per session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    pub working_directory: String,
    pub platform: String,           // "darwin", "linux", "windows"
    pub os_version: String,
    pub date: String,               // ISO 8601 date
}

impl EnvironmentContext {
    /// Collect from the current environment.
    pub fn collect(cwd: &Path) -> Self;
}
```

**Example: Assembling a prompt with environment and instruction files**

```rust
let cwd = std::env::current_dir()?;

let mut prompt_builder = PromptBuilder::new(
    "You are a helpful coding assistant. Follow the user's instructions carefully.".into(),
);

// Add environment context (auto-detected)
let env = EnvironmentContext::collect(&cwd);
prompt_builder.environment_context(&env);

// Discover and load project instruction files (walks cwd → root)
prompt_builder.instruction_files(&cwd)?;

// Add a custom section
prompt_builder.section("guidelines", "Always write tests for new code.".into());

// Build the final prompt pieces
let system_prompt = prompt_builder.build_system_prompt();
// → "You are a helpful coding assistant...\n\n<environment>\nWorking directory: /home/user/project\n..."

let context_message = prompt_builder.build_context_message();
// → Some(Message::User { content: "<context>\nContents of INSTRUCTIONS.md (project instructions):\n..." })

// Pass to an agent via AgentBuilder
let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .prompt_builder(prompt_builder)
    .build()?;
```

#### 3.3 How Context Flows Through the Agent Loop

The context assembly proceeds in a pipeline from build-time configuration to runtime API call.

The `AgentBuilder` is configured with a base system prompt and a `PromptBuilder` (referred to as context). When `Agent.run(user_message)` is called, two build steps happen in sequence:

1. `ctx.build_system_prompt()` produces the system prompt string for the API. It concatenates the base prompt, followed by each section's content separated by double newlines.

2. `ctx.build_context_message()` produces an optional `Message::User` containing memory, instruction files, and user context wrapped in XML tags. If present, this message is prepended before the user's actual message.

These are assembled into a `CompletionRequest` containing: the system prompt from step 1; a messages array starting with the optional context message, then the user message, then any conversation history; and the tool definitions from the registry (sent separately, not embedded in the prompt text). This request is then passed to `provider.complete(request)`.

#### 3.4 Instruction File Discovery (Nice-to-have)

Walks from working directory up to filesystem root, collecting instruction files:

```
cwd/INSTRUCTIONS.md                    (project-level instructions)
cwd/.agent/INSTRUCTIONS.md           (project-level, hidden dir)
cwd/.agent/rules/*.md                (project-level rules)
../INSTRUCTIONS.md                     (parent directory)
../../INSTRUCTIONS.md                  (grandparent, etc.)
~/.agent/INSTRUCTIONS.md             (user-level global instructions)
```

These are concatenated (child overrides parent for conflicts) and injected as a context user message. Later files (closer to cwd) have higher priority.

#### 3.5 Persistent Memory (Nice-to-have)

Memory files persist knowledge across sessions — coding conventions, architectural decisions, user preferences. They are discovered automatically and injected into the system prompt so the agent starts each session with project context.

##### Directory Layout

```
~/.agent/INSTRUCTIONS.md                       # user-level global instructions
project/
├── INSTRUCTIONS.md                             # project-level instructions (checked in)
├── INSTRUCTIONS.local.md                       # private user instructions (gitignored)
├── .agent/
│   ├── INSTRUCTIONS.md                         # alternative project location
│   └── rules/
│       ├── rust-conventions.md                 # conditional rule files
│       └── test-patterns.md
└── src/
    └── api/
        └── INSTRUCTIONS.md                     # subdirectory-specific instructions
```

##### Discovery Algorithm

Instruction files are discovered by walking from the working directory up to the filesystem root:

```rust
impl PromptBuilder {
    pub fn instruction_files(&mut self, cwd: &Path) -> Result<&mut Self> {
        let mut dirs = Vec::new();
        let mut current = cwd.to_path_buf();

        // Collect directories from cwd to root
        loop {
            dirs.push(current.clone());
            match current.parent() {
                Some(parent) if parent != current => current = parent.to_path_buf(),
                _ => break,
            }
        }

        // Process root-first (so cwd files have highest priority)
        dirs.reverse();
        for dir in &dirs {
            self.try_load(&dir.join("INSTRUCTIONS.md"), InstructionType::Project);
            self.try_load(&dir.join(".agent/INSTRUCTIONS.md"), InstructionType::Project);
            self.try_load(&dir.join("INSTRUCTIONS.local.md"), InstructionType::Local);

            // Load all .md files in .agent/rules/
            if let Ok(entries) = std::fs::read_dir(dir.join(".agent/rules")) {
                for entry in entries.flatten() {
                    if entry.path().extension() == Some("md".as_ref()) {
                        self.try_load(&entry.path(), InstructionType::Rule);
                    }
                }
            }
        }

        // User-level global instructions (lowest priority)
        let home = dirs!("HOME").unwrap_or_default();
        self.try_load(&Path::new(&home).join(".agent/INSTRUCTIONS.md"), InstructionType::User);

        Ok(self)
    }
}
```

##### File Format

Plain markdown. Optional YAML frontmatter for conditional rules:

```markdown
---
paths: ["src/api/**/*.rs", "tests/api_*"]
---

API endpoint handlers must validate input before processing.
Always return structured error responses with status codes.
```

The `paths` field contains glob patterns — the rule is only injected when the conversation involves files matching those patterns.

##### Injection Into System Prompt

Discovered files are appended to the system prompt with labels indicating their source:

```rust
fn format_instruction_files(&self) -> String {
    let mut result = String::new();
    result.push_str("Instructions are shown below. Adhere to these instructions.\n\n");

    for file in &self.instruction_files {
        let label = match file.instruction_type {
            InstructionType::Project => "project instructions, checked in",
            InstructionType::Local => "private user instructions, not checked in",
            InstructionType::Rule => "project rule",
            InstructionType::User => "user global instructions",
        };
        result.push_str(&format!(
            "Contents of {} ({}):\n\n{}\n\n",
            file.path.display(), label, file.content
        ));
    }
    result
}
```

##### Writing Memory

The agent writes to instruction files using the standard `write_file` and `edit_file` tools. The user tells the agent to remember something, and the agent writes it to the appropriate `INSTRUCTIONS.md` or rule file. On the next session start, the updated file is automatically discovered and injected.

## Work Items

1. **`prompt.rs`** — Spec Sections 3.2, 3.4, 3.5
   - `PromptBuilder`, `PromptSection`, `EnvironmentContext` structs
   - `section()`, `environment_context()`, `user_context()`
   - `instruction_files()` — walk from cwd up to root, discover INSTRUCTIONS.md / .agent/INSTRUCTIONS.md / .agent/rules/*.md / INSTRUCTIONS.local.md
   - Frontmatter parsing for `paths` glob patterns (conditional rules)
   - `build_system_prompt()` — concatenate base + sections
   - `build_context_message()` — format discovered files with source labels, wrap in XML tags
   - `EnvironmentContext::collect()` — read cwd, platform, OS version, date

## Tests

Tests use `tempfile` to create directory trees with instruction files.

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `build_system_prompt_concatenates_sections` | Direct | Base prompt + added sections appear in output |
| `environment_context_included` | Direct | Working directory and platform present in system prompt |
| `instruction_file_discovery_walks_tree` | Direct | Root and child INSTRUCTIONS.md both discovered |
| `agent_rules_directory_discovered` | Direct | All .md files in `.agent/rules/` loaded |
| `frontmatter_paths_parsed` | Direct | YAML frontmatter with `paths` field is parsed |
| `no_context_message_when_empty` | Direct | `build_context_message()` returns None when no context added |
| `user_context_injected` | Direct | `user_context()` content appears in context message |

## Done Criteria

- All tests pass
- `PromptBuilder` assembles system prompt from base + sections + environment context
- Instruction file discovery walks directory tree correctly from a 3-level hierarchy
- `build_context_message()` returns None when no context is configured
