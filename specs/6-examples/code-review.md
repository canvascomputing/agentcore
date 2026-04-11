# Example: Code Review CLI

## Overview

A **code review tool** — a focused example showing how to use agent-core as a library. It scans a folder, produces a JSON summary of the repository. This example lives in `examples/code-review/`, signaling it is sample code, not the library's primary binary.

**Architecture choice:** Uses `AgentBuilder` + `agent.run(ctx)` directly. This is a one-shot execution, demonstrating the simplest integration path.

## Dependencies

- [Orchestration](../4-orchestration/spawn.md) + [Tool system](../3-tools/builtins.md)
- Uses `AgentBuilder` + `agent.run(ctx)` directly for one-shot execution.

## Files

```
examples/code-review/Cargo.toml
examples/code-review/src/main.rs
examples/code-review/src/file_stats.rs
```

## Specification

### Usage

```
code-review [OPTIONS] <FOLDER>

OPTIONS:
    --prompt <TEXT>        Analysis focus (default: general architecture review)
    --model <MODEL>        Model (default: claude-sonnet-4-20250514)
    --provider <PROVIDER>  "anthropic" (default) or "litellm"
    --api-key <KEY>        API key (or ANTHROPIC_API_KEY env var)
    --base-url <URL>       Override provider base URL
    --output <PATH>        Output file (default: review.json)
    --max-cost <N>         Max cost in USD (default: 5.00)
```

### Output Schema

```json
{
  "summary": "string — detailed analysis per the user's prompt"
}
```

### Tools Registered (Read-Only Only)

- `ReadFileTool`, `GlobTool`, `GrepTool`, `ListDirectoryTool` — built-in tools from `agent-tools`
- `FileStatsTool` — custom tool defined in this project (not in `agent-tools`)
- Registered individually (not via `BuiltinToolset`) to demonstrate selective registration and custom tool creation

### System Prompt

Instructs the agent to list the top-level directory, find config files, glob for source files, read key files, then produce structured output. Uses `{folder_path}` and `{prompt}` interpolation from `InvocationContext.state`.

**Default prompt:** "Analyze this repository. Identify its purpose, the programming languages used, and the key components. Provide a detailed summary of the codebase architecture."

### What This Example Demonstrates

| Library feature | How shown |
|---|---|
| Provider setup | AnthropicProvider/LiteLlmProvider with HttpTransport |
| Agent building | AgentBuilder with name, model, system_prompt, tools, output_schema |
| Selective tool registration | 5 tools registered individually (4 built-in + 1 custom) |
| Custom tool (struct-based) | `FileStatsTool` implementing `Tool` trait, defined in example project |
| `ToolContext` usage | `FileStatsTool` uses `ctx.working_directory` for path resolution |
| Structured output | output_schema + AgentOutput.structured_output extraction |
| State interpolation | {folder_path} and {prompt} resolved from InvocationContext.state |
| One-shot execution | Direct agent.run(ctx) |
| Cost tracking | CostTracker with max_budget budget, printed at end |
| Event handling | on_event callback for tool progress logging |

### Cargo.toml

```toml
[package]
name = "code-review"
version = "0.1.0"
edition = "2021"

[dependencies]
agent-core = { path = "../../crates/agent-core" }
agent-tools = { path = "../../crates/agent-tools" }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["json"] }
```

### Implementation Sketch

```rust
// examples/code-review/src/main.rs

use agent_core::*;
use agent_tools::{ReadFileTool, GlobTool, GrepTool, ListDirectoryTool};
use std::env;
use std::path::PathBuf;

mod file_stats;
use file_stats::FileStatsTool;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args);

    // Build provider
    let transport = build_reqwest_transport();
    let provider: Arc<dyn LlmProvider> = match config.provider.as_str() {
        "litellm" => Arc::new(LiteLlmProvider::new(config.api_key, transport)
            .base_url(config.base_url.unwrap_or("http://localhost:4000".into()))),
        _ => Arc::new(AnthropicProvider::new(config.api_key, transport)
            .base_url(config.base_url.unwrap_or("https://api.anthropic.com".into()))),
    };

    // Build agent with selective tool registration (read-only only)
    let output_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string", "description": "Detailed analysis per the user's prompt" }
        },
        "required": ["summary"]
    });

    let system_prompt = format!(
        "You are a code review assistant. Analyze the repository at {folder_path}.\n\n\
         Your task: {prompt}\n\n\
         Steps:\n\
         1. Use file_stats to get an overview of file types and sizes\n\
         2. List the top-level directory to understand structure\n\
         3. Find config files (Cargo.toml, package.json, pyproject.toml, etc.)\n\
         4. Glob for source files to identify languages\n\
         5. Read key files to understand architecture\n\
         6. Produce your analysis as structured output\n\n\
         Respond ONLY with structured output matching the required schema.",
        folder_path = "{folder_path}",
        prompt = "{prompt}",
    );

    let agent = AgentBuilder::new()
        .name("code-reviewer")
        .model(&config.model)
        .system_prompt(&system_prompt)
        // Built-in tools
        .tool(ReadFileTool)
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(ListDirectoryTool)
        // Custom tool defined in this project
        .tool(FileStatsTool)
        .output_schema(output_schema)
        .max_budget(config.max_cost)
        .build()
        .expect("Failed to build agent");

    // Create cost tracker manually
    let cost_tracker = CostTracker::new();

    // Build invocation context with state interpolation
    let mut state = HashMap::new();
    state.insert("folder_path".into(), config.folder.clone());
    state.insert("prompt".into(), config.prompt.clone());

    let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| {
        match event {
            Event::ToolStart { tool, .. } => eprintln!("[tool] {tool}"),
            _ => {}
        }
    });

    let ctx = InvocationContext {
        input: config.prompt.clone(),
        state,
        working_directory: PathBuf::from(&config.folder),
        provider: provider.clone(),
        cost_tracker: cost_tracker.clone(),
        on_event,
        cancelled: Arc::new(AtomicBool::new(false)),
        session_store: None,
        command_queue: None,
        agent_id: "code-reviewer".into(),
    };

    // One-shot execution — direct agent.run(ctx)
    match agent.run(ctx).await {
        Ok(output) => {
            let json = if let Some(structured) = output.structured_output {
                serde_json::to_string(&structured).unwrap()
            } else {
                output.content.clone()
            };

            // Write to output file
            std::fs::write(&config.output, &json)
                .expect("Failed to write output file");
            eprintln!("Review written to {}", config.output);

            // Print cost summary
            eprintln!("\n{}", cost_tracker.summary());
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
```

### FileStatsTool — Custom Tool (`file_stats.rs`)

A custom tool defined in the example project (not in `agent-tools`) to demonstrate downstream tool creation. Walks a directory tree, groups files by extension, and returns per-extension statistics.

```rust
// examples/code-review/src/file_stats.rs

use agent_core::{Tool, ToolContext, ToolResult, AgenticError, Result};
use std::collections::HashMap;
use std::path::Path;

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor"];

pub struct FileStatsTool;

impl Tool for FileStatsTool {
    fn name(&self) -> &str { "file_stats" }

    fn description(&self) -> &str {
        "List all file extensions in a directory with counts and total sizes. \
         Useful for understanding the composition of a codebase."
    }

    fn is_read_only(&self) -> bool { true }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to scan. Resolved relative to working directory."
                }
            }
        })
    }

    fn call(&self, input: serde_json::Value, ctx: &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
    {
        Box::pin(async move {
            let rel_path = input.get("path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");

            let dir = ctx.working_directory.join(rel_path);
            if !dir.is_dir() {
                return Ok(ToolResult {
                    content: format!("Error: {dir:?} is not a directory"),
                    is_error: true,
                });
            }

            let mut stats: HashMap<String, (u64, u64)> = HashMap::new(); // ext → (count, bytes)
            let mut total_files: u64 = 0;
            let mut total_bytes: u64 = 0;

            walk_dir(&dir, &mut stats, &mut total_files, &mut total_bytes)?;

            let extensions: serde_json::Value = stats.iter()
                .map(|(ext, (count, bytes))| {
                    (ext.clone(), serde_json::json!({ "count": count, "total_bytes": bytes }))
                })
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();

            let result = serde_json::json!({
                "extensions": extensions,
                "total_files": total_files,
                "total_bytes": total_bytes,
            });

            Ok(ToolResult {
                content: serde_json::to_string_pretty(&result).unwrap(),
                is_error: false,
            })
        })
    }
}

fn walk_dir(
    dir: &Path,
    stats: &mut HashMap<String, (u64, u64)>,
    total_files: &mut u64,
    total_bytes: &mut u64,
) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| AgenticError::Tool {
        tool_name: "file_stats".into(),
        message: format!("Failed to read directory {dir:?}: {e}"),
    })?;

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if SKIP_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            walk_dir(&path, stats, total_files, total_bytes)?;
        } else if path.is_file() {
            // Skip binary files (null byte in first 512 bytes)
            if let Ok(bytes) = std::fs::read(&path) {
                if bytes.iter().take(512).any(|&b| b == 0) {
                    continue;
                }
            }

            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let ext = path.extension()
                .map(|e| format!(".{}", e.to_string_lossy()))
                .unwrap_or_else(|| "(no extension)".into());

            let entry = stats.entry(ext).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += size;
            *total_files += 1;
            *total_bytes += size;
        }
    }
    Ok(())
}
```

### Config and Arg Parsing

```rust
struct ReviewConfig {
    folder: String,
    prompt: String,
    model: String,
    provider: String,
    api_key: String,
    base_url: Option<String>,
    output: String,
    max_cost: f64,
}

fn parse_args(args: &[String]) -> ReviewConfig {
    let mut config = ReviewConfig {
        folder: String::new(),
        prompt: "Analyze this repository. Identify its purpose, the programming \
                 languages used, and the key components. Provide a detailed summary \
                 of the codebase architecture.".into(),
        model: "claude-sonnet-4-20250514".into(),
        provider: "anthropic".into(),
        api_key: env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
        base_url: None,
        output: "review.json".into(),
        max_cost: 5.00,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => { i += 1; config.prompt = args[i].clone(); }
            "--model" => { i += 1; config.model = args[i].clone(); }
            "--provider" => { i += 1; config.provider = args[i].clone(); }
            "--api-key" => { i += 1; config.api_key = args[i].clone(); }
            "--base-url" => { i += 1; config.base_url = Some(args[i].clone()); }
            "--output" => { i += 1; config.output = args[i].clone(); }
            "--max-cost" => { i += 1; config.max_cost = args[i].parse().expect("Invalid --max-cost"); }
            other if !other.starts_with("--") && config.folder.is_empty() => {
                config.folder = other.into();
            }
            other => {
                eprintln!("Unknown option: {other}");
                eprintln!("Usage: code-review [OPTIONS] <FOLDER>");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if config.folder.is_empty() {
        eprintln!("Error: <FOLDER> argument is required");
        eprintln!("Usage: code-review [OPTIONS] <FOLDER>");
        std::process::exit(1);
    }

    config
}
```

## Work Items

1. **`Cargo.toml`** — workspace example crate with dependencies on agent-core, agent-tools, tokio, serde_json, reqwest
2. **`file_stats.rs`** — `FileStatsTool` implementing `Tool` trait
   - Recursive directory walk, groups files by extension
   - Returns JSON with per-extension count and total_bytes
   - Uses `ctx.working_directory` to resolve relative paths
   - Skips `.git`, `target`, `node_modules`, `vendor` directories
   - Skips binary files (null-byte check in first 512 bytes)
3. **`main.rs`** — full implementation
   - Manual arg parsing (`parse_args`)
   - Provider selection (anthropic/litellm)
   - Agent building with 5 read-only tools registered individually (4 built-in + `FileStatsTool`)
   - `working_directory` set to the target folder in `InvocationContext`
   - Output schema enforcement
   - State interpolation for `{folder_path}` and `{prompt}`
   - Event handler logging tool progress to stderr
   - One-shot `agent.run(ctx)` execution
   - Structured output extraction + JSON file writing
   - Cost summary on completion

## Tests

All tests use `tempfile::TempDir` for filesystem isolation. Integration tests use `MockProvider` from [`../1-base/types.md`](../1-base/types.md) and `TestHarness` from [`../2-agent/loop.md`](../2-agent/loop.md).

Shared helpers: `test_ctx(dir) -> ToolContext` — creates a `ToolContext` with the given working directory.

### `file_stats.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|--------------------|
| `file_stats_counts_extensions` | Direct | 3 files (.rs×2, .toml×1); total_files=3; per-extension counts correct |
| `file_stats_skips_git_directory` | Direct | .git/objects/pack.idx excluded; total_files=1 (only src.rs) |
| `file_stats_skips_binary_files` | Direct | File with null byte in first 512 bytes excluded; total_files=1 (only text.txt) |
| `file_stats_nonexistent_path_returns_error` | Direct | Nonexistent path → is_error=true, content contains error message |
| `file_stats_relative_path_resolved` | Direct | Subdirectory "project/src" resolved via ctx.working_directory; total_files=1 |

### `main.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|--------------------|
| `parse_args_table` | Table-driven (3 cases) | "folder only" → defaults applied; "all flags" → all overrides parsed; "base url only" → partial override |

### Integration Tests

| Test | Pattern | What it verifies |
|------|---------|--------------------|
| `end_to_end_produces_structured_output` | Direct | Temp Rust project + MockProvider with file_stats→StructuredOutput sequence; output.structured_output contains expected JSON summary |

## Done Criteria

- `cargo build -p code-review` compiles
- `cargo test -p code-review` passes
- `cargo run -p code-review -- ./some-repo` with real API key produces review.json
- review.json matches output schema
- Tool progress logged to stderr
- Cost summary printed on completion
