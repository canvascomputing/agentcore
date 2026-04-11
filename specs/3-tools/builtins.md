# Built-in Tool Implementations

## Overview

Eight built-in tools providing file operations, search, directory listing, shell execution, and tool discovery. These are packaged in the `agent-tools` crate and exposed via `BuiltinToolset`.

## Dependencies

- [Tool trait system](../3-tools/traits.md): `Tool`, `ToolContext`, `ToolResult`, `Toolset`
- [Core types](../1-base/types.md): `AgenticError`, `Result`

## Files

```
crates/agent-tools/Cargo.toml
crates/agent-tools/src/lib.rs
crates/agent-tools/src/read_file.rs
crates/agent-tools/src/write_file.rs
crates/agent-tools/src/edit_file.rs
crates/agent-tools/src/list_directory.rs
crates/agent-tools/src/glob.rs
crates/agent-tools/src/grep.rs
crates/agent-tools/src/bash.rs
crates/agent-tools/src/tool_search.rs
```

## Specification

### 7.5 Built-in Tools

These are the essential tools for a minimal agentic core, selected for maximum utility with minimum complexity:

#### File Operations

| Tool | Read-only | Input | Behavior |
|------|-----------|-------|----------|
| `read_file` | yes | `path`, optional `offset`, `limit` | Read file with line numbers. Supports offset/limit for large files. Returns `"line_num\tcontent"` format. |
| `write_file` | no | `path`, `content` | Create or overwrite a file. Creates parent directories if needed. Returns success/error. |
| `edit_file` | no | `path`, `old_string`, `new_string`, optional `replace_all` | Exact string replacement. `old_string` must be unique in file (unless `replace_all`). Fails if not found or ambiguous. |

#### File Discovery and Search

| Tool | Read-only | Input | Behavior |
|------|-----------|-------|----------|
| `glob` | yes | `pattern`, optional `path` | Fast file pattern matching. Supports glob patterns like `**/*.rs`, `src/**/*.ts`, `*.{json,toml}`. Recursive directory walk with manual glob matching (no external crate). Returns matching file paths sorted by modification time. Max 200 results. Ideal for finding files by name/extension. |
| `grep` | yes | `pattern`, optional `path`, `glob`, `output_mode`, `context_lines`, `case_insensitive`, `max_results` | Search file contents for a pattern. Recursive directory walk. `output_mode`: `"content"` (matching lines with line numbers, supports context lines before/after), `"files"` (file paths only, default), `"count"` (match counts per file). `glob` filters files by pattern (e.g., `"*.rs"`). `case_insensitive` flag. `context_lines` shows N lines before and after each match (like `grep -C`). `max_results` caps output (default 100). Substring matching — no regex crate needed. |

#### Directory Operations

| Tool | Read-only | Input | Behavior |
|------|-----------|-------|----------|
| `list_directory` | yes | `path`, optional `recursive` | List files and directories. Shows file type (file/dir/symlink) and size. Non-recursive by default. |

#### Shell Execution

| Tool | Read-only | Input | Behavior |
|------|-----------|-------|----------|
| `bash` | no | `command`, optional `timeout_ms` | Execute a shell command via `tokio::process::Command`. Captures stdout + stderr. Default timeout: 120s. Returns combined output. Working directory inherited from agent. |

#### Tool Search

| Tool | Read-only | Input | Behavior |
|------|-----------|-------|----------|
| `tool_search` | yes | `query` | Search registered tools by keyword. Returns full definitions of matching tools as formatted text. The LLM uses this to discover deferred tools whose schemas are not included by default. This tool itself is never deferred. |

**Implementation: `crates/agent-tools/src/tool_search.rs`**

```rust
pub struct ToolSearchTool;

impl Tool for ToolSearchTool {
    fn name(&self) -> &str { "tool_search" }
    fn description(&self) -> &str {
        "Search for available tools by keyword. Returns matching tool names, \
         descriptions, and input schemas so you can use them in subsequent turns."
    }
    fn is_read_only(&self) -> bool { true }
    fn should_defer(&self) -> bool { false }  // never deferred

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — a keyword, tool name, or capability"
                }
            },
            "required": ["query"]
        })
    }

    fn call(&self, input: serde_json::Value, ctx: &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
    {
        Box::pin(async move {
            let query = input.get("query").and_then(|v| v.as_str())
                .ok_or_else(|| AgenticError::Tool {
                    tool_name: "tool_search".into(),
                    message: "Missing required field: query".into(),
                })?;

            let registry = ctx.tool_registry.as_ref()
                .ok_or_else(|| AgenticError::Tool {
                    tool_name: "tool_search".into(),
                    message: "No tool registry available in context".into(),
                })?;

            let results = registry.search(query);
            if results.is_empty() {
                return Ok(ToolResult {
                    content: format!("No tools found matching '{query}'."),
                    is_error: false,
                });
            }

            let mut output = format!("Found {} tool(s) matching '{query}':\n\n", results.len());
            for r in &results {
                output.push_str(&format!(
                    "## {}\n\n{}\n\nInput schema:\n```json\n{}\n```\n\n---\n\n",
                    r.definition.name,
                    r.definition.description,
                    serde_json::to_string_pretty(&r.definition.input_schema)
                        .unwrap_or_else(|_| "{}".to_string()),
                ));
            }
            Ok(ToolResult { content: output, is_error: false })
        })
    }
}
```

#### `BuiltinToolset`

```rust
// agent-tools/src/lib.rs
pub struct BuiltinToolset;

impl Toolset for BuiltinToolset {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(EditFileTool),
            Box::new(GlobTool),
            Box::new(GrepTool),
            Box::new(ListDirectoryTool),
            Box::new(BashTool::new()),
            Box::new(SpawnAgentTool::new()),
            Box::new(TaskCreateTool),
            Box::new(TaskUpdateTool),
            Box::new(TaskListTool),
            Box::new(TaskGetTool),
            Box::new(ToolSearchTool),
        ]
    }
}
```

### Dependencies

```toml
[dependencies]
agent-core = { path = "../agent-core" }
tokio = { version = "1", features = ["fs", "process"] }
serde_json = "1"
```

## Work Items

1. **File tools** — `read_file.rs`, `write_file.rs`, `edit_file.rs`
   - `ReadFileTool` — is_read_only=true, returns lines with line numbers, supports offset/limit, resolves paths via `ctx.working_directory`
   - `WriteFileTool` — creates parent dirs, writes content, resolves paths via `ctx.working_directory`
   - `EditFileTool` — exact string replacement, validates old_string uniqueness (or replace_all), resolves paths via `ctx.working_directory`

2. **Discovery tools** — `glob.rs`, `grep.rs`, `list_directory.rs`
   - `GlobTool` — recursive directory walk with manual glob pattern matching, sorted by mtime, max 200 results, resolves paths via `ctx.working_directory`
   - `GrepTool` — recursive content search, substring matching, output modes (content/files/count), context lines, case insensitive flag, max results, resolves paths via `ctx.working_directory`
   - `ListDirectoryTool` — list entries with type and size, optional recursive, resolves paths via `ctx.working_directory`

3. **Shell tool** — `bash.rs`
   - `BashTool` — `tokio::process::Command`, captures stdout+stderr, configurable timeout (default 120s), uses `ctx.working_directory` as working directory for spawned processes

4. **Tool search** — `tool_search.rs`
   - `ToolSearchTool` — is_read_only=true, never deferred, searches `ctx.tool_registry` by keyword, returns formatted tool definitions

5. **`lib.rs`** — `BuiltinToolset` implementing `Toolset` (registers all tools above + placeholders for spawn_agent and task tools from later increments)

## Tests

Use `tempfile` crate in dev-dependencies. Shared helper: `test_ctx(dir) -> ToolContext` — creates a `ToolContext` with the given working directory.

### `read_file.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `read_file_table` | Table-driven (3 cases) | Full file read, offset+limit range, nonexistent file returns error |

### `write_file.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `write_file_creates_new` | Tempfile | Create new file, verify contents via `std::fs::read_to_string()` |
| `write_file_overwrites_existing` | Tempfile | Overwrite existing file, verify new contents |
| `write_file_creates_parent_dirs` | Tempfile | Nested path creates parent directories |

### `edit_file.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `edit_unique_match` | Tempfile | Unique `old_string` replaced correctly |
| `edit_non_unique_match_errors` | Tempfile | Non-unique `old_string` produces `is_error: true` |
| `edit_replace_all` | Tempfile | `replace_all: true` replaces all occurrences |
| `edit_not_found_errors` | Tempfile | Missing `old_string` produces error |

### `glob.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `glob_matches_pattern` | Tempfile | `**/*.rs` returns only `.rs` files from mixed tree |
| `glob_max_results_cap` | Tempfile | Max results cap respected |

### `grep.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `grep_substring_match` | Tempfile | Substring found in file content |
| `grep_case_insensitive` | Tempfile | Case-insensitive flag matches mixed case |
| `grep_context_lines` | Tempfile | Context lines before/after match included |
| `grep_output_modes` | Tempfile | content, files, and count modes produce correct output |

### `list_directory.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `list_flat` | Tempfile | Lists entries with file type (file/dir) |
| `list_recursive` | Tempfile | Recursive mode includes nested entries |

### `bash.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `bash_echo` | Process | `echo hello` produces `"hello\n"` |
| `bash_timeout` | Process | Long sleep with short timeout returns error |
| `bash_bad_command` | Process | Nonexistent command produces error |

### `tool_search.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `tool_search_finds_by_name` | Direct | Query `"read_file"` returns tool with matching name and full schema |
| `tool_search_finds_by_keyword` | Direct | Query `"file"` returns tools with `file` in name segments |
| `tool_search_no_results` | Direct | Query `"nonexistent_xyz"` returns "No tools found" (not is_error) |
| `tool_search_returns_schema` | Direct | Result content includes JSON schema for matched tool |
| `tool_search_missing_query_errors` | Direct | No `query` field returns error |
| `tool_search_without_registry_errors` | Direct | `tool_registry: None` returns error |
| `tool_search_is_never_deferred` | Direct | `should_defer()` returns false |
| `tool_search_is_read_only` | Direct | `is_read_only()` returns true |

### Test Summary Table

| Module | Tests | Key patterns |
|--------|-------|-------------|
| `read_file.rs` | 1 | Table-driven: 3 cases (full, offset, nonexistent) |
| `write_file.rs` | 3 | Tempfile: create, overwrite, nested dirs |
| `edit_file.rs` | 4 | Tempfile: unique, non-unique, replace_all, not found |
| `glob.rs` | 2 | Tempfile: pattern match, max results |
| `grep.rs` | 4 | Tempfile: match, case, context, modes |
| `list_directory.rs` | 2 | Tempfile: flat, recursive |
| `bash.rs` | 3 | Process: echo, timeout, bad command |
| `tool_search.rs` | 8 | Direct: name/keyword search, no results, schema, errors, trait flags |

## Done Criteria

- `cargo build -p agent-tools` compiles
- `cargo test -p agent-tools` passes all tests above
- All 8 tools register and execute correctly via `BuiltinToolset`
