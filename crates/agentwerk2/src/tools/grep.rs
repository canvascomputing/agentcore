//! Content search across files. Gives a model a way to locate the relevant code before opening any single file.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;
use super::util::glob_match;
use crate::providers::ProviderResult as Result;

/// Search file contents by substring under the working directory. Read-only.
/// Returns matching line snippets with file paths and line numbers; capped
/// at 100 hits by default.
pub struct GrepTool;

const DEFAULT_MAX_RESULTS: u64 = 100;
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor"];

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("grep.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for GrepTool {
    fn name(&self) -> &str {
        &tool_file().name
    }

    fn description(&self) -> &str {
        description()
    }

    fn input_schema(&self) -> Value {
        tool_file().input_schema.clone()
    }

    fn is_read_only(&self) -> bool {
        tool_file().read_only
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let pattern = match input["pattern"].as_str() {
                Some(p) => p,
                None => return Ok(ToolResult::error("Missing required parameter: pattern")),
            };

            let base = ctx.working_dir.join(input["path"].as_str().unwrap_or("."));
            let glob_filter = input["glob"].as_str().map(|s| s.to_string());
            let output_mode = input["output_mode"].as_str().unwrap_or("files");
            let context_lines = input["context_lines"].as_u64().unwrap_or(0) as usize;
            let case_insensitive = input["case_insensitive"].as_bool().unwrap_or(false);
            let max_results = input["max_results"].as_u64().unwrap_or(DEFAULT_MAX_RESULTS) as usize;

            let needle = if case_insensitive {
                pattern.to_lowercase()
            } else {
                pattern.to_string()
            };

            let mut files = Vec::new();
            collect_files(&base, &glob_filter, &mut files);

            let result = match output_mode {
                "content" => search_content(
                    &files,
                    &base,
                    &needle,
                    case_insensitive,
                    context_lines,
                    max_results,
                ),
                "count" => search_count(&files, &base, &needle, case_insensitive, max_results),
                _ => search_files(&files, &base, &needle, case_insensitive, max_results),
            };

            Ok(ToolResult::success(result))
        })
    }
}

fn search_content(
    files: &[PathBuf],
    base: &Path,
    needle: &str,
    case_insensitive: bool,
    context_lines: usize,
    max_results: usize,
) -> String {
    let mut output = Vec::new();

    'outer: for file_path in files {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let lines: Vec<&str> = content.lines().collect();
        let rel = relative_path(file_path, base);

        let match_indices: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line_matches(line, needle, case_insensitive))
            .map(|(i, _)| i)
            .collect();

        let mut emitted = std::collections::BTreeSet::new();
        for &idx in &match_indices {
            let start = idx.saturating_sub(context_lines);
            let end = (idx + context_lines + 1).min(lines.len());
            for (li, line) in lines.iter().enumerate().take(end).skip(start) {
                if !emitted.insert(li) {
                    continue;
                }
                output.push(format!("{}:{}: {}", rel, li + 1, line));
                if output.len() >= max_results {
                    break 'outer;
                }
            }
        }
    }

    output.join("\n")
}

fn search_count(
    files: &[PathBuf],
    base: &Path,
    needle: &str,
    case_insensitive: bool,
    max_results: usize,
) -> String {
    let mut counts = Vec::new();

    for file_path in files {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let n = content
            .lines()
            .filter(|line| line_matches(line, needle, case_insensitive))
            .count();
        if n > 0 {
            counts.push(format!("{}: {n} matches", relative_path(file_path, base)));
        }
        if counts.len() >= max_results {
            break;
        }
    }

    counts.join("\n")
}

fn search_files(
    files: &[PathBuf],
    base: &Path,
    needle: &str,
    case_insensitive: bool,
    max_results: usize,
) -> String {
    let mut matched = Vec::new();

    for file_path in files {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if content
            .lines()
            .any(|line| line_matches(line, needle, case_insensitive))
        {
            matched.push(relative_path(file_path, base));
            if matched.len() >= max_results {
                break;
            }
        }
    }

    matched.join("\n")
}

fn line_matches(line: &str, needle: &str, case_insensitive: bool) -> bool {
    if case_insensitive {
        line.to_lowercase().contains(needle)
    } else {
        line.contains(needle)
    }
}

fn relative_path(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn collect_files(dir: &Path, glob_filter: &Option<String>, results: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) {
                collect_files(&path, glob_filter, results);
            }
            continue;
        }

        if let Some(ref filter) = glob_filter {
            if !glob_match(filter, &name) {
                continue;
            }
        }
        results.push(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_ctx(path: &std::path::Path) -> ToolContext {
        ToolContext::new(path.to_path_buf())
    }

    fn setup_test_dir() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("src/main.rs"),
            "fn main() {\n    println!(\"Hello world\");\n}\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("src/lib.rs"),
            "pub fn greet() {\n    println!(\"Hello\");\n}\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("readme.md"),
            "# Hello Project\nThis is a test.\n",
        )
        .unwrap();
        tmp
    }

    #[tokio::test]
    async fn substring_match_found() {
        let tmp = setup_test_dir();
        let tool = GrepTool;
        let ctx = test_ctx(tmp.path());

        let result = tool
            .call(
                serde_json::json!({"pattern": "Hello world", "output_mode": "files"}),
                &ctx,
            )
            .await
            .unwrap();

        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(content.contains("main.rs"));
        // lib.rs has "Hello" but not "Hello world"
        assert!(!content.contains("lib.rs"));
    }

    #[tokio::test]
    async fn case_insensitive_search() {
        let tmp = setup_test_dir();
        let tool = GrepTool;
        let ctx = test_ctx(tmp.path());

        let result = tool
            .call(
                serde_json::json!({
                    "pattern": "hello world",
                    "case_insensitive": true,
                    "output_mode": "files"
                }),
                &ctx,
            )
            .await
            .unwrap();

        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        assert!(content.contains("main.rs"));
    }

    #[tokio::test]
    async fn context_lines_included() {
        let tmp = setup_test_dir();
        let tool = GrepTool;
        let ctx = test_ctx(tmp.path());

        let result = tool
            .call(
                serde_json::json!({
                    "pattern": "Hello world",
                    "output_mode": "content",
                    "context_lines": 1
                }),
                &ctx,
            )
            .await
            .unwrap();

        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        // Should include the matching line and context
        assert!(content.contains("Hello world"));
        // With 1 context line, should also include fn main() line (line before)
        assert!(content.contains("fn main()"));
    }

    #[tokio::test]
    async fn all_output_modes() {
        let tmp = setup_test_dir();
        let tool = GrepTool;
        let ctx = test_ctx(tmp.path());

        // files mode
        let result = tool
            .call(
                serde_json::json!({"pattern": "Hello", "output_mode": "files"}),
                &ctx,
            )
            .await
            .unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        let file_lines: Vec<&str> = content.lines().collect();
        // Should find matches in main.rs, lib.rs, and readme.md
        assert!(file_lines.len() >= 2);

        // content mode
        let result = tool
            .call(
                serde_json::json!({"pattern": "Hello", "output_mode": "content"}),
                &ctx,
            )
            .await
            .unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        // Content lines have format "file:line_no: content"
        for line in content.lines() {
            assert!(line.contains(':'), "Expected colon in content line: {line}");
        }

        // count mode
        let result = tool
            .call(
                serde_json::json!({"pattern": "Hello", "output_mode": "count"}),
                &ctx,
            )
            .await
            .unwrap();
        let (ToolResult::Success(content) | ToolResult::Error(content)) = &result;
        // Count lines have format "file: N matches"
        for line in content.lines() {
            assert!(
                line.contains("matches"),
                "Expected 'matches' in count line: {line}"
            );
        }
    }
}
