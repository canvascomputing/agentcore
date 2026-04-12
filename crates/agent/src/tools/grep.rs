use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

pub struct GrepTool;

const DEFAULT_MAX_RESULTS: u64 = 100;
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor"];

const DESCRIPTION: &str = "\
Search file contents using a substring pattern.

- ALWAYS use this tool for content search. NEVER invoke grep or rg as a bash command.
- Output modes: \"files\" (default, file paths only), \"content\" (matching lines with context), \"count\" (match counts per file).
- Use the glob parameter to filter by file type (e.g., \"*.rs\", \"*.ts\").
- For open-ended searches requiring multiple rounds, use spawn_agent instead.";

impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Substring to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: \".\")"
                },
                "glob": {
                    "type": "string",
                    "description": "File filter pattern (e.g. \"*.rs\")"
                },
                "output_mode": {
                    "type": "string",
                    "description": "Output mode: content, files, or count (default: files)"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines before and after each match"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to ignore case (default: false)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 100)"
                }
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let pattern = match input["pattern"].as_str() {
                Some(p) => p.to_string(),
                None => {
                    return Ok(ToolResult::error("Missing required parameter: pattern"));
                }
            };
            let base_str = input["path"].as_str().unwrap_or(".");
            let base = ctx.working_directory.join(base_str);
            let glob_filter = input["glob"].as_str().map(|s| s.to_string());
            let output_mode = input["output_mode"].as_str().unwrap_or("files");
            let context_lines = input["context_lines"].as_u64().map(|n| n as usize);
            let case_insensitive = input["case_insensitive"].as_bool().unwrap_or(false);
            let max_results = input["max_results"].as_u64().unwrap_or(DEFAULT_MAX_RESULTS) as usize;

            let search_pattern = if case_insensitive {
                pattern.to_lowercase()
            } else {
                pattern.clone()
            };

            let mut files_to_search = Vec::new();
            collect_files(&base, &glob_filter, &mut files_to_search);

            match output_mode {
                "content" => {
                    let mut output_lines = Vec::new();
                    let mut count = 0usize;

                    'outer: for file_path in &files_to_search {
                        let content = match std::fs::read_to_string(file_path) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };
                        let lines: Vec<&str> = content.lines().collect();
                        let rel = file_path
                            .strip_prefix(&base)
                            .unwrap_or(file_path)
                            .to_string_lossy();

                        // Find matching line indices
                        let mut match_indices = Vec::new();
                        for (i, line) in lines.iter().enumerate() {
                            let haystack = if case_insensitive {
                                line.to_lowercase()
                            } else {
                                line.to_string()
                            };
                            if haystack.contains(&search_pattern) {
                                match_indices.push(i);
                            }
                        }

                        // Build output with context
                        let mut emitted = std::collections::BTreeSet::new();
                        for &idx in &match_indices {
                            let ctx_n = context_lines.unwrap_or(0);
                            let start = idx.saturating_sub(ctx_n);
                            let end = (idx + ctx_n + 1).min(lines.len());
                            for li in start..end {
                                if emitted.insert(li) {
                                    output_lines
                                        .push(format!("{}:{}: {}", rel, li + 1, lines[li]));
                                    count += 1;
                                    if count >= max_results {
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }

                    Ok(ToolResult::success(output_lines.join("\n")))
                }
                "count" => {
                    let mut file_counts: Vec<(String, usize)> = Vec::new();

                    for file_path in &files_to_search {
                        let content = match std::fs::read_to_string(file_path) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };
                        let rel = file_path
                            .strip_prefix(&base)
                            .unwrap_or(file_path)
                            .to_string_lossy()
                            .to_string();

                        let mut n = 0usize;
                        for line in content.lines() {
                            let haystack = if case_insensitive {
                                line.to_lowercase()
                            } else {
                                line.to_string()
                            };
                            if haystack.contains(&search_pattern) {
                                n += 1;
                            }
                        }
                        if n > 0 {
                            file_counts.push((rel, n));
                        }
                        if file_counts.len() >= max_results {
                            break;
                        }
                    }

                    let lines: Vec<String> = file_counts
                        .iter()
                        .map(|(f, c)| format!("{f}: {c} matches"))
                        .collect();

                    Ok(ToolResult::success(lines.join("\n")))
                }
                // "files" mode (default)
                _ => {
                    let mut matched_files = Vec::new();

                    for file_path in &files_to_search {
                        let content = match std::fs::read_to_string(file_path) {
                            Ok(c) => c,
                            Err(_) => continue,
                        };

                        let found = content.lines().any(|line| {
                            let haystack = if case_insensitive {
                                line.to_lowercase()
                            } else {
                                line.to_string()
                            };
                            haystack.contains(&search_pattern)
                        });

                        if found {
                            let rel = file_path
                                .strip_prefix(&base)
                                .unwrap_or(file_path)
                                .to_string_lossy()
                                .to_string();
                            matched_files.push(rel);
                            if matched_files.len() >= max_results {
                                break;
                            }
                        }
                    }

                    Ok(ToolResult::success(matched_files.join("\n")))
                }
            }
        })
    }
}

/// Recursively collect files, skipping ignored directories and applying glob filter.
fn collect_files(dir: &Path, glob_filter: &Option<String>, results: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        if path.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }
            collect_files(&path, glob_filter, results);
        } else {
            if let Some(ref filter) = glob_filter {
                if !simple_glob_match(filter, &name) {
                    continue;
                }
            }
            // Skip binary-looking files by checking if the file can be read as text
            results.push(path);
        }
    }
}

/// Simple glob matching for file name filters (e.g. "*.rs", "*.txt").
/// Supports `*` (any chars) and `?` (single char) within a single filename segment.
fn simple_glob_match(pattern: &str, name: &str) -> bool {
    seg_match(pattern.as_bytes(), name.as_bytes())
}

fn seg_match(pat: &[u8], txt: &[u8]) -> bool {
    if pat.is_empty() {
        return txt.is_empty();
    }
    match pat[0] {
        b'*' => {
            if seg_match(&pat[1..], txt) {
                return true;
            }
            if !txt.is_empty() && seg_match(pat, &txt[1..]) {
                return true;
            }
            false
        }
        b'?' => {
            if txt.is_empty() {
                false
            } else {
                seg_match(&pat[1..], &txt[1..])
            }
        }
        c => {
            if txt.is_empty() || txt[0] != c {
                false
            } else {
                seg_match(&pat[1..], &txt[1..])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tool::ToolContext;
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
        fs::write(tmp.path().join("readme.md"), "# Hello Project\nThis is a test.\n").unwrap();
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

        assert!(!result.is_error);
        assert!(result.content.contains("main.rs"));
        // lib.rs has "Hello" but not "Hello world"
        assert!(!result.content.contains("lib.rs"));
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

        assert!(!result.is_error);
        assert!(result.content.contains("main.rs"));
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

        assert!(!result.is_error);
        // Should include the matching line and context
        assert!(result.content.contains("Hello world"));
        // With 1 context line, should also include fn main() line (line before)
        assert!(result.content.contains("fn main()"));
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
        assert!(!result.is_error);
        let file_lines: Vec<&str> = result.content.lines().collect();
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
        assert!(!result.is_error);
        // Content lines have format "file:line_no: content"
        for line in result.content.lines() {
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
        assert!(!result.is_error);
        // Count lines have format "file: N matches"
        for line in result.content.lines() {
            assert!(
                line.contains("matches"),
                "Expected 'matches' in count line: {line}"
            );
        }
    }
}
