use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::SystemTime;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

pub struct GlobTool;

const MAX_RESULTS: usize = 200;

const DESCRIPTION: &str = "\
Fast file pattern matching tool that works with any codebase size.

- Returns matching file paths sorted by modification time (newest first).
- Use this when you need to find files by name or extension patterns.
- For open-ended searches that may require multiple rounds, use spawn_agent instead.";

impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
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
                    "description": "Glob pattern (e.g. **/*.rs)"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: \".\")"
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
                Some(p) => p,
                None => {
                    return Ok(ToolResult::error("Missing required parameter: pattern"));
                }
            };
            let base_str = input["path"].as_str().unwrap_or(".");
            let base = ctx.working_directory.join(base_str);

            let pattern_segments: Vec<&str> = pattern.split('/').collect();

            let mut matches: Vec<(PathBuf, SystemTime)> = Vec::new();
            collect_matches(&base, &base, &pattern_segments, &mut matches);

            // Sort by modification time, newest first
            matches.sort_by(|a, b| b.1.cmp(&a.1));

            // Cap at MAX_RESULTS
            matches.truncate(MAX_RESULTS);

            let lines: Vec<String> = matches
                .iter()
                .map(|(p, _)| {
                    p.strip_prefix(&base)
                        .unwrap_or(p)
                        .to_string_lossy()
                        .to_string()
                })
                .collect();

            Ok(ToolResult::success(lines.join("\n")))
        })
    }
}

/// Recursively walk the directory tree and collect files matching the glob pattern.
fn collect_matches(
    current: &Path,
    base: &Path,
    pattern_segments: &[&str],
    results: &mut Vec<(PathBuf, SystemTime)>,
) {
    if pattern_segments.is_empty() {
        return;
    }

    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let is_dir = path.is_dir();

        let rel_path = path
            .strip_prefix(base)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let rel_segments: Vec<&str> = rel_path.split('/').collect();

        if glob_matches(pattern_segments, &rel_segments) {
            if !is_dir {
                let mtime = path
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                results.push((path.clone(), mtime));
            }
        }

        // Recurse into directories if pattern contains ** or more segments remain
        if is_dir {
            let has_doublestar = pattern_segments.iter().any(|s| *s == "**");
            let could_match_deeper = pattern_segments.len() > 1 || has_doublestar;
            if could_match_deeper {
                collect_matches(&path, base, pattern_segments, results);
            }
        }
    }
}

/// Match a sequence of path segments against a glob pattern's segments.
///
/// Supports:
/// - `*` matches any sequence of non-`/` characters within a segment
/// - `**` matches zero or more path segments
/// - `?` matches exactly one character
pub fn glob_matches(pattern: &[&str], path: &[&str]) -> bool {
    glob_match_recursive(pattern, path)
}

fn glob_match_recursive(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }

    let seg = pattern[0];

    if seg == "**" {
        // ** can match zero segments
        if glob_match_recursive(&pattern[1..], path) {
            return true;
        }
        // ** can consume one segment and keep ** active
        if !path.is_empty() && glob_match_recursive(pattern, &path[1..]) {
            return true;
        }
        return false;
    }

    if path.is_empty() {
        return false;
    }

    if segment_matches(seg, path[0]) {
        return glob_match_recursive(&pattern[1..], &path[1..]);
    }

    false
}

/// Match a single path segment against a pattern segment with `*` and `?` wildcards.
fn segment_matches(pattern: &str, text: &str) -> bool {
    seg_match_recursive(pattern.as_bytes(), text.as_bytes())
}

fn seg_match_recursive(pat: &[u8], txt: &[u8]) -> bool {
    if pat.is_empty() {
        return txt.is_empty();
    }

    match pat[0] {
        b'*' => {
            // * matches zero characters
            if seg_match_recursive(&pat[1..], txt) {
                return true;
            }
            // * matches one character and continues
            if !txt.is_empty() && seg_match_recursive(pat, &txt[1..]) {
                return true;
            }
            false
        }
        b'?' => {
            if txt.is_empty() {
                false
            } else {
                seg_match_recursive(&pat[1..], &txt[1..])
            }
        }
        c => {
            if txt.is_empty() || txt[0] != c {
                false
            } else {
                seg_match_recursive(&pat[1..], &txt[1..])
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

    #[tokio::test]
    async fn doublestar_rs_finds_nested() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src/sub")).unwrap();
        fs::write(tmp.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "// lib").unwrap();
        fs::write(tmp.path().join("src/sub/deep.rs"), "// deep").unwrap();
        fs::write(tmp.path().join("readme.md"), "# hi").unwrap();

        let tool = GlobTool;
        let ctx = test_ctx(tmp.path());
        let result = tool
            .call(serde_json::json!({"pattern": "**/*.rs"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        let lines: Vec<&str> = result.content.lines().collect();
        assert_eq!(lines.len(), 3);
        for line in &lines {
            assert!(line.ends_with(".rs"), "Expected .rs file, got: {line}");
        }
        // Should NOT include readme.md
        assert!(!result.content.contains("readme.md"));
    }

    #[tokio::test]
    async fn max_results_cap() {
        let tmp = tempfile::tempdir().unwrap();
        // Create 210 .txt files
        for i in 0..210 {
            fs::write(tmp.path().join(format!("file_{i:04}.txt")), "x").unwrap();
        }

        let tool = GlobTool;
        let ctx = test_ctx(tmp.path());
        let result = tool
            .call(serde_json::json!({"pattern": "*.txt"}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        let lines: Vec<&str> = result.content.lines().collect();
        assert_eq!(lines.len(), MAX_RESULTS);
    }
}
