use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

pub struct ListDirectoryTool;

const DESCRIPTION: &str = "\
List the contents of a directory. Returns file and directory names.

- Use this for a quick overview of directory structure.
- For finding files by pattern across the tree, use glob instead.";

impl Tool for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: \".\")"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively (default: false)"
                }
            }
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
            let path_str = input["path"].as_str().unwrap_or(".");
            let recursive = input["recursive"].as_bool().unwrap_or(false);
            let base = ctx.working_directory.join(path_str);

            match list_entries(&base, &base, recursive) {
                Ok(mut entries) => {
                    entries.sort_by(|a, b| a.display_name.cmp(&b.display_name));
                    let lines: Vec<String> = entries
                        .iter()
                        .map(|e| {
                            if e.size.is_some() {
                                format!("{}  {}  {}", e.display_name, e.kind, e.size.unwrap())
                            } else {
                                format!("{}  {}", e.display_name, e.kind)
                            }
                        })
                        .collect();
                    Ok(ToolResult::success(lines.join("\n")))
                }
                Err(e) => Ok(ToolResult::error(format!("Error listing directory: {e}"))),
            }
        })
    }
}

struct EntryInfo {
    display_name: String,
    kind: &'static str,
    size: Option<u64>,
}

fn list_entries(
    dir: &PathBuf,
    base: &PathBuf,
    recursive: bool,
) -> std::io::Result<Vec<EntryInfo>> {
    let mut results = Vec::new();
    let read_dir = std::fs::read_dir(dir)?;

    for entry in read_dir {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let file_type = metadata.file_type();

        let display_name = if recursive {
            entry
                .path()
                .strip_prefix(base)
                .unwrap_or(&entry.path())
                .to_string_lossy()
                .to_string()
        } else {
            entry.file_name().to_string_lossy().to_string()
        };

        let (kind, size) = if file_type.is_symlink() {
            ("symlink", None)
        } else if file_type.is_dir() {
            ("dir", None)
        } else {
            ("file", Some(metadata.len()))
        };

        results.push(EntryInfo {
            display_name,
            kind,
            size,
        });

        if recursive && file_type.is_dir() {
            let sub = list_entries(&entry.path(), base, true)?;
            results.extend(sub);
        }
    }

    Ok(results)
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
    async fn flat_listing() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("alpha.txt"), "hello").unwrap();
        fs::write(tmp.path().join("beta.txt"), "world").unwrap();
        fs::create_dir(tmp.path().join("subdir")).unwrap();

        let tool = ListDirectoryTool;
        let ctx = test_ctx(tmp.path());
        let result = tool
            .call(serde_json::json!({}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        let lines: Vec<&str> = result.content.lines().collect();
        assert_eq!(lines.len(), 3);
        // Sorted alphabetically
        assert!(lines[0].starts_with("alpha.txt"));
        assert!(lines[0].contains("file"));
        assert!(lines[0].contains("5")); // 5 bytes
        assert!(lines[1].starts_with("beta.txt"));
        assert!(lines[2].starts_with("subdir"));
        assert!(lines[2].contains("dir"));
    }

    #[tokio::test]
    async fn recursive_listing() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("root.txt"), "r").unwrap();
        fs::create_dir(tmp.path().join("child")).unwrap();
        fs::write(tmp.path().join("child").join("nested.txt"), "n").unwrap();

        let tool = ListDirectoryTool;
        let ctx = test_ctx(tmp.path());
        let result = tool
            .call(serde_json::json!({"recursive": true}), &ctx)
            .await
            .unwrap();

        assert!(!result.is_error);
        let content = &result.content;
        assert!(content.contains("child/nested.txt") || content.contains("child\\nested.txt"));
        assert!(content.contains("root.txt"));
        // Should have at least 3 entries: root.txt, child, child/nested.txt
        assert!(content.lines().count() >= 3);
    }
}
