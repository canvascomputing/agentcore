use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use agent_core::{AgenticError, Result, Tool, ToolContext, ToolResult};

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor", ".build", "dist"];

pub struct FileStatsTool;

impl Tool for FileStatsTool {
    fn name(&self) -> &str {
        "file_stats"
    }

    fn description(&self) -> &str {
        "List all file extensions in a directory with counts and total sizes. \
         Useful for understanding the composition of a codebase."
    }

    fn is_read_only(&self) -> bool {
        true
    }

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

    fn call<'a>(
        &'a self,
        input: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let rel_path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let dir = ctx.working_directory.join(rel_path);

            if !dir.is_dir() {
                return Ok(ToolResult {
                    content: format!("Error: {} is not a directory", dir.display()),
                    is_error: true,
                });
            }

            let mut stats: HashMap<String, (u64, u64)> = HashMap::new();
            let mut total_files: u64 = 0;
            let mut total_bytes: u64 = 0;

            walk_dir(&dir, &mut stats, &mut total_files, &mut total_bytes)?;

            let mut extensions: Vec<_> = stats.into_iter().collect();
            extensions.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

            let ext_json: serde_json::Value = extensions
                .iter()
                .map(|(ext, (count, bytes))| {
                    (
                        ext.clone(),
                        serde_json::json!({"count": count, "total_bytes": bytes}),
                    )
                })
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();

            let result = serde_json::json!({
                "extensions": ext_json,
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
        message: format!("Failed to read directory {}: {e}", dir.display()),
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
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let ext = path
                .extension()
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
