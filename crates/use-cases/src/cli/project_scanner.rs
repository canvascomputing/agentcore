//! Scans a project directory and outputs a JSON summary with description and languages.
//!
//! Usage: project-scanner [OPTIONS] [FOLDER]

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent::{
    AgentBuilder, AgenticError, Event, GlobTool, InvocationContext, ListDirectoryTool,
    ReadFileTool, Result, Tool, ToolContext, ToolResult, generate_agent_name,
};

// ---------------------------------------------------------------------------
// Prompt & Schema — the core of what this use case does
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are a project scanner. Analyze the repository at {folder_path}.

Steps:
1. Use file_stats to get an overview of file types and sizes
2. Use list_directory to understand the top-level structure
3. Read key files (README, Cargo.toml, package.json, etc.) to understand the project purpose

Then produce structured output with:
- summary: A concise description of what the project is about (max 200 characters)
- languages: An array of programming languages used in the project

Respond ONLY with structured output matching the required schema.";

fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "What the project is about, max 200 characters"
            },
            "languages": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Programming languages used in the project"
            }
        },
        "required": ["summary", "languages"]
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let (folder, model_override, output_path, max_cost) = parse_args();
    let (provider, default_model) = use_cases::auto_detect_provider();
    let model = if model_override.is_empty() { default_model } else { model_override };

    eprintln!("Scanning: {}\n", folder.display());

    let agent = AgentBuilder::new()
        .name("project-scanner")
        .model(&model)
        .system_prompt(SYSTEM_PROMPT)
        .tool(FileStatsTool)
        .tool(ReadFileTool)
        .tool(ListDirectoryTool)
        .tool(GlobTool)
        .output_schema(output_schema())
        .max_budget(max_cost)
        .build()
        .expect("Failed to build agent");

    let mut vars = HashMap::new();
    vars.insert("folder_path".into(), serde_json::Value::String(folder.display().to_string()));

    let cancel = Arc::new(AtomicBool::new(false));
    let c = cancel.clone();
    tokio::spawn(async move { tokio::signal::ctrl_c().await.ok(); c.store(true, std::sync::atomic::Ordering::Relaxed); });

    let ctx = InvocationContext {
        prompt: "Scan this project and identify what it does and what languages it uses.".into(),
        template_variables: vars,
        working_directory: folder,
        provider,
        event_handler: Arc::new(|event| {
            if let Event::ToolCallStart { tool_name, .. } = &event {
                eprintln!("[tool] {tool_name}");
            }
        }),
        cancel_signal: cancel,
        session_store: None,
        command_queue: None,
        agent_name: generate_agent_name("project-scanner"),
    };

    match agent.run(ctx).await {
        Ok(output) => {
            let json = if let Some(data) = output.response {
                serde_json::to_string_pretty(&data).unwrap()
            } else {
                serde_json::to_string_pretty(
                    &serde_json::json!({"summary": output.response_raw, "languages": []}),
                ).unwrap()
            };
            std::fs::write(&output_path, &json).expect("Failed to write output file");
            eprintln!("\nResult written to {output_path}");
        }
        Err(e) => {
            eprintln!("\nError: {e}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> (PathBuf, String, String, f64) {
    let args: Vec<String> = std::env::args().collect();
    let mut folder = ".".to_string();
    let mut model = String::new();
    let mut output = "project.json".to_string();
    let mut max_cost = 1.00;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model = args[i].clone(); }
            "-o" | "--output" => { i += 1; output = args[i].clone(); }
            "--max-cost" => { i += 1; max_cost = args[i].parse().expect("Invalid --max-cost"); }
            "-h" | "--help" => {
                eprintln!("Scan a project and output a JSON summary.\n");
                eprintln!("Usage: project-scanner [OPTIONS] [FOLDER]\n");
                eprintln!("Options:");
                eprintln!("  --model <MODEL>      Model override");
                eprintln!("  -o, --output <PATH>  Output file (default: project.json)");
                eprintln!("  --max-cost <USD>     Cost limit (default: 1.00)");
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') && folder == "." => folder = arg.into(),
            arg => {
                eprintln!("Unknown option: {arg}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let path = std::fs::canonicalize(&folder).unwrap_or_else(|_| PathBuf::from(&folder));
    (path, model, output, max_cost)
}

// ---------------------------------------------------------------------------
// FileStatsTool — custom tool for directory scanning
// ---------------------------------------------------------------------------

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor", ".build", "dist"];

struct FileStatsTool;

impl Tool for FileStatsTool {
    fn name(&self) -> &str { "file_stats" }
    fn description(&self) -> &str { "List all file extensions in a directory with counts and total sizes." }
    fn is_read_only(&self) -> bool { true }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to scan (default: '.')" }
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
                    (ext.clone(), serde_json::json!({"count": count, "total_bytes": bytes}))
                })
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();

            Ok(ToolResult {
                content: serde_json::to_string_pretty(&serde_json::json!({
                    "extensions": ext_json,
                    "total_files": total_files,
                    "total_bytes": total_bytes,
                }))
                .unwrap(),
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
        message: format!("Failed to read {}: {e}", dir.display()),
    })?;

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if !SKIP_DIRS.contains(&name_str.as_ref()) {
                walk_dir(&path, stats, total_files, total_bytes)?;
            }
        } else if path.is_file() {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let ext = path
                .extension()
                .map(|e| format!(".{}", e.to_string_lossy()))
                .unwrap_or_else(|| "(no ext)".into());
            let e = stats.entry(ext).or_insert((0, 0));
            e.0 += 1;
            e.1 += size;
            *total_files += 1;
            *total_bytes += size;
        }
    }
    Ok(())
}
