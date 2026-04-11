//! Code review CLI — analyzes a repository using an LLM agent with tools.
//!
//! Usage: cargo run -p agent-core --example code_review -- [OPTIONS] [FOLDER]
//!
//! Requires ANTHROPIC_API_KEY.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent::{
    AgentBuilder, AgenticError, AnthropicProvider, CostTracker, Event, GlobTool, GrepTool,
    HttpTransport, InvocationContext, LiteLlmProvider, ListDirectoryTool, LlmProvider,
    ReadFileTool, Result, Tool, ToolContext, ToolResult, generate_agent_id,
};

// ---------------------------------------------------------------------------
// FileStatsTool — custom tool (demonstrates struct-based Tool impl)
// ---------------------------------------------------------------------------

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "vendor", ".build", "dist"];

struct FileStatsTool;

impl Tool for FileStatsTool {
    fn name(&self) -> &str {
        "file_stats"
    }

    fn description(&self) -> &str {
        "List all file extensions in a directory with counts and total sizes."
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
                    "description": "Directory to scan (default: '.')"
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
            if SKIP_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            walk_dir(&path, stats, total_files, total_bytes)?;
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

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    folder: String,
    prompt: String,
    model: String,
    provider: String,
    api_key: String,
    base_url: Option<String>,
    output: String,
    max_cost: f64,
}

fn parse_args(args: &[String]) -> Config {
    // Auto-detect provider:
    //   LITELLM_API_URL set → litellm
    //   ANTHROPIC_API_KEY set → anthropic
    //   localhost:4000 reachable → litellm (default)
    let (default_provider, default_api_key, default_base_url, default_model) =
        if let Ok(url) = std::env::var("LITELLM_API_URL") {
            let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
            let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
            ("litellm".into(), key, Some(url), model)
        } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            let base = std::env::var("ANTHROPIC_BASE_URL").ok();
            let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
            ("anthropic".into(), key, base, model)
        } else if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
            let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
            let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
            ("litellm".into(), key, Some("http://localhost:4000".into()), model)
        } else {
            ("anthropic".into(), String::new(), None, "claude-sonnet-4-20250514".into())
        };

    let mut c = Config {
        folder: String::new(),
        prompt: "Analyze this repository. Identify its purpose, languages, and key components. \
                 Provide a detailed architecture summary."
            .into(),
        model: default_model,
        provider: default_provider,
        api_key: default_api_key,
        base_url: default_base_url,
        output: "review.json".into(),
        max_cost: 5.00,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => { i += 1; c.prompt = args[i].clone(); }
            "--model" => { i += 1; c.model = args[i].clone(); }
            "--provider" => { i += 1; c.provider = args[i].clone(); }
            "--api-key" => { i += 1; c.api_key = args[i].clone(); }
            "--base-url" => { i += 1; c.base_url = Some(args[i].clone()); }
            "--output" | "-o" => { i += 1; c.output = args[i].clone(); }
            "--max-cost" => { i += 1; c.max_cost = args[i].parse().expect("Invalid --max-cost"); }
            "--help" | "-h" => {
                eprintln!("Usage: code_review [OPTIONS] [FOLDER]");
                eprintln!("\n  [FOLDER]  Directory to review (default: .)");
                eprintln!("\nOptions:");
                eprintln!("  --prompt <TEXT>     Analysis focus");
                eprintln!("  --model <MODEL>     Model (default: claude-sonnet-4-20250514)");
                eprintln!("  --provider <NAME>   'anthropic' (default) or 'litellm'");
                eprintln!("  --api-key <KEY>     API key (or ANTHROPIC_API_KEY env)");
                eprintln!("  --base-url <URL>    Override provider URL");
                eprintln!("  -o, --output <PATH> Output file (default: review.json)");
                eprintln!("  --max-cost <N>      Max cost in USD (default: 5.00)");
                eprintln!("\nEnvironment:");
                eprintln!("  ANTHROPIC_API_KEY   Use Anthropic directly");
                eprintln!("  LITELLM_API_URL     Use LiteLLM proxy (e.g. http://localhost:4000)");
                std::process::exit(0);
            }
            other if !other.starts_with('-') && c.folder.is_empty() => c.folder = other.into(),
            other => {
                eprintln!("Unknown option: {other}\nUsage: code_review [OPTIONS] [FOLDER]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if c.folder.is_empty() { c.folder = ".".into(); }
    if c.api_key.is_empty() {
        let supported = ["ANTHROPIC_API_KEY", "LITELLM_API_URL"];
        let names = supported.join(" or ");
        eprintln!("Error: Set {names}");
        std::process::exit(1);
    }
    c
}

fn build_transport() -> HttpTransport {
    Box::new(|url, headers, body| {
        let url = url.to_string();
        let headers: Vec<(String, String)> = headers
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Box::pin(async move {
            let client = reqwest::Client::new();
            let mut req = client.post(&url).json(&body);
            for (k, v) in &headers {
                req = req.header(k.as_str(), v.as_str());
            }
            let resp = req.send().await.map_err(|e| AgenticError::Other(e.to_string()))?;
            resp.json().await.map_err(|e| AgenticError::Other(e.to_string()))
        })
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let config = parse_args(&std::env::args().collect::<Vec<_>>());
    let folder = std::fs::canonicalize(&config.folder)
        .unwrap_or_else(|_| PathBuf::from(&config.folder));

    eprintln!("Reviewing: {}\n", folder.display());

    let transport = build_transport();
    let provider: Arc<dyn LlmProvider> = match config.provider.as_str() {
        "litellm" => Arc::new(
            LiteLlmProvider::new(config.api_key.clone(), transport)
                .base_url(config.base_url.unwrap_or("http://localhost:4000".into())),
        ),
        _ => {
            let mut p = AnthropicProvider::new(config.api_key.clone(), transport);
            if let Some(url) = config.base_url { p = p.base_url(url); }
            Arc::new(p)
        }
    };

    let agent = AgentBuilder::new()
        .name("code-reviewer")
        .model(&config.model)
        .system_prompt(
            "You are a code review assistant. Analyze the repository at {folder_path}.\n\n\
             Your task: {prompt}\n\n\
             Steps:\n\
             1. Use file_stats to get an overview of file types and sizes\n\
             2. Use list_directory to understand the top-level structure\n\
             3. Use glob to find config files and source files\n\
             4. Use read_file to read key files\n\
             5. Use grep to find important patterns if needed\n\
             6. Produce your analysis as structured output\n\n\
             Respond ONLY with structured output matching the required schema.",
        )
        .tool(FileStatsTool)
        .tool(ReadFileTool)
        .tool(ListDirectoryTool)
        .tool(GlobTool)
        .tool(GrepTool)
        .output_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string", "description": "Detailed analysis" }
            },
            "required": ["summary"]
        }))
        .max_budget(config.max_cost)
        .build()
        .expect("Failed to build agent");

    let cost_tracker = CostTracker::new();

    let mut state = HashMap::new();
    state.insert("folder_path".into(), serde_json::Value::String(folder.display().to_string()));
    state.insert("prompt".into(), serde_json::Value::String(config.prompt.clone()));

    let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match &event {
        Event::ToolStart { tool, input, .. } => {
            eprintln!("[tool] {tool}({})", serde_json::to_string(input).unwrap_or_default());
        }
        Event::ToolEnd { tool, is_error, .. } if *is_error => eprintln!("[error] {tool}"),
        _ => {}
    });

    let cancelled = Arc::new(AtomicBool::new(false));
    {
        let c = cancelled.clone();
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            c.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }

    let ctx = InvocationContext {
        input: config.prompt.clone(),
        state,
        working_directory: folder,
        provider,
        cost_tracker: cost_tracker.clone(),
        on_event,
        cancelled,
        session_store: None,
        command_queue: None,
        agent_id: generate_agent_id("code-reviewer"),
    };

    match agent.run(ctx).await {
        Ok(output) => {
            let json = if let Some(structured) = output.structured_output {
                serde_json::to_string_pretty(&structured).unwrap()
            } else {
                serde_json::to_string_pretty(&serde_json::json!({"summary": output.content})).unwrap()
            };
            std::fs::write(&config.output, &json).expect("Failed to write output file");
            eprintln!("\nReview written to {}\n", config.output);
            eprintln!("{}", cost_tracker.summary());
        }
        Err(e) => {
            eprintln!("\nError: {e}\n");
            eprintln!("{}", cost_tracker.summary());
            std::process::exit(1);
        }
    }
}
