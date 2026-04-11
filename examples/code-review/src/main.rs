mod file_stats;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent_core::{
    AgentBuilder, AgenticError, AnthropicProvider, CostTracker, Event, HttpTransport,
    InvocationContext, LiteLlmProvider, LlmProvider, ToolBuilder, ToolResult, generate_agent_id,
};

use file_stats::FileStatsTool;

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
            for (key, value) in &headers {
                req = req.header(key.as_str(), value.as_str());
            }
            let resp = req
                .send()
                .await
                .map_err(|e| AgenticError::Other(e.to_string()))?;
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| AgenticError::Other(e.to_string()))?;
            Ok(json)
        })
    })
}

fn read_file_tool() -> impl agent_core::Tool {
    ToolBuilder::new("read_file", "Read a file's contents. Path resolved relative to working directory.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to read" }
            },
            "required": ["path"]
        }))
        .read_only(true)
        .handler(|input, ctx| {
            let path = ctx.working_directory.join(
                input["path"].as_str().unwrap_or("")
            );
            Box::pin(async move {
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        // Truncate large files
                        let content = if content.len() > 10_000 {
                            format!("{}...\n[truncated, {} bytes total]", &content[..10_000], content.len())
                        } else {
                            content
                        };
                        Ok(ToolResult { content, is_error: false })
                    }
                    Err(e) => Ok(ToolResult {
                        content: format!("Error reading {}: {e}", path.display()),
                        is_error: true,
                    }),
                }
            })
        })
        .build()
}

fn list_dir_tool() -> impl agent_core::Tool {
    ToolBuilder::new("list_dir", "List files and directories at a path. Path resolved relative to working directory.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path (default: '.')" }
            }
        }))
        .read_only(true)
        .handler(|input, ctx| {
            let path = ctx.working_directory.join(
                input["path"].as_str().unwrap_or(".")
            );
            Box::pin(async move {
                match std::fs::read_dir(&path) {
                    Ok(entries) => {
                        let mut items: Vec<String> = entries
                            .flatten()
                            .map(|e| {
                                let name = e.file_name().to_string_lossy().to_string();
                                if e.path().is_dir() {
                                    format!("{name}/")
                                } else {
                                    name
                                }
                            })
                            .collect();
                        items.sort();
                        Ok(ToolResult { content: items.join("\n"), is_error: false })
                    }
                    Err(e) => Ok(ToolResult {
                        content: format!("Error listing {}: {e}", path.display()),
                        is_error: true,
                    }),
                }
            })
        })
        .build()
}

fn grep_tool() -> impl agent_core::Tool {
    ToolBuilder::new("grep", "Search for a pattern in files under a directory. Returns matching lines with file paths.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Text pattern to search for" },
                "path": { "type": "string", "description": "Directory to search (default: '.')" }
            },
            "required": ["pattern"]
        }))
        .read_only(true)
        .handler(|input, ctx| {
            let pattern = input["pattern"].as_str().unwrap_or("").to_string();
            let path = ctx.working_directory.join(
                input["path"].as_str().unwrap_or(".")
            );
            Box::pin(async move {
                let mut matches = Vec::new();
                grep_recursive(&path, &pattern, &path, &mut matches, 0);
                if matches.is_empty() {
                    Ok(ToolResult { content: "No matches found.".into(), is_error: false })
                } else {
                    // Limit output
                    matches.truncate(50);
                    Ok(ToolResult { content: matches.join("\n"), is_error: false })
                }
            })
        })
        .build()
}

fn grep_recursive(
    dir: &std::path::Path,
    pattern: &str,
    base: &std::path::Path,
    matches: &mut Vec<String>,
    depth: u32,
) {
    if depth > 10 || matches.len() >= 50 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if [".git", "target", "node_modules", "vendor"].contains(&name.as_str()) {
                continue;
            }
            grep_recursive(&path, pattern, base, matches, depth + 1);
        } else if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let rel = path.strip_prefix(base).unwrap_or(&path);
                for (i, line) in content.lines().enumerate() {
                    if line.contains(pattern) {
                        matches.push(format!("{}:{}: {}", rel.display(), i + 1, line.trim()));
                        if matches.len() >= 50 {
                            return;
                        }
                    }
                }
            }
        }
    }
}

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
                 of the codebase architecture."
            .into(),
        model: "claude-sonnet-4-20250514".into(),
        provider: "anthropic".into(),
        api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
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
            "--output" | "-o" => { i += 1; config.output = args[i].clone(); }
            "--max-cost" => { i += 1; config.max_cost = args[i].parse().expect("Invalid --max-cost"); }
            "--help" | "-h" => {
                eprintln!("Usage: code-review [OPTIONS] [FOLDER]");
                eprintln!();
                eprintln!("Arguments:");
                eprintln!("  [FOLDER]              Directory to review (default: current dir)");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --prompt <TEXT>        Analysis focus");
                eprintln!("  --model <MODEL>        Model (default: claude-sonnet-4-20250514)");
                eprintln!("  --provider <NAME>      'anthropic' (default) or 'litellm'");
                eprintln!("  --api-key <KEY>        API key (or ANTHROPIC_API_KEY env)");
                eprintln!("  --base-url <URL>       Override provider URL");
                eprintln!("  -o, --output <PATH>    Output file (default: review.json)");
                eprintln!("  --max-cost <N>         Max cost in USD (default: 5.00)");
                std::process::exit(0);
            }
            other if !other.starts_with('-') && config.folder.is_empty() => {
                config.folder = other.into();
            }
            other => {
                eprintln!("Unknown option: {other}");
                eprintln!("Usage: code-review [OPTIONS] [FOLDER]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if config.folder.is_empty() {
        config.folder = ".".into();
    }

    if config.api_key.is_empty() {
        eprintln!("Error: API key required. Set ANTHROPIC_API_KEY or use --api-key");
        std::process::exit(1);
    }

    config
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args);

    let folder_path = std::fs::canonicalize(&config.folder)
        .unwrap_or_else(|_| PathBuf::from(&config.folder));

    eprintln!("Reviewing: {}", folder_path.display());

    // Build provider
    let transport = build_transport();
    let provider: Arc<dyn LlmProvider> = match config.provider.as_str() {
        "litellm" => Arc::new(
            LiteLlmProvider::new(config.api_key.clone(), transport)
                .base_url(config.base_url.unwrap_or("http://localhost:4000".into())),
        ),
        _ => {
            let mut p = AnthropicProvider::new(config.api_key.clone(), transport);
            if let Some(url) = config.base_url {
                p = p.base_url(url);
            }
            Arc::new(p)
        }
    };

    // Build agent with read-only tools
    let system_prompt = "\
        You are a code review assistant. Analyze the repository at {folder_path}.\n\n\
        Your task: {prompt}\n\n\
        Steps:\n\
        1. Use file_stats to get an overview of file types and sizes\n\
        2. List the top-level directory to understand structure\n\
        3. Find config files (Cargo.toml, package.json, pyproject.toml, etc.)\n\
        4. Read key files to understand architecture\n\
        5. Use grep to find important patterns if needed\n\
        6. Produce your analysis as structured output\n\n\
        Respond ONLY with structured output matching the required schema.";

    let agent = AgentBuilder::new()
        .name("code-reviewer")
        .model(&config.model)
        .system_prompt(system_prompt)
        .tool(FileStatsTool)
        .tool(read_file_tool())
        .tool(list_dir_tool())
        .tool(grep_tool())
        .output_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Detailed analysis per the user's prompt"
                }
            },
            "required": ["summary"]
        }))
        .max_budget(config.max_cost)
        .build()
        .expect("Failed to build agent");

    let cost_tracker = CostTracker::new();

    let mut state = HashMap::new();
    state.insert(
        "folder_path".into(),
        serde_json::Value::String(folder_path.display().to_string()),
    );
    state.insert(
        "prompt".into(),
        serde_json::Value::String(config.prompt.clone()),
    );

    let on_event: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|event| match &event {
        Event::ToolStart { tool, .. } => eprintln!("[tool] {tool}"),
        Event::ToolEnd { tool, is_error, .. } if *is_error => eprintln!("[error] {tool}"),
        _ => {}
    });

    let ctx = InvocationContext {
        input: config.prompt.clone(),
        state,
        working_directory: folder_path,
        provider,
        cost_tracker: cost_tracker.clone(),
        on_event,
        cancelled: Arc::new(AtomicBool::new(false)),
        session_store: None,
        command_queue: None,
        agent_id: generate_agent_id("code-reviewer"),
    };

    match agent.run(ctx).await {
        Ok(output) => {
            let json = if let Some(structured) = output.structured_output {
                serde_json::to_string_pretty(&structured).unwrap()
            } else {
                serde_json::to_string_pretty(&serde_json::json!({
                    "summary": output.content
                }))
                .unwrap()
            };

            std::fs::write(&config.output, &json).expect("Failed to write output file");
            eprintln!("\nReview written to {}", config.output);
            eprintln!("{}", cost_tracker.summary());
        }
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!("{}", cost_tracker.summary());
            std::process::exit(1);
        }
    }
}
