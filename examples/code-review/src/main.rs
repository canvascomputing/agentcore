mod file_stats;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use agent_core::{
    AgentBuilder, AgenticError, AnthropicProvider, CostTracker, Event, GlobTool, GrepTool,
    HttpTransport, InvocationContext, LiteLlmProvider, ListDirectoryTool, LlmProvider,
    ReadFileTool, generate_agent_id,
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

    eprintln!("Reviewing: {}\n", folder_path.display());

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

    // Build agent with built-in read-only tools + custom FileStatsTool
    let system_prompt = "\
        You are a code review assistant. Analyze the repository at {folder_path}.\n\n\
        Your task: {prompt}\n\n\
        Steps:\n\
        1. Use file_stats to get an overview of file types and sizes\n\
        2. Use list_directory to understand the top-level structure\n\
        3. Use glob to find config files and source files\n\
        4. Use read_file to read key files\n\
        5. Use grep to find important patterns if needed\n\
        6. Produce your analysis as structured output\n\n\
        Respond ONLY with structured output matching the required schema.";

    let agent = AgentBuilder::new()
        .name("code-reviewer")
        .model(&config.model)
        .system_prompt(system_prompt)
        .tool(FileStatsTool)
        .tool(ReadFileTool)
        .tool(ListDirectoryTool)
        .tool(GlobTool)
        .tool(GrepTool)
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
        Event::ToolStart { tool, input, .. } => {
            let args = serde_json::to_string(input).unwrap_or_default();
            eprintln!("[tool] {tool}({args})");
        }
        Event::ToolEnd { tool, is_error, .. } if *is_error => eprintln!("[error] {tool}"),
        _ => {}
    });

    let cancelled = Arc::new(AtomicBool::new(false));
    {
        let cancelled = cancelled.clone();
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }

    let ctx = InvocationContext {
        input: config.prompt.clone(),
        state,
        working_directory: folder_path,
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
                serde_json::to_string_pretty(&serde_json::json!({
                    "summary": output.content
                }))
                .unwrap()
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
