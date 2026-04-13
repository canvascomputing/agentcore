//! Scans a project directory and outputs a JSON summary with description and languages.
//!
//! Usage: project-scanner [OPTIONS] [FOLDER]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agent::{
    AgentBuilder, Event, GlobTool, ListDirectoryTool,
    ReadFileTool,
};

// ---------------------------------------------------------------------------
// Prompt & Schema — the core of what this use case does
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are a project scanner. Analyze the repository at {folder_path}.

Steps:
1. Use list_directory to understand the top-level structure
2. Use glob to find files by extension and identify languages used
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
    let config = parse_args();
    let (provider, default_model) = use_cases::auto_detect_provider();
    let model = if config.model.is_empty() { default_model } else { config.model };

    eprintln!("Scanning: {}\n", config.folder.display());

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_handle = cancel.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        cancel_handle.store(true, Ordering::Relaxed);
    });

    let output = match AgentBuilder::new()
        .name("project-scanner")
        .model(&model)
        .system_prompt(SYSTEM_PROMPT)
        .tool(ReadFileTool)
        .tool(ListDirectoryTool)
        .tool(GlobTool)
        .output_schema(output_schema())
        .max_budget(config.max_cost)
        .provider(provider)
        .prompt("Scan this project and identify what it does and what languages it uses.")
        .template_var("folder_path", serde_json::Value::String(config.folder.display().to_string()))
        .working_directory(config.folder)
        .event_handler(Arc::new(log_event))
        .cancel_signal(cancel)
        .run()
        .await
    {
        Ok(output) => output,
        Err(e) => {
            eprintln!("\nError: {e}");
            std::process::exit(1);
        }
    };

    let json = match output.response {
        Some(data) => serde_json::to_string_pretty(&data).unwrap(),
        None => serde_json::to_string_pretty(
            &serde_json::json!({"summary": output.response_raw, "languages": []}),
        ).unwrap(),
    };
    std::fs::write(&config.output, &json).expect("Failed to write output file");
    eprintln!("\nResult written to {}", config.output);
}

fn log_event(event: Event) {
    match &event {
        Event::ToolCallStart { tool_name, input, .. } => {
            let detail = input["path"].as_str()
                .or(input["pattern"].as_str())
                .unwrap_or("");
            if detail.is_empty() {
                eprintln!("[tool] {tool_name}");
            } else {
                eprintln!("[tool] {tool_name}: {detail}");
            }
        }
        Event::AgentEnd { turns, .. } => eprintln!("[done] {turns} turns"),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CliConfig {
    folder: PathBuf,
    model: String,
    output: String,
    max_cost: f64,
}

fn parse_args() -> CliConfig {
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

    let folder = std::fs::canonicalize(&folder).unwrap_or_else(|_| PathBuf::from(&folder));
    CliConfig { folder, model, output, max_cost }
}

