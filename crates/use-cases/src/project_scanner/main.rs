//! Scans a project directory in two phases:
//! 1. Discovery agent finds files worth reading
//! 2. Werk of agents summarizes each file in parallel
//!
//! Usage: project-scanner [OPTIONS] [DIR]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use agentwerk::event::EventKind;
use agentwerk::tools::{GlobTool, ListDirectoryTool, ReadFileTool};
use agentwerk::{Agent, Werk};
use serde_json::{json, Value};

const DISCOVERY_PROMPT: &str = "\
You are a file discovery agent. Analyze the repository at {dir_path}.

1. Use list_directory to see the top-level structure
2. Use glob to find source files, config files, and documentation

Return a list of files worth reading to understand the project.
Only include text files (source code, config, docs). Skip binaries, lock files, and generated files.
Limit to at most 6 files. Prioritize the most important ones.";

fn discovery_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": { "type": "string" },
                "description": "File paths relative to project root"
            }
        },
        "required": ["files"]
    })
}

const SUMMARIZE_PROMPT: &str = "\
Read the file and describe what it does.";

fn summarize_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One-sentence description of the file's purpose"
            },
            "language": {
                "type": "string",
                "description": "Programming language, or 'config' / 'docs' for non-code files"
            }
        },
        "required": ["summary", "language"]
    })
}

#[tokio::main]
async fn main() {
    let config = parse_args();
    let provider = agentwerk::provider::from_env().expect("LLM provider required");
    let model = if config.model.is_empty() {
        agentwerk::provider::model_from_env().expect("model name required")
    } else {
        config.model
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_handle = cancel.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        cancel_handle.store(true, Ordering::Relaxed);
    });

    // Phase 1: Discover files
    eprintln!("[discover] scanning {}", config.dir.display());

    let discovery = Agent::new()
        .provider(provider.clone())
        .model_name(&model)
        .role(DISCOVERY_PROMPT)
        .instruction("Find all files worth reading to understand this project.")
        .tool(ListDirectoryTool)
        .tool(GlobTool)
        .output_schema(discovery_schema())
        .template("dir_path", json!(config.dir.display().to_string()))
        .working_dir(config.dir.clone())
        .cancel_signal(cancel.clone())
        .max_turns(20)
        .event_handler(Arc::new(|event| match &event.kind {
            EventKind::ToolCallStarted {
                tool_name, input, ..
            } => {
                let detail = input["path"]
                    .as_str()
                    .or(input["pattern"].as_str())
                    .unwrap_or("");
                if detail.is_empty() {
                    eprintln!("[discover] {tool_name}");
                } else {
                    eprintln!("[discover] {tool_name}({detail})");
                }
            }
            EventKind::ToolCallFailed {
                tool_name, message, ..
            } => {
                eprintln!("[discover] error in {tool_name}: {message}");
            }
            _ => {}
        }))
        .work()
        .await;

    let files: Vec<String> = match discovery {
        Ok(output) => match output.response {
            Some(data) => data["files"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default(),
            None => {
                eprintln!("Error: discovery agent returned no structured output");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error in discovery: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("[discover] found {} files\n", files.len());

    if files.is_empty() {
        write_output(&config.output, &json!({ "files": [], "languages": [] }));
        return;
    }

    // Phase 2: Summarize each file in parallel
    let summarizer = Agent::new()
        .provider(provider.clone())
        .model_name(&model)
        .role(SUMMARIZE_PROMPT)
        .tool(ReadFileTool)
        .output_schema(summarize_schema())
        .max_turns(5);

    let total = files.len();
    let progress = Arc::new(AtomicUsize::new(0));

    let agents = files.iter().map(|file| {
        let file_name = file.clone();
        let progress = progress.clone();
        summarizer
            .clone()
            .name(format!("summarize-{file}"))
            .instruction(format!("Read and summarize: {file}"))
            .working_dir(config.dir.clone())
            .event_handler(Arc::new(move |event| match &event.kind {
                EventKind::ToolCallFailed { message, .. } => {
                    eprintln!("[summarize] {file_name}: error: {message}");
                }
                EventKind::AgentFinished { .. } => {
                    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!("[summarize] {done}/{total} {file_name}");
                }
                _ => {}
            }))
    });

    let results = Werk::new()
        .lines(config.batch_size)
        .cancel_signal(cancel.clone())
        .workers(agents)
        .produce()
        .await;

    // Phase 3: Aggregate — `results` is in hire order, so indices align with `files`.
    let mut languages = BTreeSet::new();
    let mut file_summaries: Vec<Value> = Vec::new();

    for (file, result) in files.iter().zip(results.iter()) {
        let Ok(output) = result else {
            eprintln!("[error] {}", result.as_ref().err().unwrap());
            continue;
        };
        let Some(ref data) = output.response else {
            continue;
        };
        let lang = data["language"].as_str().unwrap_or("unknown");
        let summary = data["summary"].as_str().unwrap_or("");
        languages.insert(lang.to_string());
        file_summaries.push(json!({
            "file": file,
            "summary": summary,
            "language": lang,
        }));
    }

    let output = json!({
        "languages": languages.into_iter().collect::<Vec<_>>(),
        "files": file_summaries,
    });

    write_output(&config.output, &output);
}

fn write_output(path: &str, json: &Value) {
    let pretty = serde_json::to_string_pretty(json).unwrap();
    std::fs::write(path, &pretty).expect("Failed to write output file");
    eprintln!("\nResult written to {path}");
}

struct CliConfig {
    dir: PathBuf,
    model: String,
    output: String,
    batch_size: usize,
}

fn parse_args() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut dir = ".".to_string();
    let mut model = String::new();
    let mut output = "project.json".to_string();
    let mut batch_size = 5;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args[i].clone();
            }
            "-o" | "--output" => {
                i += 1;
                output = args[i].clone();
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().expect("batch-size must be a number");
            }
            "-h" | "--help" => {
                eprintln!("Scan a project and output a JSON summary.\n");
                eprintln!("Usage: project-scanner [OPTIONS] [DIR]\n");
                eprintln!("Options:");
                eprintln!("  --model <MODEL>        Model override");
                eprintln!("  --batch-size <N>       Parallel summarizations (default: 5)");
                eprintln!("  -o, --output <PATH>    Output file (default: project.json)");
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') && dir == "." => dir = arg.into(),
            arg => {
                eprintln!("Unknown option: {arg}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let dir = std::fs::canonicalize(&dir).unwrap_or_else(|_| PathBuf::from(&dir));
    CliConfig {
        dir,
        model,
        output,
        batch_size,
    }
}
