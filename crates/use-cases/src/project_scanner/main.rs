//! Scans a project for two classes of security issue: data exfiltration paths
//! and persistence backdoors. Two scouts run in parallel to pick the files
//! each specialty should audit; a Werk dispatches (file, specialty) pairs to
//! specialist auditors in parallel. Each auditor reports findings via a
//! `report_issue` tool that appends NDJSON immediately, so a Ctrl-C or a
//! `max_steps` cutoff still leaves every issue found so far on disk.
//!
//! Usage: project-scanner [OPTIONS] [DIR]

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use agentwerk::event::EventKind;
use agentwerk::output::Outcome;
use agentwerk::tools::{GlobTool, ListDirectoryTool, ReadFileTool, Tool, ToolResult};
use agentwerk::{Agent, Event, Werk};
use serde_json::json;

const MAX_STEPS: u32 = 200;

struct Specialty {
    name: &'static str,
    criteria: &'static str,
}

const SPECIALTIES: &[Specialty] = &[
    Specialty {
        name: "exfiltration",
        criteria: "code that ships sensitive data off the host: network requests that include credentials, tokens, env vars, PII, or filesystem contents; log/print statements that emit secrets; URLs built with secret query params; telemetry/analytics with overbroad payloads; reads of .env, ~/.ssh, ~/.aws, keychain, or browser data forwarded outward; crash dumps that capture in-memory secrets",
    },
    Specialty {
        name: "persistence_backdoors",
        criteria: "code that creates a foothold that survives restart or hides ongoing access: hardcoded admin credentials and env-var auth bypasses, debug shortcuts that skip authentication, writes to authorized_keys / ~/.bashrc / startup scripts / systemd units / cron entries, self-installing services, replacement of system binaries or PATH shims, hidden HTTP routes, callbacks to attacker-controlled URLs",
    },
];

const SCOUT_PROMPT: &str = "\
You are a {specialty} scout at {dir_path}.

Your specialty looks for: {criteria}

Tools:
- glob:           your primary tool. ONE broad call usually suffices, e.g.
                  '**/*.{{rs,py,ts,tsx,js,go,java,rb}}' or scope it to a subdir
- list_directory: optional, only if glob isn't enough
- report_file:    record one file you want audited. Call once per file.

Process:
1. Run ONE glob to get the candidate file list.
2. For each source file (at most 4) where {specialty} issues are most likely,
   call report_file with the path. Skip vendored, generated, binary, lock, and
   test files.
3. After your last report_file call, end with a short text reply summarizing
   your picks. Do not keep exploring.

You CANNOT read file contents.";

const AUDIT_PROMPT: &str = "\
You are a {specialty} auditor.

Your specialty looks for: {criteria}

Tools:
- read_file:    read the file you have been assigned.
- report_issue: record one issue. Call once per issue, immediately when you
                find it. Do NOT batch issues until the end.

Process:
1. Read the file with read_file (once is usually enough).
2. For each {specialty} issue you find, call report_issue with:
     line     - 1-based line number
     severity - 'high', 'medium', or 'low'
     category - short tag, e.g. 'unsafe-block', 'await-while-locked'
     message  - one sentence: what is wrong and why
3. After your last report_issue call, end with a short text reply summarizing
   what you found.

Report only {specialty} issues. Each issue must point to a concrete line.
If the file has no issues for your specialty, just end with a text reply
saying so without calling report_issue.";

type AssignmentBuf = Arc<Mutex<Vec<(String, String)>>>;
type IssueWriter = Arc<Mutex<std::fs::File>>;
type Totals = Arc<Mutex<TotalsInner>>;
type Affected = Arc<Mutex<BTreeSet<String>>>;

#[derive(Default)]
struct TotalsInner {
    input_tokens: u64,
    output_tokens: u64,
    failed: usize,
    high: usize,
    medium: usize,
    low: usize,
    issues: usize,
}

fn report_file_tool(specialty: &'static str, assignments: AssignmentBuf) -> Tool {
    Tool::new(
        "report_file",
        "Record one source file that your specialty should audit. \
         Call once per file you want audited.",
    )
    .contract(json!({
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Path relative to the project root"
            }
        },
        "required": ["file"]
    }))
    .handler(move |input, _ctx| {
        let assignments = assignments.clone();
        Box::pin(async move {
            let Some(file) = input["file"].as_str() else {
                return Ok(ToolResult::error("missing required field 'file'"));
            };
            assignments
                .lock()
                .unwrap()
                .push((file.to_string(), specialty.to_string()));
            Ok(ToolResult::success(format!("recorded: {file}")))
        })
    })
}

fn report_issue_tool(
    file: String,
    specialty: String,
    writer: IssueWriter,
    totals: Totals,
    affected: Affected,
) -> Tool {
    Tool::new(
        "report_issue",
        "Record one issue you found in the file. Call once per issue, the \
         moment you spot it. Do not batch.",
    )
    .contract(json!({
        "type": "object",
        "properties": {
            "line": {
                "type": "integer",
                "description": "1-based line number where the issue is"
            },
            "severity": {
                "type": "string",
                "enum": ["high", "medium", "low"]
            },
            "category": {
                "type": "string",
                "description": "Short tag, e.g. 'unsafe-block'"
            },
            "message": {
                "type": "string",
                "description": "One sentence: what is wrong and why"
            }
        },
        "required": ["line", "severity", "category", "message"]
    }))
    .handler(move |input, _ctx| {
        let file = file.clone();
        let specialty = specialty.clone();
        let writer = writer.clone();
        let totals = totals.clone();
        let affected = affected.clone();
        Box::pin(async move {
            let record = json!({
                "file": file,
                "specialist": specialty,
                "line": input["line"],
                "severity": input["severity"],
                "category": input["category"],
                "message": input["message"],
            });
            let line = serde_json::to_string(&record).unwrap();
            {
                let mut w = writer.lock().unwrap();
                if let Err(e) = writeln!(&mut *w, "{line}") {
                    return Ok(ToolResult::error(format!("write failed: {e}")));
                }
                w.flush().ok();
            }
            {
                let mut t = totals.lock().unwrap();
                t.issues += 1;
                match input["severity"].as_str() {
                    Some("high") => t.high += 1,
                    Some("medium") => t.medium += 1,
                    Some("low") => t.low += 1,
                    _ => {}
                }
            }
            affected.lock().unwrap().insert(file);
            Ok(ToolResult::success("recorded"))
        })
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

    let started = Instant::now();

    // Shared state. The output file is opened once and written to incrementally
    // by every auditor's report_issue tool, so a Ctrl-C or max_steps cutoff
    // still leaves a valid NDJSON file with everything found so far.
    let writer: IssueWriter = Arc::new(Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&config.output)
            .expect("Failed to open output file"),
    ));
    let assignments_buf: AssignmentBuf = Arc::new(Mutex::new(Vec::new()));
    let totals: Totals = Arc::new(Mutex::new(TotalsInner::default()));
    let affected: Affected = Arc::new(Mutex::new(BTreeSet::new()));

    // Phase 1: 8 scouts run in parallel; each calls report_file for its picks.
    eprintln!(
        "┌ scouts   {} specialists scanning {}",
        SPECIALTIES.len(),
        config.dir.display(),
    );

    let phase1_started = Instant::now();
    let scouts = SPECIALTIES.iter().map(|s| {
        Agent::new()
            .name(format!("{}-scout", s.name))
            .provider(provider.clone())
            .model(&model)
            .role(SCOUT_PROMPT)
            .tool(ListDirectoryTool)
            .tool(GlobTool)
            .tool(report_file_tool(s.name, assignments_buf.clone()))
            .max_steps(MAX_STEPS)
            .template("specialty", json!(s.name))
            .template("criteria", json!(s.criteria))
            .template("dir_path", json!(config.dir.display().to_string()))
            .working_dir(config.dir.clone())
            .interrupt_signal(cancel.clone())
            .event_handler(scout_logger(s.name.to_string()))
            .task("Find files most relevant to your specialty.")
    });

    let scout_results = Werk::new()
        .lines(SPECIALTIES.len())
        .interrupt_signal(cancel.clone())
        .hire_and_fire(scouts)
        .await;

    let mut scout_in_tokens = 0u64;
    let mut scout_out_tokens = 0u64;
    let mut scout_failures = 0usize;
    for (i, result) in scout_results.iter().enumerate() {
        let specialty = SPECIALTIES[i].name;
        match result {
            Ok(o) => {
                scout_in_tokens += o.statistics.input_tokens;
                scout_out_tokens += o.statistics.output_tokens;
                if o.outcome != Outcome::Completed {
                    eprintln!(
                        "│ {specialty}-scout {:?} after {} steps (any reported files are kept)",
                        o.outcome, o.statistics.steps,
                    );
                    scout_failures += 1;
                }
            }
            Err(e) => {
                eprintln!("│ {specialty}-scout dispatch error: {e}");
                scout_failures += 1;
            }
        }
    }

    let assignments: Vec<(String, String)> = {
        let buf = assignments_buf.lock().unwrap();
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        let mut out = Vec::with_capacity(buf.len());
        for pair in buf.iter() {
            if seen.insert(pair.clone()) {
                out.push(pair.clone());
            }
        }
        out
    };

    eprintln!(
        "└ scouts   {} ok, {} failed, {} (file,specialist) pairs, {} in / {} out tokens, {:.1}s",
        SPECIALTIES.len() - scout_failures,
        scout_failures,
        assignments.len(),
        scout_in_tokens,
        scout_out_tokens,
        phase1_started.elapsed().as_secs_f64(),
    );

    if assignments.is_empty() {
        eprintln!("\nNo assignments produced. Exiting.");
        std::process::exit(1);
    }

    print_plan(&assignments);

    // Phase 2: dispatch specialist auditors in parallel via Werk.
    let total = assignments.len();
    let phase2_started = Instant::now();
    let progress = Arc::new(AtomicUsize::new(0));

    eprintln!(
        "\n┌ audit    {} pairs, up to {} in flight (issues stream to {})",
        total, config.batch_size, config.output,
    );

    let agents = assignments.iter().enumerate().map(|(i, (file, specialty))| {
        let criteria = SPECIALTIES
            .iter()
            .find(|s| s.name == specialty)
            .map(|s| s.criteria)
            .unwrap_or("");
        Agent::new()
            .name(format!("{specialty}-{i}"))
            .provider(provider.clone())
            .model(&model)
            .role(AUDIT_PROMPT)
            .tool(ReadFileTool)
            .tool(report_issue_tool(
                file.clone(),
                specialty.clone(),
                writer.clone(),
                totals.clone(),
                affected.clone(),
            ))
            .max_steps(MAX_STEPS)
            .template("specialty", json!(specialty))
            .template("criteria", json!(criteria))
            .working_dir(config.dir.clone())
            .task(format!("Read and audit: {file}"))
            .event_handler(audit_logger(
                file.clone(),
                specialty.clone(),
                progress.clone(),
                total,
            ))
    });

    let (producing, mut stream) = Werk::new()
        .lines(config.batch_size)
        .interrupt_signal(cancel.clone())
        .open();
    producing.hire_all(agents);
    producing.close();

    while let Some((i, result)) = stream.next().await {
        let (file, specialty) = &assignments[i];
        let mut t = totals.lock().unwrap();
        match result {
            Ok(o) => {
                t.input_tokens += o.statistics.input_tokens;
                t.output_tokens += o.statistics.output_tokens;
                if o.outcome != Outcome::Completed {
                    eprintln!(
                        "│ {specialty}/{file} {:?} after {} steps (issues already on disk)",
                        o.outcome, o.statistics.steps,
                    );
                    t.failed += 1;
                }
            }
            Err(e) => {
                eprintln!("│ {specialty}/{file} dispatch error: {e}");
                t.failed += 1;
            }
        }
    }

    let summary = totals.lock().unwrap();
    eprintln!(
        "└ audit    {} ok, {} failed, {} in / {} out tokens, {:.1}s",
        total - summary.failed,
        summary.failed,
        summary.input_tokens,
        summary.output_tokens,
        phase2_started.elapsed().as_secs_f64(),
    );
    eprintln!("Result written to {}", config.output);
    eprintln!(
        "\n{} issue(s) across {} file(s): {} high, {} medium, {} low. {:.1}s total.",
        summary.issues,
        affected.lock().unwrap().len(),
        summary.high,
        summary.medium,
        summary.low,
        started.elapsed().as_secs_f64(),
    );
}

fn print_plan(assignments: &[(String, String)]) {
    let mut by_file: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (f, s) in assignments {
        by_file.entry(f).or_default().push(s);
    }
    eprintln!("triage plan:");
    for (file, specialists) in by_file {
        eprintln!("  {file}  →  {}", specialists.join(", "));
    }
}

fn scout_logger(specialty: String) -> Arc<dyn Fn(Event) + Send + Sync> {
    let tag = format!("{specialty}-scout");
    Arc::new(move |event| log_agent_event(&tag, &event.kind))
}

fn audit_logger(
    file: String,
    specialty: String,
    progress: Arc<AtomicUsize>,
    total: usize,
) -> Arc<dyn Fn(Event) + Send + Sync> {
    let tag = format!("{specialty} {file}");
    Arc::new(move |event| {
        if let EventKind::AgentFinished { steps, outcome } = &event.kind {
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!("│ {tag:<48} ▾ {done:>3}/{total} {outcome:?} in {steps} steps");
            return;
        }
        log_agent_event(&tag, &event.kind);
    })
}

// Verbose per-event log. Surfaces step boundaries, every tool call (input
// summary in, output preview out, failure reason), token usage, context
// compactions, request retries, policy hits, and run boundaries. The goal is
// "you can see what each agent is doing right now" without flag-gating.
fn log_agent_event(tag: &str, kind: &EventKind) {
    match kind {
        EventKind::AgentStarted => {
            eprintln!("│ {tag:<48} ▸ start");
        }
        EventKind::StepStarted { step } => {
            eprintln!("│ {tag:<48}   step {step}");
        }
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => {
            let detail = tool_input_summary(tool_name, input);
            if detail.is_empty() {
                eprintln!("│ {tag:<48}     → {tool_name}");
            } else {
                eprintln!("│ {tag:<48}     → {tool_name}({detail})");
            }
        }
        EventKind::ToolCallFinished {
            tool_name, output, ..
        } => {
            eprintln!(
                "│ {tag:<48}     ← {tool_name}: {}",
                preview(output, 100)
            );
        }
        EventKind::ToolCallFailed {
            tool_name, message, ..
        } => {
            eprintln!(
                "│ {tag:<48}     ✗ {tool_name}: {}",
                preview(message, 120)
            );
        }
        EventKind::TokensReported { usage, .. } => {
            eprintln!(
                "│ {tag:<48}     tokens in={} out={} cached={}",
                usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens,
            );
        }
        EventKind::ContextCompacted {
            tokens, threshold, ..
        } => {
            eprintln!("│ {tag:<48}     compact {tokens} → {threshold} tokens");
        }
        EventKind::RequestRetried {
            attempt,
            max_attempts,
            message,
            ..
        } => {
            eprintln!(
                "│ {tag:<48}     ↻ retry {attempt}/{max_attempts}: {}",
                preview(message, 100)
            );
        }
        EventKind::RequestFailed { message, .. } => {
            eprintln!(
                "│ {tag:<48}     ✗ request failed: {}",
                preview(message, 120)
            );
        }
        EventKind::PolicyViolated { kind, limit } => {
            eprintln!("│ {tag:<48}     ✗ policy {kind:?} (limit {limit})");
        }
        EventKind::OutputTruncated { step } => {
            eprintln!("│ {tag:<48}     ⚠ reply truncated at step {step}");
        }
        EventKind::AgentFinished { steps, outcome } => {
            eprintln!("│ {tag:<48} ▾ {outcome:?} in {steps} steps");
        }
        _ => {}
    }
}

fn tool_input_summary(tool: &str, input: &serde_json::Value) -> String {
    match tool {
        "glob" => input["pattern"].as_str().unwrap_or("").into(),
        "list_directory" => input["path"].as_str().unwrap_or("").into(),
        "read_file" => input["path"]
            .as_str()
            .or(input["file"].as_str())
            .unwrap_or("")
            .into(),
        "report_file" => input["file"].as_str().unwrap_or("").into(),
        "report_issue" => {
            let line = input["line"].as_i64().unwrap_or(0);
            let severity = input["severity"].as_str().unwrap_or("?");
            let category = input["category"].as_str().unwrap_or("?");
            let message = input["message"].as_str().unwrap_or("");
            format!("L{line} {severity}:{category} — {}", preview(message, 80))
        }
        _ => preview(&input.to_string(), 80),
    }
}

fn preview(s: &str, max: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= max {
        return one_line;
    }
    let cut: String = one_line.chars().take(max).collect();
    format!("{cut}…")
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
    let mut output = "issues.ndjson".to_string();
    let mut batch_size = 2;

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
                eprintln!("Scan a project for code issues. Issues stream to NDJSON: one");
                eprintln!("JSON object per line, flushed after every report_issue call so");
                eprintln!("Ctrl-C and max_steps cutoffs leave partial results on disk.\n");
                eprintln!("Usage: project-scanner [OPTIONS] [DIR]\n");
                eprintln!("Options:");
                eprintln!("  --model <MODEL>        Model override");
                eprintln!("  --batch-size <N>       Parallel auditors in flight (default: 2)");
                eprintln!("  -o, --output <PATH>    Output file (default: issues.ndjson)");
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
