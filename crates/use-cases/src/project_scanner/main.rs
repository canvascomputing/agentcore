//! Scans a project for two classes of security issue: data exfiltration paths
//! and persistence backdoors. The directory is walked programmatically to
//! enumerate source files; a Werk then dispatches (file, specialty) pairs to
//! specialist auditors in parallel.
//!
//! The output file is an audit log in NDJSON: one JSON object per line, each
//! tagged with `kind` and `ts`. Records are written and flushed at the moment
//! each event happens, so the file is a complete chronological replay even
//! after Ctrl-C or `max_steps`.
//!
//! Record kinds:
//! - Orchestrator-level: `run_start`, `triage_plan`, `run_finish`.
//! - Per-agent (every `EventKind` the loop emits, except `TextChunkReceived`):
//!   `agent_started`, `agent_finished`, `step_started`, `step_finished`,
//!   `tool_call_started`, `tool_call_finished`, `tool_call_failed`,
//!   `tokens_reported`, `request_started`, `request_finished`,
//!   `request_retried`, `request_failed`, `output_truncated`,
//!   `context_compacted`, `policy_violated`, `contract_missed`,
//!   `agent_paused`, `agent_resumed`.
//!
//! Custom-tool calls (`report_issue`, `mark_status`) come through as
//! `tool_call_started` records with `tool_name` and `input`.
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
use agentwerk::tools::{ReadFileTool, Tool, ToolResult};
use agentwerk::{Agent, Event, Werk};
use serde_json::json;

const MAX_STEPS: u32 = 200;

// Per-agent prompt budget. Threaded into the prompt via a template so the
// prompt text and the runtime limit stay in sync from a single source.
const AUDIT_TOOL_BUDGET: usize = 8;

struct Specialty {
    name: &'static str,
    role_file: &'static str,
    task_file: &'static str,
}

const SPECIALTIES: &[Specialty] = &[
    Specialty {
        name: "exfiltration",
        role_file: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/project_scanner/prompts/exfiltration-analyst.role.md",
        ),
        task_file: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/project_scanner/prompts/exfiltration-analyst.task.md",
        ),
    },
    Specialty {
        name: "persistence_backdoors",
        role_file: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/project_scanner/prompts/persistence-analyst.role.md",
        ),
        task_file: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/project_scanner/prompts/persistence-analyst.task.md",
        ),
    },
];

// Parse a JSON Schema embedded at compile time via `include_str!`.
// Schemas are stored as JSON files under `schemas/` so they can be edited
// without touching Rust source. A malformed schema fails the binary at
// startup, which is preferable to surfacing it on first tool use.
fn load_schema(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("invalid embedded JSON schema")
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// Append one audit-log record. The caller passes a JSON object payload; this
// fn injects `kind` and `ts`, serializes to one line, writes, and flushes.
// Flushing per record keeps the file a valid NDJSON stream even after Ctrl-C.
fn append_event(writer: &IssueWriter, kind: &str, mut payload: serde_json::Value) {
    if let serde_json::Value::Object(ref mut map) = payload {
        map.insert("kind".into(), json!(kind));
        map.insert("ts".into(), json!(now_unix_ms()));
    }
    let line = serde_json::to_string(&payload).unwrap_or_default();
    let mut w = writer.lock().unwrap();
    let _ = writeln!(&mut *w, "{line}");
    let _ = w.flush();
}

// Convert a loop event into an audit-log record. Returns None for events too
// noisy to record (TextChunkReceived fires per streamed sub-token). The
// `agent` field is injected by `write_audit_event`, not here.
fn serialize_event(event: &Event) -> Option<(&'static str, serde_json::Value)> {
    use EventKind::*;
    let pair = match &event.kind {
        AgentStarted => ("agent_started", json!({})),
        AgentFinished { steps, outcome } => (
            "agent_finished",
            json!({ "steps": steps, "outcome": format!("{outcome:?}") }),
        ),
        StepStarted { step } => ("step_started", json!({ "step": step })),
        StepFinished { step } => ("step_finished", json!({ "step": step })),
        ToolCallStarted {
            tool_name,
            call_id,
            input,
        } => (
            "tool_call_started",
            json!({ "tool_name": tool_name, "call_id": call_id, "input": input }),
        ),
        ToolCallFinished {
            tool_name,
            call_id,
            output,
        } => (
            "tool_call_finished",
            json!({ "tool_name": tool_name, "call_id": call_id, "output": output }),
        ),
        ToolCallFailed {
            tool_name,
            call_id,
            message,
            kind,
        } => (
            "tool_call_failed",
            json!({
                "tool_name": tool_name,
                "call_id": call_id,
                "message": message,
                "failure_kind": format!("{kind:?}"),
            }),
        ),
        TokensReported { model, usage } => {
            ("tokens_reported", json!({ "model": model, "usage": usage }))
        }
        TextChunkReceived { .. } => return None,
        RequestStarted { model } => ("request_started", json!({ "model": model })),
        RequestFinished { model } => ("request_finished", json!({ "model": model })),
        RequestRetried {
            attempt,
            max_attempts,
            kind,
            message,
        } => (
            "request_retried",
            json!({
                "attempt": attempt,
                "max_attempts": max_attempts,
                "error_kind": format!("{kind:?}"),
                "message": message,
            }),
        ),
        RequestFailed { kind, message } => (
            "request_failed",
            json!({ "error_kind": format!("{kind:?}"), "message": message }),
        ),
        OutputTruncated { step } => ("output_truncated", json!({ "step": step })),
        ContextCompacted {
            step,
            tokens,
            threshold,
            reason,
        } => (
            "context_compacted",
            json!({
                "step": step,
                "tokens": tokens,
                "threshold": threshold,
                "reason": format!("{reason:?}"),
            }),
        ),
        PolicyViolated { kind, limit } => (
            "policy_violated",
            json!({ "policy": format!("{kind:?}"), "limit": limit }),
        ),
        ContractMissed {
            attempt,
            max_attempts,
            path,
            message,
        } => (
            "contract_missed",
            json!({
                "attempt": attempt,
                "max_attempts": max_attempts,
                "path": path,
                "message": message,
            }),
        ),
        AgentPaused => ("agent_paused", json!({})),
        AgentResumed => ("agent_resumed", json!({})),
    };
    Some(pair)
}

// Audit-log adapter for the agent's event handler. Writes one NDJSON record
// per loop event, with the agent name attached.
fn write_audit_event(writer: &IssueWriter, event: &Event) {
    let Some((kind, mut payload)) = serialize_event(event) else {
        return;
    };
    if let serde_json::Value::Object(ref mut map) = payload {
        map.insert("agent".into(), json!(event.agent_name));
    }
    append_event(writer, kind, payload);
}

type IssueWriter = Arc<Mutex<std::fs::File>>;

#[derive(Clone, Debug)]
struct Assignment {
    file: String,
    specialty: String,
}
type Totals = Arc<Mutex<TotalsInner>>;
type Affected = Arc<Mutex<BTreeSet<String>>>;
type StatusSlot = Arc<Mutex<Option<RunStatus>>>;

#[derive(Clone, Debug)]
struct RunStatus {
    status: String,
    reason: String,
    trustworthy: Option<bool>,
    summary: Option<String>,
}

#[derive(Default)]
struct TotalsInner {
    input_tokens: u64,
    output_tokens: u64,
    failed: usize,
    high: usize,
    medium: usize,
    low: usize,
    issues: usize,
    trustworthy_files: usize,
    untrustworthy_files: usize,
    unjudged_files: usize,
}

// Walks `root` and returns relative paths of source files to audit. Skips
// vendored, generated, hidden, test, and minified paths. Sorted so the run is
// deterministic for the same input directory.
fn scan_source_files(root: &std::path::Path) -> Vec<String> {
    fn is_source(path: &std::path::Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".min.js") || name.ends_with(".min.css") {
            return false;
        }
        matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go" | "java" | "rb")
        )
    }
    fn skip_dir(name: &str) -> bool {
        matches!(
            name,
            "node_modules"
                | "target"
                | "dist"
                | "build"
                | "out"
                | "bin"
                | "obj"
                | "vendor"
                | "third_party"
                | "__pycache__"
                | ".venv"
                | "venv"
                | ".next"
                | ".nuxt"
                | "tests"
                | "test"
                | "__tests__"
                | "spec"
                | "specs"
                | "fixtures"
        )
    }
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            if name.starts_with('.') {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                if skip_dir(&name) {
                    continue;
                }
                stack.push(path);
            } else if meta.is_file() && is_source(&path) {
                let rel = path.strip_prefix(root).unwrap_or(&path);
                files.push(rel.to_string_lossy().into_owned());
            }
        }
    }
    files.sort();
    files
}

fn report_issue_tool(file: String, totals: Totals, affected: Affected) -> Tool {
    Tool::new(
        "report_issue",
        "Record one issue you found in the file. Call once per issue, the \
         moment you spot it. Do not batch.",
    )
    .contract(load_schema(include_str!("schemas/report_issue.json")))
    .handler(move |input, _ctx| {
        let file = file.clone();
        let totals = totals.clone();
        let affected = affected.clone();
        Box::pin(async move {
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

// Per-agent terminal status. The agent calls this exactly once as its last
// action; the slot is filled in place so main can read it after the run.
// Replaces the prior stringly-typed "AUDIT_FAILED" convention with a
// structured signal the orchestrator can aggregate without parsing prose.
fn mark_status_tool(slot: StatusSlot) -> Tool {
    Tool::new(
        "mark_status",
        "Declare how this run ended. Call exactly once as your final action \
         before your text reply.",
    )
    .contract(load_schema(include_str!("schemas/mark_status.json")))
    .handler(move |input, _ctx| {
        let slot = slot.clone();
        Box::pin(async move {
            let Some(status) = input["status"].as_str() else {
                return Ok(ToolResult::error("missing required field 'status'"));
            };
            let reason = input["reason"].as_str().unwrap_or("").to_string();
            let trustworthy = input["trustworthy"].as_bool();
            let summary = input["summary"].as_str().map(|s| s.to_string());
            if let Some(ref s) = summary {
                let len = s.chars().count();
                if len > 200 {
                    return Ok(ToolResult::error(format!(
                        "'summary' must be at most 200 chars (got {len})"
                    )));
                }
            }
            {
                let mut s = slot.lock().unwrap();
                if s.is_some() {
                    return Ok(ToolResult::error("mark_status already called once"));
                }
                *s = Some(RunStatus {
                    status: status.to_string(),
                    reason,
                    trustworthy,
                    summary,
                });
            }
            Ok(ToolResult::success(format!("status recorded: {status}")))
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
    let totals: Totals = Arc::new(Mutex::new(TotalsInner::default()));
    let affected: Affected = Arc::new(Mutex::new(BTreeSet::new()));

    append_event(
        &writer,
        "run_start",
        json!({
            "working_dir": config.dir.display().to_string(),
            "model": model,
            "batch_size": config.batch_size,
            "max_steps": MAX_STEPS,
            "specialties": SPECIALTIES.iter().map(|s| s.name).collect::<Vec<_>>(),
        }),
    );

    // Phase 1: walk the directory and pair every source file with every
    // specialty. Replaces the prior LLM-driven scout phase with a deterministic
    // scan, so the auditors get a complete and reproducible work list.
    let scan_started = Instant::now();
    let files = scan_source_files(&config.dir);
    let assignments: Vec<Assignment> = files
        .iter()
        .flat_map(|file| {
            SPECIALTIES.iter().map(|s| Assignment {
                file: file.clone(),
                specialty: s.name.to_string(),
            })
        })
        .collect();

    eprintln!(
        "┌ scan     {} source file(s) under {} -> {} pair(s) ({:.1}s)",
        files.len(),
        config.dir.display(),
        assignments.len(),
        scan_started.elapsed().as_secs_f64(),
    );

    if assignments.is_empty() {
        eprintln!("\nNo source files found. Exiting.");
        append_event(
            &writer,
            "run_finish",
            json!({
                "duration_seconds": started.elapsed().as_secs_f64(),
                "exit": "no_source_files",
            }),
        );
        std::process::exit(1);
    }

    append_event(
        &writer,
        "triage_plan",
        json!({
            "assignments": assignments
                .iter()
                .map(|a| json!({
                    "file": a.file,
                    "specialty": a.specialty,
                }))
                .collect::<Vec<_>>(),
        }),
    );

    print_plan(&assignments);

    // Phase 2: dispatch specialist auditors in parallel via Werk.
    let total = assignments.len();
    let phase2_started = Instant::now();
    let progress = Arc::new(AtomicUsize::new(0));

    eprintln!(
        "\n┌ audit    {} pairs, up to {} in flight (events stream to {})",
        total, config.batch_size, config.output,
    );

    let audit_slots: Vec<StatusSlot> = (0..assignments.len())
        .map(|_| Arc::new(Mutex::new(None)))
        .collect();
    let agents = assignments.iter().enumerate().map(|(i, a)| {
        let file = &a.file;
        let specialty = &a.specialty;
        let prompts = SPECIALTIES
            .iter()
            .find(|s| s.name == specialty.as_str())
            .expect("assignment specialty must match a SPECIALTIES entry");
        Agent::new()
            .name(format!("{specialty}-{i}"))
            .provider(provider.clone())
            .model(&model)
            .role_file(prompts.role_file)
            .tool(ReadFileTool)
            .tool(report_issue_tool(
                file.clone(),
                totals.clone(),
                affected.clone(),
            ))
            .tool(mark_status_tool(audit_slots[i].clone()))
            .max_steps(MAX_STEPS)
            .template("budget", json!(AUDIT_TOOL_BUDGET))
            .template("file", json!(file))
            .working_dir(config.dir.clone())
            .work_file(prompts.task_file)
            .event_handler(audit_logger(
                file.clone(),
                specialty.clone(),
                progress.clone(),
                total,
                writer.clone(),
            ))
    });

    let agent_index_by_name: std::collections::HashMap<String, usize> = assignments
        .iter()
        .enumerate()
        .map(|(i, a)| (format!("{}-{i}", a.specialty), i))
        .collect();

    let (producing, mut stream) = Werk::new()
        .lines(config.batch_size)
        .interrupt_signal(cancel.clone())
        .keep_working(std::iter::empty::<Agent>());
    producing.staff_more(agents);
    drop(producing);

    let mut audit_status = StatusBreakdown::default();

    while let Some((name, result)) = stream.next().await {
        let i = *agent_index_by_name
            .get(&name)
            .expect("stream name must map back to an assignment");
        let a = &assignments[i];
        let file = &a.file;
        let specialty = &a.specialty;
        let declared = audit_slots[i].lock().unwrap().clone();
        match result {
            Ok(o) => {
                if !o.response_raw.trim().is_empty() {
                    eprintln!(
                        "│ {specialty} {file} reply: {}",
                        preview(&o.response_raw, 200)
                    );
                }
                match &declared {
                    Some(s) => {
                        let trust = match s.trustworthy {
                            Some(true) => " trust=ok",
                            Some(false) => " trust=NO",
                            None => " trust=?",
                        };
                        let verdict = s.summary.as_deref().unwrap_or(&s.reason);
                        eprintln!(
                            "│ {specialty} {file} status: {}{} — {}",
                            s.status, trust, verdict,
                        );
                        audit_status.bump(&s.status);
                        let mut t = totals.lock().unwrap();
                        match s.trustworthy {
                            Some(true) => t.trustworthy_files += 1,
                            Some(false) => t.untrustworthy_files += 1,
                            None => t.unjudged_files += 1,
                        }
                    }
                    None => {
                        eprintln!("│ {specialty} {file} status: <not declared>");
                        audit_status.undeclared += 1;
                        totals.lock().unwrap().unjudged_files += 1;
                    }
                }
                let mut t = totals.lock().unwrap();
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
                let mut t = totals.lock().unwrap();
                t.failed += 1;
                audit_status.undeclared += 1;
            }
        }
    }

    let summary = totals.lock().unwrap();
    eprintln!(
        "└ audit    {} ok, {} failed | status: {complete} complete, {partial} partial, {blocked} blocked, {undeclared} undeclared | trust: {trust_ok} ok, {trust_no} NO, {trust_q} unjudged | {} in / {} out tokens, {:.1}s",
        total - summary.failed,
        summary.failed,
        summary.input_tokens,
        summary.output_tokens,
        phase2_started.elapsed().as_secs_f64(),
        complete = audit_status.complete,
        partial = audit_status.partial,
        blocked = audit_status.blocked,
        undeclared = audit_status.undeclared,
        trust_ok = summary.trustworthy_files,
        trust_no = summary.untrustworthy_files,
        trust_q = summary.unjudged_files,
    );
    eprintln!("Result written to {}", config.output);
    let affected_count = affected.lock().unwrap().len();
    eprintln!(
        "\n{} issue(s) across {} file(s): {} high, {} medium, {} low. trust: {} ok / {} NO / {} unjudged. {:.1}s total.",
        summary.issues,
        affected_count,
        summary.high,
        summary.medium,
        summary.low,
        summary.trustworthy_files,
        summary.untrustworthy_files,
        summary.unjudged_files,
        started.elapsed().as_secs_f64(),
    );
    let run_finish = json!({
        "duration_seconds": started.elapsed().as_secs_f64(),
        "issues": summary.issues,
        "high": summary.high,
        "medium": summary.medium,
        "low": summary.low,
        "files_with_issues": affected_count,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "trustworthy_files": summary.trustworthy_files,
        "untrustworthy_files": summary.untrustworthy_files,
        "unjudged_files": summary.unjudged_files,
        "audit_status": {
            "complete": audit_status.complete,
            "partial": audit_status.partial,
            "blocked": audit_status.blocked,
            "undeclared": audit_status.undeclared,
        },
    });
    drop(summary);
    append_event(&writer, "run_finish", run_finish);
}

#[derive(Default)]
struct StatusBreakdown {
    complete: usize,
    partial: usize,
    blocked: usize,
    undeclared: usize,
}

impl StatusBreakdown {
    fn bump(&mut self, status: &str) {
        match status {
            "complete" => self.complete += 1,
            "partial" => self.partial += 1,
            "blocked" => self.blocked += 1,
            _ => self.undeclared += 1,
        }
    }
}

fn print_plan(assignments: &[Assignment]) {
    let mut by_file: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for a in assignments {
        by_file.entry(&a.file).or_default().push(&a.specialty);
    }
    eprintln!("triage plan:");
    for (file, specialties) in by_file {
        for specialty in specialties {
            eprintln!("  {file}  →  {specialty}");
        }
    }
}

fn audit_logger(
    file: String,
    specialty: String,
    progress: Arc<AtomicUsize>,
    total: usize,
    writer: IssueWriter,
) -> Arc<dyn Fn(Event) + Send + Sync> {
    let tag = format!("{specialty} {file}");
    Arc::new(move |event| {
        write_audit_event(&writer, &event);
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
            eprintln!("│ {tag:<48}     ← {tool_name}: {}", preview(output, 100));
        }
        EventKind::ToolCallFailed {
            tool_name, message, ..
        } => {
            eprintln!("│ {tag:<48}     ✗ {tool_name}: {}", preview(message, 120));
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
        "read_file" => input["path"]
            .as_str()
            .or(input["file"].as_str())
            .unwrap_or("")
            .into(),
        "report_issue" => {
            let line = input["line"].as_i64().unwrap_or(0);
            let severity = input["severity"].as_str().unwrap_or("?");
            let category = input["category"].as_str().unwrap_or("?");
            let message = input["message"].as_str().unwrap_or("");
            format!("L{line} {severity}:{category} — {}", preview(message, 80))
        }
        "mark_status" => {
            let status = input["status"].as_str().unwrap_or("?");
            let reason = input["reason"].as_str().unwrap_or("");
            if reason.is_empty() {
                status.into()
            } else {
                format!("{status} — {}", preview(reason, 80))
            }
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
    let mut output = "audit.ndjson".to_string();
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
                eprintln!("Scan a project for code issues. Output is an NDJSON audit log:");
                eprintln!("one JSON object per line, each tagged with `kind` and `ts`,");
                eprintln!("flushed after every event so Ctrl-C and max_steps cutoffs leave");
                eprintln!("a complete chronological replay on disk.\n");
                eprintln!("Record kinds: orchestrator-level run_start, triage_plan,");
                eprintln!("run_finish; plus every EventKind the agent loop emits");
                eprintln!("(agent_started, step_started, tool_call_started/_finished,");
                eprintln!("tokens_reported, request_*, policy_violated, etc.).\n");
                eprintln!("Custom-tool calls (report_file, report_issue, mark_status)");
                eprintln!("appear as tool_call_started records with their inputs.\n");
                eprintln!("Usage: project-scanner [OPTIONS] [DIR]\n");
                eprintln!("Options:");
                eprintln!("  --model <MODEL>        Model override");
                eprintln!("  --batch-size <N>       Parallel auditors in flight (default: 2)");
                eprintln!("  -o, --output <PATH>    Output file (default: audit.ndjson)");
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
