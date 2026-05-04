//! Project scanner.
//!
//! Walks a directory, derives the unique set of file extensions, and runs
//! three phases on a single shared `TicketSystem` driven by `run`:
//!   1. A pool of `technology_guru_*` agents drains one ticket per
//!      extension and maps each to its primary technology, returning a
//!      JSON string `{"extension", "technology"}` validated against the
//!      ticket schema. Pool size is `--tech-concurrency` (default 4),
//!      capped at the number of unique extensions.
//!   2. One `explorer_<ext>` agent per extension scans `<ext>` files
//!      for indicators across all configured risk domains in a single
//!      sweep. For each finding, the explorer creates a Path-A ticket
//!      assigned to the matching `investigator_<ext>` agent (label
//!      `investigation`); the ticket body carries the domain name. The
//!      explorer settles its own ticket with a one-line summary string.
//!   3. One `investigator_<ext>` agent per extension processes every
//!      per-finding ticket forwarded by its paired explorer, regardless
//!      of domain. Each investigator owns one report file at
//!      `<reports-dir>/<ext-slug>.md`. Before doing any work it reads
//!      the report; if the same `(source, domain)` pair is already in
//!      there, the ticket settles as a duplicate. Otherwise the
//!      investigator pulls surrounding context, appends a Markdown entry
//!      tagged with the domain, and settles done.
//!
//! Cancel handling: a private `ctrl_c` atomic is set only by the Ctrl-C
//! task. A relay copies it into the framework's interrupt signal so the
//! cooperative-cancel path inside the loop still works. Happy-path
//! shutdown happens by setting the framework signal once all phases
//! settle.
//!
//! Usage: project-scanner <DIR> [--max-steps N] [--tech-concurrency N] [--reports-dir PATH]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agentwerk::providers::{model_from_env, provider_from_env};
use agentwerk::tools::{
    GlobTool, GrepTool, ListDirectoryTool, ManageTicketsTool, ReadFileTool, WriteFileTool,
};
use agentwerk::{Agent, Event, EventKind, Runnable, Schema, Status, Ticket, TicketSystem};
use serde_json::{json, Value};

const TECH_ROLE: &str = include_str!("prompts/technology-guru.md");
const EXPLORATION_ROLE: &str = include_str!("prompts/exploration-expert.md");
const EXPLORER_TASK: &str = include_str!("prompts/explorer.task.md");
const INVESTIGATOR_ROLE: &str = include_str!("prompts/security-investigator.md");

const DOMAINS: &[(&str, &str)] = &[
    (
        "exfiltration",
        "patterns that move data out of a process or host: outbound network requests, \
         encoded-payload writes, suspicious file uploads, DNS-tunnelling helpers.",
    ),
    (
        "persistence",
        "patterns that survive a restart or re-establish access: scheduled tasks, \
         autostart entries, dropped binaries, modified service files, hooks installed \
         into shell init.",
    ),
];

const PHASE_POLL_INTERVAL: Duration = Duration::from_millis(150);

#[tokio::main]
async fn main() {
    let args = parse_args();
    let provider = provider_from_env().expect("LLM provider required");
    let model = model_from_env().expect("model name required");
    let ctrl_c = install_interrupt_signal();

    let scan_dir = match fs::canonicalize(&args.dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot resolve directory '{}': {e}", args.dir.display());
            std::process::exit(1);
        }
    };

    let scan = collect_extensions(&scan_dir);
    if scan.extensions.is_empty() {
        eprintln!(
            "no files with extensions found under {} ({} file(s) walked)",
            scan_dir.display(),
            scan.files,
        );
        std::process::exit(1);
    }

    let extensions = scan.extensions;
    let tech_workers = args.tech_concurrency.min(extensions.len()).max(1);
    let per_extension_agents = extensions.len();

    let reports_dir = match prepare_reports_dir(&args.reports_dir, &scan_dir, &extensions) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot prepare reports directory: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "scanning {}\n  • {} files walked\n  • {} unique extension(s)\n  • {} domain(s)\n  • {} technology guru(s)\n  • {} exploration expert(s) (one per extension)\n  • {} security investigator(s) (one per extension)\n  • reports → {}\n",
        scan_dir.display(),
        scan.files,
        extensions.len(),
        DOMAINS.len(),
        tech_workers,
        per_extension_agents,
        per_extension_agents,
        reports_dir.display(),
    );

    let event_handler: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(|e: Event| log_event(&e));
    let framework_signal = framework_signal_mirroring(&ctrl_c);

    let tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&framework_signal))
        .max_steps(args.max_steps);

    let tech_schema = Schema::parse(json!({
        "type": "object",
        "properties": {
            "extension":  { "type": "string", "description": "Extension copied verbatim from the task, including the leading dot." },
            "technology": { "type": "string", "description": "Canonical English name of the language, format, or framework." }
        },
        "required": ["extension", "technology"],
        "additionalProperties": false
    }))
    .expect("technology schema is well-formed");

    eprintln!("── phase 1: technology mapping ────────────────────────────────");
    eprintln!("created {} tech-mapping ticket(s):", extensions.len());
    for ext in &extensions {
        tickets.task_schema_assigned(ext.clone(), tech_schema.clone(), "tech-mapping");
        eprintln!("  • {ext}");
    }
    eprintln!();

    for w in 0..tech_workers {
        tickets.add(
            Agent::new()
                .name(format!("technology_guru_{w}"))
                .provider(Arc::clone(&provider))
                .model(&model)
                .role(TECH_ROLE.trim())
                .label("tech-mapping")
                .tool(ManageTicketsTool)
                .event_handler(Arc::clone(&event_handler)),
        );
    }

    let explorer_names = register_exploration_agents(
        &tickets,
        &extensions,
        &scan_dir,
        &provider,
        &model,
        &event_handler,
    );

    register_investigator_agents(
        &tickets,
        &extensions,
        &scan_dir,
        &reports_dir,
        &provider,
        &model,
        &event_handler,
    );

    let run_handle = {
        let tickets = Arc::clone(&tickets);
        tokio::spawn(async move {
            tickets.run().await;
        })
    };

    if !wait_for_label(&tickets, "tech-mapping", &ctrl_c).await {
        return shutdown(framework_signal, run_handle, ExitReason::Cancelled).await;
    }

    let mappings = mappings_from_tickets(&tickets);
    if mappings.is_empty() {
        return shutdown(framework_signal, run_handle, ExitReason::NoMappings).await;
    }

    eprintln!("\n── phase 2: exploration ───────────────────────────────────────");
    eprintln!("created {} exploration ticket(s):", mappings.len());
    for (extension, technology) in &mappings {
        let agent_name = explorer_agent_name(extension);
        if !explorer_names.contains(&agent_name) {
            continue;
        }
        let body = render_explorer_body(EXPLORER_TASK, technology, extension);
        tickets.create(
            Ticket::new(body)
                .label("exploration")
                .assign_to(&agent_name),
        );
        eprintln!("  • {agent_name}  (technology={technology})");
    }
    eprintln!();

    if !wait_for_label(&tickets, "exploration", &ctrl_c).await {
        return shutdown(framework_signal, run_handle, ExitReason::Cancelled).await;
    }

    let investigation_count = count_with_label(&tickets, "investigation");
    if investigation_count > 0 {
        eprintln!("\n── phase 3: investigation ─────────────────────────────────────");
        eprintln!(
            "{investigation_count} investigation ticket(s) created by explorers; reports at {}",
            reports_dir.display()
        );
        eprintln!();

        if !wait_for_label(&tickets, "investigation", &ctrl_c).await {
            return shutdown(framework_signal, run_handle, ExitReason::Cancelled).await;
        }
    } else {
        eprintln!("\nno findings forwarded to investigators — phase 3 skipped.");
    }

    framework_signal.store(true, Ordering::Relaxed);
    let _ = run_handle.await;

    print_phase1_summary(&tickets, &mappings);
    print_phase2_summary(&tickets);
    if investigation_count > 0 {
        print_phase3_summary(&tickets);
    }
    print_aggregate_stats(
        &tickets,
        extensions.len(),
        per_extension_agents,
        investigation_count,
    );
}

fn register_exploration_agents(
    tickets: &TicketSystem,
    extensions: &[String],
    scan_dir: &Path,
    provider: &Arc<dyn agentwerk::providers::Provider>,
    model: &str,
    event_handler: &Arc<dyn Fn(Event) + Send + Sync>,
) -> Vec<String> {
    let domains_block = domains_block();
    let mut names = Vec::with_capacity(extensions.len());
    for extension in extensions {
        let agent_name = explorer_agent_name(extension);
        let investigator_name = investigator_agent_name(extension);
        tickets.add(
            Agent::new()
                .name(&agent_name)
                .provider(Arc::clone(provider))
                .model(model)
                .role(EXPLORATION_ROLE.trim())
                .template_variables([
                    ("extension", extension.as_str()),
                    ("domains_block", domains_block.as_str()),
                    ("investigator_agent_name", investigator_name.as_str()),
                ])
                .working_dir(scan_dir.to_path_buf())
                .tool(GrepTool)
                .tool(GlobTool)
                .tool(ReadFileTool)
                .tool(ListDirectoryTool)
                .tool(ManageTicketsTool)
                .event_handler(Arc::clone(event_handler)),
        );
        names.push(agent_name);
    }
    names
}

fn register_investigator_agents(
    tickets: &TicketSystem,
    extensions: &[String],
    scan_dir: &Path,
    reports_dir: &Path,
    provider: &Arc<dyn agentwerk::providers::Provider>,
    model: &str,
    event_handler: &Arc<dyn Fn(Event) + Send + Sync>,
) {
    for extension in extensions {
        let agent_name = investigator_agent_name(extension);
        let report_path = report_file_path(reports_dir, extension);
        tickets.add(
            Agent::new()
                .name(&agent_name)
                .provider(Arc::clone(provider))
                .model(model)
                .role(INVESTIGATOR_ROLE.trim())
                .template_variables([
                    ("extension", extension.as_str()),
                    ("report_file", report_path.to_string_lossy().as_ref()),
                ])
                .working_dir(scan_dir.to_path_buf())
                .tool(ReadFileTool)
                .tool(WriteFileTool)
                .tool(GrepTool)
                .tool(GlobTool)
                .tool(ListDirectoryTool)
                .tool(ManageTicketsTool)
                .event_handler(Arc::clone(event_handler)),
        );
    }
}

fn render_explorer_body(template: &str, technology: &str, extension: &str) -> String {
    template
        .replace("{technology}", technology)
        .replace("{extension}", extension)
}

fn domains_block() -> String {
    DOMAINS
        .iter()
        .map(|(name, description)| format!("- **{name}** — {description}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn explorer_agent_name(extension: &str) -> String {
    format!("explorer_{}", extension_slug(extension))
}

fn investigator_agent_name(extension: &str) -> String {
    format!("investigator_{}", extension_slug(extension))
}

fn extension_slug(extension: &str) -> String {
    extension
        .trim_start_matches('.')
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn report_file_path(reports_dir: &Path, extension: &str) -> PathBuf {
    reports_dir.join(format!("{}.md", extension_slug(extension)))
}

fn prepare_reports_dir(
    cli_path: &Option<PathBuf>,
    scan_dir: &Path,
    extensions: &[String],
) -> std::io::Result<PathBuf> {
    let dir = cli_path
        .clone()
        .unwrap_or_else(|| scan_dir.join(".scanner-reports"));
    fs::create_dir_all(&dir)?;
    let dir = fs::canonicalize(&dir)?;
    let domains_summary = DOMAINS
        .iter()
        .map(|(name, _)| format!("- {name}"))
        .collect::<Vec<_>>()
        .join("\n");
    for extension in extensions {
        let path = report_file_path(&dir, extension);
        if !path.exists() {
            fs::write(
                &path,
                format!(
                    "# Investigation report — {extension}\n\n\
                     Domains in scope:\n\
                     {domains_summary}\n\n\
                     (no findings yet)\n",
                ),
            )?;
        }
    }
    Ok(dir)
}

enum ExitReason {
    Cancelled,
    NoMappings,
}

async fn shutdown(
    framework_signal: Arc<AtomicBool>,
    run_handle: tokio::task::JoinHandle<()>,
    reason: ExitReason,
) {
    framework_signal.store(true, Ordering::Relaxed);
    match reason {
        ExitReason::Cancelled => {
            // User wanted out: don't wait for graceful agent teardown — the
            // process exit aborts in-flight tools (HTTP retries, blocked
            // grep) that the cooperative-cancel signal can't always reach.
            eprintln!("\ncancelled.");
            run_handle.abort();
            std::process::exit(130);
        }
        ExitReason::NoMappings => {
            let _ = run_handle.await;
            eprintln!("\nno extensions mapped successfully — aborting before phase 2.");
            std::process::exit(1);
        }
    }
}

/// Block until every ticket carrying `label` has reached a terminal status
/// (`Done` or `Failed`) and at least one such ticket exists. Returns
/// `false` if the Ctrl-C atomic flips before that condition is met.
async fn wait_for_label(tickets: &TicketSystem, label: &str, ctrl_c: &Arc<AtomicBool>) -> bool {
    loop {
        if ctrl_c.load(Ordering::Relaxed) {
            return false;
        }
        let labelled: Vec<_> = tickets
            .tickets()
            .into_iter()
            .filter(|t| t.has_label(label))
            .collect();
        if !labelled.is_empty() && labelled.iter().all(|t| t.is_done() || t.is_failed()) {
            return true;
        }
        tokio::time::sleep(PHASE_POLL_INTERVAL).await;
    }
}

fn count_with_label(tickets: &TicketSystem, label: &str) -> usize {
    tickets
        .tickets()
        .iter()
        .filter(|t| t.has_label(label))
        .count()
}

struct WalkSummary {
    files: usize,
    extensions: Vec<String>,
}

fn collect_extensions(root: &Path) -> WalkSummary {
    let mut set: BTreeSet<String> = BTreeSet::new();
    let mut files: usize = 0;
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(it) => it,
            Err(e) => {
                eprintln!("warn: cannot read {}: {e}", dir.display());
                continue;
            }
        };
        for entry in entries.flatten() {
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("warn: cannot stat {}: {e}", entry.path().display());
                    continue;
                }
            };
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                files += 1;
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                    set.insert(format!(".{ext}"));
                }
            }
        }
    }
    WalkSummary {
        files,
        extensions: set.into_iter().collect(),
    }
}

fn mappings_from_tickets(tickets: &TicketSystem) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for ticket in tickets.tickets() {
        if !ticket.has_label("tech-mapping") || ticket.status() != Status::Done {
            continue;
        }
        let Some(result) = ticket.result() else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(result) else {
            continue;
        };
        let ext = value.get("extension").and_then(|v| v.as_str());
        let tech = value.get("technology").and_then(|v| v.as_str());
        if let (Some(ext), Some(tech)) = (ext, tech) {
            out.push((ext.to_string(), tech.to_string()));
        }
    }
    out
}

fn print_phase1_summary(tickets: &TicketSystem, mappings: &[(String, String)]) {
    eprintln!("\nphase 1 results:");
    for ticket in tickets.tickets() {
        if !ticket.has_label("tech-mapping") {
            continue;
        }
        let key = ticket.key();
        match ticket.status() {
            Status::Done => {
                let pair = mappings
                    .iter()
                    .find(|(ext, _)| ticket.task.as_str().is_some_and(|s| s.contains(ext)));
                match pair {
                    Some((ext, tech)) => eprintln!("  {key}  {ext:<10} → {tech}"),
                    None => eprintln!("  {key}  done (unparseable result)"),
                }
            }
            other => eprintln!("  {key}  {other:?}"),
        }
    }
}

fn print_phase2_summary(tickets: &TicketSystem) {
    eprintln!("\nphase 2 results:");
    for ticket in tickets.tickets() {
        if !ticket.has_label("exploration") {
            continue;
        }
        let key = ticket.key();
        let status = ticket.status();
        let snippet = ticket
            .result()
            .map(|r| truncate(r, 120))
            .unwrap_or_else(|| "(no result)".into());
        eprintln!("  {key}  {status:?}  {snippet}");
    }
}

fn print_phase3_summary(tickets: &TicketSystem) {
    eprintln!("\nphase 3 results:");
    for ticket in tickets.tickets() {
        if !ticket.has_label("investigation") {
            continue;
        }
        let key = ticket.key();
        let status = ticket.status();
        let snippet = ticket
            .result()
            .map(|r| truncate(r, 120))
            .unwrap_or_else(|| "(no result)".into());
        eprintln!("  {key}  {status:?}  {snippet}");
    }
}

fn print_aggregate_stats(
    tickets: &TicketSystem,
    extensions: usize,
    pairs: usize,
    investigations: usize,
) {
    eprintln!("\nstats:");
    let s = tickets.stats();
    eprintln!(
        "  tickets: {} done / {} failed (phase 1 expected {}, phase 2 expected {}, phase 3 expected {})",
        s.tickets_done(),
        s.tickets_failed(),
        extensions,
        pairs,
        investigations,
    );
    eprintln!(
        "  {} requests, {} tool calls, {} in / {} out tokens",
        s.requests(),
        s.tool_calls(),
        s.input_tokens(),
        s.output_tokens(),
    );
}

fn log_event(event: &Event) {
    let agent = &event.agent_name;
    match &event.kind {
        EventKind::TicketStarted { key } => eprintln!("[{agent}] started {key}"),
        EventKind::TicketDone { key } => eprintln!("[{agent}] done {key}"),
        EventKind::TicketFailed { key } => eprintln!("[{agent}] failed {key}"),
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => eprintln!(
            "[{agent}] {tool_name}: {}",
            tool_call_summary(tool_name, input)
        ),
        EventKind::ToolCallFailed {
            tool_name,
            message,
            kind,
            ..
        } => eprintln!(
            "[{agent}] ✗ {tool_name} ({kind:?}): {}",
            truncate(message, 200)
        ),
        EventKind::RequestFailed { message, kind } => eprintln!(
            "[{agent}] ✗ request failed ({kind:?}): {}",
            truncate(message, 200)
        ),
        EventKind::RequestRetried {
            attempt,
            max_attempts,
            kind,
            message,
        } => eprintln!(
            "[{agent}] ⟳ request retry {attempt}/{max_attempts} ({kind:?}): {}",
            truncate(message, 200)
        ),
        EventKind::SchemaRetried {
            attempt,
            max_attempts,
            message,
        } => eprintln!(
            "[{agent}] ⟳ schema retry {attempt}/{max_attempts}: {}",
            truncate(message, 200)
        ),
        EventKind::PolicyViolated { kind, limit } => {
            eprintln!("[{agent}] ✗ policy violated: {kind:?} limit={limit}")
        }
        _ => {}
    }
}

fn tool_call_summary(tool_name: &str, input: &Value) -> String {
    match tool_name {
        "grep_tool" | "grep" => {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let glob = input.get("glob").and_then(|v| v.as_str());
            let path = input.get("path").and_then(|v| v.as_str());
            let mut out = format!("/{}/", truncate(pattern, 60));
            if let Some(g) = glob {
                out.push_str(&format!(" glob={g}"));
            }
            if let Some(p) = path {
                out.push_str(&format!(" path={p}"));
            }
            out
        }
        "glob_tool" | "glob" => {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let path = input.get("path").and_then(|v| v.as_str());
            let mut out = truncate(pattern, 60);
            if let Some(p) = path {
                out.push_str(&format!(" path={p}"));
            }
            out
        }
        "read_file_tool" | "read_file" | "list_directory_tool" | "list_directory" => input
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| truncate(s, 80))
            .unwrap_or_default(),
        "write_file_tool" | "write_file" => input
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| format!("write {}", truncate(s, 80)))
            .unwrap_or_default(),
        "manage_tickets_tool" => {
            let action = input.get("action").and_then(|v| v.as_str()).unwrap_or("?");
            match action {
                "done" => {
                    let result = input.get("result").and_then(|v| v.as_str()).unwrap_or("");
                    format!("done: {}", truncate(result, 80))
                }
                "create" => {
                    let assignee = input
                        .get("assignee")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unassigned)");
                    let task_preview = input
                        .get("task")
                        .map(|v| match v {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .unwrap_or_default();
                    format!("create → {assignee}: {}", truncate(&task_preview, 80))
                }
                other => other.into(),
            }
        }
        _ => truncate(&serde_json::to_string(input).unwrap_or_default(), 80),
    }
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        return s;
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…")
}

fn install_interrupt_signal() -> Arc<AtomicBool> {
    let signal = Arc::new(AtomicBool::new(false));
    let handle = signal.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        handle.store(true, Ordering::Relaxed);
    });
    signal
}

/// Mint a fresh atomic to hand to `TicketSystem::interrupt_signal`, then
/// spawn a one-shot relay so a real Ctrl-C still propagates. The driver
/// reads the original `ctrl_c` to distinguish user cancel from
/// driver-initiated shutdown (which writes to this fresh atomic only).
fn framework_signal_mirroring(ctrl_c: &Arc<AtomicBool>) -> Arc<AtomicBool> {
    let mirror = Arc::new(AtomicBool::new(false));
    let watch = Arc::clone(ctrl_c);
    let target = Arc::clone(&mirror);
    tokio::spawn(async move {
        loop {
            if watch.load(Ordering::Relaxed) {
                target.store(true, Ordering::Relaxed);
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });
    mirror
}

struct CliArgs {
    dir: PathBuf,
    max_steps: u32,
    tech_concurrency: usize,
    reports_dir: Option<PathBuf>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut dir: Option<PathBuf> = None;
    let mut max_steps: u32 = 500;
    let mut tech_concurrency: usize = 4;
    let mut reports_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--max-steps" => {
                i += 1;
                max_steps = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| bad_arg("--max-steps expects a positive number"));
            }
            "--tech-concurrency" => {
                i += 1;
                tech_concurrency = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| bad_arg("--tech-concurrency expects a positive number"));
            }
            "--reports-dir" => {
                i += 1;
                reports_dir = Some(PathBuf::from(
                    args.get(i)
                        .map(String::as_str)
                        .unwrap_or_else(|| bad_arg("--reports-dir expects a path")),
                ));
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            arg if arg.starts_with('-') => bad_arg(&format!("unknown flag: {arg}")),
            _ => dir = Some(PathBuf::from(&args[i])),
        }
        i += 1;
    }

    CliArgs {
        dir: dir.unwrap_or_else(|| {
            print_help();
            std::process::exit(1)
        }),
        max_steps,
        tech_concurrency,
        reports_dir,
    }
}

fn print_help() {
    eprintln!(
        "Project scanner. Maps file extensions to technologies, explores files for domain-specific patterns,\n\
         and writes per-(extension, domain) investigation reports.\n"
    );
    eprintln!("Usage: project-scanner <DIR> [OPTIONS]\n");
    eprintln!("Options:");
    eprintln!("      --max-steps <N>          Per-system step cap (default: 500)");
    eprintln!("      --tech-concurrency <N>   Number of technology-guru agents (default: 4)");
    eprintln!(
        "      --reports-dir <PATH>     Where to write investigation reports (default: <DIR>/.scanner-reports)"
    );
    eprintln!("  -h, --help                   Show this help\n");
    eprintln!("Example:");
    eprintln!("  project-scanner ./src");
}

fn bad_arg(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
