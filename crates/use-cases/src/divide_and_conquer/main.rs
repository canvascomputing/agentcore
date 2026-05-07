//! Divide-and-conquer sum of squares.
//!
//! Partitions `[1, N]` into K subranges and creates one ticket per
//! subrange. Workers share the labelled queue, call the `python` tool
//! for an exact integer, and finish via `write_result_tool` with a
//! schema-validated `{"idx", "partial_sum"}`. The driver aggregates
//! after `run_dry` returns and verifies the total against the
//! closed-form `N(N+1)(2N+1)/6`.
//!
//! Usage: divide-and-conquer [OPTIONS] [N]
//!
//! Example:
//!   divide-and-conquer 10000                # default: 16 partitions, 8 workers
//!   divide-and-conquer -p 32 -c 16 100000

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use agentwerk::providers::{model_from_env, provider_from_env};
use agentwerk::tools::ManageTicketsTool;
use agentwerk::{Agent, Event, EventKind, Schema, Ticket, TicketSystem, Tool, ToolResult};
use serde_json::{json, Value};

const ROLE: &str = include_str!("prompts/worker.role.md");

#[tokio::main]
async fn main() {
    let args = CliArgs::parse();
    let provider = provider_from_env().expect("LLM provider required");
    let model = model_from_env().expect("model name required");
    let cancel = install_ctrl_c();
    let style = Style::detect();

    let partitions = partition(args.n, args.partitions);
    let workers = args.concurrency.min(partitions.len());
    print_intro(args.n, partitions.len(), workers, &style);

    let schema = partial_sum_schema();
    let mut tickets = TicketSystem::new().interrupt_signal(Arc::clone(&cancel));
    if let Some(n) = args.max_steps {
        tickets = tickets.max_steps(n);
    }

    for (idx, (lo, hi)) in partitions.iter().enumerate() {
        let body = format!(
            "Compute the partial sum S = sum_{{k={lo}}}^{{{hi}}} k^2.\n\
             lo={lo}\nhi={hi}\nidx={idx}",
        );
        tickets.task_schema_labeled(body, schema.clone(), "worker");
    }

    let event_handler = build_event_handler(args.verbose, style.clone(), partitions.len());
    for w in 0..workers {
        tickets.agent(
            Agent::new()
                .name(format!("worker_{w}"))
                .provider(Arc::clone(&provider))
                .model(&model)
                .role(ROLE.trim())
                .label("worker")
                .tool(python_tool())
                .tool(ManageTicketsTool)
                .event_handler(Arc::clone(&event_handler)),
        );
    }

    let started = Instant::now();
    tickets.run_dry().await;
    let elapsed = started.elapsed().as_secs_f64();

    aggregate_and_report(&tickets, &partitions, args.n, elapsed, &style);
}

fn aggregate_and_report(
    tickets: &TicketSystem,
    partitions: &[(u64, u64)],
    n: u64,
    elapsed: f64,
    style: &Style,
) {
    let total = partitions.len();
    let mut partials: Vec<Option<i128>> = vec![None; total];
    let mut failures = 0usize;

    for ticket in tickets.tickets() {
        match extract_partial(&ticket, total) {
            Ok((idx, sum)) => {
                let (lo, hi) = partitions[idx];
                eprintln!(
                    "{dim}│{reset} chunk_{idx:<3}  {lo:>9}..{hi:<9}  {green}={reset} {sum:>20}",
                    dim = style.dim,
                    green = style.green,
                    reset = style.reset,
                );
                partials[idx] = Some(sum);
            }
            Err(reason) => {
                failures += 1;
                eprintln!(
                    "{red}│{reset} {key:<8} ✗ {reason}",
                    key = ticket.key(),
                    red = style.red,
                    reset = style.reset,
                );
            }
        }
    }

    let total_sum: i128 = partials.iter().flatten().sum();
    let expected = closed_form(n);
    let stats = tickets.stats();

    eprintln!(
        "{dim}└ aggregated in {elapsed:.1}s · {} done, {failures} failed · {} in / {} out tokens{reset}",
        stats.tickets_done(),
        stats.input_tokens(),
        stats.output_tokens(),
        dim = style.dim,
        reset = style.reset,
    );
    println!();
    println!("aggregated sum : {total_sum}");
    println!("closed form    : {expected}");

    if failures > 0 {
        println!(
            "{red}✗{reset} {failures} partition(s) failed: aggregate incomplete",
            red = style.red,
            reset = style.reset,
        );
        std::process::exit(1);
    }
    if total_sum != expected {
        println!(
            "{red}✗{reset} mismatch: off by {}",
            total_sum - expected,
            red = style.red,
            reset = style.reset,
        );
        std::process::exit(1);
    }
    println!(
        "{green}✓ verified{reset}",
        green = style.green,
        reset = style.reset,
    );
}

/// Pull a `(idx, partial_sum)` pair off a finished ticket. The schema
/// already guarantees the field shape; this also cross-checks `idx`
/// against the `idx=` line in the task body so a misrouted result
/// can't quietly slot into the wrong partition.
fn extract_partial(ticket: &Ticket, total: usize) -> Result<(usize, i128), String> {
    if ticket.status() != "done" {
        return Err(ticket.status().to_string());
    }
    let attached = ticket.result().ok_or("no result attached")?;
    let idx = attached
        .result
        .get("idx")
        .and_then(|v| v.as_u64())
        .ok_or("idx missing")? as usize;
    let sum = attached
        .result
        .get("partial_sum")
        .and_then(|v| v.as_i64())
        .ok_or("partial_sum missing")? as i128;

    if idx >= total {
        return Err(format!("idx {idx} out of range"));
    }
    let body_idx = parse_idx_from_body(&ticket.task);
    if body_idx != Some(idx) {
        return Err(format!(
            "idx mismatch: body={body_idx:?}, result={idx}"
        ));
    }
    Ok((idx, sum))
}

fn parse_idx_from_body(task: &Value) -> Option<usize> {
    task.as_str()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("idx=")))
        .and_then(|n| n.trim().parse().ok())
}

fn partial_sum_schema() -> Schema {
    Schema::parse(json!({
        "type": "object",
        "properties": {
            "idx": {
                "type": "integer",
                "description": "Partition index, copied verbatim from the task"
            },
            "partial_sum": {
                "type": "integer",
                "description": "Exact integer value of the partial sum"
            }
        },
        "required": ["idx", "partial_sum"],
        "additionalProperties": false
    }))
    .expect("partial-sum schema is well-formed")
}

fn python_tool() -> Tool {
    Tool::new(
        "python",
        "Run a short Python 3 snippet. The `code` field is passed directly to \
         `python3 -c`. Return value is the snippet's stdout, trimmed. Use this \
         for exact integer arithmetic.",
    )
    .schema(json!({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python 3 source. Must print the result to stdout."
            }
        },
        "required": ["code"]
    }))
    .read_only(true)
    .handler(|input, ctx| {
        Box::pin(async move {
            let code = input
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if code.is_empty() {
                return Ok(ToolResult::error("missing required field `code`"));
            }

            let output_fut = tokio::process::Command::new("python3")
                .arg("-c")
                .arg(code)
                .kill_on_drop(true)
                .output();

            tokio::select! {
                biased;
                _ = ctx.wait_for_cancel() => Ok(ToolResult::error("cancelled")),
                result = output_fut => match result {
                    Err(e) => Ok(ToolResult::error(format!("failed to spawn python3: {e}"))),
                    Ok(out) if out.status.success() => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        Ok(ToolResult::success(stdout.trim().to_string()))
                    }
                    Ok(out) => {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        Ok(ToolResult::error(format!("python error: {stderr}")))
                    }
                }
            }
        })
    })
}

fn build_event_handler(
    verbose: bool,
    style: Style,
    total: usize,
) -> Arc<dyn Fn(Event) + Send + Sync> {
    let done = Arc::new(AtomicUsize::new(0));
    let width = digit_width(total);
    Arc::new(move |event: Event| {
        let agent = &event.agent_name;
        match &event.kind {
            EventKind::TicketStarted { key } => eprintln!(
                "{dim}│       ▶ {agent:<10} {key} dispatched{reset}",
                dim = style.dim,
                reset = style.reset,
            ),
            EventKind::TicketDone { key } | EventKind::TicketFailed { key } => {
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                let outcome = if matches!(event.kind, EventKind::TicketDone { .. }) {
                    "done"
                } else {
                    "failed"
                };
                eprintln!(
                    "{dim}│ {n:>width$}/{total} ▾ {agent:<10} {key} {outcome}{reset}",
                    dim = style.dim,
                    reset = style.reset,
                );
            }
            EventKind::ToolCallStarted { tool_name, input, .. } if verbose => {
                let snippet = input
                    .get("code")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                eprintln!(
                    "{dim}│    {agent} → {tool_name}({}){reset}",
                    truncate(snippet, 70),
                    dim = style.dim,
                    reset = style.reset,
                );
            }
            EventKind::ToolCallFailed { tool_name, message, .. } => eprintln!(
                "{red}│    {agent} ✗ {tool_name}: {}{reset}",
                truncate(message, 120),
                red = style.red,
                reset = style.reset,
            ),
            EventKind::RequestFailed { message, .. } => eprintln!(
                "{red}│    {agent} ✗ request failed: {}{reset}",
                truncate(message, 120),
                red = style.red,
                reset = style.reset,
            ),
            EventKind::PolicyViolated { kind, limit } => eprintln!(
                "{red}│    {agent} ✗ policy {kind:?} (limit {limit}){reset}",
                red = style.red,
                reset = style.reset,
            ),
            _ => {}
        }
    })
}

fn print_intro(n: u64, partitions: usize, workers: usize, style: &Style) {
    eprintln!("divide-and-conquer   sum_{{k=1}}^{{{n}}} k^2   (verified via N(N+1)(2N+1)/6)\n");
    eprintln!("  Split [1, {n}] into {partitions} contiguous subranges and enqueue one ticket per");
    eprintln!("  subrange. {workers} worker agent(s) share the queue, each calling a `python` tool");
    eprintln!("  to compute its partial sum exactly. Workers finish their tickets via");
    eprintln!("  `write_result_tool` with `{{\"idx\", \"partial_sum\"}}`; the driver aggregates");
    eprintln!("  once every ticket is finished and verifies against the closed-form total.\n");
    eprintln!(
        "{dim}┌ {partitions} partitions · {workers} worker(s) sharing the queue{reset}",
        dim = style.dim,
        reset = style.reset,
    );
}

fn install_ctrl_c() -> Arc<AtomicBool> {
    let signal = Arc::new(AtomicBool::new(false));
    let handle = Arc::clone(&signal);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        handle.store(true, Ordering::Relaxed);
    });
    signal
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        return s;
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…")
}

fn digit_width(n: usize) -> usize {
    let mut n = n.max(1);
    let mut w = 0;
    while n > 0 {
        n /= 10;
        w += 1;
    }
    w
}

fn partition(n: u64, k: usize) -> Vec<(u64, u64)> {
    let k = k.max(1).min(n.max(1) as usize);
    let base = n / k as u64;
    let extra = n % k as u64;
    let mut out = Vec::with_capacity(k);
    let mut lo = 1u64;
    for i in 0..k {
        let size = base + if (i as u64) < extra { 1 } else { 0 };
        let hi = lo + size - 1;
        out.push((lo, hi));
        lo = hi + 1;
    }
    out
}

fn closed_form(n: u64) -> i128 {
    let n = i128::from(n);
    n * (n + 1) * (2 * n + 1) / 6
}

#[derive(Clone)]
struct Style {
    dim: &'static str,
    green: &'static str,
    red: &'static str,
    reset: &'static str,
}

impl Style {
    fn detect() -> Self {
        if std::io::stderr().is_terminal() {
            Self {
                dim: "\x1b[2m",
                green: "\x1b[32m",
                red: "\x1b[31m",
                reset: "\x1b[0m",
            }
        } else {
            Self { dim: "", green: "", red: "", reset: "" }
        }
    }
}

struct CliArgs {
    n: u64,
    partitions: usize,
    concurrency: usize,
    max_steps: Option<u32>,
    verbose: bool,
}

impl CliArgs {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut n: Option<u64> = None;
        let mut partitions: usize = 16;
        let mut concurrency: usize = 8;
        let mut max_steps: Option<u32> = None;
        let mut verbose = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "-p" | "--partitions" => {
                    i += 1;
                    partitions = parse_value(args.get(i), "--partitions");
                }
                "-c" | "--concurrency" => {
                    i += 1;
                    concurrency = parse_value(args.get(i), "--concurrency");
                }
                "--max-steps" => {
                    i += 1;
                    max_steps = Some(parse_value(args.get(i), "--max-steps"));
                }
                "-v" | "--verbose" => verbose = true,
                "-h" | "--help" => {
                    Self::print_help();
                    std::process::exit(0);
                }
                arg if arg.starts_with('-') => bad_arg(&format!("unknown flag: {arg}")),
                _ => {
                    n = Some(args[i].parse().unwrap_or_else(|_| {
                        bad_arg("N must be a positive integer")
                    }));
                }
            }
            i += 1;
        }

        Self {
            n: n.unwrap_or(10_000),
            partitions,
            concurrency,
            max_steps,
            verbose,
        }
    }

    fn print_help() {
        eprintln!("Divide-and-conquer sum of squares.\n");
        eprintln!("Usage: divide-and-conquer [OPTIONS] [N]\n");
        eprintln!("Options:");
        eprintln!("  -p, --partitions <K>   Number of ticket partitions (default: 16)");
        eprintln!("  -c, --concurrency <N>  Number of worker agents sharing the queue (default: 8)");
        eprintln!("      --max-steps <N>    Per-system step cap (default: unlimited)");
        eprintln!("  -v, --verbose          Stream per-worker tool calls");
        eprintln!("  -h, --help             Show this help\n");
        eprintln!("Examples:");
        eprintln!("  divide-and-conquer 10000");
        eprintln!("  divide-and-conquer -p 32 -c 16 100000");
    }
}

fn parse_value<T: std::str::FromStr>(value: Option<&String>, flag: &str) -> T {
    value
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| bad_arg(&format!("{flag} expects a positive number")))
}

fn bad_arg(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
