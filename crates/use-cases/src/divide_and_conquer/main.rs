//! Divide-and-conquer sum of squares.
//!
//! Partitions `[1, N]` into K subranges, registers C worker agents on a
//! shared `TicketSystem`, and enqueues K tickets. Workers pick tickets
//! from the shared queue (Path B label routing), compute their partial
//! sum via the `python` tool, and settle the ticket via
//! `manage_tickets_tool` with a JSON result `{"idx", "partial_sum"}`
//! validated against the ticket's schema. The driver aggregates after
//! `run_dry().await` returns.
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

use agentwerk::providers::{from_env, model_from_env};
use agentwerk::tools::ManageTicketsTool;
use agentwerk::{
    Agent, Event, EventKind, Runnable, Schema, Status, TicketSystem, Tool, ToolResult,
};
use serde_json::{json, Value};

const ROLE: &str = include_str!("prompts/worker.role.md");
const BEHAVIOR: &str = include_str!("prompts/worker.behavior.md");

#[tokio::main]
async fn main() {
    let args = parse_args();
    let provider = from_env().expect("LLM provider required");
    let model = model_from_env().expect("model name required");
    let style = Style::detect();
    let cancel = install_interrupt_signal();

    let partitions = partition(args.n, args.partitions);
    let total = partitions.len();
    let width = digit_width(total);

    print_intro(&args, total, &style);

    let role = format!("{}\n\n{}", ROLE.trim(), BEHAVIOR.trim());

    let schema = Schema::parse(json!({
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
    .expect("partial-sum schema is well-formed");

    let started = Instant::now();
    let done = Arc::new(AtomicUsize::new(0));

    let log_style = style.clone();
    let log_done = Arc::clone(&done);
    let event_handler: Arc<dyn Fn(Event) + Send + Sync> = Arc::new(move |e: Event| {
        log_worker_event(&e, args.verbose, &log_style, total, width, &log_done)
    });

    let tickets = TicketSystem::new()
        .interrupt_signal(Arc::clone(&cancel))
        .max_steps(args.max_steps);

    for (idx, (lo, hi)) in partitions.iter().enumerate() {
        let body = format!(
            "Compute the partial sum S = sum_{{k={lo}}}^{{{hi}}} k^2.\n\
             lo={lo}\nhi={hi}\nidx={idx}",
        );
        tickets.task_schema_assigned(body, schema.clone(), "worker");
    }

    for w in 0..args.concurrency.min(total) {
        let agent = Agent::new()
            .name(format!("worker_{w}"))
            .provider(Arc::clone(&provider))
            .model(&model)
            .role(&role)
            .label("worker")
            .tool(python_tool())
            .tool(ManageTicketsTool)
            .event_handler(Arc::clone(&event_handler));
        tickets.add(agent);
    }

    tickets.run_dry().await;

    let mut partials: Vec<Option<i128>> = vec![None; total];
    let mut failures = 0usize;
    for ticket in tickets.tickets() {
        let idx_from_body = parse_idx_from_body(&ticket.task);
        let parsed = ticket
            .result()
            .and_then(|r| serde_json::from_str::<Value>(r).ok())
            .and_then(|v| {
                let idx = v.get("idx").and_then(|x| x.as_u64()).map(|n| n as usize)?;
                let sum = v.get("partial_sum").and_then(|x| x.as_i64()).map(i128::from)?;
                Some((idx, sum))
            });

        match (ticket.status(), parsed) {
            (Status::Done, Some((idx, sum))) if Some(idx) == idx_from_body && idx < total => {
                let (lo, hi) = partitions[idx];
                let range = format!("{lo:>9}‥{hi:<9}");
                eprintln!(
                    "{dim}│{reset} chunk_{idx:<3}  {range}  {green}={reset} {sum:>20}",
                    dim = style.dim,
                    green = style.green,
                    reset = style.reset,
                );
                partials[idx] = Some(sum);
            }
            _ => {
                failures += 1;
                let detail = match (ticket.status(), parsed) {
                    (Status::Done, Some((idx, _))) => {
                        format!("{:?}, idx mismatch: body={idx_from_body:?}, result={idx}", Status::Done)
                    }
                    (status, None) => format!(
                        "{status:?}; result not parseable as {{idx, partial_sum}}"
                    ),
                    (status, Some(_)) => format!("{status:?}"),
                };
                let body_idx = idx_from_body
                    .map(|i| format!("idx={i}"))
                    .unwrap_or_else(|| "idx=?".into());
                eprintln!(
                    "{red}│{reset} {body_idx:<7}  ✗ {detail}",
                    red = style.red,
                    reset = style.reset,
                );
            }
        }
    }

    let total_sum: i128 = partials.iter().flatten().sum();
    let expected = closed_form(args.n);
    let elapsed = started.elapsed().as_secs_f64();
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
            "{red}✗{reset} {failures} partition(s) failed — aggregate incomplete",
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

fn print_intro(args: &CliArgs, total_chunks: usize, style: &Style) {
    let n = args.n;
    let k = total_chunks;
    let c = args.concurrency.min(total_chunks);

    eprintln!("divide-and-conquer   sum_{{k=1}}^{{{n}}} k^2   (verified via N(N+1)(2N+1)/6)\n",);
    eprintln!("  Split [1, {n}] into {k} contiguous subranges and enqueue one ticket per");
    eprintln!("  subrange. {c} worker agent(s) share the queue, each calling a `python` tool");
    eprintln!("  to compute its partial sum exactly. Workers settle their tickets via");
    eprintln!("  `manage_tickets_tool` with `{{\"idx\", \"partial_sum\"}}`; the driver aggregates");
    eprintln!("  after the queue settles and verifies against the closed-form total.\n");
    eprintln!(
        "{dim}┌ {k} partitions · {c} worker(s) sharing the queue{reset}",
        dim = style.dim,
        reset = style.reset,
    );
}

fn install_interrupt_signal() -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let handle = cancel.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        handle.store(true, Ordering::Relaxed);
    });
    cancel
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

fn parse_idx_from_body(task: &Value) -> Option<usize> {
    task.as_str()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("idx=")))
        .and_then(|n| n.trim().parse().ok())
}

fn log_worker_event(
    event: &Event,
    verbose: bool,
    style: &Style,
    total: usize,
    width: usize,
    done: &Arc<AtomicUsize>,
) {
    let agent = &event.agent_name;
    match &event.kind {
        EventKind::TicketClaimed { key } => {
            eprintln!(
                "{dim}│       ▶ {agent:<10} {key} dispatched{reset}",
                dim = style.dim,
                reset = style.reset,
            );
        }
        EventKind::TicketFinished { key } => {
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!(
                "{dim}│ {n:>width$}/{total} ▾ {agent:<10} {key} finished{reset}",
                dim = style.dim,
                reset = style.reset,
            );
        }
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } if verbose => {
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
        EventKind::ToolCallFailed {
            tool_name, message, ..
        } => eprintln!(
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
            Self {
                dim: "",
                green: "",
                red: "",
                reset: "",
            }
        }
    }
}

struct CliArgs {
    n: u64,
    partitions: usize,
    concurrency: usize,
    max_steps: u32,
    verbose: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut n: Option<u64> = None;
    let mut partitions: usize = 16;
    let mut concurrency: usize = 8;
    let mut max_steps: u32 = 8;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--partitions" => {
                i += 1;
                partitions = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| bad_arg("--partitions expects a positive number"));
            }
            "-c" | "--concurrency" => {
                i += 1;
                concurrency = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| bad_arg("--concurrency expects a positive number"));
            }
            "--max-steps" => {
                i += 1;
                max_steps = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| bad_arg("--max-steps expects a positive number"));
            }
            "-v" | "--verbose" => {
                verbose = true;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            arg if arg.starts_with('-') => bad_arg(&format!("unknown flag: {arg}")),
            _ => {
                n = Some(
                    args[i]
                        .parse()
                        .unwrap_or_else(|_| bad_arg("N must be a positive integer")),
                );
            }
        }
        i += 1;
    }

    CliArgs {
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
    eprintln!("      --max-steps <N>    Per-system step cap (default: 8)");
    eprintln!("  -v, --verbose          Stream per-worker tool calls");
    eprintln!("  -h, --help             Show this help\n");
    eprintln!("Examples:");
    eprintln!("  divide-and-conquer 10000");
    eprintln!("  divide-and-conquer -p 32 -c 16 100000");
}

fn bad_arg(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
