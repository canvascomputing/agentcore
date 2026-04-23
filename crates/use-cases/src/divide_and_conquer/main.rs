//! Divide-and-conquer sum of squares.
//!
//! Partitions `[1, N]` into K subranges, dispatches one worker agent per
//! partition through a `Batch`, aggregates the partial sums, and verifies the
//! result against the closed-form identity `N(N+1)(2N+1)/6`.
//!
//! Usage: divide-and-conquer [OPTIONS] [N]
//!
//! Example:
//!   divide-and-conquer 10000                # default: 16 partitions, concurrency 8
//!   divide-and-conquer -p 32 -c 16 100000

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use agentwerk::{Agent, Batch, Event, EventKind, Output, Tool, ToolResult};
use serde_json::{json, Value};

const WORKER_PROMPT: &str = "\
You are a precise arithmetic worker in a divide-and-conquer pipeline.

Given integers LO and HI, compute the exact integer value of
    S = sum_{k=LO}^{HI} k^2

You have a `python` tool: pass `{\"code\": \"<python source>\"}` and it returns
whatever the snippet prints to stdout. The canonical call is:
    {\"code\": \"print(sum(k*k for k in range(LO, HI + 1)))\"}

When the tool prints an integer N, your FINAL reply MUST be exactly the JSON
object {\"partial_sum\": N} and nothing else — no prose, no code fences, no
units. Use the raw integer with no commas or spaces.";

#[tokio::main]
async fn main() {
    let args = parse_args();
    let provider = agentwerk::provider::from_env().expect("LLM provider required");
    let model = agentwerk::provider::model_from_env().expect("model name required");
    let style = Style::detect();
    let cancel = install_cancel_signal();

    let partitions = partition(args.n, args.partitions);
    let total_chunks = partitions.len();
    let width = digit_width(total_chunks);

    print_intro(&args, total_chunks, &style);

    let ranges: Arc<std::collections::HashMap<String, (u64, u64)>> = Arc::new(
        partitions
            .iter()
            .enumerate()
            .map(|(i, &(lo, hi))| (format!("chunk_{i}"), (lo, hi)))
            .collect(),
    );

    let log_style = style.clone();
    let log_ranges = ranges.clone();
    let log = Arc::new(move |event: Event| {
        log_worker_event(&event, args.verbose, &log_style, &log_ranges)
    });
    let agents = partitions.iter().enumerate().map(|(i, (lo, hi))| {
        build_worker(
            i,
            *lo,
            *hi,
            provider.clone(),
            &model,
            args.max_turns,
            log.clone(),
        )
    });

    let started = Instant::now();
    let (pool, mut stream) = Batch::new()
        .concurrency(args.concurrency)
        .cancel_signal(cancel)
        .agents(agents)
        .spawn();
    pool.drain();

    let mut partial_sums: Vec<Option<i128>> = vec![None; total_chunks];
    let mut failures = 0usize;
    let mut done = 0usize;

    while let Some((i, result)) = stream.next().await {
        done += 1;
        let progress = format!("{done:>width$}/{total_chunks}");
        let (lo, hi) = partitions[i];
        let range = format!("{lo:>9}‥{hi:<9}");
        let name = format!("chunk_{i}");

        let output = match result {
            Ok(output) => output,
            Err(e) => {
                failures += 1;
                eprintln!(
                    "{red}│ {progress}  ✗ {name:<9}  {range}  batch error: {e}{reset}",
                    red = style.red,
                    reset = style.reset,
                );
                continue;
            }
        };

        let Some(partial) = extract_partial_sum(&output.response) else {
            failures += 1;
            eprintln!(
                "{red}│ {progress}  ✗ {name:<9}  {range}  {reason}{reset}",
                reason = failure_reason(&output),
                red = style.red,
                reset = style.reset,
            );
            continue;
        };

        partial_sums[i] = Some(partial);
        eprintln!(
            "{dim}│{reset} {progress}  {green}✓{reset} {name:<9}  {range}  = {partial:>20}",
            dim = style.dim,
            green = style.green,
            reset = style.reset,
        );
    }

    let total: i128 = partial_sums.iter().flatten().sum();
    let expected = closed_form(args.n);
    let elapsed = started.elapsed().as_secs_f64();

    eprintln!(
        "{dim}└ aggregated in {elapsed:.1}s{reset}",
        dim = style.dim,
        reset = style.reset,
    );
    println!();
    println!("aggregated sum : {total}");
    println!("closed form    : {expected}");

    if failures > 0 {
        println!(
            "{red}✗{reset} {failures} partition(s) failed — aggregate incomplete",
            red = style.red,
            reset = style.reset,
        );
        std::process::exit(1);
    }
    if total != expected {
        println!(
            "{red}✗{reset} mismatch: off by {}",
            total - expected,
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
    let c = args.concurrency;

    eprintln!("divide-and-conquer   sum_{{k=1}}^{{{n}}} k^2   (verified via N(N+1)(2N+1)/6)\n",);
    eprintln!("  Split [1, {n}] into {k} contiguous subranges and launch one LLM agent per");
    eprintln!(
        "  subrange. Each worker calls a `python` tool with {{\"code\": \"...\"}} to compute"
    );
    eprintln!("  its partial sum exactly, then returns JSON {{\"partial_sum\": N}}. Up to {c}");
    eprintln!("  agents run concurrently; results stream back in completion order and the");
    eprintln!("  aggregate is checked against the closed-form total.\n");
    eprintln!(
        "{dim}┌ {k} partitions · up to {c} in flight{reset}",
        dim = style.dim,
        reset = style.reset,
    );
}

fn install_cancel_signal() -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let handle = cancel.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        handle.store(true, Ordering::Relaxed);
    });
    cancel
}

fn build_worker(
    i: usize,
    lo: u64,
    hi: u64,
    provider: Arc<dyn agentwerk::Provider>,
    model: &str,
    max_turns: u32,
    event_handler: Arc<dyn Fn(Event) + Send + Sync>,
) -> Agent {
    let schema = json!({
        "type": "object",
        "properties": {
            "partial_sum": {
                "type": "integer",
                "description": "Exact integer value of the partial sum"
            }
        },
        "required": ["partial_sum"]
    });

    Agent::new()
        .name(format!("chunk_{i}"))
        .provider(provider)
        .model_name(model)
        .identity_prompt(WORKER_PROMPT)
        .instruction_prompt(format!("Compute S = sum_{{k={lo}}}^{{{hi}}} k^2."))
        .tool(python_tool())
        .output_schema(schema)
        .max_turns(max_turns)
        .event_handler(event_handler)
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

fn extract_partial_sum(response: &Option<Value>) -> Option<i128> {
    response
        .as_ref()?
        .get("partial_sum")?
        .as_i64()
        .map(i128::from)
}

fn log_worker_event(
    event: &Event,
    verbose: bool,
    style: &Style,
    ranges: &std::collections::HashMap<String, (u64, u64)>,
) {
    let agent = &event.agent_name;
    match &event.kind {
        EventKind::AgentStarted { .. } => {
            let range = format_range(ranges.get(agent));
            eprintln!(
                "{dim}│       ▶ {agent:<9}  {range}  dispatched{reset}",
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
        _ => {}
    }
}

fn format_range(range: Option<&(u64, u64)>) -> String {
    match range {
        Some(&(lo, hi)) => format!("{lo:>9}‥{hi:<9}"),
        None => " ".repeat(19),
    }
}

fn failure_reason(output: &Output) -> String {
    let status = format!("{:?}", output.status);
    let turns = output.statistics.turns;
    let preview = truncate(output.response_raw.trim(), 100);
    if preview.is_empty() {
        format!("[{status}, {turns} turns, no final text]")
    } else {
        format!("[{status}, {turns} turns] final text: {preview}")
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
    max_turns: u32,
    verbose: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut n: Option<u64> = None;
    let mut partitions: usize = 16;
    let mut concurrency: usize = 8;
    let mut max_turns: u32 = 8;
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
            "--max-turns" => {
                i += 1;
                max_turns = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| bad_arg("--max-turns expects a positive number"));
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
        max_turns,
        verbose,
    }
}

fn print_help() {
    eprintln!("Divide-and-conquer sum of squares.\n");
    eprintln!("Usage: divide-and-conquer [OPTIONS] [N]\n");
    eprintln!("Options:");
    eprintln!("  -p, --partitions <K>   Number of worker agents (default: 16)");
    eprintln!("  -c, --concurrency <N>  Max in-flight agents (default: 8)");
    eprintln!("      --max-turns <N>    Per-agent turn cap (default: 8)");
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
