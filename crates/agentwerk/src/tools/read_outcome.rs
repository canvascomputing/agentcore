//! Read the result of a background sub-agent on demand. Pairs with `agent_tool` (`background: true`) to give the parent model an explicit handle on the spawned run.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::work::{SpawnState, Work};
use crate::error::Result;
use crate::tools::error::ToolError;
use crate::tools::tool::{ToolContext, ToolLike, ToolResult};
use crate::tools::tool_file::ToolFile;

/// Look up a background sub-agent by id and return its result, optionally
/// blocking until it finishes.
pub struct ReadOutcomeTool;

#[derive(Deserialize)]
struct ReadArgs {
    task_id: String,
    #[serde(default)]
    timeout_secs: Option<f64>,
}

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("read_outcome.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for ReadOutcomeTool {
    fn name(&self) -> &str {
        &tool_file().name
    }

    fn description(&self) -> &str {
        description()
    }

    fn is_read_only(&self) -> bool {
        tool_file().read_only
    }

    fn input_schema(&self) -> Value {
        tool_file().input_schema.clone()
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let args: ReadArgs = match serde_json::from_value(input) {
                Ok(a) => a,
                Err(e) => return Ok(ToolResult::error(format!("Invalid input: {e}"))),
            };

            let work = match ctx.runtime.as_ref().and_then(|r| r.incoming_work.clone()) {
                Some(w) => w,
                None => {
                    return Err(ToolError::ExecutionFailed {
                        tool_name: tool_file().name.clone(),
                        message: "Work inbox not available on LoopRuntime".into(),
                    }
                    .into());
                }
            };

            let timeout = args.timeout_secs.unwrap_or(0.0);
            if timeout <= 0.0 {
                return Ok(snapshot(&args.task_id, work.spawn_state(&args.task_id)));
            }

            Ok(wait(&work, &args.task_id, Duration::from_secs_f64(timeout), ctx).await)
        })
    }
}

/// Format a single-shot snapshot of the spawn state.
fn snapshot(task_id: &str, state: Option<SpawnState>) -> ToolResult {
    match state {
        None => ToolResult::success(format!("Unknown task id: {task_id}.")),
        Some(None) => ToolResult::success(format!("Task {task_id} is still running.")),
        Some(Some((_, text, structured))) => render(text, structured),
    }
}

/// Block until the spawn settles, the deadline passes, or the run is cancelled.
async fn wait(work: &Work, task_id: &str, timeout: Duration, ctx: &ToolContext) -> ToolResult {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let notified = work.notified();

        match work.spawn_state(task_id) {
            None => return ToolResult::success(format!("Unknown task id: {task_id}.")),
            Some(Some((_, text, structured))) => return render(text, structured),
            Some(None) => {}
        }

        let Some(remaining) = deadline.checked_duration_since(tokio::time::Instant::now()) else {
            return ToolResult::success(format!(
                "Timed out waiting for task {task_id} after {}s.",
                timeout.as_secs_f64()
            ));
        };

        tokio::select! {
            biased;
            _ = ctx.wait_for_cancel() => {
                return ToolResult::success(format!("Cancelled while waiting for task {task_id}."));
            }
            _ = notified => {}
            _ = tokio::time::sleep(remaining) => {}
        }
    }
}

/// Prefer the validated structured value when the child carried a contract;
/// fall back to the raw text otherwise.
fn render(text: String, structured: Option<Value>) -> ToolResult {
    match structured {
        Some(v) => ToolResult::success(serde_json::to_string(&v).unwrap_or(text)),
        None => ToolResult::success(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentSpec};
    use crate::output::Outcome;
    use crate::testutil::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn ctx_with(work: Arc<Work>) -> (ToolContext, Arc<AgentSpec>) {
        let caller = Agent::new()
            .name("parent")
            .model("mock")
            .role("")
            .provider(Arc::new(MockProvider::text("unused")))
            .incoming_work(work);
        let (spec, runtime) = caller.compile(None);
        let ctx = ToolContext::new(PathBuf::from("."))
            .runtime(Arc::new(runtime))
            .caller_spec(spec.clone());
        (ctx, spec)
    }

    fn input(task_id: &str, timeout: Option<f64>) -> Value {
        let mut v = serde_json::json!({"task_id": task_id});
        if let Some(t) = timeout {
            v["timeout_secs"] = serde_json::json!(t);
        }
        v
    }

    #[tokio::test]
    async fn unknown_id_returns_message() {
        let work = Arc::new(Work::new());
        let (ctx, _) = ctx_with(work);
        let result = ReadOutcomeTool
            .call(input("ghost", None), &ctx)
            .await
            .unwrap();
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        assert!(s.contains("Unknown task id: ghost"));
    }

    #[tokio::test]
    async fn running_task_reports_still_running() {
        let work = Arc::new(Work::new());
        work.spawned("t1");
        let (ctx, _) = ctx_with(work);
        let result = ReadOutcomeTool.call(input("t1", None), &ctx).await.unwrap();
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        assert!(s.contains("still running"));
    }

    #[tokio::test]
    async fn completed_returns_text() {
        let work = Arc::new(Work::new());
        work.spawned("t1");
        work.settled("t1", Outcome::Completed, "the answer".into(), None);
        let (ctx, _) = ctx_with(work);
        let result = ReadOutcomeTool.call(input("t1", None), &ctx).await.unwrap();
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        assert_eq!(s, "the answer");
    }

    #[tokio::test]
    async fn completed_with_contract_returns_validated_json() {
        let work = Arc::new(Work::new());
        work.spawned("t1");
        let structured = serde_json::json!({"answer": 42});
        work.settled(
            "t1",
            Outcome::Completed,
            "{\"answer\":42}".into(),
            Some(structured),
        );
        let (ctx, _) = ctx_with(work);
        let result = ReadOutcomeTool.call(input("t1", None), &ctx).await.unwrap();
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        let parsed: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed["answer"], 42);
    }

    #[tokio::test]
    async fn failed_returns_error_text_in_band() {
        let work = Arc::new(Work::new());
        work.spawned("t1");
        work.settled("t1", Outcome::Failed, "Failed: boom".into(), None);
        let (ctx, _) = ctx_with(work);
        let result = ReadOutcomeTool.call(input("t1", None), &ctx).await.unwrap();
        // Failed runs surface the message as Success so the model can read it
        // as ordinary content.
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        assert!(s.contains("Failed: boom"));
    }

    #[tokio::test]
    async fn blocking_returns_when_settle_arrives() {
        let work = Arc::new(Work::new());
        work.spawned("t1");
        let (ctx, _) = ctx_with(work.clone());

        let waker = work.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            waker.settled("t1", Outcome::Completed, "done".into(), None);
        });

        let started = std::time::Instant::now();
        let result = ReadOutcomeTool
            .call(input("t1", Some(5.0)), &ctx)
            .await
            .unwrap();
        let elapsed = started.elapsed();
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        assert_eq!(s, "done");
        assert!(
            elapsed < Duration::from_secs(2),
            "should wake well under timeout, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn blocking_times_out() {
        let work = Arc::new(Work::new());
        work.spawned("t1");
        let (ctx, _) = ctx_with(work);
        let result = ReadOutcomeTool
            .call(input("t1", Some(0.05)), &ctx)
            .await
            .unwrap();
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        assert!(s.contains("Timed out"), "got: {s}");
    }

    #[tokio::test]
    async fn blocking_cancels() {
        let work = Arc::new(Work::new());
        work.spawned("t1");
        let (ctx, _) = ctx_with(work);
        let signal = ctx.interrupt_signal.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            signal.store(true, std::sync::atomic::Ordering::Relaxed);
        });

        let result = ReadOutcomeTool
            .call(input("t1", Some(5.0)), &ctx)
            .await
            .unwrap();
        let (ToolResult::Success(s) | ToolResult::Error(s)) = result;
        assert!(s.contains("Cancelled"), "got: {s}");
    }
}
