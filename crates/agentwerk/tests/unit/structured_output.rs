//! How structured output flows from declaration to validated `Output`.
//!
//! `output_schema` turns the agent loop into a tiny state machine on top of
//! the regular run loop:
//!
//! ```text
//!   S0 (instruction) ──turn 1──▶ S1 (model text) ──validate─┐
//!                                                           │
//!                                 ┌─── valid ───────────────┘
//!                                 ▼
//!                       Output { response: Some(value) }
//!
//!                                 ┌─── invalid ─────────────┐
//!                                 ▼                         │
//!                  S2 (text + validator correction) ──turn 2──▶ S1' ─▶ ...
//! ```
//!
//! The validator only fires at the natural end of turn — no tool calls, not
//! truncated, no pending peer messages. Tools, truncation, peer messages,
//! and idle wait all short-circuit *before* validation, so a schema agent
//! can interleave tool work freely; only its terminal text reply is parsed.
//!
//! Run with `cargo test --test structured_output -- --nocapture` to inspect
//! the per-turn snapshots.
//!
//! Mirrors the structure of `context_window_events.rs`: a file-level
//! state-machine thesis, one snapshot test that walks every node, and
//! focused tests grouped by concern.
//!
//! - **Centerpiece** — the snapshot state machine.
//! - **A. Happy paths** — single-turn valid output, lenient code-fence handling.
//! - **B. Retry / failure** — corrective messages, schema violations, exhaustion.
//! - **C. Tools and end-conditions** — tools run first, truncation defers, guards skip.
//! - **D. Sub-agent boundary** — the propagation bug: registered, ad-hoc, and background.

use agentwerk::testutil::{
    text_response, tool_response, truncated_response, MockProvider, MockTool, TestHarness,
};
use agentwerk::{Agent, CompletionRequest, ContentBlock, Error, Message, SpawnAgentTool};

fn answer_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "integer" } },
        "required": ["answer"]
    })
}

const VALID_JSON: &str = r#"{"answer":42}"#;

fn report_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "category": { "type": "string" },
            "priority": { "type": "string" },
            "summary": { "type": "string" },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file":     { "type": "string" },
                        "line":     { "type": "integer" },
                        "severity": { "type": "string" }
                    },
                    "required": ["file", "line", "severity"]
                }
            }
        },
        "required": ["category", "priority", "summary", "findings"]
    })
}

const VALID_REPORT_JSON: &str = r#"{"category":"security","priority":"high","summary":"Two SQL-injection paths in the auth flow.","findings":[{"file":"src/auth.rs","line":47,"severity":"critical"},{"file":"src/db.rs","line":112,"severity":"high"}]}"#;

fn report_agent() -> Agent {
    Agent::new()
        .name("reviewer")
        .model_name("mock")
        .identity_prompt("You are a code reviewer. Reply with a structured report.")
        .behavior_prompt("")
        .output_schema(report_schema())
}

const INVALID_REPORT_JSON: &str = r#"{"category":"security","priority":"high","findings":[]}"#;

const S0_INITIAL: &str = "\
=== system ===
You are a code reviewer. Reply with a structured report.

IMPORTANT: You MUST return your final response as a single JSON value that \
conforms to the declared output schema. After using any tools needed to complete \
the task, your last message MUST be the JSON value, exactly once. Do not wrap it \
in markdown code fences. Do not include any text before or after the JSON.

=== messages[0] user ===
<environment>
Working directory: <WORKING_DIRECTORY>
Platform: <PLATFORM>
OS version: <OS_VERSION>
Date: <DATE>
</environment>

=== messages[1] user ===
Review the auth module.
";

const S1_AFTER_INVALID_REPLY: &str = "\
=== system ===
You are a code reviewer. Reply with a structured report.

IMPORTANT: You MUST return your final response as a single JSON value that \
conforms to the declared output schema. After using any tools needed to complete \
the task, your last message MUST be the JSON value, exactly once. Do not wrap it \
in markdown code fences. Do not include any text before or after the JSON.

=== messages[0] user ===
<environment>
Working directory: <WORKING_DIRECTORY>
Platform: <PLATFORM>
OS version: <OS_VERSION>
Date: <DATE>
</environment>

=== messages[1] user ===
Review the auth module.

=== messages[2] assistant ===
{\"category\":\"security\",\"priority\":\"high\",\"findings\":[]}

=== messages[3] user ===
Your last reply did not match the required output schema. You MUST reply with a \
single JSON value conforming to the schema, with no surrounding text and no code \
fences.

Validator said: Schema validation error at summary: missing required field
";

const S2_AFTER_VALID_REPLY: &str = "\
=== system ===
You are a code reviewer. Reply with a structured report.

IMPORTANT: You MUST return your final response as a single JSON value that \
conforms to the declared output schema. After using any tools needed to complete \
the task, your last message MUST be the JSON value, exactly once. Do not wrap it \
in markdown code fences. Do not include any text before or after the JSON.

=== messages[0] user ===
<environment>
Working directory: <WORKING_DIRECTORY>
Platform: <PLATFORM>
OS version: <OS_VERSION>
Date: <DATE>
</environment>

=== messages[1] user ===
Review the auth module.

=== messages[2] assistant ===
{\"category\":\"security\",\"priority\":\"high\",\"findings\":[]}

=== messages[3] user ===
Your last reply did not match the required output schema. You MUST reply with a \
single JSON value conforming to the schema, with no surrounding text and no code \
fences.

Validator said: Schema validation error at summary: missing required field

=== messages[4] assistant ===
{\"category\":\"security\",\"priority\":\"high\",\"summary\":\"Two SQL-injection paths in the auth flow.\",\"findings\":[{\"file\":\"src/auth.rs\",\"line\":47,\"severity\":\"critical\"},{\"file\":\"src/db.rs\",\"line\":112,\"severity\":\"high\"}]}
";

#[tokio::test]
async fn state_machine_advances_invalid_then_valid() {
    // Two turns:
    //   turn 1 → INVALID_REPORT_JSON  → schema mismatch → push corrective msg
    //   turn 2 → VALID_REPORT_JSON    → validates       → terminate
    let provider = MockProvider::new(vec![
        text_response(INVALID_REPORT_JSON),
        text_response(VALID_REPORT_JSON),
    ]);

    let harness = TestHarness::new(provider);
    let output = harness
        .run_agent(&report_agent(), "Review the auth module.")
        .await
        .unwrap();

    assert_eq!(
        harness.provider().request_count(),
        2,
        "two turns: invalid + valid"
    );
    assert_eq!(output.response_raw, VALID_REPORT_JSON);
    assert_eq!(
        output.response,
        Some(serde_json::from_str::<serde_json::Value>(VALID_REPORT_JSON).unwrap()),
    );

    let reqs = harness.provider().requests.lock().unwrap();
    let state = |i: usize| canonicalize(&render(&reqs[i]));
    assert_eq!(state(0), S0_INITIAL);
    assert_eq!(state(1), S1_AFTER_INVALID_REPLY);

    // Loop terminated on turn 2 without a third request. Reconstruct what req[2] would have been
    // (req[1]'s context plus the final assistant reply) to snapshot the end-to-end state machine.
    let mut terminal_messages = reqs[1].messages.clone();
    terminal_messages.push(Message::Assistant {
        content: vec![ContentBlock::Text {
            text: output.response_raw.clone(),
        }],
    });
    let terminal = render_conversation(&reqs[1].system_prompt, &terminal_messages);
    assert_eq!(canonicalize(&terminal), S2_AFTER_VALID_REPLY);
}

#[tokio::test]
async fn schema_agent_does_not_inject_synthetic_tool() {
    // The rewrite removed `StructuredOutputTool` from the registry and the
    // forced `tool_choice = Specific("…")` that came with it; validation
    // moved into `try_finish` against the model's natural text reply. Every
    // test in this file goes green even if a future commit silently puts a
    // synthetic tool back — the model would just call it and the validated
    // value would still flow. Pin the contract here so a regression to the
    // old "fake tool" approach fails immediately and loudly.
    let provider = MockProvider::new(vec![text_response(VALID_JSON)]);
    let harness = TestHarness::new(provider);
    harness.run_agent(&schema_agent(), "go").await.unwrap();

    let req = harness.provider().last_request().unwrap();
    let advertised: Vec<&String> = req.tools.iter().map(|t| &t.name).collect();
    assert!(
        req.tools.is_empty(),
        "schema agent must advertise no tool to the LLM, got: {advertised:?}"
    );
    assert!(
        req.tool_choice.is_none(),
        "schema agent must not force a tool_choice (the model speaks JSON freely)"
    );
    assert!(
        req.system_prompt
            .contains("You MUST return your final response as a single JSON value"),
        "schema contract must be carried by the system prompt — got:\n{}",
        req.system_prompt,
    );
}

#[tokio::test]
async fn schema_agent_with_user_tools_still_advertises_no_synthetic_tool() {
    // Same invariant holds when the user added their own tools — the schema
    // contract lives in the prompt, not the registry, regardless of what
    // else is in the registry. We don't actually use the tool here; we just
    // need to prove the user tool is the *only* one advertised.
    let provider = MockProvider::new(vec![text_response(VALID_JSON)]);
    let agent = schema_agent().tool(MockTool::new("lookup", true, "ok"));

    let harness = TestHarness::new(provider);
    harness.run_agent(&agent, "go").await.unwrap();

    let req = harness.provider().last_request().unwrap();
    let names: Vec<&String> = req.tools.iter().map(|t| &t.name).collect();
    assert_eq!(
        names,
        vec![&"lookup".to_string()],
        "only the user-registered tool should be advertised, got: {names:?}"
    );
    assert!(req.tool_choice.is_none());
}

#[tokio::test]
async fn valid_json_terminates_in_one_turn() {
    // The whole point of going text-based: a valid reply terminates the loop
    // immediately. No follow-up round-trip.
    let provider = MockProvider::new(vec![text_response(VALID_JSON)]);
    let harness = TestHarness::new(provider);
    let output = harness
        .run_agent(&schema_agent(), "What is the answer?")
        .await
        .unwrap();

    assert_eq!(harness.provider().request_count(), 1);
    assert_eq!(output.response, Some(serde_json::json!({"answer": 42})));
    assert_eq!(output.response_raw, VALID_JSON);
}

#[tokio::test]
async fn code_fenced_json_accepted_leniently() {
    // The most common model mistake: wrapping JSON in a ```json fence.
    // strip_code_fences peels the wrapper so we don't burn a retry.
    let fenced = "```json\n{\"answer\":42}\n```";
    let provider = MockProvider::new(vec![text_response(fenced)]);
    let harness = TestHarness::new(provider);
    let output = harness
        .run_agent(&schema_agent(), "What is the answer?")
        .await
        .unwrap();

    assert_eq!(harness.provider().request_count(), 1);
    assert_eq!(output.response, Some(serde_json::json!({"answer": 42})));
}

#[tokio::test]
async fn valid_complex_report_terminates_in_one_turn() {
    // Realistic case: a structured report with nested objects in an array and
    // mixed primitive types. Proves the loop handles non-trivial JSON the same
    // way it handles the toy `{answer:42}` — a single round-trip, with the
    // parsed value reachable via `output.response` and the raw text preserved
    // verbatim in `output.response_raw`.
    let provider = MockProvider::new(vec![text_response(VALID_REPORT_JSON)]);
    let harness = TestHarness::new(provider);
    let output = harness
        .run_agent(&report_agent(), "Review the auth module.")
        .await
        .unwrap();

    assert_eq!(harness.provider().request_count(), 1);
    assert_eq!(output.response_raw, VALID_REPORT_JSON);

    // Walk into the parsed value to prove every layer survived intact —
    // not just that the top-level keys are present.
    let value = output.response.expect("response must be Some");
    assert_eq!(value["category"], "security");
    assert_eq!(value["priority"], "high");

    let findings = value["findings"].as_array().expect("findings is an array");
    assert_eq!(findings.len(), 2);
    assert_eq!(findings[0]["file"], "src/auth.rs");
    assert_eq!(findings[0]["line"], 47);
    assert_eq!(findings[0]["severity"], "critical");
    assert_eq!(findings[1]["file"], "src/db.rs");
    assert_eq!(findings[1]["line"], 112);
}

#[tokio::test]
async fn non_json_reply_triggers_retry_with_validator_detail() {
    // Loop must surface the parse error to the model so it can self-correct.
    // The next user message bundles both the static retry instruction and a
    // dynamic "Validator said: …" line naming the failure.
    let provider = MockProvider::new(vec![
        text_response("the answer is 42"),
        text_response(VALID_JSON),
    ]);
    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&schema_agent(), "go").await.unwrap();

    let req2 = &harness.provider().requests.lock().unwrap()[1];
    let last_user = last_user_text(req2).expect("expected a user message in turn 2 input");
    assert!(
        last_user.contains("did not match the required output schema"),
        "expected the static retry copy in: {last_user}"
    );
    assert!(
        last_user.contains("Validator said:"),
        "expected validator detail in: {last_user}"
    );
    assert_eq!(output.response, Some(serde_json::json!({"answer": 42})));
}

#[tokio::test]
async fn schema_violation_triggers_retry() {
    // Same retry mechanism, but the failure surfaces from validate_value
    // (well-formed JSON, wrong shape) rather than serde's parser.
    let provider = MockProvider::new(vec![
        text_response(r#"{"answer":"not a number"}"#),
        text_response(VALID_JSON),
    ]);
    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&schema_agent(), "go").await.unwrap();

    let req2 = &harness.provider().requests.lock().unwrap()[1];
    let last_user = last_user_text(req2).unwrap();
    assert!(
        last_user.contains("Validator said:") && last_user.contains("answer"),
        "expected validator detail naming the field: {last_user}"
    );
    assert_eq!(output.response, Some(serde_json::json!({"answer": 42})));
}

#[tokio::test]
async fn validator_path_reaches_into_array_items() {
    // The validator walks recursively into nested objects and array items,
    // accumulating a dotted path so the retry message points the model at the
    // exact field that broke. Here `findings[0].line` is a string instead of
    // an integer — the corrective message must surface that path so the model
    // can fix the right field on its next attempt.
    let bad = r#"{"category":"perf","priority":"low","summary":"slow query","findings":[{"file":"src/db.rs","line":"forty-two","severity":"low"}]}"#;
    let provider = MockProvider::new(vec![text_response(bad), text_response(VALID_REPORT_JSON)]);
    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&report_agent(), "review").await.unwrap();

    let req2 = &harness.provider().requests.lock().unwrap()[1];
    let last_user = last_user_text(req2).unwrap();
    assert!(
        last_user.contains("findings.[0].line"),
        "validator detail must name the deep path, got: {last_user}"
    );
    assert!(
        last_user.contains("expected integer"),
        "validator detail must name the type mismatch, got: {last_user}"
    );
    assert!(output.response.is_some());
}

#[tokio::test]
async fn missing_required_field_at_depth_triggers_retry() {
    // Required-field check applies inside nested array items too. Drop
    // `severity` from the first finding — the validator must report the
    // missing field with its full path, not just say "something is off".
    let bad = r#"{"category":"bug","priority":"med","summary":"flaky test","findings":[{"file":"src/x.rs","line":10}]}"#;
    let provider = MockProvider::new(vec![text_response(bad), text_response(VALID_REPORT_JSON)]);
    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&report_agent(), "review").await.unwrap();

    let req2 = &harness.provider().requests.lock().unwrap()[1];
    let last_user = last_user_text(req2).unwrap();
    assert!(
        last_user.contains("findings.[0].severity"),
        "expected nested path to the missing field, got: {last_user}"
    );
    assert!(
        last_user.contains("missing required field"),
        "expected the required-field message, got: {last_user}"
    );
    assert!(output.response.is_some());
}

#[tokio::test]
async fn retry_exhaustion_returns_schema_retry_exhausted() {
    // Cap retries at 2 → after 3 failures, the loop bails with the original
    // limit attached to the error.
    let provider = MockProvider::new(vec![
        text_response("nope"),
        text_response("still nope"),
        text_response("never going to be json"),
    ]);
    let agent = schema_agent().max_schema_retries(2);
    let harness = TestHarness::new(provider);
    let err = harness.run_agent(&agent, "go").await.unwrap_err();

    assert!(matches!(err, Error::SchemaRetryExhausted { retries: 2 }));
}

#[tokio::test]
async fn tools_run_first_then_validation_at_natural_end() {
    // Schema validation must NOT run while the loop is mid-tool-use. The
    // model's terminal text reply is the only thing parsed.
    let provider = MockProvider::new(vec![
        tool_response("lookup", "c1", serde_json::json!({"q": "answer"})),
        text_response(VALID_JSON),
    ]);
    let agent = schema_agent().tool(MockTool::new("lookup", true, "answer=42"));

    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&agent, "go").await.unwrap();

    // Tool ran on turn 1; validation on turn 2; total 2 requests.
    assert_eq!(harness.provider().request_count(), 2);
    assert_eq!(output.response, Some(serde_json::json!({"answer": 42})));
}

#[tokio::test]
async fn validation_deferred_through_truncation() {
    // Truncated turns push MAX_TOKENS_CONTINUATION instead of try_finish'ing,
    // so the schema retry counter doesn't move.
    let provider = MockProvider::new(vec![
        truncated_response("partial JSON…"),
        text_response(VALID_JSON),
    ]);
    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&schema_agent(), "go").await.unwrap();

    assert_eq!(harness.provider().request_count(), 2);
    assert_eq!(output.response, Some(serde_json::json!({"answer": 42})));

    // Turn 2's input must contain the continuation prompt, not the schema
    // retry copy — proving the deferral happened.
    let req2 = &harness.provider().requests.lock().unwrap()[1];
    let last_user = last_user_text(req2).unwrap();
    assert!(
        last_user.contains("cut off"),
        "expected MAX_TOKENS_CONTINUATION, got: {last_user}"
    );
    assert!(
        !last_user.contains("required output schema"),
        "schema retry should not fire on truncation, got: {last_user}"
    );
}

#[tokio::test]
async fn cancel_before_any_reply_skips_validation() {
    // Pre-cancelled run: check_guards fires on the first iteration; finish_early
    // returns Status::Cancelled with response=None and never calls validate_value.
    let provider = MockProvider::new(vec![text_response("would-be JSON if we got here")]);
    let harness = TestHarness::new(provider);
    harness.cancel();
    let output = harness.run_agent(&schema_agent(), "go").await.unwrap();

    use agentwerk::Status;
    assert_eq!(output.status, Status::Cancelled);
    assert_eq!(output.response, None);
    // No request was sent at all (guard fires before the first turn body).
    assert_eq!(harness.provider().request_count(), 0);
}

#[tokio::test]
async fn turn_limit_skips_validation() {
    // Same shape as cancel: guard short-circuits on turn 2 before try_finish
    // can validate (max_turns(1) means turn=1 is the last iteration; the guard
    // catches turn=2 at the top of the loop).
    let provider = MockProvider::new(vec![
        text_response("not json"),
        text_response("still not json"),
    ]);
    let agent = schema_agent().max_turns(1);
    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&agent, "go").await.unwrap();

    use agentwerk::Status;
    assert_eq!(output.status, Status::TurnLimitReached { limit: 1 });
    assert_eq!(output.response, None);
}

#[tokio::test]
async fn sub_agent_with_schema_returns_json_in_tool_result() {
    // Registered sub-agent declares a schema. Its validated JSON must reach
    // the parent's tool_result content verbatim — including the nested
    // findings array, so we know the boundary doesn't accidentally re-pretty
    // or strip whitespace.
    let child = Agent::new()
        .name("reviewer")
        .model_name("mock")
        .identity_prompt("You are a code reviewer. Reply with a structured report.")
        .behavior_prompt("")
        .output_schema(report_schema());

    let parent = Agent::new()
        .name("orchestrator")
        .model_name("mock")
        .identity_prompt("Coordinate.")
        .behavior_prompt("")
        .sub_agents([child]);

    let provider = MockProvider::new(vec![
        // parent turn 1: spawn the registered reviewer
        tool_response(
            "spawn_agent",
            "sa1",
            serde_json::json!({
                "description": "ask reviewer",
                "instruction": "Review the auth module.",
                "agent": "reviewer"
            }),
        ),
        // child turn 1: structured report → validates → terminates immediately
        text_response(VALID_REPORT_JSON),
        // parent turn 2: wraps up
        text_response("done"),
    ]);

    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&parent, "go").await.unwrap();
    assert_eq!(output.response_raw, "done");

    // The parent's input on turn 2 must contain a tool_result whose content
    // is the child's JSON answer byte-for-byte — proving the boundary
    // forwarded the full nested structure without re-formatting.
    let req2 = &harness.provider().requests.lock().unwrap()[2];
    let tool_result = last_tool_result_content(req2).expect("expected a tool_result");
    assert_eq!(tool_result, VALID_REPORT_JSON);

    // And the parsed view round-trips: parent reads what child wrote.
    let parsed: serde_json::Value = serde_json::from_str(&tool_result).unwrap();
    assert_eq!(parsed["findings"][1]["file"], "src/db.rs");
}

#[tokio::test]
async fn ad_hoc_spawned_agent_declares_schema_via_overrides() {
    // No registered sub-agent. The parent passes output_schema in the spawn
    // tool's JSON args; apply_overrides wires it onto the ad-hoc child.
    let parent = Agent::new()
        .name("orchestrator")
        .model_name("mock")
        .identity_prompt("")
        .behavior_prompt("")
        .tool(SpawnAgentTool);

    let provider = MockProvider::new(vec![
        // parent turn 1: ad-hoc spawn with output_schema in args
        tool_response(
            "spawn_agent",
            "sa1",
            serde_json::json!({
                "description": "ad-hoc classifier",
                "instruction": "Reply with the answer.",
                "identity": "You answer with JSON.",
                "model": "mock",
                "output_schema": answer_schema(),
            }),
        ),
        // child turn 1: invalid → triggers schema retry inside the child
        text_response("just kidding"),
        // child turn 2: valid → terminates the child
        text_response(VALID_JSON),
        // parent turn 2: wraps up
        text_response("done"),
    ]);

    let harness = TestHarness::new(provider);
    let output = harness.run_agent(&parent, "go").await.unwrap();
    assert_eq!(output.response_raw, "done");

    // Total 4 requests: parent(2) + child(2). Confirms the child enforced
    // its schema and retried before terminating.
    assert_eq!(harness.provider().request_count(), 4);

    let req4 = &harness.provider().requests.lock().unwrap()[3];
    let tool_result = last_tool_result_content(req4).expect("expected a tool_result");
    assert_eq!(tool_result, VALID_JSON);
}

// (Background sub-agent + schema is covered by an inline test in
// `spawn_agent.rs` — that test has crate-private access to the command queue
// and can assert directly on the notification content. From here the queue
// is unreachable, so an integration test would only duplicate coverage
// while introducing tokio-spawn races between parent and child mock pops.)

fn schema_agent() -> Agent {
    Agent::new()
        .name("classifier")
        .model_name("mock")
        .identity_prompt("You answer with JSON.")
        .behavior_prompt("")
        .output_schema(answer_schema())
}

/// Render a CompletionRequest as plain text. Mirrors `context_window_events.rs`'s
/// helper of the same name — same shape so snapshots are visually comparable.
fn render(req: &CompletionRequest) -> String {
    render_conversation(&req.system_prompt, &req.messages)
}

/// Render a `(system_prompt, messages)` pair. Used by `render` for actual
/// requests and by the centerpiece test to synthesize the post-terminal
/// state — the loop doesn't send a request after the final valid reply, so
/// there's no `req[2]` to inspect; we reconstruct what it would have looked
/// like by appending the final assistant message to `req[1]`'s context.
fn render_conversation(system_prompt: &str, messages: &[Message]) -> String {
    let mut out = String::new();
    out.push_str("=== system ===\n");
    out.push_str(system_prompt);
    out.push('\n');
    for (i, msg) in messages.iter().enumerate() {
        let (role, body) = match msg {
            Message::System { content } => ("system", content.clone()),
            Message::User { content } => ("user", render_blocks(content)),
            Message::Assistant { content } => ("assistant", render_blocks(content)),
        };
        out.push_str(&format!("\n=== messages[{i}] {role} ===\n{body}\n"));
    }
    out
}

fn render_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::ToolUse { id, name, input } => {
                format!("[tool_use {id}] {name}({input})")
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let tag = if *is_error { "ERR" } else { "ok" };
                format!("[tool_result {tool_use_id} {tag}] {content}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Replace the four dynamic values inside the `<environment>` block (cwd,
/// platform, OS version, date) with stable placeholders so snapshots are
/// reproducible on any host.
fn canonicalize(rendered: &str) -> String {
    let mut out: Vec<String> = Vec::with_capacity(rendered.lines().count());
    let mut in_env = false;
    for line in rendered.lines() {
        match line {
            "<environment>" => {
                in_env = true;
                out.push(line.to_string());
            }
            "</environment>" => {
                in_env = false;
                out.push(line.to_string());
            }
            _ if in_env => out.push(replace_value_with_placeholder(line)),
            _ => out.push(line.to_string()),
        }
    }
    let mut joined = out.join("\n");
    if rendered.ends_with('\n') {
        joined.push('\n');
    }
    joined
}

fn replace_value_with_placeholder(line: &str) -> String {
    let Some(colon) = line.find(':') else {
        return line.to_string();
    };
    let key = &line[..colon];
    let placeholder = key.to_uppercase().replace(' ', "_");
    format!("{key}: <{placeholder}>")
}

/// Pull the text of the most recent `Message::User` entry — useful for tests
/// that want to check what the loop just told the model.
fn last_user_text(req: &CompletionRequest) -> Option<String> {
    req.messages.iter().rev().find_map(|m| match m {
        Message::User { content } => {
            let text: String = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    })
}

/// Pull the content of the most recent `tool_result` block in the conversation.
/// Used by the sub-agent boundary tests to assert the child's JSON reached the
/// parent intact.
fn last_tool_result_content(req: &CompletionRequest) -> Option<String> {
    req.messages.iter().rev().find_map(|m| {
        if let Message::User { content } = m {
            content.iter().rev().find_map(|b| match b {
                ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
        } else {
            None
        }
    })
}
