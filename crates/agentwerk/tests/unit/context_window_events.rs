//! How message state accumulates across agent turns.
//!
//! The conversation is a linear state machine. Each turn is one step:
//!
//! ```text
//!   S0 (initial) ──turn 1──▶ S1 ──turn 2──▶ S2 ──turn 3──▶ (terminal)
//! ```
//!
//! Every `MockProvider` call captures the input state of that turn, so we
//! can read back `provider.requests[i]` to see the state at step `i`. The
//! snapshot constants below ARE the expected states; the test drives the
//! machine to completion and asserts each one.
//!
//! Run with `cargo test --test context_accumulation -- --nocapture` to print
//! the dumps to stderr.

use std::sync::Arc;

use agentwerk::event::{CompactReason, EventKind};
use agentwerk::provider::types::ResponseStatus;
use agentwerk::provider::{ContentBlock, Message, ModelRequest, ProviderError, TokenUsage};
use agentwerk::testutil::{text_response, tool_response, MockProvider, MockTool, TestHarness};
use agentwerk::{Agent, Error, Event, Model};

/// Local helper: compact threshold for a known window size. Mirrors the
/// crate-internal constants `RESERVED_RESPONSE_TOKENS` (20_000) +
/// `COMPACTION_HEADROOM_TOKENS` (13_000) = 33_000 below the hard limit.
fn compact_threshold(window: u64) -> u64 {
    window.saturating_sub(20_000).saturating_sub(13_000)
}

const S0_INITIAL: &str = "\
=== system ===
You are Ada, a concise assistant.

Be terse.

=== messages[0] user ===
<environment>
Working directory: <WORKING_DIRECTORY>
Platform: <PLATFORM>
OS version: <OS_VERSION>
Date: <DATE>
</environment>

<context>
Working directory: /tmp/demo
</context>

=== messages[1] user ===
What is the answer?
";

const S1_AFTER_TURN_1: &str = "\
=== system ===
You are Ada, a concise assistant.

Be terse.

=== messages[0] user ===
<environment>
Working directory: <WORKING_DIRECTORY>
Platform: <PLATFORM>
OS version: <OS_VERSION>
Date: <DATE>
</environment>

<context>
Working directory: /tmp/demo
</context>

=== messages[1] user ===
What is the answer?

=== messages[2] assistant ===
[tool_use call_1] lookup({\"q\":\"answer\"})

=== messages[3] user ===
[tool_result call_1 ok] answer=42
";

const S2_AFTER_TURN_2: &str = "\
=== system ===
You are Ada, a concise assistant.

Be terse.

=== messages[0] user ===
<environment>
Working directory: <WORKING_DIRECTORY>
Platform: <PLATFORM>
OS version: <OS_VERSION>
Date: <DATE>
</environment>

<context>
Working directory: /tmp/demo
</context>

=== messages[1] user ===
What is the answer?

=== messages[2] assistant ===
[tool_use call_1] lookup({\"q\":\"answer\"})

=== messages[3] user ===
[tool_result call_1 ok] answer=42

=== messages[4] assistant ===
[tool_use call_2] lookup({\"q\":\"confirm\"})

=== messages[5] user ===
[tool_result call_2 ok] answer=42
";

#[tokio::test]
async fn state_machine_advances_one_turn_at_a_time() {
    // Pre-script the three turns the LLM will drive:
    //   turn 1 → tool_use(call_1) → advances S0 to S1
    //   turn 2 → tool_use(call_2) → advances S1 to S2
    //   turn 3 → final text        → terminal (no S3 sent)
    let provider = MockProvider::new(vec![
        tool_response("lookup", "call_1", serde_json::json!({"q": "answer"})),
        tool_response("lookup", "call_2", serde_json::json!({"q": "confirm"})),
        text_response("The answer is 42."),
    ]);

    let agent = Agent::new()
        .name("demo")
        .model_name("mock")
        .identity_prompt("You are Ada, a concise assistant.")
        .behavior_prompt("Be terse.")
        .context_prompt("Working directory: /tmp/demo")
        .tool(MockTool::new("lookup", true, "answer=42"));

    // Run the machine to completion; every input state lands in
    // `provider.requests` so we can read back each node.
    let harness = TestHarness::new(provider);
    harness
        .run_agent(&agent, "What is the answer?")
        .await
        .unwrap();
    let reqs = harness.provider().requests.lock().unwrap();
    let state = |i: usize| canonicalize(&render(&reqs[i]));

    assert_eq!(state(0), S0_INITIAL);
    assert_eq!(state(1), S1_AFTER_TURN_1);
    assert_eq!(state(2), S2_AFTER_TURN_2);

    // Terminal: last turn returned EndTurn with no tool calls, so the loop exited without a 4th request.
    assert_eq!(reqs.len(), 3, "no further request after EndTurn");
}

/// Context window used by the proactive tests. Paired with
/// `compact_threshold(CONTEXT_WINDOW_SIZE)` so the
/// relationship between window, threshold, and usage is spelled out in
/// every test body.
const CONTEXT_WINDOW_SIZE: u64 = 50_000;

#[tokio::test]
async fn proactive_compact_fires_when_threshold_crossed() {
    let threshold = compact_threshold(CONTEXT_WINDOW_SIZE);
    let mut response = text_response("done");
    response.usage = TokenUsage {
        input_tokens: threshold + 1_000,
        output_tokens: 10,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    };

    let agent = Agent::new()
        .name("demo")
        .model(Model::from_name("mock").context_window_size(CONTEXT_WINDOW_SIZE))
        .identity_prompt("");

    let harness = TestHarness::new(MockProvider::new(vec![response]));
    let _ = harness.run_agent(&agent, "hi").await;

    let (turn, tokens, threshold_in_event, reason) = first_compact(&harness.events().all());
    assert_eq!(reason, CompactReason::Proactive);
    assert_eq!(turn, 1);
    assert_eq!(threshold_in_event, threshold);
    assert!(
        tokens > threshold,
        "tokens={tokens} should exceed threshold={threshold}",
    );
}

#[tokio::test]
async fn proactive_compact_suppressed_when_model_has_no_window() {
    // Usage that would cross the threshold if the model had a known window.
    // "mock" isn't in any provider's registry, so the seam stays dormant.
    let mut response = text_response("done");
    response.usage = TokenUsage {
        input_tokens: 60_000,
        output_tokens: 10,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    };

    let agent = Agent::new()
        .name("demo")
        .model_name("mock")
        .identity_prompt("");

    let harness = TestHarness::new(MockProvider::new(vec![response]));
    let output = harness.run_agent(&agent, "hi").await.unwrap();

    assert_eq!(output.response_raw, "done");
    assert!(
        compact_reasons(&harness.events().all()).is_empty(),
        "no ContextCompacted when the model has no known window",
    );
}

#[tokio::test]
async fn reactive_compact_fires_on_context_window_exceeded_error() {
    let provider = Arc::new(MockProvider::with_results(vec![Err(
        ProviderError::ContextWindowExceeded {
            message: "prompt is too long: 205000 > 200000".into(),
        },
    )]));

    let agent = Agent::new()
        .name("demo")
        .model(Model::from_name("mock").context_window_size(200_000))
        .identity_prompt("");

    let harness = TestHarness::with_provider(provider);
    let _ = harness.run_agent(&agent, "hi").await;

    let (turn, tokens, threshold, reason) = first_compact(&harness.events().all());
    assert_eq!(reason, CompactReason::Reactive);
    assert_eq!(turn, 1);
    assert_eq!(tokens, 0, "reactive event sentinels tokens to 0");
    assert_eq!(threshold, 0, "reactive event sentinels threshold to 0");
}

#[tokio::test]
async fn reactive_compact_fires_on_mid_generation_context_window_exceeded() {
    // Rare path: provider returned Ok(response) with a stop_reason of
    // `model_context_window_exceeded`. Loop should route through the same
    // reactive seam as the pre-flight error.
    let mut response = text_response("");
    response.status = ResponseStatus::ContextWindowExceeded;

    let agent = Agent::new()
        .name("demo")
        .model(Model::from_name("mock").context_window_size(200_000))
        .identity_prompt("");

    let harness = TestHarness::new(MockProvider::new(vec![response]));
    let _ = harness.run_agent(&agent, "hi").await;

    let (turn, tokens, threshold, reason) = first_compact(&harness.events().all());
    assert_eq!(reason, CompactReason::Reactive);
    assert_eq!(turn, 1);
    assert_eq!(tokens, 0);
    assert_eq!(threshold, 0);
}

#[tokio::test]
async fn sub_agent_compaction_uses_own_model_window() {
    // Parent and child carry independent `Model`s with different windows.
    // Each one's ContextCompacted event should reflect its own threshold,
    // and the `event.agent_name` should match the emitter — proving windows
    // don't leak across the parent/child boundary.
    let parent_threshold = compact_threshold(200_000);
    let child_threshold = compact_threshold(50_000);

    let child = Agent::new()
        .name("child")
        .model(Model::from_name("child-mock").context_window_size(50_000))
        .identity_prompt("");

    let parent = Agent::new()
        .name("parent")
        .model(Model::from_name("parent-mock").context_window_size(200_000))
        .identity_prompt("")
        .sub_agents([child]);

    // Response script (shared across parent + child runs via the same mock):
    //   1. parent turn 1 — small usage, stays under parent threshold, spawns child
    //   2. child  turn 1 — usage crosses child threshold  → child's  ContextCompacted
    //   3. parent turn 2 — usage crosses parent threshold → parent's ContextCompacted
    let parent_turn1 = tool_response(
        "spawn_agent",
        "sa1",
        serde_json::json!({
            "description": "delegate",
            "instruction": "go",
            "agent": "child"
        }),
    );
    let mut child_turn1 = text_response("child done");
    child_turn1.usage = TokenUsage {
        input_tokens: child_threshold + 1_000,
        output_tokens: 10,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    };
    let mut parent_turn2 = text_response("parent done");
    // Parent accumulates usage across turns; turn 2 on top of turn 1's
    // (negligible) baseline must still overshoot the parent threshold.
    parent_turn2.usage = TokenUsage {
        input_tokens: parent_threshold + 1_000,
        output_tokens: 10,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    };

    let harness = TestHarness::new(MockProvider::new(vec![
        parent_turn1,
        child_turn1,
        parent_turn2,
    ]));
    let _ = harness.run_agent(&parent, "delegate").await;

    let events = harness.events().all();
    let child_event = events
        .iter()
        .find_map(|e| match &e.kind {
            EventKind::ContextCompacted {
                threshold, reason, ..
            } if e.agent_name == "child" => Some((*threshold, *reason)),
            _ => None,
        })
        .expect("child should emit ContextCompacted");
    let parent_event = events
        .iter()
        .find_map(|e| match &e.kind {
            EventKind::ContextCompacted {
                threshold, reason, ..
            } if e.agent_name == "parent" => Some((*threshold, *reason)),
            _ => None,
        })
        .expect("parent should emit ContextCompacted");

    assert_eq!(child_event, (child_threshold, CompactReason::Proactive));
    assert_eq!(parent_event, (parent_threshold, CompactReason::Proactive));
}

#[tokio::test]
async fn registry_populates_context_window_for_known_model_name() {
    // `.model_name("claude-…")` consults the built-in registry via
    // Model::from_name, so the compaction seam fires at the threshold derived
    // from 200k.
    let expected_threshold = compact_threshold(200_000);
    let mut response = text_response("done");
    response.usage = TokenUsage {
        input_tokens: expected_threshold + 1_000,
        output_tokens: 10,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    };

    let agent = Agent::new()
        .name("claude-agent")
        .model_name("claude-sonnet-4-20250514")
        .identity_prompt("");

    let harness = TestHarness::new(MockProvider::new(vec![response]));
    let _ = harness.run_agent(&agent, "hi").await;

    let (_, _, threshold_in_event, reason) = first_compact(&harness.events().all());
    assert_eq!(reason, CompactReason::Proactive);
    assert_eq!(threshold_in_event, expected_threshold);
}

#[tokio::test]
async fn explicit_context_window_size_bypasses_registry() {
    // The override lets callers name any model (private proxy, local
    // deployment) and set the window explicitly — without touching the registry.
    let expected_threshold = compact_threshold(30_000);
    let mut response = text_response("done");
    response.usage = TokenUsage {
        input_tokens: expected_threshold + 500,
        output_tokens: 10,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
    };

    let agent = Agent::new()
        .name("custom")
        .model(Model::from_name("custom-proxy").context_window_size(30_000))
        .identity_prompt("");

    let harness = TestHarness::new(MockProvider::new(vec![response]));
    let _ = harness.run_agent(&agent, "hi").await;

    let (_, _, threshold_in_event, reason) = first_compact(&harness.events().all());
    assert_eq!(reason, CompactReason::Proactive);
    assert_eq!(threshold_in_event, expected_threshold);
}

#[tokio::test]
async fn reactive_compact_suppressed_when_model_has_no_window() {
    let provider = Arc::new(MockProvider::with_results(vec![Err(
        ProviderError::ContextWindowExceeded {
            message: "prompt is too long".into(),
        },
    )]));

    // "mock" has no known window — reactive seam stays dormant and the
    // ContextWindowExceeded error propagates as-is.
    let agent = Agent::new()
        .name("demo")
        .model_name("mock")
        .identity_prompt("");

    let harness = TestHarness::with_provider(provider);
    let output = harness.run_agent(&agent, "hi").await.unwrap();

    assert_eq!(output.outcome, agentwerk::output::Outcome::Failed);
    assert!(
        matches!(
            output.errors.last(),
            Some(Error::Provider(ProviderError::ContextWindowExceeded { .. }))
        ),
        "expected ContextWindowExceeded error in output.errors, got {:?}",
        output.errors
    );
    assert!(
        compact_reasons(&harness.events().all()).is_empty(),
        "no ContextCompacted when the model has no known window"
    );
}

/// Render a ModelRequest as plain text. The output structure mirrors
/// what the provider receives: a `system` section followed by every message
/// in order, each labelled with its index and role.
fn render(req: &ModelRequest) -> String {
    let mut out = String::new();
    out.push_str("=== system ===\n");
    out.push_str(&req.system_prompt);
    out.push('\n');
    for (i, msg) in req.messages.iter().enumerate() {
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
/// platform, OS version, date) with stable placeholders so the snapshot is
/// reproducible on any host. Lines outside the environment block are left
/// alone — including any user-supplied `<context>` block that happens to
/// mention `Working directory:`.
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

fn compact_reasons(events: &[Event]) -> Vec<CompactReason> {
    events
        .iter()
        .filter_map(|e| match e.kind {
            EventKind::ContextCompacted { reason, .. } => Some(reason),
            _ => None,
        })
        .collect()
}

/// Extract `(turn, tokens, threshold, reason)` of the first
/// `ContextCompacted` event. Panics if none was emitted — callers use this
/// when the event is the behavior under test.
fn first_compact(events: &[Event]) -> (u32, u64, u64, CompactReason) {
    events
        .iter()
        .find_map(|e| match e.kind {
            EventKind::ContextCompacted {
                turn,
                tokens,
                threshold,
                reason,
            } => Some((turn, tokens, threshold, reason)),
            _ => None,
        })
        .expect("ContextCompacted event must be emitted")
}
