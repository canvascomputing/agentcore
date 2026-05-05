//! End-to-end: a real LLM is asked to find the file that contains a
//! specific code pattern with regex-special characters (`<`, `>`, `(`,
//! `)`, `,`, `:`). The role does NOT name `grep_tool`, does NOT describe
//! its argument shape, and does NOT mention literal-vs-regex semantics.
//! Proves the tool's *description* is good enough for a model to pick
//! content search and pass code as a literal substring (no regex escaping
//! that would zero out the match).

use std::fs;
use std::sync::{Arc, Mutex};

use super::common;

use agentwerk::event::EventKind;
use agentwerk::tools::{GlobTool, GrepTool, ListDirectoryTool, ReadFileTool};
use agentwerk::{default_logger, Agent, Event, Runnable, TicketSystem};

/// The exact substring the model must locate. Contains regex metachars
/// (`<`, `>`, `(`, `)`, `,`) so any regex-style escaping by the model
/// turns the search into a no-match.
const TARGET_SIGNATURE: &str = "fn calculate(items: Vec<(String, i32)>)";

#[derive(Clone)]
struct CapturedCall {
    name: String,
    input: serde_json::Value,
    output: Option<String>,
}

#[tokio::test]
async fn finds_code_pattern_with_special_chars(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let dir = tempfile::tempdir()?;
    let root = dir.path();
    fs::create_dir_all(root.join("src"))?;

    // Three look-alike signatures plus the target. Only one file contains
    // the exact substring `fn calculate(items: Vec<(String, i32)>)`.
    fs::write(
        root.join("src/render.rs"),
        "pub fn render(node: &Node<'a>) -> Result<String, Error> { todo!() }\n",
    )?;
    fs::write(
        root.join("src/merge.rs"),
        "pub fn merge(items: Vec<(String, String)>) -> Vec<String> { todo!() }\n",
    )?;
    fs::write(
        root.join("src/process.rs"),
        "pub fn process(items: Vec<String>) -> Result<i32, Error> { todo!() }\n",
    )?;
    fs::write(
        root.join("src/calc.rs"),
        format!("pub {TARGET_SIGNATURE} -> Result<i32, Error> {{ Ok(0) }}\n"),
    )?;
    fs::write(
        root.join("README.md"),
        "# project\n\nSome notes about Vec and tuples.\n",
    )?;

    let calls: Arc<Mutex<Vec<CapturedCall>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = Arc::clone(&calls);
    let logger = default_logger();
    let event_handler = Arc::new(move |e: Event| {
        match &e.kind {
            EventKind::ToolCallStarted {
                tool_name, input, ..
            } => {
                collected.lock().unwrap().push(CapturedCall {
                    name: tool_name.clone(),
                    input: input.clone(),
                    output: None,
                });
            }
            EventKind::ToolCallFinished {
                tool_name, output, ..
            } => {
                let mut g = collected.lock().unwrap();
                if let Some(slot) = g
                    .iter_mut()
                    .rev()
                    .find(|c| &c.name == tool_name && c.output.is_none())
                {
                    slot.output = Some(output.clone());
                }
            }
            _ => {}
        }
        logger(e);
    });

    let tickets = TicketSystem::new().max_steps(10);
    let agent = Agent::new()
        .provider(provider)
        .model(&model)
        .working_dir(root)
        .role(
            "Investigate the working directory and answer the user's question. \
             Use the available tools — pick whichever one fits the question. \
             When you have the answer, settle the ticket via \
             `mark_ticket_done_tool`.",
        )
        .tool(GrepTool)
        .tool(GlobTool)
        .tool(ListDirectoryTool)
        .tool(ReadFileTool)
        .event_handler(event_handler);
    tickets.add(agent);
    tickets.task(format!(
        "Which source file in this project contains the exact code \
         `{TARGET_SIGNATURE}`? Answer with the file's path."
    ));

    let result = tickets.run_dry().await;
    common::print_result(&result, tickets.stats());

    let recorded = calls.lock().unwrap().clone();

    // The model must have called grep_tool and passed the signature literally
    // (no `\(`, `\<`, etc. — those would make substring search miss).
    let grep_call = recorded
        .iter()
        .find(|c| {
            c.name == "grep_tool"
                && c.input
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .is_some_and(|p| p.contains("Vec<(String, i32)>"))
        })
        .unwrap_or_else(|| {
            panic!(
                "model should call `grep_tool` with the literal signature in `pattern` \
                 (no regex escaping); instead called: {:?}",
                recorded
                    .iter()
                    .map(|c| (&c.name, &c.input))
                    .collect::<Vec<_>>()
            )
        });

    let pattern = grep_call.input["pattern"].as_str().unwrap_or("");
    assert!(
        !pattern.contains('\\'),
        "model regex-escaped the pattern, but `grep_tool` is literal substring \
         (description says 'Literal match; not a regex'); pattern was: {pattern:?}"
    );

    let output = grep_call
        .output
        .as_deref()
        .expect("grep_tool call should have produced output");
    assert!(
        output.contains("calc.rs"),
        "grep_tool output should locate the signature in `src/calc.rs`; \
         got: {output:?}"
    );
    assert!(
        !output.contains("merge.rs")
            && !output.contains("process.rs")
            && !output.contains("render.rs"),
        "grep_tool output should NOT match the look-alike signatures; got: {output:?}"
    );

    Ok(())
}
