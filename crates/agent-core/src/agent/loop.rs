use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::message::{ContentBlock, Message, StopReason, Usage};
use crate::prompt::PromptBuilder;
use crate::provider::{CompletionRequest, ToolChoice};
use crate::tool::{ToolCall, ToolContext, ToolRegistry, execute_tool_calls};

use crate::session::{EntryType, TranscriptEntry};

use super::context::{InvocationContext, now_millis};
use super::event::Event;
use super::output::{
    AgentOutput, OutputSchema, StructuredOutputTool, STRUCTURED_OUTPUT_TOOL_NAME,
};
use super::queue::QueuePriority;
use super::Agent;

pub(crate) struct LlmAgent {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) model: String,
    pub(crate) system_prompt: String,
    pub(crate) max_tokens: u32,
    pub(crate) max_turns: Option<u32>,
    pub(crate) max_budget: Option<f64>,
    pub(crate) output_schema: Option<OutputSchema>,
    pub(crate) max_schema_retries: u32,
    pub(crate) prompt_builder: Option<PromptBuilder>,
    pub(crate) tools: ToolRegistry,
    #[allow(dead_code)]
    pub(crate) sub_agents: Vec<Arc<dyn Agent>>,
}

impl Agent for LlmAgent {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>> {
        Box::pin(async move { self.run_loop(ctx).await })
    }
}

impl LlmAgent {
    async fn run_loop(&self, ctx: InvocationContext) -> Result<AgentOutput> {
        let mut messages: Vec<Message> = Vec::new();
        let mut total_usage = Usage::default();
        let mut structured_output: Option<Value> = None;
        let mut schema_retries: u32 = 0;
        let mut discovered_tools: HashSet<String> = HashSet::new();

        // 1. Interpolate system prompt
        let mut system_prompt = interpolate(&self.system_prompt, &ctx.state);

        // 1b. Append structured output instruction if output_schema is set
        if self.output_schema.is_some() {
            system_prompt.push_str(
                "\n\nIMPORTANT: You must provide your final response using the StructuredOutput tool \
                 with the required structured format. After using any other tools needed to complete \
                 the task, always call StructuredOutput with your final answer in the specified schema.",
            );
        }

        // 2. Inject context message
        if let Some(ref pb) = self.prompt_builder {
            if let Some(context_msg) = pb.build_context_message() {
                messages.push(context_msg);
            }
        }

        // 3. Add user message
        messages.push(Message::User {
            content: vec![ContentBlock::Text {
                text: ctx.input.clone(),
            }],
        });

        // 3b. Record user message in transcript
        if let Some(ref store) = ctx.session_store {
            store
                .lock()
                .unwrap()
                .record(TranscriptEntry {
                    recorded_at: now_millis(),
                    entry_type: EntryType::UserMessage,
                    message: messages.last().unwrap().clone(),
                    usage: None,
                    model: None,
                })
                .ok();
        }

        // 4. Prepare tools (with structured output tool if needed)
        let (tools, tool_choice) = if let Some(ref schema) = self.output_schema {
            let mut tools = self.tools.clone();
            tools.register(StructuredOutputTool::new(schema.clone()));
            let choice = if self.tools.is_empty() {
                Some(ToolChoice::Specific {
                    name: STRUCTURED_OUTPUT_TOOL_NAME.into(),
                })
            } else {
                None
            };
            (tools, choice)
        } else {
            (self.tools.clone(), None)
        };

        (ctx.on_event)(Event::AgentStart {
            agent: self.name.clone(),
        });
        let mut turn: u32 = 0;

        loop {
            // === GUARDS ===
            if ctx.cancelled.load(Ordering::Relaxed) {
                return Err(AgenticError::Aborted);
            }
            turn += 1;
            if let Some(max) = self.max_turns {
                if turn > max {
                    return Err(AgenticError::MaxTurnsExceeded(max));
                }
            }
            if let Some(limit) = self.max_budget {
                if ctx.cost_tracker.total_cost_usd() >= limit {
                    return Err(AgenticError::BudgetExceeded {
                        spent: ctx.cost_tracker.total_cost_usd(),
                        limit,
                    });
                }
            }

            (ctx.on_event)(Event::TurnStart {
                agent: self.name.clone(),
                turn,
            });

            // === LLM CALL ===
            let response = ctx
                .provider
                .complete(CompletionRequest {
                    model: self.model.clone(),
                    system_prompt: system_prompt.clone(),
                    messages: messages.clone(),
                    tools: if tools.has_deferred_tools() {
                        tools.definitions_filtered(&discovered_tools)
                    } else {
                        tools.definitions()
                    },
                    max_tokens: self.max_tokens,
                    tool_choice: tool_choice.clone(),
                })
                .await?;

            // === RECORD USAGE ===
            total_usage.add(&response.usage);
            ctx.cost_tracker
                .record_usage(&response.model, &response.usage);
            (ctx.on_event)(Event::Usage {
                agent: self.name.clone(),
                model: response.model.clone(),
                usage: response.usage.clone(),
            });

            // === PARSE RESPONSE ===
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for block in &response.content {
                match block {
                    ContentBlock::Text { text: t } => {
                        text.push_str(t);
                        (ctx.on_event)(Event::Text {
                            agent: self.name.clone(),
                            text: t.clone(),
                        });
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                    }
                    _ => {}
                }
            }
            messages.push(Message::Assistant {
                content: response.content.clone(),
            });

            // Record assistant message in transcript
            if let Some(ref store) = ctx.session_store {
                store
                    .lock()
                    .unwrap()
                    .record(TranscriptEntry {
                        recorded_at: now_millis(),
                        entry_type: EntryType::AssistantMessage,
                        message: Message::Assistant {
                            content: response.content.clone(),
                        },
                        usage: Some(response.usage.clone()),
                        model: Some(response.model.clone()),
                    })
                    .ok();
            }

            // === STOP CHECK ===
            if response.stop_reason != StopReason::ToolUse || tool_calls.is_empty() {
                // Structured output retry enforcement
                if self.output_schema.is_some() && structured_output.is_none() {
                    schema_retries += 1;
                    if schema_retries > self.max_schema_retries {
                        return Err(AgenticError::SchemaRetryExhausted {
                            retries: self.max_schema_retries,
                        });
                    }
                    messages.push(Message::User {
                        content: vec![ContentBlock::Text {
                            text: "You MUST call the StructuredOutput tool to complete \
                                   this request. Call this tool now with the required schema."
                                .to_string(),
                        }],
                    });
                    continue;
                }

                (ctx.on_event)(Event::AgentEnd {
                    agent: self.name.clone(),
                    turns: turn,
                });
                return Ok(AgentOutput {
                    content: text,
                    usage: total_usage,
                    structured_output,
                });
            }

            // === EXECUTE TOOLS ===
            // Emit tool start events
            for call in &tool_calls {
                (ctx.on_event)(Event::ToolStart {
                    agent: self.name.clone(),
                    tool: call.name.clone(),
                    id: call.id.clone(),
                });
                ctx.cost_tracker.record_tool_calls(1);
            }

            let mut tool_ctx = ToolContext::new(ctx.working_directory.clone())
                .with_registry(Arc::new(tools.clone()));
            tool_ctx.set_extension(ctx.clone());
            let tool_results = execute_tool_calls(&tool_calls, &tools, &tool_ctx).await;

            // Emit tool end events
            for block in &tool_results {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } = block
                {
                    let tool_name = tool_calls
                        .iter()
                        .find(|c| c.id == *tool_use_id)
                        .map(|c| c.name.clone())
                        .unwrap_or_default();
                    (ctx.on_event)(Event::ToolEnd {
                        agent: self.name.clone(),
                        tool: tool_name,
                        id: tool_use_id.clone(),
                        result: content.clone(),
                        is_error: *is_error,
                    });
                }
            }

            // Extract discovered tool names from tool_search results
            for call in &tool_calls {
                if call.name == "tool_search" {
                    for block in &tool_results {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error: false,
                        } = block
                        {
                            if *tool_use_id == call.id {
                                extract_discovered_tool_names(content, &mut discovered_tools);
                            }
                        }
                    }
                }
            }

            // Extract structured output from StructuredOutput tool
            for call in &tool_calls {
                if call.name == STRUCTURED_OUTPUT_TOOL_NAME {
                    structured_output = Some(call.input.clone());
                }
            }

            messages.push(Message::User {
                content: tool_results,
            });

            // Record tool results in transcript
            if let Some(ref store) = ctx.session_store {
                store
                    .lock()
                    .unwrap()
                    .record(TranscriptEntry {
                        recorded_at: now_millis(),
                        entry_type: EntryType::ToolResult,
                        message: messages.last().unwrap().clone(),
                        usage: None,
                        model: None,
                    })
                    .ok();
            }

            // === DRAIN COMMAND QUEUE ===
            if let Some(ref queue) = ctx.command_queue {
                while let Some(cmd) = queue.dequeue(Some(&ctx.agent_id)) {
                    match cmd.priority {
                        QueuePriority::Now | QueuePriority::Next => {
                            messages.push(Message::User {
                                content: vec![ContentBlock::Text {
                                    text: cmd.content,
                                }],
                            });
                        }
                        QueuePriority::Later => {
                            queue.enqueue(cmd);
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Replace {key} placeholders in a template with values from state.
fn interpolate(template: &str, state: &HashMap<String, Value>) -> String {
    let mut result = template.to_string();
    for (key, value) in state {
        let replacement = match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        result = result.replace(&format!("{{{key}}}"), &replacement);
    }
    result
}

/// Extract tool names from tool_search result content.
fn extract_discovered_tool_names(content: &str, discovered: &mut HashSet<String>) {
    for line in content.lines() {
        if let Some(name) = line.strip_prefix("## ") {
            let name = name.trim();
            if !name.is_empty() {
                discovered.insert(name.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentBuilder, CommandQueue, CommandSource, QueuedCommand};
    use crate::error::AgenticError;
    use crate::message::ContentBlock;
    use crate::testutil::*;
    use std::sync::Arc;

    fn build_simple_agent() -> Arc<dyn Agent> {
        AgentBuilder::new()
            .name("test-agent")
            .model("mock-model")
            .system_prompt("You are a test assistant.")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn agent_loop_text_response() {
        let harness = TestHarness::new(MockProvider::text("Hello, world!"));
        let agent = build_simple_agent();

        let output = harness.run_agent(agent.as_ref(), "Hi").await.unwrap();
        assert_eq!(output.content, "Hello, world!");
        assert!(output.structured_output.is_none());
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn agent_loop_with_tool_execution() {
        let provider = MockProvider::tool_then_text(
            "echo_tool",
            serde_json::json!({"text": "ping"}),
            "Done!",
        );
        let agent = AgentBuilder::new()
            .name("test-agent")
            .model("mock-model")
            .system_prompt("You are helpful.")
            .tool(MockTool::new("echo_tool", false, "pong"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "Echo test").await.unwrap();
        assert_eq!(output.content, "Done!");
        assert_eq!(harness.provider().request_count(), 2);
    }

    #[tokio::test]
    async fn agent_guards_table() {
        // max_turns exceeded
        {
            let provider = MockProvider::new(vec![
                tool_response("t", "c1", serde_json::json!({})),
                tool_response("t", "c2", serde_json::json!({})),
                tool_response("t", "c3", serde_json::json!({})),
            ]);
            let agent = AgentBuilder::new()
                .name("test")
                .model("mock")
                .system_prompt("")
                .max_turns(2)
                .tool(MockTool::new("t", false, "ok"))
                .build()
                .unwrap();
            let harness = TestHarness::new(provider);
            let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
            assert!(matches!(err, AgenticError::MaxTurnsExceeded(2)));
        }

        // budget exceeded
        {
            let provider = MockProvider::new(vec![
                tool_response("t", "c1", serde_json::json!({})),
                text_response("done"),
            ]);
            let agent = AgentBuilder::new()
                .name("test")
                .model("mock")
                .system_prompt("")
                .max_budget(0.0)
                .tool(MockTool::new("t", false, "ok"))
                .build()
                .unwrap();
            let harness = TestHarness::new(provider);
            let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
            assert!(matches!(err, AgenticError::BudgetExceeded { .. }));
        }

        // cancellation
        {
            let provider = MockProvider::new(vec![
                tool_response("t", "c1", serde_json::json!({})),
                text_response("done"),
            ]);
            let agent = AgentBuilder::new()
                .name("test")
                .model("mock")
                .system_prompt("")
                .tool(MockTool::new("t", false, "ok"))
                .build()
                .unwrap();
            let harness = TestHarness::new(provider);
            harness.cancel();
            let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
            assert!(matches!(err, AgenticError::Aborted));
        }
    }

    #[tokio::test]
    async fn state_interpolation_in_system_prompt() {
        let provider = MockProvider::text("Answer about rust");
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("You are an expert on {topic}.")
            .build()
            .unwrap();

        let harness = TestHarness::new(provider).with_state("topic", serde_json::json!("rust"));
        harness.run_agent(agent.as_ref(), "Tell me").await.unwrap();

        let prompts = harness.provider().system_prompts();
        assert!(prompts[0].contains("expert on rust"));
    }

    #[tokio::test]
    async fn events_emitted_during_agent_run() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "Done");
        let agent = AgentBuilder::new()
            .name("assistant")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("read", true, "file contents"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "read it").await.unwrap();

        let events = harness.events();
        assert_eq!(events.agent_starts(), vec!["assistant"]);
        assert!(!events.tool_starts().is_empty());
        assert!(events.texts().contains(&"Done".to_string()));
        assert_eq!(events.agent_ends().len(), 1);
    }

    #[tokio::test]
    async fn agent_drains_command_queue() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build()
            .unwrap();

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "extra instruction".into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_id: Some("test".into()),
        });

        let harness = TestHarness::new(provider);
        let mut ctx = harness.build_context("start");
        ctx.command_queue = Some(queue);
        ctx.agent_id = "test".into();

        let output = agent.run(ctx).await.unwrap();
        assert_eq!(output.content, "final");
        let requests = harness.provider().requests.lock().unwrap();
        let second_req = &requests[1];
        let has_extra = second_req.messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains("extra instruction"),
                _ => false,
            }),
            _ => false,
        });
        assert!(has_extra, "Extra instruction should be in second request");
    }

    #[tokio::test]
    async fn agent_requeues_later_commands() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build()
            .unwrap();

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "later task".into(),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification {
                task_id: "42".into(),
            },
            agent_id: Some("test".into()),
        });

        let harness = TestHarness::new(provider);
        let mut ctx = harness.build_context("start");
        ctx.command_queue = Some(queue.clone());
        ctx.agent_id = "test".into();

        agent.run(ctx).await.unwrap();

        let cmd = queue.dequeue(Some("test"));
        assert!(cmd.is_some());
        assert_eq!(cmd.unwrap().content, "later task");
    }

    #[tokio::test]
    async fn agent_sends_filtered_definitions_when_deferred() {
        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("always", true, "ok"))
            .tool(DeferredMockTool::new("deferred"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let deferred_def = req.tools.iter().find(|t| t.name == "deferred").unwrap();
        assert!(
            deferred_def.description.is_empty(),
            "Deferred tool should have empty description"
        );
    }

    #[tokio::test]
    async fn agent_no_filtering_without_deferred() {
        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .tool(MockTool::new("read", true, "ok"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let def = req.tools.iter().find(|t| t.name == "read").unwrap();
        assert!(!def.description.is_empty());
    }

    #[test]
    fn extract_discovered_tool_names_parses_headers() {
        let mut discovered = HashSet::new();
        let content = "## read_file\nReads a file.\n\n## grep\nSearches content.";
        extract_discovered_tool_names(content, &mut discovered);
        assert!(discovered.contains("read_file"));
        assert!(discovered.contains("grep"));
        assert_eq!(discovered.len(), 2);
    }

    // --- Structured output tests ---

    #[tokio::test]
    async fn structured_output_extracted() {
        let schema_input = serde_json::json!({"category": "billing", "priority": "high"});
        let provider = MockProvider::new(vec![
            tool_response(STRUCTURED_OUTPUT_TOOL_NAME, "so1", schema_input.clone()),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("classifier")
            .model("mock")
            .system_prompt("Classify.")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "priority": {"type": "string"}
                },
                "required": ["category", "priority"]
            }))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "ticket").await.unwrap();
        assert!(output.structured_output.is_some());
        let so = output.structured_output.unwrap();
        assert_eq!(so["category"], "billing");
        assert_eq!(so["priority"], "high");
    }

    #[tokio::test]
    async fn structured_output_retry_on_noncompliance() {
        let provider = MockProvider::new(vec![
            text_response("thinking..."),
            text_response("still thinking..."),
            tool_response(
                STRUCTURED_OUTPUT_TOOL_NAME,
                "so1",
                serde_json::json!({"answer": "yes"}),
            ),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "question").await.unwrap();
        assert!(output.structured_output.is_some());
        assert!(harness.provider().request_count() >= 3);
    }

    #[tokio::test]
    async fn structured_output_retry_exhausted() {
        let provider = MockProvider::new(vec![
            text_response("nope"),
            text_response("still nope"),
            text_response("nope again"),
            text_response("last nope"),
        ]);
        let agent = AgentBuilder::new()
            .name("test")
            .model("mock")
            .system_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
        assert!(matches!(
            err,
            AgenticError::SchemaRetryExhausted { retries: 3 }
        ));
    }

    #[test]
    fn validate_value_table() {
        use crate::agent::validate_value;

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "age"]
        });

        assert!(validate_value(
            &serde_json::json!({"name": "Alice", "age": 30, "score": 9.5, "active": true, "tags": ["a", "b"]}),
            &schema
        ).is_ok());

        assert!(validate_value(
            &serde_json::json!({"name": "Bob", "age": 25}),
            &schema
        ).is_ok());

        assert!(validate_value(
            &serde_json::json!({"name": "Carol"}),
            &schema
        ).is_err());

        assert!(validate_value(
            &serde_json::json!({"name": 123, "age": 25}),
            &schema
        ).is_err());

        assert!(validate_value(
            &serde_json::json!({"name": "Dave", "age": "old"}),
            &schema
        ).is_err());

        assert!(validate_value(
            &serde_json::json!({"name": "Eve", "age": 20, "active": "yes"}),
            &schema
        ).is_err());

        assert!(validate_value(
            &serde_json::json!({"name": "Frank", "age": 40, "tags": [1, 2]}),
            &schema
        ).is_err());

        assert!(validate_value(
            &serde_json::json!("not an object"),
            &schema
        ).is_err());
    }
}
