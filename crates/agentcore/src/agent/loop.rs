use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::persistence::session::{EntryType, TranscriptEntry};
use crate::provider::model::ModelSpec;
use crate::provider::types::{ContentBlock, Message, ModelResponse, StopReason, StreamEvent, TokenUsage};
use crate::provider::{CompletionRequest, ToolChoice};
use crate::tools::{ToolCall, ToolContext, ToolRegistry, execute_tool_calls};

use super::r#trait::Agent;
use super::context::{InvocationContext, now_millis};
use super::event::Event;
use super::output::{AgentOutput, OutputSchema, Statistics, StructuredOutputTool};
use super::prompts::{self as prompts, BehaviorPrompt, ContextBuilder, STRUCTURED_OUTPUT_TOOL_NAME, interpolate};
use super::queue::QueuePriority;

// ---------------------------------------------------------------------------
// AgentLoop — the LLM-powered agent implementation
// ---------------------------------------------------------------------------

/// An LLM-powered agent. Calls an LLM in a loop, executing tools until done.
/// Created via `AgentBuilder::build()`.
pub(crate) struct AgentLoop {
    pub(crate) name: String,
    pub(crate) model: ModelSpec,
    pub(crate) identity_prompt: String,
    pub(crate) max_tokens: u32,
    pub(crate) max_turns: Option<u32>,
    pub(crate) output_schema: Option<OutputSchema>,
    pub(crate) max_schema_retries: u32,
    pub(crate) behavior_prompts: Vec<(BehaviorPrompt, String)>,
    pub(crate) context_builder: ContextBuilder,
    pub(crate) tools: ToolRegistry,
    #[allow(dead_code)]
    pub(crate) sub_agents: Vec<Arc<dyn Agent>>,
}

impl Agent for AgentLoop {
    fn name(&self) -> &str {
        &self.name
    }
    fn run(
        &self,
        ctx: InvocationContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>> {
        Box::pin(async move { self.execute(ctx).await })
    }
}

// ---------------------------------------------------------------------------
// Agent loop execution
// ---------------------------------------------------------------------------

/// Mutable state carried across turns of the agent loop.
struct LoopState {
    messages: Vec<Message>,
    total_usage: TokenUsage,
    request_count: u64,
    tool_call_count: u64,
    structured_output: Option<Value>,
    schema_retries: u32,
    discovered_tools: HashSet<String>,
    turn: u32,
    system_prompt: String,
    tools: ToolRegistry,
    tool_choice: Option<ToolChoice>,
}

impl AgentLoop {
    pub(crate) async fn execute(&self, ctx: InvocationContext) -> Result<AgentOutput> {
        let mut state = self.init_state(&ctx);
        self.emit(&ctx, Event::AgentStart { agent_name: self.name.clone() });

        loop {
            // Guards: cancellation, turn limit, estimated costs
            self.check_guards(&ctx, &state)?;
            state.turn += 1;
            self.emit(&ctx, Event::TurnStart { agent_name: self.name.clone(), turn: state.turn });

            // LLM call
            let resolved_model = self.model.resolve(&ctx.model);
            self.emit(&ctx, Event::RequestStart { agent_name: self.name.clone(), model: resolved_model.clone() });
            let response = self.call_llm(&ctx, &state).await?;
            self.emit(&ctx, Event::RequestEnd { agent_name: self.name.clone(), model: resolved_model });
            self.record_usage(&ctx, &response, &mut state);

            // Parse response into text and tool calls
            let (text, tool_calls) = self.parse_response(&ctx, &response);
            state.messages.push(Message::Assistant { content: response.content.clone() });
            self.record_transcript(&ctx, EntryType::AssistantMessage, &state, Some(&response));

            // Done? Return or retry structured output
            if response.stop_reason != StopReason::ToolUse || tool_calls.is_empty() {
                if let Some(output) = self.try_finish(&ctx, &mut state, text)? {
                    return Ok(output);
                }
                continue;
            }

            // Execute tools and collect results
            let results = self.execute_tools(&ctx, &mut state, &tool_calls).await;
            self.extract_discoveries(&tool_calls, &results, &mut state);

            // Add tool results to conversation
            state.messages.push(Message::User { content: results });
            self.record_transcript(&ctx, EntryType::ToolResult, &state, None);
            self.drain_command_queue(&ctx, &mut state);

            self.emit(&ctx, Event::TurnEnd { agent_name: self.name.clone(), turn: state.turn });
        }
    }

    fn init_state(&self, ctx: &InvocationContext) -> LoopState {
        let mut system_prompt = interpolate(&self.identity_prompt, &ctx.template_variables);
        for (_, content) in &self.behavior_prompts {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(content);
        }
        if self.output_schema.is_some() {
            system_prompt.push_str(prompts::STRUCTURED_OUTPUT_INSTRUCTION);
        }
        let (tools, tool_choice) = self.build_tool_config();

        let mut messages = Vec::new();
        if let Some(context_msg) = self.context_builder.build_context_message() {
            messages.push(context_msg);
        }
        messages.push(Message::user(ctx.instruction_prompt.clone()));
        self.record_initial_message(ctx, &messages);

        LoopState {
            messages,
            total_usage: TokenUsage::default(),
            request_count: 0,
            tool_call_count: 0,
            structured_output: None,
            schema_retries: 0,
            discovered_tools: HashSet::new(),
            turn: 0,
            system_prompt,
            tools,
            tool_choice,
        }
    }

    fn record_initial_message(&self, ctx: &InvocationContext, messages: &[Message]) {
        let Some(ref store) = ctx.session_store else { return };
        let Some(message) = messages.last() else { return };
        store.lock().unwrap().record(TranscriptEntry {
            recorded_at: now_millis(),
            entry_type: EntryType::UserMessage,
            message: message.clone(),
            usage: None,
            model: None,
        }).ok();
    }

    fn build_tool_config(&self) -> (ToolRegistry, Option<ToolChoice>) {
        let Some(ref schema) = self.output_schema else {
            return (self.tools.clone(), None);
        };

        let mut tools = self.tools.clone();
        tools.register(StructuredOutputTool::new(schema.clone()));

        let tool_choice = if self.tools.is_empty() {
            Some(ToolChoice::Specific { name: STRUCTURED_OUTPUT_TOOL_NAME.into() })
        } else {
            None
        };

        (tools, tool_choice)
    }

    fn check_guards(&self, ctx: &InvocationContext, state: &LoopState) -> Result<()> {
        if ctx.cancel_signal.load(Ordering::Relaxed) {
            return Err(AgenticError::Aborted);
        }
        if self.max_turns.is_some_and(|max| state.turn >= max) {
            return Err(AgenticError::MaxTurnsExceeded(self.max_turns.unwrap()));
        }
        Ok(())
    }

    async fn call_llm(&self, ctx: &InvocationContext, state: &LoopState) -> Result<ModelResponse> {
        let tools = &state.tools;
        let tool_defs = if tools.has_deferred_tools() {
            tools.definitions_filtered(&state.discovered_tools)
        } else {
            tools.definitions()
        };

        let resolved_model = self.model.resolve(&ctx.model);

        let request = CompletionRequest {
            model: resolved_model,
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tools: tool_defs,
            max_tokens: self.max_tokens,
            tool_choice: state.tool_choice.clone(),
        };

        let event_handler = ctx.event_handler.clone();
        let agent_name = self.name.clone();
        let on_event = Arc::new(move |event: StreamEvent| {
            if let StreamEvent::TextDelta { text, .. } = &event {
                event_handler(Event::ResponseTextChunk {
                    agent_name: agent_name.clone(),
                    content: text.clone(),
                });
            }
        });

        ctx.provider.complete_streaming(request, on_event).await
    }

    fn record_usage(&self, ctx: &InvocationContext, response: &ModelResponse, state: &mut LoopState) {
        state.total_usage.add(&response.usage);
        state.request_count += 1;
        self.emit(ctx, Event::TokenUsage {
            agent_name: self.name.clone(),
            model: response.model.clone(),
            usage: response.usage.clone(),
        });
    }

    fn parse_response(&self, ctx: &InvocationContext, response: &ModelResponse) -> (String, Vec<ToolCall>) {
        let mut text = String::new();
        let mut tool_calls = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text: chunk } => {
                    text.push_str(chunk);
                    self.emit(ctx, Event::ResponseTextChunk { agent_name: self.name.clone(), content: chunk.clone() });
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

        (text, tool_calls)
    }

    fn try_finish(
        &self,
        ctx: &InvocationContext,
        state: &mut LoopState,
        text: String,
    ) -> Result<Option<AgentOutput>> {
        // Structured output retry
        if self.output_schema.is_some() && state.structured_output.is_none() {
            state.schema_retries += 1;
            if state.schema_retries > self.max_schema_retries {
                return Err(AgenticError::SchemaRetryExhausted { retries: self.max_schema_retries });
            }
            state.messages.push(Message::user(prompts::STRUCTURED_OUTPUT_RETRY));
            return Ok(None); // continue loop
        }

        self.emit(ctx, Event::TurnEnd { agent_name: self.name.clone(), turn: state.turn });
        self.emit(ctx, Event::AgentEnd { agent_name: self.name.clone(), turns: state.turn });
        Ok(Some(AgentOutput {
            response: state.structured_output.take(),
            response_raw: text,
            statistics: Statistics {
                input_tokens: state.total_usage.input_tokens,
                output_tokens: state.total_usage.output_tokens,
                cache_read_tokens: state.total_usage.cache_read_input_tokens,
                cache_write_tokens: state.total_usage.cache_creation_input_tokens,
                requests: state.request_count,
                tool_calls: state.tool_call_count,
                turns: state.turn,
            },
        }))
    }

    async fn execute_tools(
        &self,
        ctx: &InvocationContext,
        state: &mut LoopState,
        tool_calls: &[ToolCall],
    ) -> Vec<ContentBlock> {
        state.tool_call_count += tool_calls.len() as u64;
        for call in tool_calls {
            self.emit(ctx, Event::ToolCallStart {
                agent_name: self.name.clone(),
                tool_name: call.name.clone(),
                call_id: call.id.clone(),
                input: call.input.clone(),
            });
        }

        let mut tool_ctx = ToolContext::new(ctx.working_directory.clone())
            .registry(Arc::new(state.tools.clone()));
        tool_ctx.set_extension(ctx.clone());
        let results = execute_tool_calls(tool_calls, &state.tools, &tool_ctx).await;

        self.emit_tool_results(ctx, tool_calls, &results);
        results
    }

    fn emit_tool_results(&self, ctx: &InvocationContext, tool_calls: &[ToolCall], results: &[ContentBlock]) {
        for block in results {
            let ContentBlock::ToolResult { tool_use_id, content, is_error } = block else {
                continue;
            };
            let tool_name = tool_calls
                .iter()
                .find(|c| c.id == *tool_use_id)
                .map(|c| c.name.clone())
                .unwrap_or_default();
            self.emit(ctx, Event::ToolCallEnd {
                agent_name: self.name.clone(),
                tool_name,
                call_id: tool_use_id.clone(),
                output: content.clone(),
                is_error: *is_error,
            });
        }
    }

    fn extract_discoveries(
        &self,
        tool_calls: &[ToolCall],
        results: &[ContentBlock],
        state: &mut LoopState,
    ) {
        for call in tool_calls {
            if call.name == STRUCTURED_OUTPUT_TOOL_NAME {
                state.structured_output = Some(call.input.clone());
                continue;
            }
            if call.name != "tool_search" {
                continue;
            }

            let search_output = find_tool_result(results, &call.id);
            if let Some(content) = search_output {
                extract_discovered_tool_names(content, &mut state.discovered_tools);
            }
        }
    }

    fn record_transcript(
        &self,
        ctx: &InvocationContext,
        entry_type: EntryType,
        state: &LoopState,
        response: Option<&ModelResponse>,
    ) {
        let Some(ref store) = ctx.session_store else { return };
        let Some(message) = state.messages.last() else { return };
        store.lock().unwrap().record(TranscriptEntry {
            recorded_at: now_millis(),
            entry_type,
            message: message.clone(),
            usage: response.map(|r| r.usage.clone()),
            model: response.map(|r| r.model.clone()),
        }).ok();
    }

    fn drain_command_queue(&self, ctx: &InvocationContext, state: &mut LoopState) {
        let Some(ref queue) = ctx.command_queue else { return };
        while let Some(cmd) = queue.dequeue(Some(&ctx.agent_name)) {
            match cmd.priority {
                QueuePriority::Now | QueuePriority::Next => {
                    state.messages.push(Message::user(cmd.content));
                }
                QueuePriority::Later => {
                    queue.enqueue(cmd);
                    break;
                }
            }
        }
    }

    fn emit(&self, ctx: &InvocationContext, event: Event) {
        (ctx.event_handler)(event);
    }
}

fn find_tool_result<'a>(results: &'a [ContentBlock], call_id: &str) -> Option<&'a str> {
    results.iter().find_map(|block| match block {
        ContentBlock::ToolResult { tool_use_id, content, is_error: false }
            if *tool_use_id == call_id => Some(content.as_str()),
        _ => None,
    })
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
    use crate::agent::{Agent, AgentBuilder};
    use crate::agent::queue::{CommandQueue, CommandSource, QueuePriority, QueuedCommand};
    use crate::error::AgenticError;
    use crate::provider::types::ContentBlock;
    use crate::testutil::*;
    use std::sync::Arc;

    fn build_simple_agent() -> Arc<dyn Agent> {
        AgentBuilder::new()
            .name("test-agent")
            .model("mock-model")
            .identity_prompt("You are a test assistant.")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn simple_text_response() {
        let harness = TestHarness::new(MockProvider::text("Hello, world!"));
        let agent = build_simple_agent();

        let output = harness.run_agent(agent.as_ref(), "Hi").await.unwrap();
        assert_eq!(output.response_raw, "Hello, world!");
        assert!(output.response.is_none());
        assert_eq!(harness.provider().request_count(), 1);
    }

    #[tokio::test]
    async fn simple_tool_execution() {
        let provider = MockProvider::tool_then_text(
            "echo_tool",
            serde_json::json!({"text": "ping"}),
            "Done!",
        );
        let agent = AgentBuilder::new()
            .name("test-agent")
            .model("mock-model")
            .identity_prompt("You are helpful.")
            .tool(MockTool::new("echo_tool", false, "pong"))
            .build()
            .unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "Echo test").await.unwrap();
        assert_eq!(output.response_raw, "Done!");
        assert_eq!(harness.provider().request_count(), 2);
    }

    #[tokio::test]
    async fn guard_max_turns() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            tool_response("t", "c2", serde_json::json!({})),
            tool_response("t", "c3", serde_json::json!({})),
        ]);
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .max_turns(2)
            .tool(MockTool::new("t", false, "ok"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
        assert!(matches!(err, AgenticError::MaxTurnsExceeded(2)));
    }

    #[tokio::test]
    async fn guard_cancellation() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        harness.cancel();
        let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
        assert!(matches!(err, AgenticError::Aborted));
    }

    #[tokio::test]
    async fn state_interpolation_in_system_prompt() {
        let provider = MockProvider::text("Answer about rust");
        let agent = AgentBuilder::new()
            .name("test").model("mock")
            .identity_prompt("You are an expert on {topic}.")
            .build().unwrap();

        let harness = TestHarness::new(provider).with_state("topic", serde_json::json!("rust"));
        harness.run_agent(agent.as_ref(), "Tell me").await.unwrap();

        let prompts = harness.provider().system_prompts();
        assert!(prompts[0].contains("expert on rust"));
    }

    #[tokio::test]
    async fn events_emitted() {
        let provider = MockProvider::tool_then_text("read", serde_json::json!({}), "Done");
        let agent = AgentBuilder::new()
            .name("assistant").model("mock").identity_prompt("")
            .tool(MockTool::new("read", true, "file contents"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "read it").await.unwrap();

        let events = harness.events();
        assert_eq!(events.agent_starts(), vec!["assistant"]);
        assert!(!events.tool_starts().is_empty());
        assert!(events.texts().contains(&"Done".to_string()));
        assert_eq!(events.agent_ends().len(), 1);
    }

    #[tokio::test]
    async fn command_queue_drains_next() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build().unwrap();

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "extra instruction".into(),
            priority: QueuePriority::Next,
            source: CommandSource::UserInput,
            agent_name: Some("test".into()),
        });

        let harness = TestHarness::new(provider);
        let mut ctx = harness.build_context("start");
        ctx.command_queue = Some(queue);
        ctx.agent_name = "test".into();

        let output = agent.run(ctx).await.unwrap();
        assert_eq!(output.response_raw, "final");

        let requests = harness.provider().requests.lock().unwrap();
        let has_extra = requests[1].messages.iter().any(|m| match m {
            Message::User { content } => content.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains("extra instruction"),
                _ => false,
            }),
            _ => false,
        });
        assert!(has_extra);
    }

    #[tokio::test]
    async fn command_queue_requeues_later() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            text_response("final"),
        ]);
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build().unwrap();

        let queue = Arc::new(CommandQueue::new());
        queue.enqueue(QueuedCommand {
            content: "later task".into(),
            priority: QueuePriority::Later,
            source: CommandSource::TaskNotification { task_id: "42".into() },
            agent_name: Some("test".into()),
        });

        let harness = TestHarness::new(provider);
        let mut ctx = harness.build_context("start");
        ctx.command_queue = Some(queue.clone());
        ctx.agent_name = "test".into();

        agent.run(ctx).await.unwrap();

        let cmd = queue.dequeue(Some("test"));
        assert!(cmd.is_some());
        assert_eq!(cmd.unwrap().content, "later task");
    }

    #[tokio::test]
    async fn deferred_tool_filtering() {
        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("always", true, "ok"))
            .tool(DeferredMockTool::new("deferred"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let deferred_def = req.tools.iter().find(|t| t.name == "deferred").unwrap();
        assert!(deferred_def.description.is_empty());
    }

    #[tokio::test]
    async fn no_filtering_without_deferred() {
        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("read", true, "ok"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        let def = req.tools.iter().find(|t| t.name == "read").unwrap();
        assert!(!def.description.is_empty());
    }

    #[test]
    fn extract_discovered_tool_names_parses_headers() {
        let mut discovered = HashSet::new();
        extract_discovered_tool_names("## read_file\nReads a file.\n\n## grep\nSearches.", &mut discovered);
        assert!(discovered.contains("read_file"));
        assert!(discovered.contains("grep"));
        assert_eq!(discovered.len(), 2);
    }

    // --- Structured output ---

    #[tokio::test]
    async fn structured_output_extracted() {
        let schema_input = serde_json::json!({"category": "billing", "priority": "high"});
        let provider = MockProvider::new(vec![
            tool_response(STRUCTURED_OUTPUT_TOOL_NAME, "so1", schema_input.clone()),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("classifier").model("mock").identity_prompt("Classify.")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": { "category": {"type": "string"}, "priority": {"type": "string"} },
                "required": ["category", "priority"]
            }))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "ticket").await.unwrap();
        let so = output.response.unwrap();
        assert_eq!(so["category"], "billing");
        assert_eq!(so["priority"], "high");
    }

    #[tokio::test]
    async fn structured_output_retry_on_noncompliance() {
        let provider = MockProvider::new(vec![
            text_response("thinking..."),
            text_response("still thinking..."),
            tool_response(STRUCTURED_OUTPUT_TOOL_NAME, "so1", serde_json::json!({"answer": "yes"})),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "question").await.unwrap();
        assert!(output.response.is_some());
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
            .name("test").model("mock").identity_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
        assert!(matches!(err, AgenticError::SchemaRetryExhausted { retries: 3 }));
    }

    #[test]
    fn validate_value_table() {
        use crate::agent::output::validate_value;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}, "age": {"type": "integer"},
                "score": {"type": "number"}, "active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age"]
        });

        assert!(validate_value(&serde_json::json!({"name": "Alice", "age": 30, "score": 9.5, "active": true, "tags": ["a", "b"]}), &schema).is_ok());
        assert!(validate_value(&serde_json::json!({"name": "Bob", "age": 25}), &schema).is_ok());
        assert!(validate_value(&serde_json::json!({"name": "Carol"}), &schema).is_err());
        assert!(validate_value(&serde_json::json!({"name": 123, "age": 25}), &schema).is_err());
        assert!(validate_value(&serde_json::json!({"name": "Dave", "age": "old"}), &schema).is_err());
        assert!(validate_value(&serde_json::json!({"name": "Eve", "age": 20, "active": "yes"}), &schema).is_err());
        assert!(validate_value(&serde_json::json!({"name": "Frank", "age": 40, "tags": [1, 2]}), &schema).is_err());
        assert!(validate_value(&serde_json::json!("not an object"), &schema).is_err());
    }

    // --- New test coverage ---

    #[tokio::test]
    async fn llm_error_mid_loop() {
        let provider = MockProvider::new(vec![
            tool_response("t", "c1", serde_json::json!({})),
            // Second call will fail — no more responses
        ]);
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let err = harness.run_agent(agent.as_ref(), "go").await.unwrap_err();
        assert!(format!("{err}").contains("no more mock responses"));
    }

    #[tokio::test]
    async fn multiple_tool_calls_in_one_response() {
        let response = ModelResponse {
            content: vec![
                ContentBlock::ToolUse { id: "c1".into(), name: "a".into(), input: serde_json::json!({}) },
                ContentBlock::ToolUse { id: "c2".into(), name: "b".into(), input: serde_json::json!({}) },
            ],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: "mock".into(),
        };
        let provider = MockProvider::new(vec![response, text_response("done")]);

        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("a", true, "result_a"))
            .tool(MockTool::new("b", true, "result_b"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "go").await.unwrap();
        assert_eq!(output.response_raw, "done");

        // Both tools should have produced results in the second request
        let requests = harness.provider().requests.lock().unwrap();
        let tool_results: Vec<_> = requests[1].messages.iter()
            .filter_map(|m| match m {
                Message::User { content } => Some(content),
                _ => None,
            })
            .flatten()
            .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
            .collect();
        assert_eq!(tool_results.len(), 2);
    }

    #[tokio::test]
    async fn usage_accumulates_across_turns() {
        use crate::provider::types::ModelResponse;

        let r1 = ModelResponse {
            content: vec![ContentBlock::ToolUse { id: "c1".into(), name: "t".into(), input: serde_json::json!({}) }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage { input_tokens: 100, output_tokens: 50, ..Default::default() },
            model: "mock".into(),
        };
        let r2 = ModelResponse {
            content: vec![ContentBlock::Text { text: "done".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage { input_tokens: 200, output_tokens: 80, ..Default::default() },
            model: "mock".into(),
        };
        let provider = MockProvider::new(vec![r1, r2]);

        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("t", false, "ok"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "go").await.unwrap();
        assert_eq!(output.statistics.input_tokens, 300);
        assert_eq!(output.statistics.output_tokens, 130);
    }

    #[tokio::test]
    async fn user_context_injects_into_messages() {
        let provider = MockProvider::text("ok");
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("System.")
            .context_prompt("Project context here")
            .build().unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().last_request().unwrap();
        // First message should be the context message from ContextBuilder
        let first_msg = &req.messages[0];
        match first_msg {
            Message::User { content } => {
                let text = match &content[0] {
                    ContentBlock::Text { text } => text.clone(),
                    _ => panic!("Expected text"),
                };
                assert!(text.contains("Project context here"));
            }
            _ => panic!("Expected user message"),
        }
    }

    #[tokio::test]
    async fn forced_tool_choice_with_no_other_tools() {
        let provider = MockProvider::new(vec![
            tool_response(STRUCTURED_OUTPUT_TOOL_NAME, "so1", serde_json::json!({"x": "y"})),
            text_response("done"),
        ]);
        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .output_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        harness.run_agent(agent.as_ref(), "go").await.unwrap();

        let req = harness.provider().requests.lock().unwrap();
        match &req[0].tool_choice {
            Some(ToolChoice::Specific { name }) => assert_eq!(name, STRUCTURED_OUTPUT_TOOL_NAME),
            other => panic!("Expected Specific tool_choice, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_error_continues_loop() {
        let error_response = ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: "c1".into(), name: "failing".into(), input: serde_json::json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: "mock".into(),
        };
        let provider = MockProvider::new(vec![error_response, text_response("recovered")]);

        let agent = AgentBuilder::new()
            .name("test").model("mock").identity_prompt("")
            .tool(MockTool::new("failing", false, "error: something broke"))
            .build().unwrap();

        let harness = TestHarness::new(provider);
        let output = harness.run_agent(agent.as_ref(), "go").await.unwrap();
        assert_eq!(output.response_raw, "recovered");
        assert_eq!(harness.provider().request_count(), 2);
    }
}
