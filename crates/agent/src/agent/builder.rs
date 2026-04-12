use std::sync::Arc;

use serde_json::Value;

use crate::error::{AgenticError, Result};
use super::prompts::{BehaviorPrompt, ContextBuilder, EnvironmentContext};
use crate::tools::{Tool, ToolRegistry};

use super::agent::{Agent, AgentLoop};
use super::output::OutputSchema;

pub struct AgentBuilder {
    name: Option<String>,
    description: String,
    model: Option<String>,
    system_prompt: String,
    max_tokens: u32,
    max_turns: Option<u32>,
    max_budget: Option<f64>,
    output_schema: Option<OutputSchema>,
    max_schema_retries: u32,
    behavior_prompts: Vec<(BehaviorPrompt, String)>,
    context_builder: ContextBuilder,
    tools: ToolRegistry,
    sub_agents: Vec<Arc<dyn Agent>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        let behavior_prompts = BehaviorPrompt::all()
            .iter()
            .map(|kind| (*kind, kind.default_content().to_string()))
            .collect();

        Self {
            name: None,
            description: String::new(),
            model: None,
            system_prompt: String::new(),
            max_tokens: 4096,
            max_turns: None,
            max_budget: None,
            output_schema: None,
            max_schema_retries: 3,
            behavior_prompts,
            context_builder: ContextBuilder::new(),
            tools: ToolRegistry::new(),
            sub_agents: Vec::new(),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    pub fn max_turns(mut self, max: u32) -> Self {
        self.max_turns = Some(max);
        self
    }

    pub fn max_budget(mut self, budget: f64) -> Self {
        self.max_budget = Some(budget);
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(OutputSchema::new(schema).expect("invalid output schema"));
        self
    }

    pub fn max_schema_retries(mut self, retries: u32) -> Self {
        self.max_schema_retries = retries;
        self
    }

    pub fn behavior_prompt(mut self, kind: BehaviorPrompt, content: impl Into<String>) -> Self {
        if let Some(entry) = self.behavior_prompts.iter_mut().find(|(k, _)| *k == kind) {
            entry.1 = content.into();
        }
        self
    }

    pub fn environment_context(mut self, env: &EnvironmentContext) -> Self {
        self.context_builder.environment_context(env);
        self
    }

    pub fn instruction_files(mut self, cwd: &std::path::Path) -> Self {
        self.context_builder.instruction_files(cwd).ok();
        self
    }

    pub fn memory(mut self, memory_dir: &std::path::Path) -> Self {
        self.context_builder.memory(memory_dir).ok();
        self
    }

    pub fn user_context(mut self, context: impl Into<String>) -> Self {
        self.context_builder.user_context(context.into());
        self
    }

    pub fn sub_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.sub_agents.push(agent);
        self
    }

    pub fn build(self) -> Result<Arc<dyn Agent>> {
        let name = self
            .name
            .ok_or_else(|| AgenticError::Other("AgentBuilder requires a name".into()))?;
        let model = self
            .model
            .ok_or_else(|| AgenticError::Other("AgentBuilder requires a model".into()))?;

        Ok(Arc::new(AgentLoop {
            name,
            description: self.description,
            model,
            system_prompt: self.system_prompt,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            max_budget: self.max_budget,
            output_schema: self.output_schema,
            max_schema_retries: self.max_schema_retries,
            behavior_prompts: self.behavior_prompts,
            context_builder: self.context_builder,
            tools: self.tools,
            sub_agents: self.sub_agents,
        }))
    }
}
