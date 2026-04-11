use std::sync::Arc;

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::prompt::PromptBuilder;
use crate::tool::{Tool, ToolRegistry};

use super::r#loop::LlmAgent;
use super::output::OutputSchema;
use super::Agent;

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
    prompt_builder: Option<PromptBuilder>,
    tools: ToolRegistry,
    sub_agents: Vec<Arc<dyn Agent>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
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
            prompt_builder: None,
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

    pub fn prompt_builder(mut self, pb: PromptBuilder) -> Self {
        self.prompt_builder = Some(pb);
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

        Ok(Arc::new(LlmAgent {
            name,
            description: self.description,
            model,
            system_prompt: self.system_prompt,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            max_budget: self.max_budget,
            output_schema: self.output_schema,
            max_schema_retries: self.max_schema_retries,
            prompt_builder: self.prompt_builder,
            tools: self.tools,
            sub_agents: self.sub_agents,
        }))
    }
}
