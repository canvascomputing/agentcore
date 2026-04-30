//! Agent: identity + prompt parts + provider/model. Constructed
//! independently of any ticket queue and paired with one via
//! `TicketSystem::assign`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::prompts::{PromptBuilder, Section, DEFAULT_BEHAVIOR};
use crate::providers::{Provider, ProviderToolDefinition};
use crate::tools::{ToolLike, ToolRegistry};

use super::r#loop::Runnable;
use super::tickets::TicketSystem;

#[derive(Default, Clone)]
pub struct Agent {
    name: String,
    provider: Option<Arc<dyn Provider>>,
    model: Option<String>,
    role: Option<String>,
    behavior: Option<String>,
    context: Option<String>,
    ticket_types: Vec<String>,
    tools: ToolRegistry,
    working_dir: Option<PathBuf>,
}

impl Agent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, n: impl Into<String>) -> Self {
        self.name = n.into();
        self
    }

    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }

    pub fn role(mut self, r: impl Into<String>) -> Self {
        self.role = Some(r.into());
        self
    }

    pub fn behavior(mut self, b: impl Into<String>) -> Self {
        self.behavior = Some(b.into());
        self
    }

    pub fn context(mut self, c: impl Into<String>) -> Self {
        self.context = Some(c.into());
        self
    }

    /// Restrict the agent to handle only tickets whose type matches
    /// one of the strings supplied. Calling more than once accumulates;
    /// an agent with no calls handles every type.
    pub fn ticket_type(mut self, t: impl Into<String>) -> Self {
        self.ticket_types.push(t.into());
        self
    }

    /// Register a tool the agent may call.
    pub fn tool(mut self, tool: impl ToolLike + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    /// Working directory tools resolve filesystem paths against. Defaults
    /// to the process's current directory when unset.
    pub fn working_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(p.into());
        self
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Empty allow-list = handle any type.
    pub fn handles(&self, ticket_type: &str) -> bool {
        self.ticket_types.is_empty()
            || self.ticket_types.iter().any(|t| t == ticket_type)
    }

    pub(super) fn handles_any_type(&self) -> bool {
        self.ticket_types.is_empty()
    }

    pub(super) fn allowed_ticket_types(&self) -> &[String] {
        &self.ticket_types
    }

    pub(super) fn tool_definitions(&self) -> Vec<ProviderToolDefinition> {
        self.tools.definitions()
    }

    pub(super) fn tool_registry(&self) -> &ToolRegistry {
        &self.tools
    }

    pub(super) fn working_dir_or_default(&self) -> PathBuf {
        self.working_dir.clone().unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        })
    }

    pub(super) fn provider_handle(&self) -> Arc<dyn Provider> {
        Arc::clone(
            self.provider
                .as_ref()
                .expect("Agent::run requires .provider(...) to be set"),
        )
    }

    pub(super) fn model_str(&self) -> &str {
        self.model
            .as_deref()
            .expect("Agent::run requires .model(...) to be set")
    }

    pub(super) fn system_prompt(&self) -> String {
        let mut b = PromptBuilder::default();
        if let Some(role) = &self.role {
            b = b.role(role.clone());
        }
        let behavior = self
            .behavior
            .clone()
            .unwrap_or_else(|| DEFAULT_BEHAVIOR.to_string());
        if !behavior.is_empty() {
            b = b.behavior(behavior);
        }
        b.build().system
    }

    /// Render the agent's `context` (if set) as a `## Context\n\n…`
    /// section, ready to be pushed as the first user message in the
    /// loop. Returns `None` when no context was configured.
    pub(super) fn context_message(&self) -> Option<String> {
        self.context
            .as_ref()
            .map(|body| Section::context(body.clone()).render())
    }

    /// Sugar: build a default `TicketSystem`, add one Todo ticket
    /// carrying `task`, drive this agent until the queue settles,
    /// return the final `TicketSystem`. Panics if `provider` or
    /// `model` is missing.
    pub async fn run(self, task: impl Into<String>) -> TicketSystem {
        let task = task.into();
        let reporter = if self.name.is_empty() {
            "user".to_string()
        } else {
            self.name.clone()
        };
        let mut tickets = TicketSystem::new();
        tickets.create(task.clone(), task, "task", reporter);
        tickets.assign(self).run_until_empty().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_returns_true_when_allow_list_empty() {
        let agent = Agent::new();
        assert!(agent.handles("anything"));
        assert!(agent.handles("task"));
        assert!(agent.handles(""));
    }

    #[test]
    fn handles_filters_by_configured_types() {
        let agent = Agent::new().ticket_type("task").ticket_type("research");
        assert!(agent.handles("task"));
        assert!(agent.handles("research"));
        assert!(!agent.handles("bug"));
        assert!(!agent.handles("review"));
    }

    #[test]
    fn get_name_returns_configured_name() {
        let agent = Agent::new().name("alice");
        assert_eq!(agent.get_name(), "alice");
    }

    #[test]
    fn default_get_name_is_empty() {
        let agent = Agent::new();
        assert_eq!(agent.get_name(), "");
    }

    #[test]
    fn context_message_returns_none_when_unset() {
        let agent = Agent::new().role("R");
        assert!(agent.context_message().is_none());
    }

    #[test]
    fn context_message_renders_h2_heading_when_set() {
        let agent = Agent::new().context("- Working directory: /tmp");
        assert_eq!(
            agent.context_message().as_deref(),
            Some("## Context\n\n- Working directory: /tmp"),
        );
    }

    #[test]
    fn system_prompt_does_not_include_context() {
        let agent = Agent::new()
            .role("ROLE")
            .behavior("BEH")
            .context("CTX");
        let prompt = agent.system_prompt();
        assert!(prompt.contains("ROLE"));
        assert!(prompt.contains("BEH"));
        assert!(!prompt.contains("CTX"));
        assert!(!prompt.contains("## Context"));
    }

    #[test]
    fn system_prompt_uses_default_behavior_when_unset() {
        let agent = Agent::new().role("ROLE");
        let prompt = agent.system_prompt();
        assert!(prompt.contains("ROLE"));
        assert!(prompt.contains(DEFAULT_BEHAVIOR.trim()));
    }
}
