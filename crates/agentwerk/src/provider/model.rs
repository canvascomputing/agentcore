//! Per-model knowledge of context window size and compaction thresholds. The loop consults this to decide when a conversation must be shrunk.

use super::{AnthropicProvider, MistralProvider, OpenAiProvider};

/// Model metadata: the name plus anything we know about its capabilities.
///
/// Built by [`Model::from_name`] (registry-backed) or
/// [`Model::context_window_size`] (explicit override). `Model` is what
/// `AgentSpec.model` holds at runtime and what the compaction seams read to
/// decide when to fire. Agents express "inherit from parent" via
/// `AgentSpec.model: Option<Model>` (`None` = inherit) — there's no separate
/// spec enum.
#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub context_window_size: Option<u64>,
}

impl Model {
    /// Build a `Model` by asking each provider in turn for its known context
    /// window. Unknown names produce a `Model` with `context_window_size:
    /// None` — compaction stays dormant, no error.
    pub fn from_name(name: impl Into<String>) -> Self {
        let name = name.into();
        let context_window_size = AnthropicProvider::lookup_context_window_size(&name)
            .or_else(|| OpenAiProvider::lookup_context_window_size(&name))
            .or_else(|| MistralProvider::lookup_context_window_size(&name));
        Self {
            name,
            context_window_size,
        }
    }

    /// Explicit override — skips the registry. Useful for local proxies or
    /// private deployments whose name isn't in any provider's table.
    pub fn context_window_size(mut self, size: u64) -> Self {
        self.context_window_size = Some(size);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_resolves_claude_models() {
        assert_eq!(
            Model::from_name("claude-sonnet-4-20250514").context_window_size,
            Some(200_000)
        );
    }

    #[test]
    fn from_name_resolves_openai_models() {
        assert_eq!(Model::from_name("gpt-5").context_window_size, Some(400_000));
        assert_eq!(
            Model::from_name("gpt-4o").context_window_size,
            Some(128_000)
        );
    }

    #[test]
    fn from_name_resolves_mistral_models() {
        assert_eq!(
            Model::from_name("mistral-large-2411").context_window_size,
            Some(131_072)
        );
    }

    #[test]
    fn from_name_unknown_has_no_context_window_size() {
        assert_eq!(Model::from_name("unknown").context_window_size, None);
        assert_eq!(Model::from_name("mock").context_window_size, None);
    }

    #[test]
    fn context_window_size_overrides() {
        let m = Model::from_name("unknown").context_window_size(50_000);
        assert_eq!(m.context_window_size, Some(50_000));
    }
}
