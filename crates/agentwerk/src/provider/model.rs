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
    /// Tokens set aside for the model's response. The context window holds
    /// input + output combined, so input must leave at least this much room
    /// for the next reply. Treated as an upper bound on one response.
    pub const RESERVED_RESPONSE_TOKENS: u64 = 20_000;

    /// Headroom reserved below the hard window limit so compaction has room
    /// to fire *and* finish before the real overflow. Also absorbs drift in
    /// the `bytes / 4` token estimate, which usually under-counts code and
    /// JSON.
    pub const COMPACTION_HEADROOM_TOKENS: u64 = 13_000;

    /// Build a `Model` by looking up the name in each provider's
    /// [`ModelLookup`] impl. Unknown names produce a `Model` with
    /// `context_window_size: None` — compaction stays dormant, no error.
    pub fn from_name(name: impl Into<String>) -> Self {
        let name = name.into();
        let context_window_size =
            <AnthropicProvider as ModelLookup>::lookup_context_window_size(&name)
                .or_else(|| <OpenAiProvider as ModelLookup>::lookup_context_window_size(&name))
                .or_else(|| <MistralProvider as ModelLookup>::lookup_context_window_size(&name));
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

    /// Token count at which proactive compaction fires. Returns `None` when
    /// the model's context window size is unknown — callers treat that as
    /// "no threshold; compaction is dormant".
    pub fn compact_threshold(&self) -> Option<u64> {
        self.context_window_size.map(|size| {
            size.saturating_sub(Self::RESERVED_RESPONSE_TOKENS)
                .saturating_sub(Self::COMPACTION_HEADROOM_TOKENS)
        })
    }
}

/// Model-name knowledge, implemented by each provider for the families it owns.
///
/// Associated function (no `&self`) because the lookup is static — it maps
/// a model name to its published context window size, which doesn't depend
/// on any provider instance.
pub trait ModelLookup {
    fn lookup_context_window_size(name: &str) -> Option<u64>;
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

    #[test]
    fn compact_threshold_200k_model() {
        let m = Model::from_name("unknown").context_window_size(200_000);
        assert_eq!(m.compact_threshold(), Some(167_000));
    }

    #[test]
    fn compact_threshold_saturates_on_tiny_window() {
        let tiny = Model::from_name("unknown").context_window_size(100);
        let zero = Model::from_name("unknown").context_window_size(0);
        assert_eq!(tiny.compact_threshold(), Some(0));
        assert_eq!(zero.compact_threshold(), Some(0));
    }

    #[test]
    fn compact_threshold_none_for_unknown_window() {
        assert_eq!(Model::from_name("unknown").compact_threshold(), None);
    }
}
