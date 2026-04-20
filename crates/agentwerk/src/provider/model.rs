use super::{AnthropicProvider, MistralProvider, OpenAiProvider};

/// Model metadata: the id plus anything we know about its capabilities.
///
/// Built by [`Model::from_id`] (registry-backed) or
/// [`Model::with_context_window_size`] (explicit override). `Model` is what
/// `ModelSpec::Exact` stores, what `LoopSpec.model` holds at runtime, and
/// what the compaction seams read to decide when to fire.
#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
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

    /// Build a `Model` by looking up the id in each provider's
    /// [`ModelLookup`] impl. Unknown ids produce a `Model` with
    /// `context_window_size: None` — compaction stays dormant, no error.
    pub fn from_id(id: impl Into<String>) -> Self {
        let id = id.into();
        let context_window_size =
            <AnthropicProvider as ModelLookup>::lookup_context_window_size(&id)
                .or_else(|| <OpenAiProvider as ModelLookup>::lookup_context_window_size(&id))
                .or_else(|| <MistralProvider as ModelLookup>::lookup_context_window_size(&id));
        Self {
            id,
            context_window_size,
        }
    }

    /// Explicit override — skips the registry. Useful for local proxies or
    /// private deployments whose id isn't in any provider's table.
    pub fn with_context_window_size(mut self, size: Option<u64>) -> Self {
        self.context_window_size = size;
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

/// How an agent specifies which model to use.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// A specific model (id + metadata).
    Exact(Model),
    /// Use the parent agent's model.
    Inherit,
}

impl ModelSpec {
    /// Resolve to a concrete `Model`. `Inherit` clones the parent's full
    /// metadata (id *and* `context_window_size`) — the child inherits both
    /// pieces, not just the id.
    pub fn resolve(&self, parent: &Model) -> Model {
        match self {
            Self::Exact(m) => m.clone(),
            Self::Inherit => parent.clone(),
        }
    }
}

/// Model-id knowledge, implemented by each provider for the families it owns.
///
/// Associated function (no `&self`) because the lookup is static — it maps
/// a model id to its published context window size, which doesn't depend
/// on any provider instance.
pub trait ModelLookup {
    fn lookup_context_window_size(id: &str) -> Option<u64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_id_resolves_claude_models() {
        assert_eq!(
            Model::from_id("claude-sonnet-4-20250514").context_window_size,
            Some(200_000)
        );
    }

    #[test]
    fn from_id_resolves_openai_models() {
        assert_eq!(Model::from_id("gpt-5").context_window_size, Some(400_000));
        assert_eq!(Model::from_id("gpt-4o").context_window_size, Some(128_000));
    }

    #[test]
    fn from_id_resolves_mistral_models() {
        assert_eq!(
            Model::from_id("mistral-large-2411").context_window_size,
            Some(131_072)
        );
    }

    #[test]
    fn from_id_unknown_has_no_context_window_size() {
        assert_eq!(Model::from_id("unknown").context_window_size, None);
        assert_eq!(Model::from_id("mock").context_window_size, None);
    }

    #[test]
    fn with_context_window_size_overrides() {
        let m = Model::from_id("unknown").with_context_window_size(Some(50_000));
        assert_eq!(m.context_window_size, Some(50_000));
    }

    #[test]
    fn resolve_exact_clones_the_model() {
        let parent = Model::from_id("claude-sonnet-4-20250514");
        let child = ModelSpec::Exact(Model::from_id("gpt-5")).resolve(&parent);
        assert_eq!(child.id, "gpt-5");
        assert_eq!(child.context_window_size, Some(400_000));
    }

    #[test]
    fn resolve_inherit_propagates_parent_model_and_window() {
        let parent = Model::from_id("claude-sonnet-4-20250514");
        let child = ModelSpec::Inherit.resolve(&parent);
        assert_eq!(child.id, "claude-sonnet-4-20250514");
        assert_eq!(child.context_window_size, Some(200_000));
    }

    #[test]
    fn compact_threshold_200k_model() {
        let m = Model::from_id("unknown").with_context_window_size(Some(200_000));
        assert_eq!(m.compact_threshold(), Some(167_000));
    }

    #[test]
    fn compact_threshold_saturates_on_tiny_window() {
        let tiny = Model::from_id("unknown").with_context_window_size(Some(100));
        let zero = Model::from_id("unknown").with_context_window_size(Some(0));
        assert_eq!(tiny.compact_threshold(), Some(0));
        assert_eq!(zero.compact_threshold(), Some(0));
    }

    #[test]
    fn compact_threshold_none_for_unknown_window() {
        assert_eq!(Model::from_id("unknown").compact_threshold(), None);
    }
}
