//! Static context-window table for Anthropic models.
//!
//! Values track Anthropic's published per-model limits. Update here when
//! a new Claude model ships or its window changes.

/// Context window for Claude models, in input tokens. Matches by canonical
/// substring of the model id. Returns `None` for unknown ids so the
/// compaction seam stays dormant instead of firing on a guess.
///
/// The `[1m]` suffix (used in the Anthropic 1M-context beta) is recognised
/// irrespective of the base model family.
pub fn context_window(model: &str) -> Option<u64> {
    let m = model.to_ascii_lowercase();
    if m.contains("[1m]") {
        return Some(1_000_000);
    }
    if m.contains("claude-opus-4")
        || m.contains("claude-sonnet-4")
        || m.contains("claude-haiku-4")
        || m.contains("claude-3-7-sonnet")
        || m.contains("claude-3-5-sonnet")
        || m.contains("claude-3-5-haiku")
        || m.contains("claude-3-opus")
        || m.contains("claude-3-sonnet")
        || m.contains("claude-3-haiku")
    {
        return Some(200_000);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_4_family_returns_200k() {
        assert_eq!(context_window("claude-sonnet-4-20250514"), Some(200_000));
        assert_eq!(context_window("claude-opus-4-20250101"), Some(200_000));
        assert_eq!(context_window("claude-haiku-4-5-20251001"), Some(200_000));
    }

    #[test]
    fn claude_3_family_returns_200k() {
        assert_eq!(context_window("claude-3-5-sonnet-20241022"), Some(200_000));
        assert_eq!(context_window("claude-3-opus-20240229"), Some(200_000));
    }

    #[test]
    fn one_million_suffix_overrides_base_family() {
        assert_eq!(
            context_window("claude-opus-4-7[1m]"),
            Some(1_000_000),
            "explicit [1m] opt-in promotes to 1M"
        );
        assert_eq!(
            context_window("claude-sonnet-4-20250514[1m]"),
            Some(1_000_000)
        );
    }

    #[test]
    fn unknown_models_return_none() {
        assert_eq!(context_window("gpt-4"), None);
        assert_eq!(context_window("mistral-large-2411"), None);
        assert_eq!(context_window("some-future-model"), None);
    }
}
