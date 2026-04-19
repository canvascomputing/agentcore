//! Static context-window table for OpenAI-compatible models.
//!
//! Covers OpenAI's own models plus the Mistral model ids — both speak the
//! same OpenAI wire format, so [`OpenAiProvider`] and [`MistralProvider`]
//! share this lookup. Update here when a new model ships or its window
//! changes.

/// Context window for OpenAI-compatible models, in input tokens. Matches
/// by canonical substring of the model id. Returns `None` for unknown ids
/// so the compaction seam stays dormant instead of firing on a guess.
pub fn context_window(model: &str) -> Option<u64> {
    let m = model.to_ascii_lowercase();

    // OpenAI — newest first so substrings like "gpt-4" don't shadow "gpt-4.1"
    if m.contains("gpt-4.1") || m.contains("gpt-5") || m.starts_with("o3") || m.starts_with("o1") {
        return Some(200_000);
    }
    if m.contains("gpt-4o") || m.contains("gpt-4-turbo") {
        return Some(128_000);
    }
    if m.contains("gpt-4-32k") {
        return Some(32_768);
    }
    if m.contains("gpt-4") {
        return Some(8_192);
    }
    if m.contains("gpt-3.5-turbo-16k") {
        return Some(16_385);
    }
    if m.contains("gpt-3.5-turbo") {
        return Some(16_385);
    }

    // Mistral — served via Mistral's own endpoint or via LiteLLM.
    if m.contains("mistral-large") || m.contains("codestral") {
        return Some(131_072);
    }
    if m.contains("mistral-medium") || m.contains("mistral-small") {
        return Some(32_768);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_5_family_returns_200k() {
        assert_eq!(context_window("gpt-5"), Some(200_000));
        assert_eq!(context_window("gpt-5-mini"), Some(200_000));
    }

    #[test]
    fn gpt_4_1_and_o_series_return_200k() {
        assert_eq!(context_window("gpt-4.1"), Some(200_000));
        assert_eq!(context_window("o3-mini"), Some(200_000));
        assert_eq!(context_window("o1-preview"), Some(200_000));
    }

    #[test]
    fn gpt_4o_and_turbo_return_128k() {
        assert_eq!(context_window("gpt-4o"), Some(128_000));
        assert_eq!(context_window("gpt-4o-mini"), Some(128_000));
        assert_eq!(context_window("gpt-4-turbo-2024-04-09"), Some(128_000));
    }

    #[test]
    fn legacy_gpt_4_returns_8k() {
        assert_eq!(context_window("gpt-4"), Some(8_192));
        assert_eq!(context_window("gpt-4-32k"), Some(32_768));
    }

    #[test]
    fn mistral_family_returns_published_windows() {
        assert_eq!(context_window("mistral-large-2411"), Some(131_072));
        assert_eq!(context_window("mistral-medium-2508"), Some(32_768));
        assert_eq!(context_window("mistral-small-latest"), Some(32_768));
        assert_eq!(context_window("codestral-latest"), Some(131_072));
    }

    #[test]
    fn unknown_models_return_none() {
        assert_eq!(context_window("claude-sonnet-4-20250514"), None);
        assert_eq!(context_window("llama-3-70b"), None);
        assert_eq!(context_window("unknown-model"), None);
    }
}
