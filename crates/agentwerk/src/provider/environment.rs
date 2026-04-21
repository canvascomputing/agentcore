//! Picks a provider from environment variables, so callers can say `Provider::from_env()` without coding the detection matrix themselves.

use std::sync::Arc;

use crate::error::{AgenticError, Result};

use super::{AnthropicProvider, LiteLlmProvider, MistralProvider, OpenAiProvider, Provider};

/// Detected provider name, before constructing the actual provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DetectedProvider {
    Anthropic,
    Mistral,
    OpenAi,
    LiteLlm,
}

/// Read an env var, treating empty values as unset. Returns `default` if missing or empty.
pub(crate) fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.into())
}

/// Read a required env var. Returns `Err` if missing or empty.
pub(crate) fn env_required(name: &str) -> Result<String> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| AgenticError::Other(format!("{name} environment variable not set")))
}

/// Detect an LLM provider from environment variables and return `(provider, model)`.
///
/// Detection order:
///   0. `LITELLM_PROVIDER` → explicit selection (`anthropic`, `mistral`, `openai`, `litellm`)
///   1. `LITELLM_API_KEY`  → LiteLLM proxy (URL from `LITELLM_BASE_URL`, default `http://localhost:4000`)
///   2. `MISTRAL_API_KEY`  → Mistral
///   3. `ANTHROPIC_API_KEY` → Anthropic
///   4. `OPENAI_API_KEY`   → OpenAI
///
/// Empty env vars are treated as unset.
pub fn from_env() -> Result<(Arc<dyn Provider>, String)> {
    let detected = detect_provider_name(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))?;

    let (provider, model): (Arc<dyn Provider>, String) = match detected {
        DetectedProvider::Anthropic => {
            let (p, m) = AnthropicProvider::from_env_with_model()?;
            (Arc::new(p), m)
        }
        DetectedProvider::Mistral => {
            let (p, m) = MistralProvider::from_env_with_model()?;
            (Arc::new(p), m)
        }
        DetectedProvider::OpenAi => {
            let (p, m) = OpenAiProvider::from_env_with_model()?;
            (Arc::new(p), m)
        }
        DetectedProvider::LiteLlm => {
            let (p, m) = LiteLlmProvider::from_env_with_model()?;
            (Arc::new(p), m)
        }
    };
    Ok((provider, model))
}

/// Pure detection logic. Determines which provider to use based on env var values.
pub(crate) fn detect_provider_name<F>(get_env: F) -> Result<DetectedProvider>
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(name) = get_env("LITELLM_PROVIDER") {
        return match name.as_str() {
            "anthropic" => Ok(DetectedProvider::Anthropic),
            "mistral" => Ok(DetectedProvider::Mistral),
            "openai" => Ok(DetectedProvider::OpenAi),
            "litellm" => Ok(DetectedProvider::LiteLlm),
            other => Err(AgenticError::Other(format!(
                "Unknown LITELLM_PROVIDER \"{other}\". Supported: anthropic, mistral, openai, litellm"
            ))),
        };
    }

    if get_env("LITELLM_API_KEY").is_some() {
        return Ok(DetectedProvider::LiteLlm);
    }

    if get_env("MISTRAL_API_KEY").is_some() {
        return Ok(DetectedProvider::Mistral);
    }

    if get_env("ANTHROPIC_API_KEY").is_some() {
        return Ok(DetectedProvider::Anthropic);
    }

    if get_env("OPENAI_API_KEY").is_some() {
        return Ok(DetectedProvider::OpenAi);
    }

    Err(AgenticError::Other(
        "No LLM provider found. \
         Set one of: LITELLM_PROVIDER, LITELLM_API_KEY, MISTRAL_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_map<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
                .filter(|v| !v.is_empty())
        }
    }

    #[test]
    fn explicit_anthropic() {
        let result = detect_provider_name(env_map(&[
            ("LITELLM_PROVIDER", "anthropic"),
            ("ANTHROPIC_API_KEY", "key"),
        ]))
        .unwrap();
        assert_eq!(result, DetectedProvider::Anthropic);
    }

    #[test]
    fn explicit_mistral() {
        let result = detect_provider_name(env_map(&[
            ("LITELLM_PROVIDER", "mistral"),
            ("MISTRAL_API_KEY", "key"),
        ]))
        .unwrap();
        assert_eq!(result, DetectedProvider::Mistral);
    }

    #[test]
    fn explicit_openai() {
        let result = detect_provider_name(env_map(&[
            ("LITELLM_PROVIDER", "openai"),
            ("OPENAI_API_KEY", "key"),
        ]))
        .unwrap();
        assert_eq!(result, DetectedProvider::OpenAi);
    }

    #[test]
    fn explicit_litellm() {
        let result = detect_provider_name(env_map(&[("LITELLM_PROVIDER", "litellm")])).unwrap();
        assert_eq!(result, DetectedProvider::LiteLlm);
    }

    #[test]
    fn explicit_overrides_auto_detection() {
        let result = detect_provider_name(env_map(&[
            ("LITELLM_PROVIDER", "anthropic"),
            ("ANTHROPIC_API_KEY", "key"),
            ("OPENAI_API_KEY", "key"),
        ]))
        .unwrap();
        assert_eq!(result, DetectedProvider::Anthropic);
    }

    #[test]
    fn auto_litellm_api_key() {
        let result = detect_provider_name(env_map(&[("LITELLM_API_KEY", "key")])).unwrap();
        assert_eq!(result, DetectedProvider::LiteLlm);
    }

    #[test]
    fn auto_mistral() {
        let result = detect_provider_name(env_map(&[("MISTRAL_API_KEY", "key")])).unwrap();
        assert_eq!(result, DetectedProvider::Mistral);
    }

    #[test]
    fn auto_anthropic() {
        let result = detect_provider_name(env_map(&[("ANTHROPIC_API_KEY", "key")])).unwrap();
        assert_eq!(result, DetectedProvider::Anthropic);
    }

    #[test]
    fn auto_openai() {
        let result = detect_provider_name(env_map(&[("OPENAI_API_KEY", "key")])).unwrap();
        assert_eq!(result, DetectedProvider::OpenAi);
    }

    #[test]
    fn litellm_key_wins_over_others() {
        let result = detect_provider_name(env_map(&[
            ("LITELLM_API_KEY", "key"),
            ("MISTRAL_API_KEY", "key"),
            ("ANTHROPIC_API_KEY", "key"),
        ]))
        .unwrap();
        assert_eq!(result, DetectedProvider::LiteLlm);
    }

    #[test]
    fn mistral_wins_over_anthropic() {
        let result = detect_provider_name(env_map(&[
            ("MISTRAL_API_KEY", "key"),
            ("ANTHROPIC_API_KEY", "key"),
        ]))
        .unwrap();
        assert_eq!(result, DetectedProvider::Mistral);
    }

    #[test]
    fn invalid_provider_returns_error() {
        let err = detect_provider_name(env_map(&[
            ("LITELLM_PROVIDER", "invalid"),
            ("ANTHROPIC_API_KEY", "key"),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("Unknown LITELLM_PROVIDER"));
    }

    #[test]
    fn no_provider_returns_error() {
        let err = detect_provider_name(env_map(&[])).unwrap_err();
        assert!(err.to_string().contains("No LLM provider found"));
    }

    #[test]
    fn empty_values_treated_as_unset() {
        let err = detect_provider_name(env_map(&[("ANTHROPIC_API_KEY", "")])).unwrap_err();
        assert!(err.to_string().contains("No LLM provider found"));
    }
}
