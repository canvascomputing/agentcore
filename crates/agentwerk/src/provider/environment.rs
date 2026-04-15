use std::sync::Arc;

use super::{AnthropicProvider, LiteLlmProvider, LlmProvider, MistralProvider, OpenAiProvider};

/// Provider and model detected from environment variables.
pub struct Environment {
    pub provider: Arc<dyn LlmProvider>,
    pub model: String,
}

impl Environment {
    /// Detect an LLM provider from environment variables.
    ///
    /// Detection order:
    ///   1. `LITELLM_API_KEY`  → LiteLLM proxy (URL from `LITELLM_API_URL`, default `http://localhost:4000`)
    ///   2. `MISTRAL_API_KEY`  → Mistral
    ///   3. `ANTHROPIC_API_KEY` → Anthropic
    ///   4. `OPENAI_API_KEY`   → OpenAI
    ///   5. localhost:4000 reachable → LiteLLM fallback (no auth)
    ///
    /// Empty env vars are treated as unset. Panics if no provider can be detected.
    pub fn detect_provider() -> Self {
        if env("LITELLM_API_KEY").is_some() {
            let (provider, model) = LiteLlmProvider::from_env();
            return Self { provider: Arc::new(provider), model };
        }

        if let Ok((provider, model)) = MistralProvider::from_env() {
            return Self { provider: Arc::new(provider), model };
        }

        if let Ok((provider, model)) = AnthropicProvider::from_env() {
            return Self { provider: Arc::new(provider), model };
        }

        if let Ok((provider, model)) = OpenAiProvider::from_env() {
            return Self { provider: Arc::new(provider), model };
        }

        if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
            let (provider, model) = LiteLlmProvider::from_env();
            return Self { provider: Arc::new(provider), model };
        }

        panic!(
            "No LLM provider found.\n\
             Set one of: LITELLM_API_KEY, MISTRAL_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY\n\
             Or start a LiteLLM proxy on localhost:4000"
        );
    }
}

/// Read an environment variable, treating empty strings as unset.
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

