use std::sync::Arc;

use agent::{
    AnthropicProvider, LiteLlmProvider, LlmProvider, MistralProvider, default_transport,
};

/// Read an environment variable, treating empty strings as unset.
pub fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Read an environment variable with a fallback default.
pub fn env_or(name: &str, default: &str) -> String {
    env(name).unwrap_or_else(|| default.into())
}

/// Auto-detect an LLM provider from environment variables.
///
/// Detection order:
///   1. `LITELLM_API_URL` → LiteLLM proxy
///   2. `MISTRAL_API_KEY` → Mistral API
///   3. `ANTHROPIC_API_KEY` → Anthropic API
///   4. localhost:4000 reachable → LiteLLM proxy (no key needed)
///
/// Returns `(provider, model)`. Exits with an error if no provider is found.
pub fn auto_detect_provider() -> (Arc<dyn LlmProvider>, String) {
    let transport = default_transport();

    if let Some(url) = env("LITELLM_API_URL") {
        let key = env_or("LITELLM_API_KEY", "");
        let model = env_or("LITELLM_MODEL", "claude-sonnet-4-20250514");
        return (Arc::new(LiteLlmProvider::new(key, transport).base_url(url)), model);
    }

    if let Some(key) = env("MISTRAL_API_KEY") {
        let mut p = MistralProvider::new(key, transport);
        if let Some(url) = env("MISTRAL_BASE_URL") {
            p = p.base_url(url);
        }
        let model = env_or("MISTRAL_MODEL", "mistral-medium-2508");
        return (Arc::new(p), model);
    }

    if let Some(key) = env("ANTHROPIC_API_KEY") {
        let mut p = AnthropicProvider::new(key, transport);
        if let Some(url) = env("ANTHROPIC_BASE_URL") {
            p = p.base_url(url);
        }
        let model = env_or("ANTHROPIC_MODEL", "claude-sonnet-4-20250514");
        return (Arc::new(p), model);
    }

    if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
        let key = env_or("LITELLM_API_KEY", "");
        let model = env_or("LITELLM_MODEL", "claude-sonnet-4-20250514");
        return (
            Arc::new(LiteLlmProvider::new(key, transport).base_url("http://localhost:4000".into())),
            model,
        );
    }

    eprintln!("Error: No LLM provider found.");
    eprintln!("Set one of: ANTHROPIC_API_KEY, MISTRAL_API_KEY, LITELLM_API_URL");
    eprintln!("Or start a LiteLLM proxy on localhost:4000");
    std::process::exit(1);
}
