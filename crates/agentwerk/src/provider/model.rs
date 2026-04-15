/// How an agent specifies which model to use.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// A specific model ID (e.g., `"claude-sonnet-4-20250514"`).
    Exact(String),
    /// Use the parent agent's model.
    Inherit,
}

impl ModelSpec {
    pub fn parse(s: &str) -> Self {
        match s {
            "inherit" => Self::Inherit,
            other => Self::Exact(other.to_string()),
        }
    }

    pub fn resolve(&self, parent_model: &str) -> String {
        match self {
            Self::Exact(id) => id.clone(),
            Self::Inherit => parent_model.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_exact_returns_id() {
        assert_eq!(ModelSpec::Exact("custom".into()).resolve("parent"), "custom");
    }

    #[test]
    fn resolve_inherit_uses_parent() {
        assert_eq!(ModelSpec::Inherit.resolve("my-parent-model"), "my-parent-model");
    }

    #[test]
    fn parse_inherit() {
        assert!(matches!(ModelSpec::parse("inherit"), ModelSpec::Inherit));
    }

    #[test]
    fn parse_exact_model_id() {
        match ModelSpec::parse("claude-sonnet-4-20250514") {
            ModelSpec::Exact(id) => assert_eq!(id, "claude-sonnet-4-20250514"),
            other => panic!("Expected Exact, got {other:?}"),
        }
    }
}
