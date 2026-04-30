//! View-model for one section of an assembled prompt: a body of markdown plus an optional `## Heading` the builder injects at render time.

use std::borrow::Cow;

/// One section of an assembled prompt. Knows whether it should render under a
/// `## Heading` (Context) or as bare body (Role, Behavior, Directive). Keeping
/// this concern here means the source `.md` files contain only body content —
/// the structural markdown is added by the builder.
#[derive(Debug, Clone)]
pub(crate) struct Section {
    pub heading: Option<&'static str>,
    pub body: Cow<'static, str>,
}

impl Section {
    pub fn role(body: impl Into<Cow<'static, str>>) -> Self {
        Self {
            heading: None,
            body: body.into(),
        }
    }

    pub fn behavior(body: impl Into<Cow<'static, str>>) -> Self {
        Self {
            heading: None,
            body: body.into(),
        }
    }

    pub fn context(body: impl Into<Cow<'static, str>>) -> Self {
        Self {
            heading: Some("Context"),
            body: body.into(),
        }
    }

    #[allow(dead_code)]
    pub fn work(body: impl Into<Cow<'static, str>>) -> Self {
        Self {
            heading: None,
            body: body.into(),
        }
    }

    #[allow(dead_code)]
    pub fn directive(body: impl Into<Cow<'static, str>>) -> Self {
        Self {
            heading: None,
            body: body.into(),
        }
    }

    /// Trim leading and trailing newlines off the body so the builder controls
    /// section spacing — empty leading/trailing lines in source files do not
    /// leak into the final prompt.
    pub fn render(&self) -> String {
        let body = self.body.trim_matches('\n');
        match self.heading {
            Some(h) => format!("## {h}\n\n{body}"),
            None => body.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_renders_body_without_heading() {
        let s = Section::role("You are a senior reviewer.");
        assert_eq!(s.render(), "You are a senior reviewer.");
    }

    #[test]
    fn context_wraps_body_in_h2_heading() {
        let s = Section::context("- Working directory: /tmp/test");
        assert_eq!(s.render(), "## Context\n\n- Working directory: /tmp/test");
    }

    #[test]
    fn surrounding_newlines_in_body_are_trimmed() {
        let s = Section::behavior("\n\n- MUST do the thing.\n\n");
        assert_eq!(s.render(), "- MUST do the thing.");
    }
}
