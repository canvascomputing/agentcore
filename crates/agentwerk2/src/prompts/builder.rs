//! Builder that assembles a prompt envelope from typed sections, mirroring `Agent::context/role/behavior/work` and the canonical Context → Role → Behavior → Tools → Task order.

use std::borrow::Cow;

use super::section::Section;

/// Assembled prompt envelope. Field order follows the canonical spec
/// section order: context first, then the system message (role + behavior +
/// appended directives), then work. Tools are not present here — they reach
/// the model as structured data via the registry, not as a section in the
/// prompt envelope.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct Prompt {
    /// First user message; `None` means "no context block sent".
    pub context: Option<String>,
    /// Assembled system message: role + behavior + appended directives,
    /// joined by blank lines. May be empty when neither role nor behavior
    /// is set.
    pub system: String,
    /// Task message; `None` means the caller will supply a task later.
    pub work: Option<String>,
}

/// Inverse of `Agent::{context, role, behavior, work}`. Owns spacing rules
/// and the canonical section order; call sites never concatenate strings by
/// hand.
#[derive(Default)]
pub struct PromptBuilder {
    context: Option<Section>,
    role: Option<Section>,
    behavior: Option<Section>,
    work: Option<Section>,
    directives: Vec<Section>,
}

impl PromptBuilder {
    #[allow(dead_code)]
    pub fn context(mut self, body: impl Into<Cow<'static, str>>) -> Self {
        self.context = Some(Section::context(body));
        self
    }

    pub fn role(mut self, body: impl Into<Cow<'static, str>>) -> Self {
        self.role = Some(Section::role(body));
        self
    }

    pub fn behavior(mut self, body: impl Into<Cow<'static, str>>) -> Self {
        self.behavior = Some(Section::behavior(body));
        self
    }

    #[allow(dead_code)]
    pub fn work(mut self, body: impl Into<Cow<'static, str>>) -> Self {
        self.work = Some(Section::work(body));
        self
    }

    /// Append a directive (e.g. structured-output instruction) to the
    /// system message after role and behavior.
    #[allow(dead_code)]
    pub fn append_directive(mut self, body: impl Into<Cow<'static, str>>) -> Self {
        self.directives.push(Section::directive(body));
        self
    }

    pub fn build(self) -> Prompt {
        let context = self.context.map(|s| s.render()).filter(|s| !s.is_empty());
        let work = self.work.map(|s| s.render()).filter(|s| !s.is_empty());

        let mut system_parts: Vec<String> = Vec::new();
        if let Some(role) = self.role {
            let r = role.render();
            if !r.is_empty() {
                system_parts.push(r);
            }
        }
        if let Some(behavior) = self.behavior {
            let b = behavior.render();
            if !b.is_empty() {
                system_parts.push(b);
            }
        }
        for d in self.directives {
            let r = d.render();
            if !r.is_empty() {
                system_parts.push(r);
            }
        }

        Prompt {
            context,
            system: system_parts.join("\n\n"),
            work,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_alone_produces_role_in_system_only() {
        let p = PromptBuilder::default()
            .role("You are a senior reviewer.")
            .build();
        assert_eq!(p.system, "You are a senior reviewer.");
        assert!(p.context.is_none());
        assert!(p.work.is_none());
    }

    #[test]
    fn role_and_behavior_join_with_blank_line() {
        let p = PromptBuilder::default()
            .role("You are a senior reviewer.")
            .behavior("- MUST cite file:line.")
            .build();
        assert_eq!(
            p.system,
            "You are a senior reviewer.\n\n- MUST cite file:line."
        );
    }

    #[test]
    fn empty_behavior_is_skipped() {
        let p = PromptBuilder::default()
            .role("You are a senior reviewer.")
            .behavior("")
            .build();
        assert_eq!(p.system, "You are a senior reviewer.");
    }

    #[test]
    fn directive_appends_after_role_and_behavior() {
        let p = PromptBuilder::default()
            .role("You answer with JSON.")
            .behavior("")
            .append_directive("- MUST return JSON.")
            .build();
        assert_eq!(p.system, "You answer with JSON.\n\n- MUST return JSON.");
    }

    #[test]
    fn context_renders_with_h2_heading() {
        let p = PromptBuilder::default()
            .context("- Working directory: /tmp/test")
            .role("R")
            .build();
        assert_eq!(
            p.context.as_deref(),
            Some("## Context\n\n- Working directory: /tmp/test"),
        );
        assert_eq!(p.system, "R");
    }

    #[test]
    fn work_renders_bare() {
        let p = PromptBuilder::default()
            .role("R")
            .work("Review the auth module.")
            .build();
        assert_eq!(p.work.as_deref(), Some("Review the auth module."));
    }

    #[test]
    fn full_envelope_in_canonical_order() {
        let p = PromptBuilder::default()
            .context("C")
            .role("R")
            .behavior("B")
            .work("W")
            .build();
        assert_eq!(p.context.as_deref(), Some("## Context\n\nC"));
        assert_eq!(p.system, "R\n\nB");
        assert_eq!(p.work.as_deref(), Some("W"));
    }
}
