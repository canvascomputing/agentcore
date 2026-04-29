//! Single-file tool definition: typed sections (summary, constraints, anti-patterns, cautions, output) plus `read_only` and the JSON Schema, deserialized once and rendered to markdown for the LLM.

use serde::Deserialize;
use serde_json::Value;

/// Typed view of a tool's `.tool.json` file. Every prose field is a list of
/// strings — the renderer joins them with whatever separator the target
/// format requires (newlines for cautions, blank lines for paragraphs,
/// bullets for lists). Authors edit content; the framework owns formatting.
///
/// The shape is the public contract for `.tool.json` files: a tool can be
/// authored as JSON and turned into a [`Tool`](crate::Tool) via
/// [`Tool::from_tool_file`](crate::Tool::from_tool_file). The 12 built-in
/// tools all ship as `{name}.tool.json` files alongside their Rust sources.
#[derive(Debug, Deserialize)]
pub struct ToolFile {
    pub name: String,
    #[serde(default)]
    pub summary: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub anti_patterns: Vec<String>,
    #[serde(default)]
    pub cautions: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
    pub read_only: bool,
    pub input_schema: Value,
}

impl ToolFile {
    /// Parse the embedded JSON. Panics on malformed input — same fail-fast
    /// posture as `include_str!()` of a JSON Schema elsewhere in the crate.
    pub fn parse(json: &str) -> Self {
        serde_json::from_str(json).expect("invalid tool definition JSON")
    }

    /// Render the prose sections as markdown — what the LLM sees today.
    /// Sections are emitted only when non-empty; empty sections do not
    /// produce stray headings or trailing blank lines.
    pub fn render_markdown(&self) -> String {
        let mut sections: Vec<String> = Vec::new();

        if !self.summary.is_empty() {
            sections.push(self.summary.join(" "));
        }

        if !self.constraints.is_empty() {
            sections.push(bullet_list(&self.constraints));
        }

        if !self.anti_patterns.is_empty() {
            let mut s = String::from("## When NOT to use\n\n");
            s.push_str(&bullet_list(&self.anti_patterns));
            sections.push(s);
        }

        if !self.cautions.is_empty() {
            sections.push(self.cautions.join("\n"));
        }

        if !self.output.is_empty() {
            let mut s = String::from("## Output\n\n");
            s.push_str(&self.output.join("\n\n"));
            sections.push(s);
        }

        sections.join("\n\n")
    }
}

fn bullet_list(items: &[String]) -> String {
    items
        .iter()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ToolFile {
        serde_json::from_value(serde_json::json!({
            "name": "fixture_tool",
            "summary": ["First sentence.", "Second sentence."],
            "constraints": ["A constraint.", "Another constraint."],
            "anti_patterns": ["Use X instead.", "Use Y instead."],
            "cautions": ["ALWAYS read first.", "IMPORTANT: do not break."],
            "output": ["Output paragraph one.", "Output paragraph two."],
            "read_only": true,
            "input_schema": { "type": "object" }
        }))
        .unwrap()
    }

    #[test]
    fn renders_full_definition() {
        let rendered = fixture().render_markdown();
        let expected = "First sentence. Second sentence.\n\
                        \n\
                        - A constraint.\n\
                        - Another constraint.\n\
                        \n\
                        ## When NOT to use\n\
                        \n\
                        - Use X instead.\n\
                        - Use Y instead.\n\
                        \n\
                        ALWAYS read first.\n\
                        IMPORTANT: do not break.\n\
                        \n\
                        ## Output\n\
                        \n\
                        Output paragraph one.\n\
                        \n\
                        Output paragraph two.";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn skips_empty_sections() {
        let mut tf = fixture();
        tf.anti_patterns.clear();
        tf.cautions.clear();
        tf.output.clear();
        let rendered = tf.render_markdown();
        let expected = "First sentence. Second sentence.\n\
                        \n\
                        - A constraint.\n\
                        - Another constraint.";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn read_only_flag_round_trips() {
        let tf: ToolFile = serde_json::from_value(serde_json::json!({
            "name": "minimal",
            "read_only": false,
            "input_schema": { "type": "object" }
        }))
        .unwrap();
        assert_eq!(tf.name, "minimal");
        assert!(!tf.read_only);
        assert!(tf.summary.is_empty());
    }
}
