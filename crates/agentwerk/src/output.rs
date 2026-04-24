//! Final payload of an agent run: response text, outcome, statistics, collected errors, and optional structured-output validation.

use std::fmt;

use serde_json::Value;

use crate::error::Error;

/// The high-level verdict of an agent run. Three categories, no failure
/// detail: that lives in [`Output::errors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The model ended turn naturally and the loop exited cleanly. `errors`
    /// may still contain retried transient failures that didn't terminate.
    Completed,
    /// An external cancel signal was set before natural completion.
    Cancelled,
    /// A terminal error stopped the run. The last entry in `errors` is the
    /// cause.
    Failed,
}

/// Per-run counters covering token usage, provider requests, tool calls,
/// and agentic loop turns. Zero-initialized via [`Statistics::default`].
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Cumulative input tokens across every provider request this run.
    pub input_tokens: u64,
    /// Cumulative output tokens across every provider request this run.
    pub output_tokens: u64,
    /// Number of provider requests made.
    pub requests: u64,
    /// Number of tool calls invoked.
    pub tool_calls: u64,
    /// Number of agentic loop turns executed.
    pub turns: u32,
}

/// The result of an agent run.
///
/// `response_raw` always holds the model's final reply text. `response` is
/// `Some` only when an [`Agent::output_schema`](crate::Agent::output_schema)
/// was set and the reply parsed and validated against it. `errors` captures
/// every error observed during the run: retried transient failures,
/// tool-call failures bubbled as `Err`, schema-retry misses, budget hits. On
/// `Outcome::Failed` the last entry is the cause; on `Completed` / `Cancelled`
/// the list may still contain non-terminal errors.
#[derive(Debug)]
pub struct Output {
    /// Name of the agent that produced this output.
    pub name: String,
    /// Validated JSON when an output schema is configured; `None` otherwise.
    pub response: Option<Value>,
    /// The model's final reply text, before any schema validation.
    pub response_raw: String,
    /// Counters covering the run.
    pub statistics: Statistics,
    /// Verdict: did the run complete, get cancelled, or fail?
    pub outcome: Outcome,
    /// Every error observed during the run, in emission order.
    pub errors: Vec<Error>,
}

/// Per-attempt schema mismatch carried between [`OutputSchema::validate`] and
/// the agent loop. Internal: the model-visible consequence flows through
/// [`EventKind::SchemaRetried`](crate::event::EventKind::SchemaRetried) per
/// attempt, and on exhaustion as [`AgentError::PolicyViolated`](crate::agent::AgentError::PolicyViolated)
/// with `kind: PolicyKind::SchemaRetries`.
#[derive(Debug)]
pub(crate) struct SchemaViolation {
    /// Dotted field path of the violation (empty string = the whole value).
    pub path: String,
    /// Human-readable explanation of which rule failed.
    pub message: String,
}

impl fmt::Display for SchemaViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Schema violated at {}: {}", self.path, self.message)
    }
}

pub(crate) type SchemaResult<T> = std::result::Result<T, SchemaViolation>;

/// A validated JSON Schema for structured output.
#[derive(Debug, Clone)]
pub(crate) struct OutputSchema {
    pub schema: Value,
}

impl OutputSchema {
    pub(crate) fn new(schema: Value) -> SchemaResult<Self> {
        if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
            return Err(SchemaViolation {
                path: String::new(),
                message: "output schema must have \"type\": \"object\"".into(),
            });
        }
        if schema.get("properties").is_none() {
            return Err(SchemaViolation {
                path: String::new(),
                message: "output schema must have \"properties\"".into(),
            });
        }
        Ok(Self { schema })
    }

    /// Parse the model's terminal text as JSON and validate it against this
    /// schema. Strips an optional outer ```json … ``` (or bare ``` … ```)
    /// fence so the most common formatting mistake doesn't force a retry
    /// round-trip.
    pub(crate) fn validate(&self, text: &str) -> SchemaResult<Value> {
        let body = strip_code_fences(text);
        let value: Value = serde_json::from_str(body).map_err(|e| SchemaViolation {
            path: String::new(),
            message: format!("not valid JSON: {e}"),
        })?;
        validate_value(&value, &self.schema)?;
        Ok(value)
    }

}

fn strip_code_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s)
        .trim_start();
    s.strip_suffix("```").unwrap_or(s).trim()
}

/// Recursive walker: validate a JSON value against a schema. Internal to
/// this module — `OutputSchema::validate` is the only external entry point.
fn validate_value(value: &Value, schema: &Value) -> SchemaResult<()> {
    let schema_type = schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("object");

    match schema_type {
        "object" => validate_object(value, schema),
        "array" => validate_array(value, schema),
        "string" if !value.is_string() => Err(type_error("expected string")),
        "number" if !value.is_number() => Err(type_error("expected number")),
        "integer" if !(value.is_i64() || value.is_u64()) => Err(type_error("expected integer")),
        "boolean" if !value.is_boolean() => Err(type_error("expected boolean")),
        _ => Ok(()),
    }
}

fn type_error(message: &str) -> SchemaViolation {
    SchemaViolation {
        path: String::new(),
        message: message.into(),
    }
}

fn prepend_path(prefix: &str, mut violation: SchemaViolation) -> SchemaViolation {
    violation.path = if violation.path.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{}", violation.path)
    };
    violation
}

fn validate_object(value: &Value, schema: &Value) -> SchemaResult<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| type_error("expected object"))?;

    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for key in required.iter().filter_map(|k| k.as_str()) {
            if !obj.contains_key(key) {
                return Err(SchemaViolation {
                    path: key.into(),
                    message: "missing required field".into(),
                });
            }
        }
    }

    let properties = schema.get("properties").and_then(|p| p.as_object());
    for (key, prop_schema) in properties.into_iter().flatten() {
        if let Some(prop_value) = obj.get(key) {
            validate_value(prop_value, prop_schema).map_err(|e| prepend_path(key, e))?;
        }
    }

    Ok(())
}

fn validate_array(value: &Value, schema: &Value) -> SchemaResult<()> {
    let arr = value
        .as_array()
        .ok_or_else(|| type_error("expected array"))?;

    let Some(items_schema) = schema.get("items") else {
        return Ok(());
    };

    for (i, item) in arr.iter().enumerate() {
        let index = format!("[{i}]");
        validate_value(item, items_schema).map_err(|e| prepend_path(&index, e))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> OutputSchema {
        OutputSchema::new(serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "integer" } },
            "required": ["answer"]
        }))
        .unwrap()
    }

    #[test]
    fn parses_pure_json() {
        let v = schema().validate(r#"{"answer":42}"#).unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn strips_json_code_fences() {
        let v = schema().validate("```json\n{\"answer\":42}\n```").unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn strips_bare_code_fences() {
        let v = schema().validate("```\n{\"answer\":42}\n```").unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn whitespace_around_fences_ok() {
        let v = schema()
            .validate("  ```json\n  {\"answer\":42}\n```  ")
            .unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn rejects_invalid_json_with_parser_message() {
        let err = schema().validate("the answer is 42").unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("not valid JSON"),
            "expected parse-error prefix, got: {rendered}"
        );
    }

    #[test]
    fn propagates_validate_value_error_with_path() {
        let err = schema()
            .validate(r#"{"answer":"not a number"}"#)
            .unwrap_err();
        assert!(
            err.path.contains("answer"),
            "expected field path, got: {}",
            err.path
        );
    }

    #[test]
    fn rejects_missing_required_field() {
        let err = schema().validate(r#"{}"#).unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("missing required field"),
            "expected required-field error, got: {rendered}"
        );
    }
}
