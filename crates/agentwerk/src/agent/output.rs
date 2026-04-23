//! Final payload of an agent run: response text, status, statistics, and optional structured-output validation.

use std::fmt;

use serde_json::Value;

/// Why the agent loop exited.
///
/// Distinct from [`crate::provider::types::ResponseStatus`], which describes
/// what the provider reported. `Status` describes the agent-level outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// The model chose to stop: it responded without tool calls (`EndTurn`).
    Completed,
    /// External cancel signal was set.
    Cancelled,
    /// Configured `max_turns` limit reached.
    TurnLimitReached { limit: u32 },
    /// Configured `max_input_tokens` budget exceeded.
    InputBudgetExhausted { usage: u64, limit: u64 },
    /// Configured `max_output_tokens` budget exceeded.
    OutputBudgetExhausted { usage: u64, limit: u64 },
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
/// was set and the reply parsed and validated against it.
#[derive(Debug, Clone)]
pub struct Output {
    /// Name of the agent that produced this output.
    pub name: String,
    /// Validated JSON when an output schema is configured; `None` otherwise.
    pub response: Option<Value>,
    /// The model's final reply text, before any schema validation.
    pub response_raw: String,
    /// Counters covering the run.
    pub statistics: Statistics,
    /// Why the loop exited.
    pub status: Status,
}

impl Output {
    /// An empty placeholder with default statistics and [`Status::Completed`].
    /// Useful in tests and as a neutral fallback.
    pub fn empty() -> Self {
        Self {
            name: String::new(),
            response: None,
            response_raw: String::new(),
            statistics: Statistics::default(),
            status: Status::Completed,
        }
    }
}

/// Failures from structured-output schema validation.
#[derive(Debug)]
pub enum OutputError {
    /// The model's terminal reply did not satisfy the configured JSON schema.
    /// `path` is a dotted field path ("root" = the whole value); `message`
    /// explains which rule failed.
    SchemaViolated { path: String, message: String },
    /// After the configured retry budget, the model still failed to produce a
    /// schema-conforming reply.
    SchemaRetryExhausted { retries: u32 },
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputError::SchemaViolated { path, message } => {
                write!(f, "Schema violated at {path}: {message}")
            }
            OutputError::SchemaRetryExhausted { retries } => {
                write!(f, "Schema retry exhausted after {retries} attempts")
            }
        }
    }
}

impl std::error::Error for OutputError {}

pub(crate) type OutputResult<T> = std::result::Result<T, OutputError>;

/// A validated JSON Schema for structured output.
#[derive(Debug, Clone)]
pub(crate) struct OutputSchema {
    pub schema: Value,
}

impl OutputSchema {
    pub(crate) fn new(schema: Value) -> OutputResult<Self> {
        if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
            return Err(OutputError::SchemaViolated {
                path: String::new(),
                message: "output schema must have \"type\": \"object\"".into(),
            });
        }
        if schema.get("properties").is_none() {
            return Err(OutputError::SchemaViolated {
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
    pub(crate) fn validate(&self, text: &str) -> std::result::Result<Value, String> {
        let body = strip_code_fences(text.trim());
        let value: Value =
            serde_json::from_str(body).map_err(|e| format!("not valid JSON: {e}"))?;
        validate_value(&value, &self.schema).map_err(|e| e.to_string())?;
        Ok(value)
    }

    /// Build the corrective user message shown to the model after a terminal
    /// reply fails [`Self::validate`]. `detail` is the validator's
    /// human-readable error — passing it to the model lets it fix the exact
    /// field rather than guess.
    pub(crate) fn retry_message(detail: &str) -> String {
        format!(
            "Your last reply did not match the required output schema. You MUST reply with a \
             single JSON value conforming to the schema, with no surrounding text and no code \
             fences.\n\nValidator said: {detail}"
        )
    }
}

fn strip_code_fences(s: &str) -> &str {
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s)
        .trim_start();
    s.strip_suffix("```").unwrap_or(s).trim()
}

/// Recursive walker: validate a JSON value against a schema. Internal to
/// this module — `OutputSchema::validate` is the only external entry point.
fn validate_value(value: &Value, schema: &Value) -> OutputResult<()> {
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

fn type_error(message: &str) -> OutputError {
    OutputError::SchemaViolated {
        path: String::new(),
        message: message.into(),
    }
}

fn prepend_path(prefix: &str, error: OutputError) -> OutputError {
    match error {
        OutputError::SchemaViolated { path, message } => OutputError::SchemaViolated {
            path: if path.is_empty() {
                prefix.to_string()
            } else {
                format!("{prefix}.{path}")
            },
            message,
        },
        other => other,
    }
}

fn validate_object(value: &Value, schema: &Value) -> OutputResult<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| type_error("expected object"))?;

    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for key in required.iter().filter_map(|k| k.as_str()) {
            if !obj.contains_key(key) {
                return Err(OutputError::SchemaViolated {
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

fn validate_array(value: &Value, schema: &Value) -> OutputResult<()> {
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
        assert!(
            err.starts_with("not valid JSON"),
            "expected parse-error prefix, got: {err}"
        );
    }

    #[test]
    fn propagates_validate_value_error_with_path() {
        let err = schema()
            .validate(r#"{"answer":"not a number"}"#)
            .unwrap_err();
        assert!(err.contains("answer"), "expected field path, got: {err}");
    }

    #[test]
    fn rejects_missing_required_field() {
        let err = schema().validate(r#"{}"#).unwrap_err();
        assert!(
            err.contains("missing required field"),
            "expected required-field error, got: {err}"
        );
    }
}
