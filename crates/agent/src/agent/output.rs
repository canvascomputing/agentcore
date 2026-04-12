use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::{AgenticError, Result};
use super::prompts::{STRUCTURED_OUTPUT_TOOL_DESCRIPTION, STRUCTURED_OUTPUT_TOOL_NAME};
use crate::tools::{Tool, ToolContext, ToolResult};

#[derive(Debug, Clone, Default)]
pub struct Statistics {
    pub costs: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub requests: u64,
    pub tool_calls: u64,
    pub turns: u32,
}

#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub response: Option<Value>,
    pub response_raw: String,
    pub statistics: Statistics,
}

impl AgentOutput {
    pub fn empty() -> Self {
        Self {
            response: None,
            response_raw: String::new(),
            statistics: Statistics::default(),
        }
    }
}

/// A validated JSON Schema for structured output.
#[derive(Debug, Clone)]
pub struct OutputSchema {
    pub schema: Value,
}

impl OutputSchema {
    pub fn new(schema: Value) -> Result<Self> {
        if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
            return Err(AgenticError::SchemaValidation {
                path: String::new(),
                message: "output schema must have \"type\": \"object\"".into(),
            });
        }
        if schema.get("properties").is_none() {
            return Err(AgenticError::SchemaValidation {
                path: String::new(),
                message: "output schema must have \"properties\"".into(),
            });
        }
        Ok(Self { schema })
    }
}

pub(crate) struct StructuredOutputTool {
    schema: OutputSchema,
}

impl StructuredOutputTool {
    pub(crate) fn new(schema: OutputSchema) -> Self {
        Self { schema }
    }
}

impl Tool for StructuredOutputTool {
    fn name(&self) -> &str {
        STRUCTURED_OUTPUT_TOOL_NAME
    }

    fn description(&self) -> &str {
        STRUCTURED_OUTPUT_TOOL_DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        self.schema.schema.clone()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn call<'a>(
        &'a self,
        input: Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            validate_value(&input, &self.schema.schema)?;
            Ok(ToolResult::success("Structured output accepted."))
        })
    }
}

/// Validate a JSON value against a JSON Schema object.
pub fn validate_value(value: &Value, schema: &Value) -> Result<()> {
    let schema_type = schema.get("type").and_then(|t| t.as_str()).unwrap_or("object");

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

fn type_error(message: &str) -> AgenticError {
    AgenticError::SchemaValidation {
        path: String::new(),
        message: message.into(),
    }
}

fn prepend_path(prefix: &str, error: AgenticError) -> AgenticError {
    match error {
        AgenticError::SchemaValidation { path, message } => AgenticError::SchemaValidation {
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

fn validate_object(value: &Value, schema: &Value) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| type_error("expected object"))?;

    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for key in required.iter().filter_map(|k| k.as_str()) {
            if !obj.contains_key(key) {
                return Err(AgenticError::SchemaValidation {
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

fn validate_array(value: &Value, schema: &Value) -> Result<()> {
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
