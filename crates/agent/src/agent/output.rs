use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::{AgenticError, Result};
use crate::message::Usage;
use crate::tool::{Tool, ToolContext, ToolResult};

#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub content: String,
    pub usage: Usage,
    pub structured_output: Option<Value>,
}

impl AgentOutput {
    pub fn empty(usage: Usage) -> Self {
        Self {
            content: String::new(),
            usage,
            structured_output: None,
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

pub(crate) const STRUCTURED_OUTPUT_TOOL_NAME: &str = "StructuredOutput";

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
        "Return your final response using the required output schema. \
         Call this tool exactly once at the end to provide the structured result."
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
            Ok(ToolResult {
                content: "Structured output accepted.".into(),
                is_error: false,
            })
        })
    }
}

/// Validate a JSON value against a JSON Schema object.
pub fn validate_value(value: &Value, schema: &Value) -> Result<()> {
    let schema_type = schema.get("type").and_then(|t| t.as_str()).unwrap_or("object");

    match schema_type {
        "object" => {
            let obj = value.as_object().ok_or_else(|| AgenticError::SchemaValidation {
                path: String::new(),
                message: "expected object".into(),
            })?;
            if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                for key in required {
                    if let Some(key_str) = key.as_str() {
                        if !obj.contains_key(key_str) {
                            return Err(AgenticError::SchemaValidation {
                                path: key_str.into(),
                                message: "missing required field".into(),
                            });
                        }
                    }
                }
            }
            if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
                for (key, prop_schema) in properties {
                    if let Some(prop_value) = obj.get(key) {
                        validate_value(prop_value, prop_schema).map_err(|e| match e {
                            AgenticError::SchemaValidation { path, message } => {
                                AgenticError::SchemaValidation {
                                    path: if path.is_empty() {
                                        key.clone()
                                    } else {
                                        format!("{key}.{path}")
                                    },
                                    message,
                                }
                            }
                            other => other,
                        })?;
                    }
                }
            }
            Ok(())
        }
        "array" => {
            let arr = value.as_array().ok_or_else(|| AgenticError::SchemaValidation {
                path: String::new(),
                message: "expected array".into(),
            })?;
            if let Some(items_schema) = schema.get("items") {
                for (i, item) in arr.iter().enumerate() {
                    validate_value(item, items_schema).map_err(|e| match e {
                        AgenticError::SchemaValidation { path, message } => {
                            AgenticError::SchemaValidation {
                                path: format!("[{i}].{path}"),
                                message,
                            }
                        }
                        other => other,
                    })?;
                }
            }
            Ok(())
        }
        "string" => {
            if value.is_string() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected string".into(),
                })
            }
        }
        "number" => {
            if value.is_number() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected number".into(),
                })
            }
        }
        "integer" => {
            if value.is_i64() || value.is_u64() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected integer".into(),
                })
            }
        }
        "boolean" => {
            if value.is_boolean() {
                Ok(())
            } else {
                Err(AgenticError::SchemaValidation {
                    path: String::new(),
                    message: "expected boolean".into(),
                })
            }
        }
        _ => Ok(()),
    }
}
