//! Hand-rolled minimal JSON Schema validator. Covers the subset of
//! keywords agents actually use to constrain structured content:
//! `type`, `required`, `properties`, `additionalProperties` (bool
//! only), `items`, `enum`, `const`, `minimum`, `maximum`,
//! `minLength`, `maxLength`, `minItems`, `maxItems`.
//!
//! The public surface is `Schema::parse` + `Schema::validate` plus
//! [`format_violations`] — identical to what a real JSON Schema crate
//! would expose. If we later need `$ref`, `pattern`, `format`,
//! `oneOf`/`anyOf`/`allOf`, etc., the implementation behind this API
//! can be swapped for a third-party validator without callers
//! noticing.
//!
//! Unsupported keywords are rejected at parse time with a clear
//! message, not silently ignored — agents get fast feedback that
//! their schema overreached.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use serde_json::{Map, Number, Value};

/// A compiled JSON Schema. Cheap to clone (Arc-backed); validation
/// is read-only.
#[derive(Clone)]
pub struct Schema {
    document: Arc<Value>,
    compiled: Arc<Node>,
}

impl Schema {
    /// Parse and compile a schema document. Compilation failures
    /// (malformed schema, unsupported keyword, …) come back as
    /// [`SchemaParseError`] and never feed the retry budget — they
    /// are programming errors, not content violations.
    pub fn parse(document: Value) -> Result<Self, SchemaParseError> {
        let compiled = compile(&document, "")?;
        Ok(Self {
            document: Arc::new(document),
            compiled: Arc::new(compiled),
        })
    }

    /// Validate `instance` against this schema. On success returns
    /// `Ok(())`. On failure returns every violation the validator
    /// reported, each tagged with the instance path so the model can
    /// find the field it got wrong.
    pub fn validate(&self, instance: &Value) -> Result<(), Vec<SchemaViolation>> {
        let mut violations = Vec::new();
        self.compiled.check(instance, "", &mut violations);
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Raw schema document — useful for serialisation and for
    /// rendering an attachment alongside the schema it validated.
    pub fn document(&self) -> &Value {
        &self.document
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Schema")
            .field("document", &self.document)
            .finish_non_exhaustive()
    }
}

/// A single violation reported by [`Schema::validate`].
#[derive(Debug, Clone)]
pub struct SchemaViolation {
    /// JSON Pointer into the instance (e.g. `/items/0/name`).
    /// Empty string means "the root value itself".
    pub instance_path: String,
    /// JSON Pointer into the schema (e.g.
    /// `/properties/items/items/properties/name/type`).
    pub schema_path: String,
    /// One-line message describing the violation.
    pub message: String,
}

impl fmt::Display for SchemaViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = if self.instance_path.is_empty() {
            "<root>"
        } else {
            self.instance_path.as_str()
        };
        write!(f, "{path}: {}", self.message)
    }
}

/// Schema compilation failure — the caller-supplied schema is itself
/// invalid. Distinct from [`SchemaViolation`] (an instance failing a
/// valid schema).
#[derive(Debug, Clone)]
pub struct SchemaParseError {
    pub message: String,
}

impl fmt::Display for SchemaParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid schema: {}", self.message)
    }
}

impl std::error::Error for SchemaParseError {}

/// Format a slice of violations as a single model-facing error
/// string — one violation per line, leading with the instance path.
pub fn format_violations(violations: &[SchemaViolation]) -> String {
    let mut out = String::from("Schema validation failed:\n");
    for v in violations {
        out.push_str("- ");
        out.push_str(&v.to_string());
        out.push('\n');
    }
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

// ---------- compiled schema tree ----------

#[derive(Debug)]
struct Node {
    schema_path: String,
    types: Option<Vec<JsonType>>,
    enum_values: Option<Vec<Value>>,
    const_value: Option<Value>,
    properties: Option<Vec<(String, Node)>>,
    required: Option<Vec<String>>,
    additional_properties: AdditionalProperties,
    items: Option<Box<Node>>,
    minimum: Option<f64>,
    maximum: Option<f64>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    min_items: Option<usize>,
    max_items: Option<usize>,
}

#[derive(Debug)]
enum AdditionalProperties {
    /// `additionalProperties` was not set — JSON Schema default is
    /// "anything goes".
    Unset,
    /// `additionalProperties: true` — explicitly permissive.
    Allowed,
    /// `additionalProperties: false` — extra keys produce a violation.
    Forbidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

impl JsonType {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "object" => Self::Object,
            "array" => Self::Array,
            "string" => Self::String,
            "number" => Self::Number,
            "integer" => Self::Integer,
            "boolean" => Self::Boolean,
            "null" => Self::Null,
            _ => return None,
        })
    }

    fn matches(self, value: &Value) -> bool {
        match (self, value) {
            (Self::Object, Value::Object(_)) => true,
            (Self::Array, Value::Array(_)) => true,
            (Self::String, Value::String(_)) => true,
            (Self::Boolean, Value::Bool(_)) => true,
            (Self::Null, Value::Null) => true,
            (Self::Number, Value::Number(_)) => true,
            (Self::Integer, Value::Number(n)) => is_integer(n),
            _ => false,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Object => "object",
            Self::Array => "array",
            Self::String => "string",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
            Self::Null => "null",
        }
    }
}

fn is_integer(n: &Number) -> bool {
    if n.is_i64() || n.is_u64() {
        return true;
    }
    // f64 with no fractional part is still considered an integer by
    // JSON Schema (e.g. `1.0` validates against `type: integer`).
    n.as_f64().is_some_and(|f| f.is_finite() && f.fract() == 0.0)
}

const SUPPORTED_KEYWORDS: &[&str] = &[
    "type",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    // "$schema" / "$id" / "title" / "description" / "default"
    // are accepted as informational-only and don't influence validation.
    "$schema",
    "$id",
    "title",
    "description",
    "default",
    "examples",
];

fn compile(value: &Value, schema_path: &str) -> Result<Node, SchemaParseError> {
    let obj = match value {
        Value::Object(map) => map,
        Value::Bool(true) => {
            // `true` accepts everything.
            return Ok(empty_node(schema_path.to_string()));
        }
        Value::Bool(false) => {
            // `false` rejects everything — model as "type: <none>"
            // by giving the node an enum_values of `[]`. Simplest
            // way: attach a magic "always fails" marker. We use an
            // empty `types` list (which never matches anything).
            let mut node = empty_node(schema_path.to_string());
            node.types = Some(Vec::new());
            return Ok(node);
        }
        other => {
            return Err(parse_err(
                schema_path,
                format!("expected an object or boolean, got {}", value_label(other)),
            ));
        }
    };

    for key in obj.keys() {
        if !SUPPORTED_KEYWORDS.contains(&key.as_str()) {
            return Err(parse_err(
                schema_path,
                format!("unsupported keyword `{key}`"),
            ));
        }
    }

    let types = match obj.get("type") {
        None => None,
        Some(Value::String(s)) => Some(vec![parse_type(s, schema_path, "type")?]),
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, item) in arr.iter().enumerate() {
                let s = item.as_str().ok_or_else(|| {
                    parse_err(
                        schema_path,
                        format!("`type[{i}]` must be a string, got {}", value_label(item)),
                    )
                })?;
                out.push(parse_type(
                    s,
                    schema_path,
                    &format!("type[{i}]"),
                )?);
            }
            Some(out)
        }
        Some(other) => {
            return Err(parse_err(
                schema_path,
                format!(
                    "`type` must be a string or array of strings, got {}",
                    value_label(other)
                ),
            ));
        }
    };

    let enum_values = match obj.get("enum") {
        None => None,
        Some(Value::Array(arr)) if !arr.is_empty() => Some(arr.clone()),
        Some(Value::Array(_)) => {
            return Err(parse_err(
                schema_path,
                "`enum` must contain at least one value",
            ));
        }
        Some(other) => {
            return Err(parse_err(
                schema_path,
                format!(
                    "`enum` must be a non-empty array, got {}",
                    value_label(other)
                ),
            ));
        }
    };

    let const_value = obj.get("const").cloned();

    let required = match obj.get("required") {
        None => None,
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, item) in arr.iter().enumerate() {
                let s = item.as_str().ok_or_else(|| {
                    parse_err(
                        schema_path,
                        format!(
                            "`required[{i}]` must be a string, got {}",
                            value_label(item)
                        ),
                    )
                })?;
                out.push(s.to_string());
            }
            Some(out)
        }
        Some(other) => {
            return Err(parse_err(
                schema_path,
                format!(
                    "`required` must be an array of strings, got {}",
                    value_label(other)
                ),
            ));
        }
    };

    let properties = match obj.get("properties") {
        None => None,
        Some(Value::Object(props)) => {
            let mut out = Vec::with_capacity(props.len());
            for (name, sub) in props {
                let sub_path = format!("{schema_path}/properties/{}", escape_pointer(name));
                let node = compile(sub, &sub_path)?;
                out.push((name.clone(), node));
            }
            Some(out)
        }
        Some(other) => {
            return Err(parse_err(
                schema_path,
                format!(
                    "`properties` must be an object, got {}",
                    value_label(other)
                ),
            ));
        }
    };

    let additional_properties = match obj.get("additionalProperties") {
        None => AdditionalProperties::Unset,
        Some(Value::Bool(true)) => AdditionalProperties::Allowed,
        Some(Value::Bool(false)) => AdditionalProperties::Forbidden,
        Some(other) => {
            return Err(parse_err(
                schema_path,
                format!(
                    "`additionalProperties` must be a boolean (subschema form is unsupported), got {}",
                    value_label(other)
                ),
            ));
        }
    };

    let items = match obj.get("items") {
        None => None,
        Some(v @ (Value::Object(_) | Value::Bool(_))) => {
            let sub_path = format!("{schema_path}/items");
            Some(Box::new(compile(v, &sub_path)?))
        }
        Some(other) => {
            return Err(parse_err(
                schema_path,
                format!(
                    "`items` must be a schema (object or boolean); per-position arrays are unsupported, got {}",
                    value_label(other)
                ),
            ));
        }
    };

    let minimum = parse_number(obj.get("minimum"), schema_path, "minimum")?;
    let maximum = parse_number(obj.get("maximum"), schema_path, "maximum")?;
    let min_length = parse_usize(obj.get("minLength"), schema_path, "minLength")?;
    let max_length = parse_usize(obj.get("maxLength"), schema_path, "maxLength")?;
    let min_items = parse_usize(obj.get("minItems"), schema_path, "minItems")?;
    let max_items = parse_usize(obj.get("maxItems"), schema_path, "maxItems")?;

    Ok(Node {
        schema_path: schema_path.to_string(),
        types,
        enum_values,
        const_value,
        properties,
        required,
        additional_properties,
        items,
        minimum,
        maximum,
        min_length,
        max_length,
        min_items,
        max_items,
    })
}

fn empty_node(schema_path: String) -> Node {
    Node {
        schema_path,
        types: None,
        enum_values: None,
        const_value: None,
        properties: None,
        required: None,
        additional_properties: AdditionalProperties::Unset,
        items: None,
        minimum: None,
        maximum: None,
        min_length: None,
        max_length: None,
        min_items: None,
        max_items: None,
    }
}

fn parse_type(s: &str, schema_path: &str, key: &str) -> Result<JsonType, SchemaParseError> {
    JsonType::parse(s).ok_or_else(|| {
        parse_err(schema_path, format!("`{key}` has unknown type `{s}`"))
    })
}

fn parse_number(
    v: Option<&Value>,
    schema_path: &str,
    key: &str,
) -> Result<Option<f64>, SchemaParseError> {
    match v {
        None => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(other) => Err(parse_err(
            schema_path,
            format!("`{key}` must be a number, got {}", value_label(other)),
        )),
    }
}

fn parse_usize(
    v: Option<&Value>,
    schema_path: &str,
    key: &str,
) -> Result<Option<usize>, SchemaParseError> {
    match v {
        None => Ok(None),
        Some(Value::Number(n)) => n.as_u64().map(|u| u as usize).map(Some).ok_or_else(|| {
            parse_err(
                schema_path,
                format!("`{key}` must be a non-negative integer, got {n}"),
            )
        }),
        Some(other) => Err(parse_err(
            schema_path,
            format!(
                "`{key}` must be a non-negative integer, got {}",
                value_label(other)
            ),
        )),
    }
}

fn parse_err(schema_path: &str, message: impl Into<String>) -> SchemaParseError {
    let prefix = if schema_path.is_empty() {
        "<root>".to_string()
    } else {
        schema_path.to_string()
    };
    SchemaParseError {
        message: format!("at {prefix}: {}", message.into()),
    }
}

fn value_label(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn escape_pointer(segment: &str) -> String {
    // RFC 6901: ~ becomes ~0, / becomes ~1.
    segment.replace('~', "~0").replace('/', "~1")
}

// ---------- validation ----------

impl Node {
    fn check(&self, instance: &Value, instance_path: &str, out: &mut Vec<SchemaViolation>) {
        // type
        if let Some(types) = &self.types {
            if types.is_empty() {
                self.violation(
                    instance_path,
                    "/type",
                    "value is rejected by `false` schema",
                    out,
                );
                return;
            }
            if !types.iter().any(|t| t.matches(instance)) {
                let labels: Vec<&str> = types.iter().map(|t| t.label()).collect();
                self.violation(
                    instance_path,
                    "/type",
                    format!(
                        "expected type {}, got {}",
                        join_or(&labels),
                        value_label(instance)
                    ),
                    out,
                );
                // Don't probe sub-keywords if the type is wrong —
                // most of them assume a particular shape.
                return;
            }
        }

        // const
        if let Some(expected) = &self.const_value {
            if instance != expected {
                self.violation(
                    instance_path,
                    "/const",
                    format!("expected {}", expected),
                    out,
                );
            }
        }

        // enum
        if let Some(values) = &self.enum_values {
            if !values.iter().any(|v| v == instance) {
                self.violation(
                    instance_path,
                    "/enum",
                    "value is not in `enum`",
                    out,
                );
            }
        }

        match instance {
            Value::Object(map) => self.check_object(map, instance_path, out),
            Value::Array(arr) => self.check_array(arr, instance_path, out),
            Value::String(s) => self.check_string(s, instance_path, out),
            Value::Number(n) => self.check_number(n, instance_path, out),
            _ => {}
        }
    }

    fn check_object(
        &self,
        map: &Map<String, Value>,
        instance_path: &str,
        out: &mut Vec<SchemaViolation>,
    ) {
        if let Some(req) = &self.required {
            for name in req {
                if !map.contains_key(name) {
                    self.violation(
                        instance_path,
                        "/required",
                        format!("missing required property `{name}`"),
                        out,
                    );
                }
            }
        }

        let known: HashSet<&str> = self
            .properties
            .as_ref()
            .map(|ps| ps.iter().map(|(k, _)| k.as_str()).collect())
            .unwrap_or_default();

        if let Some(props) = &self.properties {
            for (name, sub) in props {
                if let Some(v) = map.get(name) {
                    let child_path =
                        format!("{instance_path}/{}", escape_pointer(name));
                    sub.check(v, &child_path, out);
                }
            }
        }

        if matches!(self.additional_properties, AdditionalProperties::Forbidden) {
            for name in map.keys() {
                if !known.contains(name.as_str()) {
                    self.violation(
                        instance_path,
                        "/additionalProperties",
                        format!("unexpected property `{name}`"),
                        out,
                    );
                }
            }
        }
    }

    fn check_array(
        &self,
        arr: &[Value],
        instance_path: &str,
        out: &mut Vec<SchemaViolation>,
    ) {
        if let Some(min) = self.min_items {
            if arr.len() < min {
                self.violation(
                    instance_path,
                    "/minItems",
                    format!("array has {} items, expected at least {min}", arr.len()),
                    out,
                );
            }
        }
        if let Some(max) = self.max_items {
            if arr.len() > max {
                self.violation(
                    instance_path,
                    "/maxItems",
                    format!("array has {} items, expected at most {max}", arr.len()),
                    out,
                );
            }
        }
        if let Some(items_schema) = &self.items {
            for (i, item) in arr.iter().enumerate() {
                let child_path = format!("{instance_path}/{i}");
                items_schema.check(item, &child_path, out);
            }
        }
    }

    fn check_string(&self, s: &str, instance_path: &str, out: &mut Vec<SchemaViolation>) {
        let len = s.chars().count();
        if let Some(min) = self.min_length {
            if len < min {
                self.violation(
                    instance_path,
                    "/minLength",
                    format!("string length {len} is below minimum {min}"),
                    out,
                );
            }
        }
        if let Some(max) = self.max_length {
            if len > max {
                self.violation(
                    instance_path,
                    "/maxLength",
                    format!("string length {len} is above maximum {max}"),
                    out,
                );
            }
        }
    }

    fn check_number(&self, n: &Number, instance_path: &str, out: &mut Vec<SchemaViolation>) {
        let Some(f) = n.as_f64() else { return };
        if let Some(min) = self.minimum {
            if f < min {
                self.violation(
                    instance_path,
                    "/minimum",
                    format!("value {f} is below minimum {min}"),
                    out,
                );
            }
        }
        if let Some(max) = self.maximum {
            if f > max {
                self.violation(
                    instance_path,
                    "/maximum",
                    format!("value {f} is above maximum {max}"),
                    out,
                );
            }
        }
    }

    fn violation(
        &self,
        instance_path: &str,
        keyword_suffix: &str,
        message: impl Into<String>,
        out: &mut Vec<SchemaViolation>,
    ) {
        out.push(SchemaViolation {
            instance_path: instance_path.to_string(),
            schema_path: format!("{}{keyword_suffix}", self.schema_path),
            message: message.into(),
        });
    }
}

fn join_or(labels: &[&str]) -> String {
    match labels.len() {
        0 => String::new(),
        1 => labels[0].to_string(),
        _ => {
            let (last, head) = labels.split_last().unwrap();
            format!("{} or {}", head.join(", "), last)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_rejects_malformed_schema() {
        let bad = json!({"type": 42});
        let err = Schema::parse(bad).unwrap_err();
        assert!(err.message.contains("`type` must be"));
    }

    #[test]
    fn parse_rejects_unknown_type_label() {
        let err = Schema::parse(json!({"type": "thingy"})).unwrap_err();
        assert!(err.message.contains("unknown type"));
    }

    #[test]
    fn parse_rejects_unsupported_keyword() {
        // `pattern` is intentionally not supported in this minimal
        // validator. We surface that at parse time so the agent
        // sees the limitation rather than silently passing values
        // that ought to fail.
        let err = Schema::parse(json!({"type": "string", "pattern": ".+"})).unwrap_err();
        assert!(err.message.contains("unsupported keyword `pattern`"));
    }

    #[test]
    fn validate_passes_conforming_object() {
        let schema = Schema::parse(json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
        }))
        .unwrap();
        assert!(schema.validate(&json!({"name": "alice"})).is_ok());
    }

    #[test]
    fn validate_reports_each_violation_with_path() {
        let schema = Schema::parse(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer", "minimum": 0 },
            },
            "required": ["name", "age"],
        }))
        .unwrap();
        let violations = schema
            .validate(&json!({"name": 7, "age": -1}))
            .unwrap_err();
        assert!(violations.len() >= 2);
        let paths: Vec<&str> = violations.iter().map(|v| v.instance_path.as_str()).collect();
        assert!(paths.iter().any(|p| p.contains("/name")));
        assert!(paths.iter().any(|p| p.contains("/age")));
    }

    #[test]
    fn validate_reports_missing_required() {
        let schema = Schema::parse(json!({
            "type": "object",
            "properties": { "x": { "type": "string" } },
            "required": ["x", "y"],
        }))
        .unwrap();
        let violations = schema.validate(&json!({"x": "hi"})).unwrap_err();
        assert!(violations.iter().any(|v| v.message.contains("`y`")));
    }

    #[test]
    fn validate_arrays_with_items_schema() {
        let schema = Schema::parse(json!({
            "type": "array",
            "items": { "type": "integer", "minimum": 0 },
            "minItems": 1,
        }))
        .unwrap();
        assert!(schema.validate(&json!([1, 2, 3])).is_ok());
        let violations = schema.validate(&json!([])).unwrap_err();
        assert!(violations.iter().any(|v| v.schema_path.ends_with("/minItems")));
        let violations = schema.validate(&json!([0, -1])).unwrap_err();
        assert!(violations.iter().any(|v| v.instance_path == "/1"));
    }

    #[test]
    fn validate_enum_and_const() {
        let enum_schema = Schema::parse(json!({"enum": ["a", "b", "c"]})).unwrap();
        assert!(enum_schema.validate(&json!("b")).is_ok());
        assert!(enum_schema.validate(&json!("z")).is_err());

        let const_schema = Schema::parse(json!({"const": 42})).unwrap();
        assert!(const_schema.validate(&json!(42)).is_ok());
        assert!(const_schema.validate(&json!(43)).is_err());
    }

    #[test]
    fn validate_string_length_bounds() {
        let schema = Schema::parse(json!({"type": "string", "minLength": 2, "maxLength": 4})).unwrap();
        assert!(schema.validate(&json!("ok")).is_ok());
        assert!(schema.validate(&json!("a")).is_err());
        assert!(schema.validate(&json!("toolong")).is_err());
    }

    #[test]
    fn validate_additional_properties_forbidden() {
        let schema = Schema::parse(json!({
            "type": "object",
            "properties": { "x": { "type": "string" } },
            "additionalProperties": false,
        }))
        .unwrap();
        assert!(schema.validate(&json!({"x": "hi"})).is_ok());
        let violations = schema.validate(&json!({"x": "hi", "y": 1})).unwrap_err();
        assert!(violations.iter().any(|v| v.message.contains("`y`")));
    }

    #[test]
    fn validate_integer_accepts_whole_floats() {
        // JSON Schema treats `1.0` as integer-valid.
        let schema = Schema::parse(json!({"type": "integer"})).unwrap();
        assert!(schema.validate(&json!(1.0)).is_ok());
        assert!(schema.validate(&json!(1.5)).is_err());
    }

    #[test]
    fn validate_type_array_accepts_any_listed() {
        let schema = Schema::parse(json!({"type": ["string", "null"]})).unwrap();
        assert!(schema.validate(&json!("hi")).is_ok());
        assert!(schema.validate(&json!(null)).is_ok());
        assert!(schema.validate(&json!(1)).is_err());
    }

    #[test]
    fn format_violations_renders_one_line_per_violation() {
        let schema = Schema::parse(json!({
            "type": "object",
            "properties": { "x": { "type": "string" } },
            "required": ["x", "y"],
        }))
        .unwrap();
        let violations = schema.validate(&json!({"x": 1})).unwrap_err();
        let formatted = format_violations(&violations);
        assert!(formatted.starts_with("Schema validation failed:\n"));
        let body = formatted.trim_start_matches("Schema validation failed:\n");
        assert!(body.lines().all(|line| line.starts_with("- ")));
    }

    #[test]
    fn clone_is_cheap() {
        let schema = Schema::parse(json!({"type": "string"})).unwrap();
        let cloned = schema.clone();
        assert!(Arc::ptr_eq(&schema.compiled, &cloned.compiled));
        assert!(Arc::ptr_eq(&schema.document, &cloned.document));
    }

    #[test]
    fn document_round_trip() {
        let doc = json!({"type": "boolean"});
        let schema = Schema::parse(doc.clone()).unwrap();
        assert_eq!(schema.document(), &doc);
    }

    #[test]
    fn boolean_true_schema_accepts_anything() {
        let schema = Schema::parse(json!(true)).unwrap();
        assert!(schema.validate(&json!(null)).is_ok());
        assert!(schema.validate(&json!({"a": [1, 2]})).is_ok());
    }

    #[test]
    fn boolean_false_schema_rejects_everything() {
        let schema = Schema::parse(json!(false)).unwrap();
        assert!(schema.validate(&json!(null)).is_err());
        assert!(schema.validate(&json!("anything")).is_err());
    }
}
