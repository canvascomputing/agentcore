# Structured Output

## Overview

Structured output forces the model to return data matching a JSON Schema, instead of free-form text. This is essential for agents whose output is consumed programmatically (API responses, data extraction, classification, etc.). This sub-plan covers `OutputSchema`, `StructuredOutputTool`, `validate_value`, and the 4-layer enforcement mechanism.

## Dependencies

- [Agent loop](./loop.md): `Agent`, `AgentBuilder`, `LlmAgent`, `InvocationContext`, `AgentOutput`
- [Tool system](../3-tools/traits.md): `Tool`, `ToolContext`, `ToolResult`
- [Core types](../1-base/types.md): `AgenticError`, `ToolChoice`

## Files

```
crates/agent-core/src/agent.rs  (OutputSchema, StructuredOutputTool, validate_value additions)
```

## Specification

### 9.8 Structured Output (Response Schema Enforcement)

#### Design (inspired by ADK-Go and Claude Code TypeScript)

Both reference projects use the same core pattern: **inject a synthetic tool whose input_schema IS the desired output schema, then extract the tool call input as the structured result.**

| Approach | ADK-Go | Claude Code TS | This library |
|----------|--------|----------------|--------------|
| Tool name | `set_model_response` | `StructuredOutput` | `StructuredOutput` |
| Schema source | `genai.Schema` on agent config | JSON Schema via `--json-schema` CLI flag | `serde_json::Value` via `AgentBuilder::output_schema()` |
| Forced selection | Native `ResponseSchema` when supported, tool fallback otherwise | `tool_choice: { type: "tool", name: "StructuredOutput" }` when no other tools | `ToolChoice::Specific` when no other tools |
| Retry on non-compliance | **None** — relies on prompt engineering only | **Stop hook** — re-prompts LLM | **Retry loop** — up to `max_schema_retries` attempts, then returns error |
| Validation | `ValidateMapOnSchema()` — recursive type + required field checks | AJV (JSON Schema validator library) | Manual recursive validation (no external crate) |

#### Enforcement Layers

Structured output enforcement uses four layers, from weakest to strongest:

1. **Tool description** — The StructuredOutputTool's description tells the model to call it. Part of the API `tools` parameter.

2. **System prompt instruction** — When `output_schema` is set, appended to the system prompt:
   ```
   IMPORTANT: You must provide your final response using the StructuredOutput tool
   with the required structured format.
   ```

3. **Retry on non-compliance** — If the model completes a turn without calling StructuredOutput, the agent loop injects a retry message and continues, up to `max_schema_retries` (default: 3). Returns `Err(AgenticError::SchemaRetryExhausted)` if exhausted.

4. **Forced tool_choice** — When the agent has no other tools, `tool_choice` is set to `ToolChoice::Specific("StructuredOutput")`, guaranteeing compliance on the first turn.

#### OutputSchema and StructuredOutputTool

```rust
/// A validated JSON Schema for structured output.
#[derive(Debug, Clone)]
pub struct OutputSchema {
    pub schema: serde_json::Value,
}

impl OutputSchema {
    /// Create from a JSON Schema value. Validates that it's a valid object schema.
    pub fn new(schema: serde_json::Value) -> Result<Self> {
        // Validate: must be { "type": "object", "properties": { ... } }
        // Returns Err(AgenticError::SchemaValidation) if invalid
    }
}

const STRUCTURED_OUTPUT_TOOL_NAME: &str = "StructuredOutput";

/// A synthetic tool injected into the agent's tool list when output_schema is set.
/// The tool's input_schema IS the output schema — the model "calls" this tool
/// to produce structured output, and the input it provides is the structured data.
struct StructuredOutputTool {
    schema: OutputSchema,
}

impl Tool for StructuredOutputTool {
    fn name(&self) -> &str { STRUCTURED_OUTPUT_TOOL_NAME }

    fn description(&self) -> &str {
        "Return your final response using the required output schema. \
         Call this tool exactly once at the end to provide the structured result."
    }

    fn input_schema(&self) -> serde_json::Value {
        self.schema.schema.clone()
    }

    fn is_read_only(&self) -> bool { true }

    fn call(&self, input: serde_json::Value, _ctx: &ToolContext)
        -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>
    {
        Box::pin(async move {
            validate_value(&input, &self.schema.schema)?;
            Ok(ToolResult { content: "Structured output accepted.".into(), is_error: false })
        })
    }
}
```

#### Schema Validation (`validate_value`)

Recursive validation without external crates. Checks type matching, required fields, and nested objects/arrays:

```rust
/// Validate a JSON value against a JSON Schema object.
/// Supports: string, number, integer, boolean, array, object.
/// Checks: type match, required fields, nested properties, array items.
pub fn validate_value(value: &serde_json::Value, schema: &serde_json::Value) -> Result<()> {
    let schema_type = schema.get("type").and_then(|t| t.as_str()).unwrap_or("object");

    match schema_type {
        "object" => {
            let obj = value.as_object()
                .ok_or_else(|| AgenticError::SchemaValidation {
                    path: String::new(), message: "expected object".into(),
                })?;
            if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                for key in required {
                    if let Some(key_str) = key.as_str() {
                        if !obj.contains_key(key_str) {
                            return Err(AgenticError::SchemaValidation {
                                path: key_str.into(), message: "missing required field".into(),
                            });
                        }
                    }
                }
            }
            if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
                for (key, prop_schema) in properties {
                    if let Some(prop_value) = obj.get(key) {
                        validate_value(prop_value, prop_schema)?;
                    }
                }
            }
            Ok(())
        }
        "array" => {
            let arr = value.as_array()
                .ok_or_else(|| AgenticError::SchemaValidation {
                    path: String::new(), message: "expected array".into(),
                })?;
            if let Some(items_schema) = schema.get("items") {
                for (i, item) in arr.iter().enumerate() {
                    validate_value(item, items_schema).map_err(|e| match e {
                        AgenticError::SchemaValidation { path, message } =>
                            AgenticError::SchemaValidation {
                                path: format!("[{i}].{path}"), message,
                            },
                        other => other,
                    })?;
                }
            }
            Ok(())
        }
        "string" => value.is_string().then_some(()).ok_or_else(|| ...),
        "number" => value.is_number().then_some(()).ok_or_else(|| ...),
        "integer" => value.is_i64().or(value.is_u64()).then_some(()).ok_or_else(|| ...),
        "boolean" => value.is_boolean().then_some(()).ok_or_else(|| ...),
        _ => Ok(()),  // unknown types pass
    }
}
```

#### How It Works in the Agent Loop

1. At build time, the user sets an output schema via `AgentBuilder::new().output_schema(json!({...})).build()`.
2. When `run_loop` begins, it appends an instruction to the system prompt (layer 2).
3. The `StructuredOutputTool` is injected into the tools list (layer 1).
4. If no other tools, `tool_choice` is forced to `ToolChoice::Specific("StructuredOutput")` (layer 4).

Three cases can occur:

**Case A: Model calls StructuredOutput (happy path).** `validate_value` succeeds, `structured_output` is set, agent returns `AgentOutput` with structured data.

**Case B: Model does not call StructuredOutput (retry, layer 3).** The stop check injects a retry message. Up to `max_schema_retries` attempts, then `Err(SchemaRetryExhausted)`.

**Case C: Model calls StructuredOutput with invalid data.** `validate_value` returns `Err(SchemaValidation)`, producing `ToolResult { is_error: true }`. Model retries with corrected data.

#### Example 1: Schema-Only Agent (No Tools, Forced `tool_choice`)

```rust
let classifier = AgentBuilder::new()
    .name("classifier")
    .model("claude-haiku-4-5-20241022")
    .system_prompt("Classify the given support ticket into a category and priority.")
    .output_schema(json!({
        "type": "object",
        "properties": {
            "category": { "type": "string" },
            "priority": { "type": "string" },
            "reasoning": { "type": "string" }
        },
        "required": ["category", "priority", "reasoning"]
    }))
    .build()?;

let output = classifier.run(ctx.with_input(
    "I've been charged twice for my subscription this month"
)).await?;
let data = output.structured_output.unwrap();
```

#### Example 2: Schema with Tools (Research, Then Structured Result)

```rust
let reviewer = AgentBuilder::new()
    .name("code_reviewer")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are a code reviewer. Read files and provide a structured review.")
    .tool(ReadFileTool).tool(GrepTool).tool(GlobTool)
    .output_schema(json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string" },
                        "line": { "type": "integer" },
                        "severity": { "type": "string" },
                        "message": { "type": "string" },
                        "suggestion": { "type": "string" }
                    },
                    "required": ["file", "severity", "message"]
                }
            },
            "score": { "type": "integer" }
        },
        "required": ["summary", "issues", "score"]
    }))
    .build()?;
```

#### Example 3: Multi-Agent Pipeline via spawn_agent

```rust
let analyzer = AgentBuilder::new()
    .name("code_analyzer")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Analyze the provided code and produce a structured analysis.")
    .output_schema(json!({
        "type": "object",
        "properties": {
            "language": { "type": "string" },
            "functions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "purpose": { "type": "string" },
                        "complexity": { "type": "string" }
                    },
                    "required": ["name", "purpose", "complexity"]
                }
            }
        },
        "required": ["language", "functions"]
    }))
    .build()?;

let orchestrator = AgentBuilder::new()
    .name("orchestrator")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Coordinate code analysis and test writing.")
    .tool(SpawnAgentTool::new())
    .sub_agent(analyzer)
    .sub_agent(test_writer)
    .build()?;
```

#### Example 4: Parallel Research via Background Agents

```rust
let orchestrator = AgentBuilder::new()
    .name("orchestrator")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Coordinate research. Spawn agents in parallel with background: true.")
    .tool(SpawnAgentTool::new())
    .sub_agent(frontend_expert)
    .sub_agent(backend_expert)
    .build()?;
// Two spawn_agent calls in one message with background: true
```

#### Example 5: Iterative Refinement via Repeated Spawning

```rust
let orchestrator = AgentBuilder::new()
    .name("orchestrator")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Iteratively refine a draft. Spawn critic then refiner. Repeat until scores >= 8.")
    .tool(SpawnAgentTool::new())
    .sub_agent(critic)
    .sub_agent(refiner)
    .build()?;
// LLM loops dynamically: critic → refiner → critic → done
```

## Work Items

1. **Structured output** — Spec Section 9.8
   - `OutputSchema`, `validate_value()` recursive validator
   - `StructuredOutputTool` — synthetic tool injected when output_schema is set
   - 4-layer enforcement: tool description, system prompt instruction, retry on non-compliance, forced tool_choice
   - `max_schema_retries` (default 3)

## Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `structured_output_extracted` | Direct | Agent with output_schema receives StructuredOutput tool call; output.structured_output contains expected JSON |
| `structured_output_retry_on_noncompliance` | Direct | First response is text-only; agent retries; second response calls StructuredOutput; request_count=3 |
| `structured_output_retry_exhausted` | Direct | All responses are text-only; after max retries, returns AgenticError::SchemaRetryExhausted |
| `validate_value_table` | Table-driven (8 cases) | valid complete, valid minimal, missing required, wrong types, wrong array item type, non-object input |

## Done Criteria

- Structured output enforcement works end-to-end with schema validation
- All 4 enforcement layers function correctly
- `validate_value()` recursively checks type, required fields, nested objects, and arrays
- Retry mechanism correctly re-prompts on non-compliance and exhausts after max retries
