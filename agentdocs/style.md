# Style

Naming and comment rules, plus README structure. Skim the section matching what is being written.

## 1. Crate root

**Only headline types live at the crate root.**

- The current root holds nine items: `Error`, `Result`, `Model`, `Provider`, `Tool`, `Agent`, `Batch`, `Event`, `Output`.
- A new item earns a root slot only when it opens a new dimension of the public API.
- Name collisions at the root are forbidden; `ToolResult` next to `Result` is not acceptable.
- Every other type lives under its domain module.

## 2. Where non-root types live

**Types live next to the abstraction, owner, or protocol they belong to.**

- Concrete implementations live with their abstraction: `AnthropicProvider` under `provider::`, `BashTool` under `tools::`.
- Companion types and handles live with their owner: `AgentHandle` under `agent::`, `BatchHandle` under `batch::`.
- Domain errors live with their domain: `ProviderError`, `ToolError`, `AgentError`.
- Wire-protocol types live with the protocol: `ModelRequest`, `Message`, `TokenUsage` under `provider::`.
- Free functions live in their module: `tool()` is at `tools::tool`, never at the root.

## 3. Name disambiguation

**Names are disambiguated through content, not through redundant prefixes.**

- Specific compound names stand alone: `LoopState`, `AgentSpec`, `OutputSchema`.
- Vendor prefixes are used only to distinguish concrete providers or tools: `AnthropicProvider`, `OpenAiProvider`, `LiteLlmProvider`.
- Acronyms follow Rust API guidelines: `OpenAi`, not `OpenAI`.
- Two structs may not share a bare name within one module; both stay qualified.

## 4. Failure variants

**Failure variants use passive-voice past-participle: `<Subject><Verb-ed>`.**

- Accepted: `RequestFailed`, `TaskNotFound`, `ContextWindowExceeded`, `PolicyViolated`.
- Rejected: adjective-first forms such as `InvalidX`, `UnexpectedX`, or `MissingX`.
- Rejected: noun-suffix forms such as `XError`; the `Error` suffix is reserved for the top-level `Error` and its domain sub-enums.
- State-transition events use the same form: `AgentStarted`, `RequestRetried`, `ContextCompacted`.
- Whether a failure is terminal is documented on the variant, not encoded in the name.

## 5. Variant shape

**Tuple for one payload. Struct for multiple fields or a meaningful field name.**

- Tuple form: `Provider(ProviderError)`, `TaskNotFound(String)`, `IoFailed(io::Error)`.
- Struct form: `AgentError::PolicyViolated { kind, limit }`.
- Struct form is also used when a single field name carries meaning the type alone does not.
- Two-arm result enums use one word per variant: `Success` / `Error`, with no `is_*` predicates.

## 6. Payload fields

**One vocabulary is used across every error type.**

- Human-readable strings are named `message: String`, never `error`.
- Wrapped underlying errors are named `source`, as in `FooFailed { source: io::Error }`.
- Typed metadata uses descriptive names: `status`, `retryable`, `retry_delay`, `tool_name`, `retries`, `after_ms`.

## 7. RAII guard fields

**Fields held only for their `Drop` behavior use a plain name and `#[allow(dead_code)]`.**

- Name the field for its purpose: `guard`, `lock`, `permit`. The `_guard` form is not used.
- `rustc`'s `dead_code` lint flags such fields because neither `Clone` nor drop glue count as reads; the attribute acknowledges this on the one field that needs it.
- `#[expect(dead_code)]` is preferred on Rust 1.81+: it self-removes if the field later does get read.

## 8. Time-typed fields

**Use `std::time::Duration`. The type is the unit.**

- No `_ms`, `_MS`, or `_seconds` suffix on public API names.
- Internal helpers and on-the-wire JSON may use raw integers where the protocol requires it.
- Example: `timeout_ms` is acceptable inside a tool input schema because the schema is a wire protocol.

## 9. Builders

**Builder methods are bare nouns. No `with_` prefix.**

- Examples: `.name()`, `.model()`, `.tool()`, `.sub_agents()`, `.read_only()`.
- The `with_` prefix is used only when a bare name clashes with a trait method, such as `with_description` on `BashTool`.

## 10. Constructors

**`new()` for the primary path. Named constructors carry semantics.**

- `new()` is the primary constructor.
- Named constructors: `open()`, `unrestricted()`, `success()`, `error()`, `empty()`, `from_id()`, `from_env()`.

## 11. Getters and setters

**Mutable accessors use `set_` and `get_` prefixes to distinguish them from builders.**

- Example: `set_extension()`, `get_extension()`.
- Builder methods remain unprefixed.

## 12. Free functions and tool structs

**Free functions are rare and snake_case. Tool structs follow `{Name}Tool`.**

- Most operations live as methods on the owning type: `OutputSchema::validate`, `ToolRegistry::execute`.
- A free function is used only when no receiver type is natural.
- Tool structs: `ReadFileTool`, `BashTool`, `SpawnAgentTool`.

## 13. Doc comments (`///`)

**State the purpose in one sentence. No "This function…" or "Returns…".**

- Noun phrase for types and fields; verb for functions.
- Additional paragraphs are added only for a constraint, invariant, or non-obvious semantic.
- Trivial getters, `Default::default`, `From` impls, and self-explanatory variants are left undocumented.
- Within one type, coverage is all-or-none: every member has a real doc comment, or none does.

## 14. Module docs (`//!`)

**Every file begins with a `//!` that states what the file contributes to the crate.**

- One sentence; two only when the second adds context the first cannot carry.
- State the problem the file solves, not the types it defines.
- Do not list the contents of the file.
- The `//!` stays even when the filename is already descriptive.

## 15. Line comments (`//`)

**Four reasons are allowed. Everything else is deleted.**

Allowed:

- Order-dependency or crash-safety, such as `Write mark BEFORE task file: crash-safe.`
- API quirk or workaround, such as `serde_json::Map is sorted alphabetically, so we format manually.`
- Non-obvious constraint, such as `Newest first so 'gpt-4' does not shadow 'gpt-4.1'.`
- Plain section label in a long function, on its own line above the block it introduces, such as `// Parse the reply, append the assistant message`.

Not allowed:

- Restating what the code does on the same line.
- Task, PR, issue, or changelog references.
- Commented-out code.
- Stub or aspirational markers; use `unimplemented!(...)` or return `Ok(())`.
- `TODO`, `FIXME`, or `NOTE`.
- Decorative banners of any kind: `// ── Title`, `// ==== Title ====`, `// ----- Title -----`.

## 16. Tests

**Test names carry intent. Setup is not narrated.**

- A comment is justified only to pin an architectural invariant the test guards.
- A module-level `//!` describing the test file's scope is acceptable.

## 17. Comment examples

**Good and bad variants of each comment type.**

Module `//!`:

```rust
// GOOD: states what the file contributes
//! The execution loop. Runs a compiled `Agent` turn by turn until it yields an `Output`.

// BAD: lists contents
//! Agent execution loop.
//! - `AgentSpec`: compiled agent definition.
//! - `LoopRuntime`: externals.
```

Doc comment `///`:

```rust
// GOOD: purpose and invariant
/// The agent's compiled definition. `model: None` means inherit from parent.
pub(crate) struct AgentSpec { ... }

// BAD: restates the name
/// The model field.
pub model: Option<Model>,
```

Function `///`:

```rust
// GOOD: verb, one line
/// Build the environment metadata block for the first user message.

// BAD: signature already says this
/// This function returns a String containing the environment metadata.
```

Line comment `//`:

```rust
// GOOD: flags an order constraint
// Write mark BEFORE task file: crash-safe.
fs::write(&mark_path, b"")?;

// GOOD: plain section header in a long function
// Parse the reply, append the assistant message

// BAD: restates the code
// Increment the counter.
counter += 1;

// BAD: decorative banner
// ── Parse the reply, append the assistant message
// ----- Core types -----
```

## 18. README structure

**Terse, example-driven, scannable.**

- Fixed section order: Installation, Quick Start, Use Cases, API, Development.
- Every subsection leads with a minimal example, then explains.
- Enumerations use bullets or grouped bullets; tables are not used.
- Facts live in one place; other sections cross-link rather than repeat.

## 19. README voice

**Direct and neutral. No marketing language.**

- "Give the agent access to filesystem tools", not "empower the application".
- Examples stay minimal; show the smallest snippet that demonstrates the feature.
- Example models are `claude-haiku-4-5-20251001` or `claude-sonnet-4-20250514`.
- Update triggers: a new builder method, a new tool, a new event kind, a new environment variable, or a changed default.
