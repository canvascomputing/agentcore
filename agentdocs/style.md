# Style

Naming and comment rules, plus README structure. Skim the section matching what is being written.

## Crate root

**Only headline types live at the crate root.**

- The current root re-exports cover the canonical caller surface: `Agent`, `Runnable`, `TicketSystem`, `Ticket`, `Status`, `Stats`; `Tool`, `ToolContext`, `ToolLike`, `ToolResult`; `Event`, `EventKind`, `PolicyKind`, `ToolFailureKind`, `default_logger`; `Schema`, `SchemaParseError`, `SchemaViolation`, `format_violations`.
- A new item earns a root slot only when it opens a new dimension of the public API.
- Name collisions at the root are forbidden; `ToolResult` next to `Result` is not acceptable.
- Every other type lives under its domain module.

## Where non-root types live

**Types live next to the abstraction, owner, or protocol they belong to.**

- Concrete implementations live with their abstraction: `AnthropicProvider` under `providers::`, `BashTool` under `tools::`.
- Companion types and handles live with their owner: `Ticket`, `Status`, and `TicketError` under `agents::tickets`; `Stats` and `LoopStats` under `agents::stats`.
- Domain errors live with their domain: `ProviderError`, `ToolError`.
- Wire-protocol types live with the protocol: `ModelRequest`, `Message`, `TokenUsage` under `providers::`.
- Free functions live in their module, never at the crate root: `from_env()` in `providers::environment`, helpers in `tools::util`.

## Name disambiguation

**Names are disambiguated through content, not through redundant prefixes.**

- Specific compound names stand alone: `TicketSystem`, `LoopStats`, `PolicyKind`.
- Vendor prefixes are used only to distinguish concrete providers or tools: `AnthropicProvider`, `OpenAiProvider`, `LiteLlmProvider`.
- Acronyms follow Rust API guidelines: `OpenAi`, not `OpenAI`.
- Two structs may not share a bare name within one module; both stay qualified.

## Failure variants

**Failure variants use passive-voice past-participle: `<Subject><Verb-ed>`.**

- Accepted: `RequestFailed`, `TodoItemNotFound`, `ContextWindowExceeded`, `PolicyViolated`.
- Rejected: adjective-first forms such as `InvalidX`, `UnexpectedX`, or `MissingX`.
- Rejected: noun-suffix forms such as `XError`; the `Error` suffix is reserved for the top-level `Error` and its domain sub-enums.
- State-transition events use the same form: `AgentStarted`, `RequestRetried`, `ContextCompacted`.
- Whether a failure is terminal is documented on the variant, not encoded in the name.

## Variant shape

**Tuple for one payload. Struct for multiple fields or a meaningful field name.**

- Tuple form: `Provider(ProviderError)`, `TodoItemNotFound(String)`, `IoFailed(io::Error)`.
- Struct form: `AgentError::PolicyViolated { kind, limit }`.
- Struct form is also used when a single field name carries meaning the type alone does not.
- Two-arm result enums use one word per variant: `Success` / `Error`, with no `is_*` predicates.

## Payload fields

**One vocabulary is used across every error type.**

- Human-readable strings MUST be named `message: String`, never `error`.
- Wrapped underlying errors MUST be named `source`, as in `FooFailed { source: io::Error }`.
- Typed metadata uses descriptive names: `status`, `retryable`, `retry_delay`, `tool_name`, `retries`, `after_ms`.

## RAII guard fields

**Fields held only for their `Drop` behavior use a plain name and `#[allow(dead_code)]`.**

- Name the field for its purpose: `guard`, `lock`, `permit`. The `_guard` form is not used.
- `rustc`'s `dead_code` lint flags such fields because neither `Clone` nor drop glue count as reads; the attribute acknowledges this on the one field that needs it.
- `#[expect(dead_code)]` is preferred on Rust 1.81+: it self-removes if the field later does get read.

## Time-typed fields

**Public API fields MUST use `std::time::Duration`. The type is the unit.**

- No `_ms`, `_MS`, or `_seconds` suffix on public API names.
- Internal helpers and on-the-wire JSON may use raw integers where the protocol requires it.
- Example: `timeout_ms` is acceptable inside a tool input schema because the schema is a wire protocol.

## Path identifiers

**A directory path uses `_dir`. A file path uses `_file`. The bare suffix `_path` is used only when the value can be either.**

- Directories: `working_dir`, `session_dir`, `base_dir`, `data_dir`. Matches `std::fs::read_dir`, `std::env::current_dir`, `std::fs::DirEntry`.
- Files: `transcript_file`, `task_file`, `lock_file`. The value is always a concrete file on disk.
- `_path` is for genuinely ambiguous cases: input that could name either, or a value passed through as opaque.
- IMPORTANT: `folder` is never used; it has no std analog.
- Doc comments and environment labels may still say "working directory" in English prose.

## Counter identifiers

**Counters use a bare plural noun. No `_count` suffix on fields or on methods that return a count.**

- `Stats` sets the vocabulary: `requests`, `tool_calls`, `steps`, `input_tokens`, `output_tokens`.
- Event payloads follow suit: `usage` on `TokensReported` carries token counts, not a `token_count`.
- Accessor methods mirror the field form: `Stats::requests()` returns the count of recorded requests.
- The `_count` suffix is reserved for the rare case where the plural would clash with a sibling collection field on the same type.

## Builders

**Builder methods are bare nouns. No `with_` prefix.**

- Examples: `.name()`, `.model()`, `.tool()`, `.label()`, `.read_only()`.
- The `with_` prefix is used only when a bare name clashes with a trait method, such as `with_description` on `BashTool`.

## Constructors

**`new()` for the primary path. Named constructors carry semantics.**

- `new()` is the primary constructor.
- Named constructors: `open()`, `unrestricted()`, `success()`, `error()`, `empty()`, `from_id()`, `from_env()`.

## Getters and setters

**Mutable accessors use `set_` and `get_` prefixes to distinguish them from builders.**

- Example: `set_extension()`, `get_extension()`.
- Builder methods remain unprefixed.

## Free functions

**A free function is used only for one of six reasons. Otherwise the function lives on a type.**

Permitted:

- **Ambient state** has no receiver: timestamp helpers and similar utilities in `tools::util` or a sibling helper module.
- **Foreign-type constructors** cannot use an inherent `impl`: `build_client()` returns a `reqwest::Client`.
- **Module entry points** drive multiple types: `run_main_loop()` in `agents::r#loop`, `from_env()` in `providers::environment`.
- **Higher-order utilities** take a closure and wrap it: `with_file_lock(path, || ...)`.
- **Shared algorithm helpers** are called by two or more sibling types in the same module: helpers in `tools::util` shared across filesystem tools; provider-side helpers shared across concrete providers.

Forbidden:

- A free function that delegates to a single method on one type. Inline it as a method instead.
- A free constructor for a local type that already has an inherent `impl`. Constructors for `Foo` live on `Foo`.
- A free helper called from exactly one private method. Make it a private method or a nested `fn`.
- An associated function that takes no `self` and does not return `Self` or `Result<Self>`. Move it to the module as a free function. Exception: a per-variant static lookup where the `Type::` prefix partitions otherwise-colliding names, such as `AnthropicProvider::lookup_context_window_size` vs `OpenAiProvider::lookup_context_window_size`.

Naming: `snake_case`. Tool structs keep the `{Name}Tool` suffix: `ReadFileTool`, `BashTool`, `ManageTicketsTool`.

## Doc comments (`///`)

**State the purpose in one sentence. No "This function…" or "Returns…".**

- Noun phrase for types and fields; verb for functions.
- Additional paragraphs are added only for a constraint, invariant, or non-obvious semantic.
- Trivial getters, `Default::default`, `From` impls, and self-explanatory variants are left undocumented.
- Within one type, coverage is all-or-none: every member has a real doc comment, or none does.

## Module docs (`//!`)

**Every file begins with a `//!` that states what the file contributes to the crate.**

- One sentence; two only when the second adds context the first cannot carry.
- State the problem the file solves, not the types it defines.
- Do not list the contents of the file.
- The `//!` stays even when the filename is already descriptive.

## Line comments (`//`)

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
- IMPORTANT: no `TODO`, `FIXME`, or `NOTE`. Fix it or file an issue.
- Decorative banners of any kind: `// ── Title`, `// ==== Title ====`, `// ----- Title -----`.

## Tests

**Test names carry intent. Setup is not narrated.**

- A comment is justified only to pin an architectural invariant the test guards.
- A module-level `//!` describing the test file's scope is acceptable.

## Comment examples

**Good and bad variants of each comment type.**

Module `//!`:

```rust
// GOOD: states what the file contributes
//! Multi-agent loop driver. Each agent runs in its own tokio task, reading the shared ticket store.

// BAD: lists contents
//! Agent loop.
//! - `Runnable`: trait.
//! - `run_main_loop`: entry point.
```

Doc comment `///`:

```rust
// GOOD: purpose and invariant
/// A ticket. Caller-settable fields: `task`, `labels`, `schema`, `assignee`. System-managed fields are stamped at insertion time.
pub struct Ticket { ... }

// BAD: restates the name
/// The task field.
pub task: serde_json::Value,
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

## README structure

**Terse, example-driven, scannable.**

- Fixed section order: Installation, Quick Start, Use Cases, API, Development.
- Every subsection leads with a minimal example, then explains.
- Enumerations use bullets or grouped bullets; tables are not used.
- Facts live in one place; other sections cross-link rather than repeat.

## README voice

**Direct and neutral. No marketing language.**

- "Give the agent access to filesystem tools", not "empower the application".
- Examples stay minimal; show the smallest snippet that demonstrates the feature.
- Example models are `claude-haiku-4-5-20251001` or `claude-sonnet-4-20250514`.
- Update triggers: a new builder method, a new tool, a new event kind, a new environment variable, or a changed default.

## README framing

**The factory-worker analogy is allowed in the README only.**

- Section openers and orchestration descriptions may use the analogy.
- Method tables, policy descriptions, and built-in I/O tools stay direct.
- Code, doc comments, and agentdocs do not use the analogy.

## README descriptions

**In README table cells, bullet descriptions, and inline `//` comments, describe what the caller gets, not how it works inside.**

- The reader may be new to agent concepts: write for them.
- The README is an abstraction: internal type names, private field names, and enum variant names do not belong there. The reference lives in the API docs.
- Accepted: `// run the task once and return the result`, "Transient provider error triggered a retry".
- Rejected: `// drive the loop`, `// one-shot`, "(carries typed `kind: RequestErrorKind`)", "`InBand` is model-fixable, `Infrastructure` is harness-level".
- Jargon and internal terms are cut even when they are shorter.
