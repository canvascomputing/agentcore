# Spec: Session Resumption

**Date**: 2026-04-16

---

## Problem

Sessions are recorded as JSONL transcripts but can never be resumed. Each `AgentBuilder::run()` starts a fresh conversation. There is no way for a caller to continue a previous session — the loaded `TranscriptEntry` data exists on disk but has no path back into the agent loop.

## How the Reference Implementation Does It

The reference implementation's sessions flow through three stages:

```
Disk (JSONL)                    In-memory                     API request
TranscriptMessage[]     →     Message[]               →     MessageParam[]
(full message + metadata)      (message only, no metadata)    (role + content only)
```

### Write path (during execution):
1. Each turn appends to `transcript.jsonl` as a `TranscriptMessage` — the full `Message` plus session metadata (`parentUuid`, `sessionId`, `cwd`, `version`, `gitBranch`)
2. Messages link via `parentUuid` to form a chain (enables branching and sidechain detection)

### Resume path:
1. `loadTranscriptFile()` reads JSONL → `Map<UUID, TranscriptMessage>`
2. `buildConversationChain()` walks `parentUuid` links → ordered `TranscriptMessage[]`
3. `deserializeMessagesWithInterruptDetection()` strips metadata → plain `Message[]`
4. Detects interrupted turns (incomplete tool_use/tool_result pairs) and truncates
5. Resulting `Message[]` seeds `State.messages` in the query loop — standard multi-turn conversation from there

### Context management (not needed immediately):
- **AutoCompact**: at ~200K tokens, summarizes old spans into a single summary message
- **ContextCollapse**: read-time projection that hides message ranges without deleting them

## Current agentcore State

### Types involved:

```rust
// What gets stored on disk (one per JSONL line)
struct TranscriptEntry {
    recorded_at: u64,
    entry_type: EntryType,   // UserMessage | AssistantMessage | ToolResult | SystemEvent
    message: Message,        // The actual conversation message
    usage: Option<TokenUsage>,
    model: Option<String>,
}

// What the agent loop works with
enum Message {
    System { content: String },
    User { content: Vec<ContentBlock> },
    Assistant { content: Vec<ContentBlock> },
}
```

### Recording (write path, works today):
- `record_initial_message()` — records the user's instruction prompt
- `record_transcript(EntryType::AssistantMessage, ...)` — after each LLM response, with `usage` and `model`
- `record_transcript(EntryType::ToolResult, ...)` — after tool execution, no usage/model

### What's missing:
- No way to load a transcript and feed it back as `state.messages`
- No interrupted-turn detection
- `AgentBuilder` always generates a fresh session ID

## Design

### Resume via AgentBuilder

```rust
AgentBuilder::new()
    .provider(provider)
    .model("claude-sonnet-4-20250514")
    .session_directory(PathBuf::from("./data"))
    .resume_session("session_abc123")           // ← new
    .instruction_prompt("Continue where we left off.")
    .run()
    .await?;
```

`resume_session(session_id)` stores the session ID. At `run()` time:
1. Load transcript: `SessionStore::load(&session_directory, &session_id)` → `Vec<TranscriptEntry>`
2. Extract messages: `entries.iter().map(|e| e.message.clone()).collect()` → `Vec<Message>`
3. Detect interrupted turns (trailing `Assistant` with `ToolUse` blocks but no subsequent `ToolResult`) — truncate those
4. Seed `state.messages` with the loaded messages instead of building from scratch
5. Reuse the same `session_id` for the `SessionStore` (appends to existing JSONL)
6. Append the new instruction prompt as a `User` message
7. Continue normal agent loop from there

### Interrupted turn detection

An interrupted turn looks like:
```
... → Assistant { content: [ToolUse { id: "x", ... }] } → (nothing, session ended)
```

The agent expects every `ToolUse` to have a matching `ToolResult`. On resume:
- Scan the last message: if it's `Assistant` with any `ToolUse` content blocks, check if the next message has `ToolResult` blocks for each `tool_use_id`
- If not, remove the trailing `Assistant` message (the LLM will regenerate it)

### Session ID handling

| Scenario | Session ID | JSONL file |
|---|---|---|
| Fresh session | Auto-generated (`session_{nanos}`) | New file |
| Resume session | Caller-provided | Appends to existing file |

### Flush on completion

The agent loop must call `session_store.flush()` when execution ends (both success and error paths). Currently this doesn't happen — `BufWriter` may drop data on crash.

## Changes

### `crates/agentcore/src/agent/builder.rs`
- Add `resume_session_id: Option<String>` field
- Add `pub fn resume_session(mut self, session_id: impl Into<String>) -> Self`
- In `run()`: if `resume_session_id` is set, load transcript and extract messages

### `crates/agentcore/src/agent/loop.rs`
- `init_state()`: accept optional `Vec<Message>` for resumed messages instead of building fresh
- Add `flush_session()` call in the finalization path of `execute()`
- Extract `detect_interrupted_turn(messages: &mut Vec<Message>)` helper

### `crates/agentcore/src/persistence/session.rs`
- Add `pub fn messages_from_transcript(entries: &[TranscriptEntry]) -> Vec<Message>` utility
- Add `pub fn detect_interrupted_turn(messages: &mut Vec<Message>)` — removes trailing incomplete turns

## Out of Scope

- Context compression / auto-compact (separate feature)
- Message UUIDs and parent chains (adds complexity, not needed for basic resumption)
- Session branching / forking
- Session search / tagging

## Verification

```bash
make test   # unit tests for interrupted turn detection, message extraction
```

Integration test:
1. Run agent with `session_directory` → produces transcript
2. Load transcript, resume with new instruction → appends to same file
3. Verify the resumed agent sees prior messages and responds coherently
