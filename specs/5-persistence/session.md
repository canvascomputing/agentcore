# SessionStore

## Overview

Sessions persist the full conversation transcript to disk as append-only JSONL files, enabling agents to resume work across process restarts.

## Dependencies

- [Core types](../1-base/types.md): `Message`, `ContentBlock`, `Usage`, `AgenticError`, `Result`

## Files

```
crates/agent-core/src/session.rs
```

## Specification

### 5. Session Persistence and Transcript (`session.rs`)

#### 5.1 Transcript Format

```rust
/// Append-only JSONL log of all messages in a session.
///
/// File: <base_dir>/sessions/<session_id>/transcript.jsonl
pub struct SessionStore {
    base_dir: PathBuf,
    session_id: String,
    writer: Option<BufWriter<File>>,  // lazy-opened on first write
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub recorded_at: u64,
    pub entry_type: EntryType,
    pub message: Message,
    pub usage: Option<Usage>,
    pub model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EntryType {
    UserMessage,
    AssistantMessage,
    ToolResult,
    SystemEvent,
}

impl SessionStore {
    pub fn new(base_dir: &Path, session_id: &str) -> Self;

    /// Append a message to the transcript. Buffered; flushed periodically or on explicit flush.
    pub fn record(&mut self, entry: TranscriptEntry) -> Result<()>;

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> Result<()>;

    /// Load all messages from a transcript file (for session resume).
    pub fn load(base_dir: &Path, session_id: &str) -> Result<Vec<TranscriptEntry>>;

    /// List available sessions with metadata.
    pub fn list_sessions(base_dir: &Path) -> Result<Vec<SessionMetadata>>;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub session_id: String,
    pub created_at: u64,
    pub last_active_at: u64,
    pub message_count: u64,
    pub total_cost_usd: f64,
}
```

**Example: Recording and resuming a session**

```rust
let mut store = SessionStore::new(Path::new("~/.agent"), "session_abc123");

store.record(TranscriptEntry {
    recorded_at: now_millis(),
    entry_type: EntryType::UserMessage,
    message: Message::User {
        content: vec![ContentBlock::Text { text: "Explain the auth module".into() }],
    },
    usage: None,
    model: None,
})?;

store.record(TranscriptEntry {
    recorded_at: now_millis(),
    entry_type: EntryType::AssistantMessage,
    message: Message::Assistant {
        content: vec![ContentBlock::Text { text: "The auth module handles...".into() }],
    },
    usage: Some(Usage { input_tokens: 1200, output_tokens: 350, ..Default::default() }),
    model: Some("claude-sonnet-4-20250514".into()),
})?;
store.flush()?;

// Later: resume the session
let entries = SessionStore::load(Path::new("~/.agent"), "session_abc123")?;
let messages: Vec<Message> = entries.iter().map(|e| e.message.clone()).collect();
```

#### 5.2 Session Resume Flow

When a session is resumed:

1. `SessionStore::load(session_id)` reads the JSONL transcript file.
2. Entries are reconstructed into a `Vec<Message>`.
3. Cost state is restored by summing `usage` fields.
4. `Agent::run_with_messages(messages, new_prompt, on_event)` feeds the restored history into the agent loop.

#### 5.3 Directory Layout

```
~/.agent/
├── sessions/
│   ├── <session_id>/
│   │   ├── transcript.jsonl
│   │   ├── metadata.json
│   │   └── subagents/
│   │       └── <agent_id>.jsonl
│   └── <session_id>/
│       └── ...
├── projects/
│   └── <project_slug>/
│       └── memory/
├── tasks/
│   └── <list_id>/
│       ├── 1.json
│       ├── 2.json
│       └── .highwatermark
└── config.json
```

## Work Items

1. **`session.rs`** — Spec Sections 5.1-5.3
   - `SessionStore` — append-only JSONL writer with `BufWriter`
   - `TranscriptEntry`, `EntryType`, `SessionMetadata` structs
   - `record()` — append entry, `flush()` — flush buffer
   - `load()` — read all entries from JSONL file
   - `list_sessions()` — scan session directory, return metadata

## Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `session_record_and_load_round_trip` | Direct | Record 3 entries, flush, load — verify count, recorded_at, usage, model |
| `session_list_returns_metadata` | Direct | Create 2 sessions, list_sessions() returns 2 metadata entries |
| `load_empty_session_returns_empty_vec` | Direct | Empty transcript.jsonl → load() returns empty vec |

## Done Criteria

- All tests pass
- Sessions survive process restart (record, flush, new process, load — data correct)
- JSONL format is append-only and human-readable
