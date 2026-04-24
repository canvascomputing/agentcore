//! On-disk JSONL transcript of an agent run. Keeps runs inspectable and replayable after the process exits.

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::persistence::error::PersistenceResult as Result;
use crate::provider::types::{Message, TokenUsage};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TranscriptEntry {
    pub(crate) recorded_at: u64,
    pub(crate) message: Message,
    pub(crate) usage: Option<TokenUsage>,
    pub(crate) model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)] // Used by list_sessions; tested but no production caller yet.
pub(crate) struct SessionMetadata {
    pub(crate) session_id: String,
    pub(crate) created_at: u64,
    pub(crate) last_active_at: u64,
    pub(crate) messages: u64,
}

/// Append-only JSONL transcript store.
///
/// Directory layout: `<base_dir>/sessions/<session_id>/transcript.jsonl`
pub(crate) struct SessionStore {
    base_dir: PathBuf,
    session_id: String,
    writer: Option<BufWriter<File>>,
}

#[allow(dead_code)] // Session resumption API — tested but not yet wired to CLI.
impl SessionStore {
    pub(crate) fn new(base_dir: &Path, session_id: &str) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
            session_id: session_id.to_string(),
            writer: None,
        }
    }

    /// Append a message to the transcript.
    pub(crate) fn record(&mut self, entry: TranscriptEntry) -> Result<()> {
        let line = serde_json::to_string(&entry)?;
        let writer = self.open_writer()?;
        writeln!(writer, "{line}")?;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub(crate) fn flush(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.writer {
            writer.flush()?;
        }
        Ok(())
    }

    /// Load all entries from a transcript file.
    pub(crate) fn load(base_dir: &Path, session_id: &str) -> Result<Vec<TranscriptEntry>> {
        let path = base_dir
            .join("sessions")
            .join(session_id)
            .join("transcript.jsonl");

        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            entries.push(serde_json::from_str(&line)?);
        }

        Ok(entries)
    }

    /// List available sessions with metadata derived from each transcript.
    pub(crate) fn list_sessions(base_dir: &Path) -> Result<Vec<SessionMetadata>> {
        let sessions_dir = base_dir.join("sessions");
        if !sessions_dir.exists() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();
        for entry in fs::read_dir(&sessions_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let session_id = entry.file_name().to_string_lossy().to_string();
            result.push(Self::metadata_from_transcript(base_dir, &session_id)?);
        }

        Ok(result)
    }

    fn session_dir(&self) -> PathBuf {
        self.base_dir.join("sessions").join(&self.session_id)
    }

    fn transcript_path(&self) -> PathBuf {
        self.session_dir().join("transcript.jsonl")
    }

    fn open_writer(&mut self) -> Result<&mut BufWriter<File>> {
        if self.writer.is_none() {
            fs::create_dir_all(self.session_dir())?;
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.transcript_path())?;
            self.writer = Some(BufWriter::new(file));
        }
        Ok(self.writer.as_mut().unwrap())
    }

    fn metadata_from_transcript(base_dir: &Path, session_id: &str) -> Result<SessionMetadata> {
        let entries = Self::load(base_dir, session_id)?;
        Ok(SessionMetadata {
            session_id: session_id.to_string(),
            created_at: entries.first().map(|e| e.recorded_at).unwrap_or(0),
            last_active_at: entries.last().map(|e| e.recorded_at).unwrap_or(0),
            messages: entries.len() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::types::{ContentBlock, Message};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_entry(message: Message) -> TranscriptEntry {
        TranscriptEntry {
            recorded_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            message,
            usage: Some(TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            }),
            model: Some("mock".into()),
        }
    }

    #[test]
    fn session_record_and_load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();

        let mut store = SessionStore::new(tmp.path(), "test-session");
        store.record(make_entry(Message::user("hello"))).unwrap();
        store
            .record(make_entry(Message::assistant("hi there")))
            .unwrap();
        store
            .record(make_entry(Message::User {
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "tool output".into(),
                    is_error: false,
                }],
            }))
            .unwrap();
        store.flush().unwrap();

        let entries = SessionStore::load(tmp.path(), "test-session").unwrap();
        assert_eq!(entries.len(), 3);
        assert!(entries[0].recorded_at > 0);
        assert!(entries[0].usage.is_some());
        assert_eq!(entries[0].model, Some("mock".into()));
    }

    #[test]
    fn session_list_returns_metadata() {
        let tmp = tempfile::tempdir().unwrap();

        let mut store1 = SessionStore::new(tmp.path(), "session-a");
        store1.record(make_entry(Message::user("a"))).unwrap();
        store1.flush().unwrap();

        let mut store2 = SessionStore::new(tmp.path(), "session-b");
        store2.record(make_entry(Message::user("b1"))).unwrap();
        store2.record(make_entry(Message::user("b2"))).unwrap();
        store2.flush().unwrap();

        let sessions = SessionStore::list_sessions(tmp.path()).unwrap();
        assert_eq!(sessions.len(), 2);

        let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
        assert!(ids.contains(&"session-a"));
        assert!(ids.contains(&"session-b"));
    }

    #[test]
    fn load_empty_session_returns_empty_vec() {
        let tmp = tempfile::tempdir().unwrap();
        let entries = SessionStore::load(tmp.path(), "nonexistent").unwrap();
        assert!(entries.is_empty());
    }
}
