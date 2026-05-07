//! File-backed memory that persists across tickets and runs. The model-facing
//! `MemoryTool` is a thin wrapper in `tools::memory` that holds an
//! `Arc<Memory>` from this module.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Yardstick the char-limit check joins entries with. Not a file format
/// detail: on disk every entry is its own JSON line. Kept stable across the
/// JSONL switch so the limit semantics do not shift under existing callers.
const LIMIT_SEPARATOR: &str = "\n§\n";
const DEFAULT_CHAR_LIMIT: usize = 2200;
const MEMORY_FILE: &str = "memory.jsonl";

/// One on-disk entry. Round-trips through `memory.jsonl`. `added_at` is
/// preserved across reads; `add` and `replace` stamp it with the current
/// time, `rewrite` stamps every entry afresh.
#[derive(Clone, Serialize, Deserialize)]
struct MemoryRecord {
    content: String,
    added_at: u64,
}

/// File-backed memory shared by every `Agent` bound to it. Mirrors `TicketSystem`
/// in shape: the caller constructs one `Arc<Memory>` and binds it to one or
/// more agents through `Agent::memory(&store)`. Two agents pointed at the same
/// `Arc` share `memory.jsonl`; two agents pointed at different stores see
/// independent memory. For the single-agent case, [`Agent::memory`] also takes
/// a path directly: it opens the store under the hood.
pub struct Memory {
    memory_dir: PathBuf,
    entries: Mutex<Vec<MemoryRecord>>,
    char_limit: usize,
    write_lock: Mutex<()>,
}

impl Memory {
    /// Open or create a memory store rooted at `memory_dir`. The directory is
    /// created if missing. `memory.jsonl` is read line by line when present:
    /// malformed lines are dropped, duplicates by `content` are collapsed.
    /// Char limit defaults to 2200.
    pub fn open(memory_dir: impl Into<PathBuf>) -> io::Result<Arc<Self>> {
        let memory_dir = memory_dir.into();
        fs::create_dir_all(&memory_dir)?;
        let entries = read_entries_from_disk(&memory_dir.join(MEMORY_FILE))?;
        Ok(Arc::new(Self {
            memory_dir,
            entries: Mutex::new(entries),
            char_limit: DEFAULT_CHAR_LIMIT,
            write_lock: Mutex::new(()),
        }))
    }

    /// A clone of the current entry contents, in insertion order. Callers
    /// that need a single string (the loop, a REPL `/memory` command)
    /// concatenate them with whatever separator suits their format. Empty
    /// when no entries.
    pub fn entries(&self) -> Vec<String> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.content.clone())
            .collect()
    }

    /// Append a new entry. Rejects empty content, verbatim duplicates, and
    /// content that would push the rendered prompt section past the char
    /// limit.
    pub fn add(&self, content: &str) -> Result<(), String> {
        let _w = self.write_lock.lock().unwrap();
        let mut entries = self.entries.lock().unwrap().clone();
        let content = content.trim();
        if content.is_empty() {
            return Err("Content must not be empty".into());
        }
        if entries.iter().any(|e| e.content == content) {
            return Err("An entry with identical content already exists".into());
        }
        let current_chars = render_for_prompt(&entries).len();
        entries.push(MemoryRecord {
            content: content.to_string(),
            added_at: now_ms(),
        });
        let rendered = render_for_prompt(&entries);
        if rendered.len() > self.char_limit {
            return Err(format!(
                "Memory at {}/{} chars. Adding this entry ({} chars) would exceed the limit. \
                 Replace or remove existing entries first.",
                current_chars,
                self.char_limit,
                content.len(),
            ));
        }
        write_entries_to_disk(&self.memory_dir.join(MEMORY_FILE), &entries)
            .map_err(|e| format!("Failed to persist memory: {e}"))?;
        *self.entries.lock().unwrap() = entries;
        Ok(())
    }

    /// Replace the unique entry containing `old_text` with `content`. Stamps
    /// `added_at` with the current time on the replacement.
    pub fn replace(&self, old_text: &str, content: &str) -> Result<(), String> {
        let _w = self.write_lock.lock().unwrap();
        let mut entries = self.entries.lock().unwrap().clone();
        let idx = unique_match(&entries, old_text)?;
        let content = content.trim();
        if content.is_empty() {
            return Err("Replacement content must not be empty".into());
        }
        entries[idx] = MemoryRecord {
            content: content.to_string(),
            added_at: now_ms(),
        };
        let rendered = render_for_prompt(&entries);
        if rendered.len() > self.char_limit {
            return Err(format!(
                "Replacement would push memory to {} chars (limit {}). \
                 Trim the new content or remove another entry first.",
                rendered.len(),
                self.char_limit,
            ));
        }
        write_entries_to_disk(&self.memory_dir.join(MEMORY_FILE), &entries)
            .map_err(|e| format!("Failed to persist memory: {e}"))?;
        *self.entries.lock().unwrap() = entries;
        Ok(())
    }

    /// Drop the unique entry containing `old_text`.
    pub fn remove(&self, old_text: &str) -> Result<(), String> {
        let _w = self.write_lock.lock().unwrap();
        let mut entries = self.entries.lock().unwrap().clone();
        let idx = unique_match(&entries, old_text)?;
        entries.remove(idx);
        write_entries_to_disk(&self.memory_dir.join(MEMORY_FILE), &entries)
            .map_err(|e| format!("Failed to persist memory: {e}"))?;
        *self.entries.lock().unwrap() = entries;
        Ok(())
    }

    /// Replace every entry with `new_entries`. Used by callers that drive
    /// their own consolidation. Skips the char-limit check: a misbehaving
    /// rewrite would still be caught at the next `add` attempt. Stamps a
    /// fresh `added_at` on every entry.
    pub fn rewrite(&self, new_entries: Vec<String>) -> Result<(), String> {
        let _w = self.write_lock.lock().unwrap();
        let now = now_ms();
        let cleaned: Vec<MemoryRecord> = new_entries
            .into_iter()
            .map(|e| e.trim().to_string())
            .filter(|e| !e.is_empty())
            .map(|content| MemoryRecord {
                content,
                added_at: now,
            })
            .collect();
        write_entries_to_disk(&self.memory_dir.join(MEMORY_FILE), &cleaned)
            .map_err(|e| format!("Failed to persist memory: {e}"))?;
        *self.entries.lock().unwrap() = cleaned;
        Ok(())
    }
}

/// What [`Agent::memory`] accepts. A `&Arc<Memory>` shares an existing store
/// across agents; a path opens a fresh store rooted at that directory. The
/// path forms map straight onto [`Memory::open`].
pub trait IntoMemory {
    fn into_memory(self) -> io::Result<Arc<Memory>>;
}

impl IntoMemory for &Arc<Memory> {
    fn into_memory(self) -> io::Result<Arc<Memory>> {
        Ok(Arc::clone(self))
    }
}

impl IntoMemory for PathBuf {
    fn into_memory(self) -> io::Result<Arc<Memory>> {
        Memory::open(self)
    }
}

impl IntoMemory for &PathBuf {
    fn into_memory(self) -> io::Result<Arc<Memory>> {
        Memory::open(self)
    }
}

impl IntoMemory for &Path {
    fn into_memory(self) -> io::Result<Arc<Memory>> {
        Memory::open(self)
    }
}

impl IntoMemory for &str {
    fn into_memory(self) -> io::Result<Arc<Memory>> {
        Memory::open(self)
    }
}

fn unique_match(entries: &[MemoryRecord], needle: &str) -> Result<usize, String> {
    if needle.is_empty() {
        return Err("`old_text` must not be empty".into());
    }
    let hits: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.content.contains(needle).then_some(i))
        .collect();
    match hits.len() {
        0 => Err(format!(
            "No memory entry contains `{needle}`. List the entries first or pick a different substring."
        )),
        1 => Ok(hits[0]),
        n => Err(format!(
            "`{needle}` matches {n} memory entries. Pick a longer unique substring."
        )),
    }
}

fn render_for_prompt(entries: &[MemoryRecord]) -> String {
    entries
        .iter()
        .map(|e| e.content.as_str())
        .collect::<Vec<_>>()
        .join(LIMIT_SEPARATOR)
}

fn read_entries_from_disk(path: &Path) -> io::Result<Vec<MemoryRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path)?;
    let mut entries: Vec<MemoryRecord> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<MemoryRecord>(trimmed) else {
            continue;
        };
        if record.content.is_empty() {
            continue;
        }
        if seen.iter().any(|s| s == &record.content) {
            continue;
        }
        seen.push(record.content.clone());
        entries.push(record);
    }
    Ok(entries)
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_entries_to_disk(path: &Path, entries: &[MemoryRecord]) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let body = serialize_to_jsonl(entries)?;
    atomic_write(path, body.as_bytes())
}

fn serialize_to_jsonl(entries: &[MemoryRecord]) -> io::Result<String> {
    let mut out = String::new();
    for entry in entries {
        let line = serde_json::to_string(entry).map_err(io::Error::other)?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn atomic_write(path: &Path, body: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "memory".to_string());
    let temp = parent.join(format!(".{file_name}.tmp.{pid}.{counter}"));
    let result = (|| -> io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        f.write_all(body)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_store() -> (Arc<Memory>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Memory::open(dir.path()).unwrap();
        (store, dir)
    }

    fn read_records(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn open_creates_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("not-yet-there");
        let _ = Memory::open(&nested).unwrap();
        assert!(nested.exists());
    }

    #[test]
    fn open_with_no_existing_file_starts_empty() {
        let (store, _dir) = fresh_store();
        assert!(store.entries().is_empty());
    }

    #[test]
    fn add_writes_one_jsonl_line_per_entry() {
        let (store, dir) = fresh_store();
        store.add("hello world").unwrap();
        assert_eq!(store.entries(), vec!["hello world".to_string()]);
        let records = read_records(&dir.path().join("memory.jsonl"));
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["content"], "hello world");
        assert!(records[0]["added_at"].as_u64().unwrap() > 0);
    }

    #[test]
    fn add_appends_one_record_per_call() {
        let (store, dir) = fresh_store();
        store.add("first").unwrap();
        store.add("second").unwrap();
        let records = read_records(&dir.path().join("memory.jsonl"));
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["content"], "first");
        assert_eq!(records[1]["content"], "second");
    }

    #[test]
    fn entries_added_in_one_run_survive_drop_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let s1 = Memory::open(dir.path()).unwrap();
        s1.add("first").unwrap();
        s1.add("second").unwrap();
        drop(s1);
        let s2 = Memory::open(dir.path()).unwrap();
        assert_eq!(
            s2.entries(),
            vec!["first".to_string(), "second".to_string()]
        );
    }

    #[test]
    fn add_rejected_when_content_is_duplicate_and_leaves_entries_unchanged() {
        let (store, _dir) = fresh_store();
        store.add("note").unwrap();
        let before = store.entries();
        let err = store.add("note").unwrap_err();
        assert!(err.contains("identical"));
        assert_eq!(store.entries(), before);
    }

    #[test]
    fn add_rejected_when_content_exceeds_char_limit_and_leaves_entries_unchanged() {
        let (store, _dir) = fresh_store();
        store.add("first").unwrap();
        let before = store.entries();
        let big = "x".repeat(DEFAULT_CHAR_LIMIT + 1);
        let err = store.add(&big).unwrap_err();
        assert!(err.contains("chars"), "{err}");
        assert!(err.contains("Replace or remove"), "{err}");
        assert_eq!(store.entries(), before);
    }

    #[test]
    fn replace_swaps_unique_entry_in_place() {
        let (store, _dir) = fresh_store();
        store.add("one").unwrap();
        store.add("two").unwrap();
        store.replace("one", "ONE updated").unwrap();
        assert_eq!(
            store.entries(),
            vec!["ONE updated".to_string(), "two".to_string()]
        );
    }

    #[test]
    fn replace_rejected_when_old_text_matches_no_entry_and_leaves_entries_unchanged() {
        let (store, _dir) = fresh_store();
        store.add("alpha note").unwrap();
        let before = store.entries();
        let err = store.replace("zeta", "x").unwrap_err();
        assert!(err.contains("No memory entry"));
        assert_eq!(store.entries(), before);
    }

    #[test]
    fn replace_rejected_when_old_text_matches_multiple_entries_and_leaves_entries_unchanged() {
        let (store, _dir) = fresh_store();
        store.add("alpha note").unwrap();
        store.add("alpha rule").unwrap();
        let before = store.entries();
        let err = store.replace("alpha", "x").unwrap_err();
        assert!(err.contains("matches 2"));
        assert_eq!(store.entries(), before);
    }

    #[test]
    fn replace_rejected_when_new_content_would_exceed_char_limit_and_leaves_entries_unchanged() {
        let (store, _dir) = fresh_store();
        store.add("seed").unwrap();
        let before = store.entries();
        let big = "x".repeat(DEFAULT_CHAR_LIMIT + 1);
        let err = store.replace("seed", &big).unwrap_err();
        assert!(err.contains("limit"), "{err}");
        assert_eq!(store.entries(), before);
    }

    #[test]
    fn remove_drops_unique_entry() {
        let (store, _dir) = fresh_store();
        store.add("one").unwrap();
        store.add("two").unwrap();
        store.remove("one").unwrap();
        assert_eq!(store.entries(), vec!["two".to_string()]);
    }

    #[test]
    fn rewrite_replaces_every_entry() {
        let (store, _dir) = fresh_store();
        store.add("one").unwrap();
        store.add("two").unwrap();
        store.add("three").unwrap();
        store.rewrite(vec!["consolidated".to_string()]).unwrap();
        assert_eq!(store.entries(), vec!["consolidated".to_string()]);
    }

    #[test]
    fn rewrite_with_empty_entries_clears_memory() {
        let (store, _dir) = fresh_store();
        store.add("one").unwrap();
        store.add("two").unwrap();
        store.rewrite(Vec::new()).unwrap();
        assert!(store.entries().is_empty());
    }

    #[test]
    fn writes_through_one_arc_clone_are_visible_through_another() {
        let (store, _dir) = fresh_store();
        let other = Arc::clone(&store);
        store.add("shared note").unwrap();
        assert_eq!(other.entries(), vec!["shared note".to_string()]);
    }

    #[test]
    fn open_dedupes_disk_entries_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let body = "{\"content\":\"same\",\"added_at\":1}\n\
                    {\"content\":\"same\",\"added_at\":2}\n\
                    {\"content\":\"other\",\"added_at\":3}\n";
        fs::write(dir.path().join("memory.jsonl"), body).unwrap();
        let store = Memory::open(dir.path()).unwrap();
        assert_eq!(
            store.entries(),
            vec!["same".to_string(), "other".to_string()]
        );
    }

    #[test]
    fn open_silently_drops_malformed_lines() {
        let dir = tempfile::tempdir().unwrap();
        let body = "{\"content\":\"good\",\"added_at\":1}\n\
                    not json at all\n\
                    {\"missing_added_at\":\"x\"}\n\
                    {\"content\":\"also good\",\"added_at\":2}\n";
        fs::write(dir.path().join("memory.jsonl"), body).unwrap();
        let store = Memory::open(dir.path()).unwrap();
        assert_eq!(
            store.entries(),
            vec!["good".to_string(), "also good".to_string()]
        );
    }

    #[test]
    fn add_stamps_added_at_to_a_non_zero_value() {
        let (store, dir) = fresh_store();
        store.add("hello").unwrap();
        let records = read_records(&dir.path().join("memory.jsonl"));
        assert!(records[0]["added_at"].as_u64().unwrap() > 0);
    }

    #[test]
    fn round_trip_preserves_added_at_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let body = "{\"content\":\"persisted\",\"added_at\":1234567890}\n";
        fs::write(dir.path().join("memory.jsonl"), body).unwrap();
        let store = Memory::open(dir.path()).unwrap();
        // A no-op operation that triggers a rewrite of the file would
        // overwrite added_at; reading without writing preserves it.
        store.add("fresh").unwrap();
        let records = read_records(&dir.path().join("memory.jsonl"));
        let persisted = records
            .iter()
            .find(|r| r["content"] == "persisted")
            .unwrap();
        assert_eq!(persisted["added_at"].as_u64().unwrap(), 1234567890);
    }
}
