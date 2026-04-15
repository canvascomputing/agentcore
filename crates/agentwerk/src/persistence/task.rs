use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{AgenticError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub subject: String,
    pub description: String,
    pub status: TaskStatus,
    pub owner: Option<String>,
    pub blocks: Vec<String>,
    pub blocked_by: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Default)]
pub struct TaskUpdate {
    pub status: Option<TaskStatus>,
    pub subject: Option<String>,
    pub description: Option<String>,
    pub owner: Option<Option<String>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Persists tasks to disk as individual JSON files.
pub struct TaskStore {
    base_dir: PathBuf,
    list_id: String,
}

impl TaskStore {
    pub fn open(base_dir: &Path, list_id: &str) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
            list_id: list_id.to_string(),
        }
    }

    pub fn create(&self, subject: &str, description: &str) -> Result<Task> {
        self.with_lock(|| {
            let mark = self.read_high_water_mark();
            let from_files = self.highest_task_id_on_disk();
            let next_id = mark.max(from_files) + 1;

            let now = now_millis();
            let task = Task {
                id: next_id.to_string(),
                subject: subject.to_string(),
                description: description.to_string(),
                status: TaskStatus::Pending,
                owner: None,
                blocks: Vec::new(),
                blocked_by: Vec::new(),
                metadata: HashMap::new(),
                created_at: now,
                updated_at: now,
            };

            // Write mark BEFORE task file — crash-safe
            self.write_high_water_mark(next_id)?;
            self.write_task(&task)?;
            Ok(task)
        })
    }

    pub fn get(&self, id: &str) -> Result<Option<Task>> {
        self.read_task(id)
    }

    pub fn list(&self) -> Result<Vec<Task>> {
        let dir = self.dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut tasks = Vec::new();
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if let Some(task) = self.read_task(id)? {
                tasks.push(task);
            }
        }

        tasks.sort_by_key(|t| t.id.parse::<u64>().unwrap_or(0));
        Ok(tasks)
    }

    pub fn update(&self, id: &str, update: TaskUpdate) -> Result<Task> {
        self.with_lock(|| {
            let mut task = self.require_task(id)?;

            if let Some(status) = update.status {
                task.status = status;
            }
            if let Some(subject) = update.subject {
                task.subject = subject;
            }
            if let Some(description) = update.description {
                task.description = description;
            }
            if let Some(owner) = update.owner {
                task.owner = owner;
            }
            if let Some(metadata) = update.metadata {
                task.metadata = metadata;
            }
            task.updated_at = now_millis();

            self.write_task(&task)?;
            Ok(task)
        })
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        self.with_lock(|| {
            let path = self.task_path(id);
            if !path.exists() {
                return Ok(());
            }
            self.remove_from_all_dependencies(id)?;
            fs::remove_file(&path)?;
            Ok(())
        })
    }

    pub fn claim(&self, id: &str, agent_name: &str) -> Result<Task> {
        self.with_lock(|| {
            let mut task = self.require_task(id)?;

            if task.status == TaskStatus::Completed {
                return Err(AgenticError::Other(format!(
                    "Task {id} already completed"
                )));
            }
            self.check_not_blocked(id, &task.blocked_by)?;

            task.status = TaskStatus::InProgress;
            task.owner = Some(agent_name.to_string());
            task.updated_at = now_millis();
            self.write_task(&task)?;
            Ok(task)
        })
    }

    pub fn add_dependency(&self, from: &str, to: &str) -> Result<()> {
        self.with_lock(|| {
            let mut from_task = self.require_task(from)?;
            let mut to_task = self.require_task(to)?;

            if !from_task.blocks.contains(&to.to_string()) {
                from_task.blocks.push(to.to_string());
            }
            if !to_task.blocked_by.contains(&from.to_string()) {
                to_task.blocked_by.push(from.to_string());
            }

            self.write_task(&from_task)?;
            self.write_task(&to_task)?;
            Ok(())
        })
    }

    fn dir(&self) -> PathBuf {
        self.base_dir.join("tasks").join(&self.list_id)
    }

    fn task_path(&self, id: &str) -> PathBuf {
        self.dir().join(format!("{id}.json"))
    }

    fn with_lock<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        fs::create_dir_all(self.dir())?;
        let lock_path = self.dir().join(".lock");
        with_file_lock(&lock_path, f)
    }

    fn read_task(&self, id: &str) -> Result<Option<Task>> {
        let path = self.task_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&path)?;
        let task: Task = serde_json::from_str(&content)?;
        Ok(Some(task))
    }

    fn require_task(&self, id: &str) -> Result<Task> {
        self.read_task(id)?
            .ok_or_else(|| AgenticError::Other(format!("Task {id} not found")))
    }

    fn write_task(&self, task: &Task) -> Result<()> {
        let json = serde_json::to_string_pretty(task)?;
        fs::write(self.task_path(&task.id), json)?;
        Ok(())
    }

    fn read_high_water_mark(&self) -> u64 {
        let path = self.dir().join(".highwatermark");
        fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    fn write_high_water_mark(&self, mark: u64) -> Result<()> {
        let path = self.dir().join(".highwatermark");
        fs::write(&path, mark.to_string())?;
        Ok(())
    }

    fn highest_task_id_on_disk(&self) -> u64 {
        let dir = self.dir();
        if !dir.exists() {
            return 0;
        }
        fs::read_dir(&dir)
            .ok()
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        name.strip_suffix(".json")
                            .and_then(|s| s.parse::<u64>().ok())
                    })
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    fn check_not_blocked(&self, task_id: &str, blocked_by: &[String]) -> Result<()> {
        for blocker_id in blocked_by {
            let Some(blocker) = self.read_task(blocker_id)? else {
                continue;
            };
            if blocker.status != TaskStatus::Completed {
                return Err(AgenticError::Other(format!(
                    "Task {task_id} blocked by unfinished task {blocker_id}"
                )));
            }
        }
        Ok(())
    }

    fn remove_from_all_dependencies(&self, deleted_id: &str) -> Result<()> {
        let dir = self.dir();
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if id == deleted_id {
                continue;
            }
            let Some(mut task) = self.read_task(id)? else {
                continue;
            };

            let references_deleted = task.blocks.contains(&deleted_id.to_string())
                || task.blocked_by.contains(&deleted_id.to_string());
            if !references_deleted {
                continue;
            }

            task.blocks.retain(|b| b != deleted_id);
            task.blocked_by.retain(|b| b != deleted_id);
            self.write_task(&task)?;
        }
        Ok(())
    }
}

const MAX_LOCK_RETRIES: u32 = 30;
const MIN_BACKOFF_MS: u64 = 5;
const MAX_BACKOFF_MS: u64 = 100;

fn with_file_lock<T>(lock_path: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;

    let mut backoff_ms = MIN_BACKOFF_MS;
    for _ in 0..MAX_LOCK_RETRIES {
        if try_lock_exclusive(&file)? {
            let result = f();
            unlock(&file)?;
            return result;
        }
        std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
        backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
    }

    Err(AgenticError::Other(format!(
        "Failed to acquire lock after {MAX_LOCK_RETRIES} attempts"
    )))
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
        Ok(false)
    } else {
        Err(AgenticError::Io(err))
    }
}

#[cfg(unix)]
fn unlock(file: &File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
    if ret == 0 {
        Ok(())
    } else {
        Err(AgenticError::Io(std::io::Error::last_os_error()))
    }
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> Result<bool> {
    Ok(true)
}

#[cfg(not(unix))]
fn unlock(_file: &File) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> (tempfile::TempDir, TaskStore) {
        let tmp = tempfile::tempdir().unwrap();
        let store = TaskStore::open(tmp.path(), "test");
        (tmp, store)
    }

    #[test]
    fn create_and_get() {
        let (_tmp, store) = test_store();
        let task = store.create("Design API", "Define endpoints").unwrap();
        assert_eq!(task.subject, "Design API");
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.owner.is_none());
        assert_eq!(task.id, "1");

        let loaded = store.get("1").unwrap().unwrap();
        assert_eq!(loaded.subject, "Design API");
        assert_eq!(loaded.description, "Define endpoints");
    }

    #[test]
    fn list_returns_all_tasks() {
        let (_tmp, store) = test_store();
        store.create("Task 1", "desc 1").unwrap();
        store.create("Task 2", "desc 2").unwrap();
        store.create("Task 3", "desc 3").unwrap();

        let tasks = store.list().unwrap();
        assert_eq!(tasks.len(), 3);
    }

    #[test]
    fn update_status() {
        let (_tmp, store) = test_store();
        store.create("Task", "desc").unwrap();

        let updated = store
            .update(
                "1",
                TaskUpdate {
                    status: Some(TaskStatus::InProgress),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(updated.status, TaskStatus::InProgress);

        let loaded = store.get("1").unwrap().unwrap();
        assert_eq!(loaded.status, TaskStatus::InProgress);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let (_tmp, store) = test_store();
        assert!(store.get("999").unwrap().is_none());
    }

    #[test]
    fn delete_removes_task() {
        let (_tmp, store) = test_store();
        store.create("Task", "desc").unwrap();
        store.delete("1").unwrap();
        assert!(store.get("1").unwrap().is_none());
    }

    #[test]
    fn ids_never_reused_after_delete() {
        let (_tmp, store) = test_store();
        store.create("Task 1", "").unwrap();
        store.create("Task 2", "").unwrap();
        store.create("Task 3", "").unwrap();
        store.delete("2").unwrap();

        let task = store.create("Task 4", "").unwrap();
        assert_eq!(task.id, "4");
    }

    #[test]
    fn high_water_mark_survives_all_deletions() {
        let (_tmp, store) = test_store();
        store.create("Task 1", "").unwrap();
        store.create("Task 2", "").unwrap();
        store.delete("1").unwrap();
        store.delete("2").unwrap();

        let task = store.create("Task 3", "").unwrap();
        assert_eq!(task.id, "3");
    }

    #[test]
    fn claim_blocked_task_fails() {
        let (_tmp, store) = test_store();
        let a = store.create("A", "").unwrap();
        let b = store.create("B", "").unwrap();
        store.add_dependency(&a.id, &b.id).unwrap();

        let err = store.claim(&b.id, "agent_1").unwrap_err();
        assert!(format!("{err}").contains("blocked"));
    }

    #[test]
    fn claim_after_blocker_completes() {
        let (_tmp, store) = test_store();
        let a = store.create("A", "").unwrap();
        let b = store.create("B", "").unwrap();
        store.add_dependency(&a.id, &b.id).unwrap();

        store
            .update(
                &a.id,
                TaskUpdate {
                    status: Some(TaskStatus::Completed),
                    ..Default::default()
                },
            )
            .unwrap();

        let claimed = store.claim(&b.id, "agent_2").unwrap();
        assert_eq!(claimed.status, TaskStatus::InProgress);
        assert_eq!(claimed.owner, Some("agent_2".into()));
    }

    #[test]
    fn delete_cascades_dependency_removal() {
        let (_tmp, store) = test_store();
        let a = store.create("A", "").unwrap();
        let b = store.create("B", "").unwrap();
        store.add_dependency(&a.id, &b.id).unwrap();

        store.delete(&a.id).unwrap();

        let b_loaded = store.get(&b.id).unwrap().unwrap();
        assert!(b_loaded.blocked_by.is_empty());
    }

    #[test]
    fn claim_completed_task_fails() {
        let (_tmp, store) = test_store();
        store.create("Task", "").unwrap();
        store
            .update(
                "1",
                TaskUpdate {
                    status: Some(TaskStatus::Completed),
                    ..Default::default()
                },
            )
            .unwrap();

        let err = store.claim("1", "agent").unwrap_err();
        assert!(format!("{err}").contains("completed"));
    }

    #[test]
    fn concurrent_creation_no_duplicate_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let base = base.clone();
                std::thread::spawn(move || {
                    let store = TaskStore::open(&base, "concurrent");
                    store.create(&format!("Task {i}"), "").unwrap()
                })
            })
            .collect();

        let mut ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap().id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 10, "Expected 10 unique IDs, got: {ids:?}");
    }
}
