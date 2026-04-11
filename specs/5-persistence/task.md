# TaskStore

## Overview

Tasks are structured work items that agents create, track, and complete. They persist to disk so they survive process restarts and can be shared across concurrent agents. This sub-plan covers the task data model, `TaskStore` with file locking, high water mark ID management, and the dependency graph.

## Dependencies

- [Core types](../1-base/types.md): `AgenticError`, `Result`

## Files

```
crates/agent-core/src/task.rs
```

## Specification

### 4. Task Persistence and Tracking (`task.rs`)

#### 4.1 Task Data Model

```rust
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
```

#### 4.2 Task Persistence (Disk-Based Store)

```rust
/// Persists tasks to disk as individual JSON files.
/// Thread-safe via file locking for multi-agent access.
///
/// Directory layout:
///   <base_dir>/tasks/<list_id>/
///     ├── 1.json
///     ├── 2.json
///     ├── .highwatermark
///     └── .lock
pub struct TaskStore {
    base_dir: PathBuf,
    list_id: String,
}

impl TaskStore {
    pub fn open(base_dir: &Path, list_id: &str) -> Self;
    pub fn create(&self, subject: &str, description: &str) -> Result<Task>;
    pub fn get(&self, id: &str) -> Result<Option<Task>>;
    pub fn list(&self) -> Result<Vec<Task>>;
    pub fn update(&self, id: &str, update: TaskUpdate) -> Result<Task>;
    pub fn delete(&self, id: &str) -> Result<()>;
    pub fn claim(&self, id: &str, agent_id: &str) -> Result<Task>;
    pub fn add_dependency(&self, from: &str, to: &str) -> Result<()>;
}

#[derive(Debug, Default)]
pub struct TaskUpdate {
    pub status: Option<TaskStatus>,
    pub subject: Option<String>,
    pub description: Option<String>,
    pub owner: Option<Option<String>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}
```

#### 4.3 File Locking (Concurrent Agent Safety)

```rust
const MAX_RETRIES: u32 = 30;
const MIN_BACKOFF_MS: u64 = 5;
const MAX_BACKOFF_MS: u64 = 100;

/// Execute `f` while holding an exclusive advisory lock on `lock_path`.
/// Retries with exponential backoff: 5ms→100ms, up to 30 attempts (~2.6s total).
pub fn with_lock<T>(lock_path: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let file = OpenOptions::new().create(true).write(true).open(lock_path)?;
    let mut backoff_ms = MIN_BACKOFF_MS;
    for _ in 0..MAX_RETRIES {
        if try_lock_exclusive(&file)? {
            let result = f();
            unlock(&file)?;
            return result;
        }
        thread::sleep(Duration::from_millis(backoff_ms));
        backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
    }
    Err(AgenticError::Other(format!("Failed to acquire lock after {MAX_RETRIES} attempts")))
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 { Ok(true) }
    else if std::io::Error::last_os_error().raw_os_error() == Some(libc::EWOULDBLOCK) { Ok(false) }
    else { Err(AgenticError::Io(std::io::Error::last_os_error())) }
}
```

#### 4.4 High Water Mark (ID Management)

Task IDs are sequential integers (1, 2, 3, ...). Deleted IDs are never reused. The `.highwatermark` file tracks the highest ID ever assigned.

```rust
impl TaskStore {
    pub fn create(&self, subject: &str, description: &str) -> Result<Task> {
        let lock_path = self.dir().join(".lock");
        with_lock(&lock_path, || {
            let mark = self.read_high_water_mark();
            let from_files = self.highest_task_id_on_disk();
            let next_id = mark.max(from_files) + 1;

            let task = Task {
                id: next_id.to_string(),
                subject: subject.to_string(),
                description: description.to_string(),
                status: TaskStatus::Pending,
                owner: None,
                blocks: Vec::new(),
                blocked_by: Vec::new(),
                metadata: HashMap::new(),
                created_at: now_millis(),
                updated_at: now_millis(),
            };

            // Write mark BEFORE task file — crash-safe
            self.write_high_water_mark(next_id)?;
            let path = self.dir().join(format!("{}.json", next_id));
            std::fs::write(&path, serde_json::to_string_pretty(&task)?)?;
            Ok(task)
        })
    }
}
```

**Crash safety**: The high water mark is written *before* the task file. If the process crashes between these writes, the mark is already updated — no future process will reuse this ID.

#### 4.5 Task Dependency Graph

Tasks form a DAG via `blocks` and `blocked_by` arrays. A task cannot be claimed until all its `blocked_by` tasks are completed. When a task is deleted, it is removed from all other tasks' dependency arrays.

```rust
impl TaskStore {
    pub fn add_dependency(&self, from: &str, to: &str) -> Result<()> {
        let lock_path = self.dir().join(".lock");
        with_lock(&lock_path, || {
            let mut from_task = self.read_task(from)?.ok_or_else(|| ...)?;
            let mut to_task = self.read_task(to)?.ok_or_else(|| ...)?;

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

    pub fn claim(&self, id: &str, agent_id: &str) -> Result<Task> {
        let lock_path = self.dir().join(".lock");
        with_lock(&lock_path, || {
            let mut task = self.read_task(id)?.ok_or_else(|| ...)?;
            if task.status == TaskStatus::Completed {
                return Err(AgenticError::Other(format!("Task {id} already completed")));
            }
            for blocker_id in &task.blocked_by {
                if let Some(blocker) = self.read_task(blocker_id)? {
                    if blocker.status != TaskStatus::Completed {
                        return Err(AgenticError::Other(format!(
                            "Task {id} blocked by unfinished task {blocker_id}"
                        )));
                    }
                }
            }
            task.status = TaskStatus::InProgress;
            task.owner = Some(agent_id.to_string());
            task.updated_at = now_millis();
            self.write_task(&task)?;
            Ok(task)
        })
    }
}
```

**Example: Multi-agent task workflow**

```rust
let store = TaskStore::open(Path::new("~/.agent/tasks"), "project_1");
let design = store.create("Design API schema", "Define REST endpoints")?;
let implement = store.create("Implement API handlers", "Build route handlers")?;
let test = store.create("Write integration tests", "Test all endpoints")?;

store.add_dependency(&design.id, &implement.id)?;
store.add_dependency(&implement.id, &test.id)?;

let task = store.claim(&design.id, "agent_1")?;
store.update(&design.id, TaskUpdate { status: Some(TaskStatus::Completed), ..Default::default() })?;
let task = store.claim(&implement.id, "agent_2")?;
assert!(store.claim(&test.id, "agent_3").is_err()); // still blocked
```

## Work Items

1. **`task.rs`** — Spec Sections 4.1-4.5
   - `Task`, `TaskStatus`, `TaskUpdate` structs
   - `TaskStore` — disk-based store with per-list directory
   - File locking: `with_lock()` using `libc::flock` (Unix), 30 retries, 5-100ms exponential backoff
   - High water mark: `.highwatermark` file, written before task file for crash safety
   - CRUD: `create()`, `get()`, `list()`, `update()`, `delete()`
   - Dependencies: `add_dependency()`, `claim()` with blocked_by validation, `remove_from_all_dependencies()`

## Tests

All tests use `tempfile::TempDir` for filesystem isolation. Shared helper: `test_store() -> (TempDir, TaskStore)`.

### `task.rs` Tests

| Test | Pattern | What it verifies |
|------|---------|-----------------|
| `create_and_get` | Direct | Create task, verify subject/status/owner; reload by ID |
| `list_returns_all_tasks` | Direct | Create 3 tasks, list() returns 3 |
| `update_status` | Direct | Create task, update to InProgress, verify persisted |
| `delete_removes_task` | Direct | Create task, delete, get() returns None |
| `get_nonexistent_returns_none` | Direct | get("999") on empty store returns None |
| `ids_never_reused_after_delete` | Direct | Create 3, delete #2, create new → ID is 4 |
| `high_water_mark_survives_all_deletions` | Direct | Create 2, delete both, create new → ID is 3 |
| `claim_blocked_task_fails` | Direct | A blocks B; claim(B) returns error |
| `claim_after_blocker_completes` | Direct | A blocks B; complete A; claim(B) succeeds |
| `delete_cascades_dependency_removal` | Direct | A blocks B; delete A → B's blocked_by empty |
| `claim_completed_task_fails` | Direct | Complete a task, then claim() returns error |
| `concurrent_creation_no_duplicate_ids` | Concurrency | 10 threads × create(): all 10 unique IDs |

## Done Criteria

- All tests pass
- Tasks survive process restart (write, new process, read — data correct)
- Concurrent task creation from 10 threads produces 10 unique IDs
