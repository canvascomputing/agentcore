//! File-based todo list with per-item locking. Survives process restarts and lets peer agents coordinate through shared item records.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs;
use std::path::{Path, PathBuf};

use crate::persistence::error::{PersistenceError, PersistenceResult as Result};
use crate::persistence::lock::with_file_lock;
use crate::util::now_millis;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TodoItem {
    pub(crate) id: String,
    pub(crate) subject: String,
    pub(crate) description: String,
    pub(crate) status: TodoItemStatus,
    pub(crate) owner: Option<String>,
    pub(crate) blocks: Vec<String>,
    pub(crate) blocked_by: Vec<String>,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    #[serde(default)]
    pub(crate) metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TodoItemStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Default)]
pub(crate) struct TodoItemUpdate {
    pub(crate) status: Option<TodoItemStatus>,
    pub(crate) subject: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) metadata: Option<Map<String, Value>>,
}

/// Persists items to disk as individual JSON files.
pub(crate) struct TodoList {
    base_dir: PathBuf,
    list_id: String,
}

impl TodoList {
    pub(crate) fn new(base_dir: &Path, list_id: &str) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
            list_id: list_id.to_string(),
        }
    }

    pub(crate) fn create(
        &self,
        subject: &str,
        description: &str,
        metadata: Map<String, Value>,
    ) -> Result<TodoItem> {
        self.with_lock(|| {
            let mark = self.read_high_water_mark();
            let from_files = self.highest_item_id_on_disk();
            let next_id = mark.max(from_files) + 1;

            let now = now_millis();
            let item = TodoItem {
                id: next_id.to_string(),
                subject: subject.to_string(),
                description: description.to_string(),
                status: TodoItemStatus::Pending,
                owner: None,
                blocks: Vec::new(),
                blocked_by: Vec::new(),
                created_at: now,
                updated_at: now,
                metadata,
            };

            // Write mark BEFORE item file: crash-safe.
            self.write_high_water_mark(next_id)?;
            self.write_item(&item)?;
            Ok(item)
        })
    }

    pub(crate) fn get(&self, id: &str) -> Result<Option<TodoItem>> {
        self.read_item(id)
    }

    pub(crate) fn list(&self) -> Result<Vec<TodoItem>> {
        let dir = self.dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut items = Vec::new();
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if let Some(item) = self.read_item(id)? {
                items.push(item);
            }
        }

        items.sort_by_key(|t| t.id.parse::<u64>().unwrap_or(0));
        Ok(items)
    }

    pub(crate) fn update(&self, id: &str, update: TodoItemUpdate) -> Result<TodoItem> {
        self.with_lock(|| {
            let mut item = self.require_item(id)?;

            if let Some(status) = update.status {
                item.status = status;
            }
            if let Some(subject) = update.subject {
                item.subject = subject;
            }
            if let Some(description) = update.description {
                item.description = description;
            }
            if let Some(metadata) = update.metadata {
                item.metadata = metadata;
            }
            item.updated_at = now_millis();

            self.write_item(&item)?;
            Ok(item)
        })
    }

    pub(crate) fn delete(&self, id: &str) -> Result<()> {
        self.with_lock(|| {
            let path = self.item_file(id);
            if !path.exists() {
                return Ok(());
            }
            self.remove_from_all_dependencies(id)?;
            fs::remove_file(&path)?;
            Ok(())
        })
    }

    pub(crate) fn claim(&self, id: &str, agent_name: &str) -> Result<TodoItem> {
        self.with_lock(|| {
            let mut item = self.require_item(id)?;

            if item.status == TodoItemStatus::Completed {
                return Err(PersistenceError::TodoItemAlreadyCompleted(id.into()));
            }
            self.check_not_blocked(id, &item.blocked_by)?;

            item.status = TodoItemStatus::InProgress;
            item.owner = Some(agent_name.to_string());
            item.updated_at = now_millis();
            self.write_item(&item)?;
            Ok(item)
        })
    }

    pub(crate) fn add_dependency(&self, from: &str, to: &str) -> Result<()> {
        self.with_lock(|| {
            let mut from_item = self.require_item(from)?;
            let mut to_item = self.require_item(to)?;

            if !from_item.blocks.iter().any(|b| b == to) {
                from_item.blocks.push(to.to_string());
            }
            if !to_item.blocked_by.iter().any(|b| b == from) {
                to_item.blocked_by.push(from.to_string());
            }

            self.write_item(&from_item)?;
            self.write_item(&to_item)?;
            Ok(())
        })
    }

    pub(crate) fn remove_dependency(&self, from: &str, to: &str) -> Result<()> {
        self.with_lock(|| {
            let mut from_item = self.require_item(from)?;
            let mut to_item = self.require_item(to)?;

            from_item.blocks.retain(|id| id != to);
            to_item.blocked_by.retain(|id| id != from);

            self.write_item(&from_item)?;
            self.write_item(&to_item)?;
            Ok(())
        })
    }

    fn dir(&self) -> PathBuf {
        self.base_dir.join("items").join(&self.list_id)
    }

    fn item_file(&self, id: &str) -> PathBuf {
        self.dir().join(format!("{id}.json"))
    }

    fn with_lock<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        fs::create_dir_all(self.dir())?;
        let lock_file = self.dir().join(".lock");
        with_file_lock(&lock_file, f)
    }

    fn read_item(&self, id: &str) -> Result<Option<TodoItem>> {
        let path = self.item_file(id);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&path)?;
        let item: TodoItem = serde_json::from_str(&content)?;
        Ok(Some(item))
    }

    fn require_item(&self, id: &str) -> Result<TodoItem> {
        self.read_item(id)?
            .ok_or_else(|| PersistenceError::TodoItemNotFound(id.into()))
    }

    fn write_item(&self, item: &TodoItem) -> Result<()> {
        let json = serde_json::to_string_pretty(item)?;
        fs::write(self.item_file(&item.id), json)?;
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

    fn highest_item_id_on_disk(&self) -> u64 {
        let Ok(entries) = fs::read_dir(self.dir()) else {
            return 0;
        };
        entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let name = name.to_str()?;
                name.strip_suffix(".json")?.parse::<u64>().ok()
            })
            .max()
            .unwrap_or(0)
    }

    fn check_not_blocked(&self, item_id: &str, blocked_by: &[String]) -> Result<()> {
        for blocker_id in blocked_by {
            let Some(blocker) = self.read_item(blocker_id)? else {
                continue;
            };
            if blocker.status != TodoItemStatus::Completed {
                return Err(PersistenceError::TodoItemBlocked {
                    item_id: item_id.into(),
                    blocker_id: blocker_id.clone(),
                });
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
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if id == deleted_id {
                continue;
            }
            let Some(mut item) = self.read_item(id)? else {
                continue;
            };

            let before = item.blocks.len() + item.blocked_by.len();
            item.blocks.retain(|b| b != deleted_id);
            item.blocked_by.retain(|b| b != deleted_id);
            if item.blocks.len() + item.blocked_by.len() != before {
                self.write_item(&item)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> (tempfile::TempDir, TodoList) {
        let tmp = tempfile::tempdir().unwrap();
        let store = TodoList::new(tmp.path(), "test");
        (tmp, store)
    }

    fn create(store: &TodoList, subject: &str, description: &str) -> TodoItem {
        store.create(subject, description, Map::new()).unwrap()
    }

    #[test]
    fn create_and_get() {
        let (_tmp, store) = test_store();
        let item = create(&store, "Design API", "Define endpoints");
        assert_eq!(item.subject, "Design API");
        assert_eq!(item.status, TodoItemStatus::Pending);
        assert!(item.owner.is_none());
        assert_eq!(item.id, "1");

        let loaded = store.get("1").unwrap().unwrap();
        assert_eq!(loaded.subject, "Design API");
        assert_eq!(loaded.description, "Define endpoints");
    }

    #[test]
    fn list_returns_all_items() {
        let (_tmp, store) = test_store();
        create(&store, "Item 1", "desc 1");
        create(&store, "Item 2", "desc 2");
        create(&store, "Item 3", "desc 3");

        let items = store.list().unwrap();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn update_status() {
        let (_tmp, store) = test_store();
        create(&store, "Item", "desc");

        let updated = store
            .update(
                "1",
                TodoItemUpdate {
                    status: Some(TodoItemStatus::InProgress),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(updated.status, TodoItemStatus::InProgress);

        let loaded = store.get("1").unwrap().unwrap();
        assert_eq!(loaded.status, TodoItemStatus::InProgress);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let (_tmp, store) = test_store();
        assert!(store.get("999").unwrap().is_none());
    }

    #[test]
    fn delete_removes_item() {
        let (_tmp, store) = test_store();
        create(&store, "Item", "desc");
        store.delete("1").unwrap();
        assert!(store.get("1").unwrap().is_none());
    }

    #[test]
    fn ids_never_reused_after_delete() {
        let (_tmp, store) = test_store();
        create(&store, "Item 1", "");
        create(&store, "Item 2", "");
        create(&store, "Item 3", "");
        store.delete("2").unwrap();

        let item = create(&store, "Item 4", "");
        assert_eq!(item.id, "4");
    }

    #[test]
    fn high_water_mark_survives_all_deletions() {
        let (_tmp, store) = test_store();
        create(&store, "Item 1", "");
        create(&store, "Item 2", "");
        store.delete("1").unwrap();
        store.delete("2").unwrap();

        let item = create(&store, "Item 3", "");
        assert_eq!(item.id, "3");
    }

    #[test]
    fn claim_blocked_item_fails() {
        let (_tmp, store) = test_store();
        let a = create(&store, "A", "");
        let b = create(&store, "B", "");
        store.add_dependency(&a.id, &b.id).unwrap();

        let err = store.claim(&b.id, "agent_1").unwrap_err();
        assert!(format!("{err}").contains("blocked"));
    }

    #[test]
    fn claim_after_blocker_completes() {
        let (_tmp, store) = test_store();
        let a = create(&store, "A", "");
        let b = create(&store, "B", "");
        store.add_dependency(&a.id, &b.id).unwrap();

        store
            .update(
                &a.id,
                TodoItemUpdate {
                    status: Some(TodoItemStatus::Completed),
                    ..Default::default()
                },
            )
            .unwrap();

        let claimed = store.claim(&b.id, "agent_2").unwrap();
        assert_eq!(claimed.status, TodoItemStatus::InProgress);
        assert_eq!(claimed.owner, Some("agent_2".into()));
    }

    #[test]
    fn delete_cascades_dependency_removal() {
        let (_tmp, store) = test_store();
        let a = create(&store, "A", "");
        let b = create(&store, "B", "");
        store.add_dependency(&a.id, &b.id).unwrap();

        store.delete(&a.id).unwrap();

        let b_loaded = store.get(&b.id).unwrap().unwrap();
        assert!(b_loaded.blocked_by.is_empty());
    }

    #[test]
    fn claim_completed_item_fails() {
        let (_tmp, store) = test_store();
        create(&store, "Item", "");
        store
            .update(
                "1",
                TodoItemUpdate {
                    status: Some(TodoItemStatus::Completed),
                    ..Default::default()
                },
            )
            .unwrap();

        let err = store.claim("1", "agent").unwrap_err();
        assert!(format!("{err}").contains("completed"));
    }

    #[test]
    fn metadata_round_trips_through_create_list_get() {
        let (_tmp, store) = test_store();
        let mut metadata = Map::new();
        metadata.insert("priority".into(), Value::String("high".into()));
        metadata.insert("ticket".into(), Value::String("ENG-42".into()));

        let created = store.create("Subject", "Body", metadata.clone()).unwrap();
        assert_eq!(created.metadata, metadata);

        let loaded = store.get(&created.id).unwrap().unwrap();
        assert_eq!(loaded.metadata, metadata);

        let listed = store.list().unwrap();
        assert_eq!(listed[0].metadata, metadata);
    }

    #[test]
    fn legacy_item_without_metadata_field_loads_with_empty_map() {
        let (tmp, store) = test_store();
        // Hand-write a JSON file lacking the metadata field, mimicking a record
        // produced by an older binary.
        let dir = tmp.path().join("items").join("test");
        fs::create_dir_all(&dir).unwrap();
        let legacy = r#"{
            "id": "1",
            "subject": "old",
            "description": "old",
            "status": "Pending",
            "owner": null,
            "blocks": [],
            "blocked_by": [],
            "created_at": 0,
            "updated_at": 0
        }"#;
        fs::write(dir.join("1.json"), legacy).unwrap();

        let loaded = store.get("1").unwrap().unwrap();
        assert!(loaded.metadata.is_empty());
    }

    #[test]
    fn update_replaces_description_and_metadata_independently() {
        let (_tmp, store) = test_store();
        let mut original = Map::new();
        original.insert("k".into(), Value::String("v".into()));
        store.create("subject", "first body", original).unwrap();

        let after_desc = store
            .update(
                "1",
                TodoItemUpdate {
                    description: Some("second body".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(after_desc.description, "second body");
        assert_eq!(after_desc.metadata.get("k").unwrap(), "v");

        let mut replacement = Map::new();
        replacement.insert("k2".into(), Value::String("v2".into()));
        let after_meta = store
            .update(
                "1",
                TodoItemUpdate {
                    metadata: Some(replacement.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(after_meta.description, "second body");
        assert_eq!(after_meta.metadata, replacement);
    }

    #[test]
    fn remove_dependency_drops_both_edges() {
        let (_tmp, store) = test_store();
        let a = create(&store, "A", "");
        let b = create(&store, "B", "");
        store.add_dependency(&a.id, &b.id).unwrap();

        store.remove_dependency(&a.id, &b.id).unwrap();

        let a_loaded = store.get(&a.id).unwrap().unwrap();
        let b_loaded = store.get(&b.id).unwrap().unwrap();
        assert!(a_loaded.blocks.is_empty());
        assert!(b_loaded.blocked_by.is_empty());

        // Idempotent: a second call is a no-op.
        store.remove_dependency(&a.id, &b.id).unwrap();
    }

    #[test]
    fn remove_dependency_unblocks_claim() {
        let (_tmp, store) = test_store();
        let a = create(&store, "A", "");
        let b = create(&store, "B", "");
        store.add_dependency(&a.id, &b.id).unwrap();

        assert!(store.claim(&b.id, "agent").is_err());
        store.remove_dependency(&a.id, &b.id).unwrap();
        assert!(store.claim(&b.id, "agent").is_ok());
    }

    #[test]
    fn concurrent_creation_no_duplicate_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let base = base.clone();
                std::thread::spawn(move || {
                    let store = TodoList::new(&base, "concurrent");
                    store.create(&format!("Item {i}"), "", Map::new()).unwrap()
                })
            })
            .collect();

        let mut ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap().id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 10, "Expected 10 unique IDs, got: {ids:?}");
    }
}
