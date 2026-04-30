//! In-process FIFO queue of work tickets.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct TicketSystem {
    tickets: HashMap<String, Ticket>,
    next_id: u32,
    #[allow(dead_code)]
    directory: PathBuf,
}

#[derive(Debug)]
pub struct Ticket {
    pub key: String,
    pub summary: String,
    pub description: String,
    pub status: Status,
    pub r#type: TicketType,
    pub assignee: Option<String>,
    pub reporter: String,
    pub comments: Vec<Comment>,
    pub attachments: Vec<Attachment>,
    pub created_at: u64,
}

#[derive(Debug)]
pub struct Comment {
    pub author: String,
    pub body: String,
    pub created_at: u64,
}

#[derive(Debug)]
pub struct Attachment {
    pub filename: String,
    pub path: PathBuf,
    pub schema: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Todo,
    InProgress,
    Done,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TicketType {
    Task,
}

#[derive(Debug)]
pub enum TicketError {
    TicketMissing { key: String },
    TransitionRejected { from: Status, to: Status },
}

impl Default for TicketSystem {
    fn default() -> Self {
        Self {
            tickets: HashMap::new(),
            next_id: 1,
            directory: PathBuf::from("./tickets"),
        }
    }
}

impl TicketSystem {
    pub fn create(
        &mut self,
        summary: String,
        description: String,
        r#type: TicketType,
        reporter: String,
    ) -> &Ticket {
        let key = format!("TICKET-{}", self.next_id);
        self.next_id += 1;
        let ticket = Ticket {
            key: key.clone(),
            summary,
            description,
            status: Status::Todo,
            r#type,
            assignee: None,
            reporter,
            comments: Vec::new(),
            attachments: Vec::new(),
            created_at: now_millis(),
        };
        self.tickets.insert(key.clone(), ticket);
        &self.tickets[&key]
    }

    pub fn get(&self, key: &str) -> Option<&Ticket> {
        self.tickets.get(key)
    }

    pub fn update_status(&mut self, key: &str, status: Status) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing { key: key.to_string() })?;
        if !is_allowed_transition(ticket.status, status) {
            return Err(TicketError::TransitionRejected {
                from: ticket.status,
                to: status,
            });
        }
        if status == Status::Todo {
            ticket.assignee = None;
        }
        ticket.status = status;
        Ok(())
    }

    pub fn assign(&mut self, key: &str, assignee: String) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing { key: key.to_string() })?;
        ticket.assignee = Some(assignee);
        Ok(())
    }

    pub fn list_by_assignee(&self, assignee: &str) -> Vec<&Ticket> {
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| t.assignee.as_deref() == Some(assignee))
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub fn list_by_status(&self, status: Status) -> Vec<&Ticket> {
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| t.status == status)
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub fn search(&self, query: &str) -> Vec<&Ticket> {
        let needle = query.to_lowercase();
        let mut out: Vec<&Ticket> = self
            .tickets
            .values()
            .filter(|t| {
                t.summary.to_lowercase().contains(&needle)
                    || t.description.to_lowercase().contains(&needle)
            })
            .collect();
        out.sort_by_key(|t| (t.created_at, numeric_id(&t.key)));
        out
    }

    pub fn add_comment(
        &mut self,
        key: &str,
        author: String,
        body: String,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing { key: key.to_string() })?;
        ticket.comments.push(Comment {
            author,
            body,
            created_at: now_millis(),
        });
        Ok(())
    }

    pub fn add_attachment(
        &mut self,
        key: &str,
        attachment: Attachment,
    ) -> Result<(), TicketError> {
        let ticket = self
            .tickets
            .get_mut(key)
            .ok_or_else(|| TicketError::TicketMissing { key: key.to_string() })?;
        ticket.attachments.push(attachment);
        Ok(())
    }

    pub fn pending(&self) -> usize {
        self.tickets
            .values()
            .filter(|t| t.status == Status::Todo)
            .count()
    }
}

fn is_allowed_transition(from: Status, to: Status) -> bool {
    matches!(
        (from, to),
        (Status::Todo, Status::InProgress)
            | (Status::InProgress, Status::Todo)
            | (Status::InProgress, Status::Done)
            | (Status::InProgress, Status::Failed)
    )
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn numeric_id(key: &str) -> u32 {
    key.strip_prefix("TICKET-")
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

impl fmt::Display for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TicketMissing { key } => write!(f, "ticket {key} not found"),
            Self::TransitionRejected { from, to } => {
                write!(f, "cannot transition ticket from {from:?} to {to:?}")
            }
        }
    }
}

impl std::error::Error for TicketError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(system: &mut TicketSystem, summary: &str) -> String {
        system
            .create(
                summary.to_string(),
                String::new(),
                TicketType::Task,
                "tester".to_string(),
            )
            .key
            .clone()
    }

    fn finish(system: &mut TicketSystem, key: &str) {
        system.update_status(key, Status::InProgress).unwrap();
        system.update_status(key, Status::Done).unwrap();
    }

    fn attachment(name: &str) -> Attachment {
        Attachment {
            filename: name.to_string(),
            path: PathBuf::from(format!("/tmp/{name}")),
            schema: "file".to_string(),
        }
    }

    #[test]
    fn create_assigns_sequential_ticket_keys() {
        let mut system = TicketSystem::default();
        let first = task(&mut system, "first");
        let second = task(&mut system, "second");
        assert_eq!(first, "TICKET-1");
        assert_eq!(second, "TICKET-2");
    }

    #[test]
    fn create_starts_ticket_in_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "new work");
        assert_eq!(system.get(&key).unwrap().status, Status::Todo);
    }

    #[test]
    fn pending_is_zero_for_default_system() {
        let system = TicketSystem::default();
        assert_eq!(system.pending(), 0);
    }

    #[test]
    fn pending_counts_only_todo_tickets() {
        let mut system = TicketSystem::default();
        let claimed = task(&mut system, "claim me");
        let _waiting = task(&mut system, "wait");
        let finished = task(&mut system, "finish me");
        system.update_status(&claimed, Status::InProgress).unwrap();
        finish(&mut system, &finished);
        assert_eq!(system.pending(), 1);
    }

    #[test]
    fn update_status_transitions_todo_to_in_progress() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "claim");
        system.update_status(&key, Status::InProgress).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::InProgress);
    }

    #[test]
    fn update_status_transitions_in_progress_to_done() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "complete");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Done).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::Done);
    }

    #[test]
    fn update_status_transitions_in_progress_to_failed() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "give up");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Failed).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::Failed);
    }

    #[test]
    fn update_status_transitions_in_progress_back_to_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "release");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Todo).unwrap();
        assert_eq!(system.get(&key).unwrap().status, Status::Todo);
    }

    #[test]
    fn update_status_rejects_todo_to_done() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "skip");
        let err = system.update_status(&key, Status::Done).unwrap_err();
        assert!(matches!(
            err,
            TicketError::TransitionRejected {
                from: Status::Todo,
                to: Status::Done
            }
        ));
        assert_eq!(system.get(&key).unwrap().status, Status::Todo);
    }

    #[test]
    fn update_status_rejects_done_to_in_progress() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "done");
        finish(&mut system, &key);
        let err = system.update_status(&key, Status::InProgress).unwrap_err();
        assert!(matches!(
            err,
            TicketError::TransitionRejected {
                from: Status::Done,
                to: Status::InProgress
            }
        ));
        assert_eq!(system.get(&key).unwrap().status, Status::Done);
    }

    #[test]
    fn update_status_rejects_failed_to_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "failed");
        system.update_status(&key, Status::InProgress).unwrap();
        system.update_status(&key, Status::Failed).unwrap();
        let err = system.update_status(&key, Status::Todo).unwrap_err();
        assert!(matches!(
            err,
            TicketError::TransitionRejected {
                from: Status::Failed,
                to: Status::Todo
            }
        ));
        assert_eq!(system.get(&key).unwrap().status, Status::Failed);
    }

    #[test]
    fn update_status_clears_assignee_when_returning_to_todo() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "release me");
        system.assign(&key, "alice".to_string()).unwrap();
        system.update_status(&key, Status::InProgress).unwrap();
        assert_eq!(
            system.get(&key).unwrap().assignee.as_deref(),
            Some("alice")
        );
        system.update_status(&key, Status::Todo).unwrap();
        assert_eq!(system.get(&key).unwrap().assignee, None);
    }

    #[test]
    fn update_status_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .update_status("TICKET-999", Status::InProgress)
            .unwrap_err();
        let TicketError::TicketMissing { key } = err else {
            panic!("expected TicketMissing");
        };
        assert_eq!(key, "TICKET-999");
    }

    #[test]
    fn assign_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .assign("TICKET-999", "alice".to_string())
            .unwrap_err();
        assert!(matches!(err, TicketError::TicketMissing { .. }));
    }

    #[test]
    fn add_comment_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .add_comment("TICKET-999", "alice".to_string(), "hi".to_string())
            .unwrap_err();
        assert!(matches!(err, TicketError::TicketMissing { .. }));
    }

    #[test]
    fn add_attachment_returns_missing_for_unknown_key() {
        let mut system = TicketSystem::default();
        let err = system
            .add_attachment("TICKET-999", attachment("a.txt"))
            .unwrap_err();
        assert!(matches!(err, TicketError::TicketMissing { .. }));
    }

    #[test]
    fn add_comment_appends_to_ticket() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "discuss");
        system
            .add_comment(&key, "alice".to_string(), "looks good".to_string())
            .unwrap();
        system
            .add_comment(&key, "bob".to_string(), "agreed".to_string())
            .unwrap();
        let comments = &system.get(&key).unwrap().comments;
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].author, "alice");
        assert_eq!(comments[1].author, "bob");
    }

    #[test]
    fn add_attachment_appends_to_ticket() {
        let mut system = TicketSystem::default();
        let key = task(&mut system, "with files");
        system.add_attachment(&key, attachment("a.txt")).unwrap();
        system.add_attachment(&key, attachment("b.txt")).unwrap();
        let attachments = &system.get(&key).unwrap().attachments;
        assert_eq!(attachments.len(), 2);
        assert_eq!(attachments[0].filename, "a.txt");
        assert_eq!(attachments[1].filename, "b.txt");
    }

    #[test]
    fn get_returns_none_for_unknown_key() {
        let system = TicketSystem::default();
        assert!(system.get("TICKET-999").is_none());
    }

    #[test]
    fn list_by_status_returns_matching_tickets_in_creation_order() {
        let mut system = TicketSystem::default();
        let _a = task(&mut system, "a");
        let b = task(&mut system, "b");
        let _c = task(&mut system, "c");
        system.update_status(&b, Status::InProgress).unwrap();
        let todos = system.list_by_status(Status::Todo);
        let summaries: Vec<&str> = todos.iter().map(|t| t.summary.as_str()).collect();
        assert_eq!(summaries, vec!["a", "c"]);
    }

    #[test]
    fn list_by_status_returns_empty_when_no_match() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "still todo");
        assert!(system.list_by_status(Status::Done).is_empty());
    }

    #[test]
    fn list_by_assignee_returns_tickets_for_named_assignee() {
        let mut system = TicketSystem::default();
        let mine_a = task(&mut system, "mine a");
        let theirs = task(&mut system, "theirs");
        let mine_b = task(&mut system, "mine b");
        system.assign(&mine_a, "alice".to_string()).unwrap();
        system.assign(&theirs, "bob".to_string()).unwrap();
        system.assign(&mine_b, "alice".to_string()).unwrap();
        let alice = system.list_by_assignee("alice");
        let summaries: Vec<&str> = alice.iter().map(|t| t.summary.as_str()).collect();
        assert_eq!(summaries, vec!["mine a", "mine b"]);
    }

    #[test]
    fn list_by_assignee_returns_empty_when_no_match() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "unassigned");
        assert!(system.list_by_assignee("nobody").is_empty());
    }

    #[test]
    fn search_matches_summary_case_insensitively() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "Fix Login Bug");
        let _ = task(&mut system, "rewrite docs");
        let hits = system.search("login");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].summary, "Fix Login Bug");
    }

    #[test]
    fn search_matches_description_field() {
        let mut system = TicketSystem::default();
        system.create(
            "summary".to_string(),
            "secret keyword inside body".to_string(),
            TicketType::Task,
            "tester".to_string(),
        );
        let hits = system.search("keyword");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn search_returns_empty_when_no_match() {
        let mut system = TicketSystem::default();
        let _ = task(&mut system, "alpha");
        let _ = task(&mut system, "beta");
        assert!(system.search("gamma").is_empty());
    }
}
