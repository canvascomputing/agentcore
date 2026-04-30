//! In-process ticket tracker for agent work items.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

#[allow(dead_code)]
pub struct TicketSystem {
    tickets: HashMap<String, Ticket>,
    next_id: u32,
    directory: PathBuf,
}

pub struct Ticket {
    pub summary: String,
    pub description: String,
    pub status: Status,
    pub r#type: TicketType,
    pub assignee: Option<String>,
    pub reporter: String,
    pub comments: Vec<Comment>,
    pub attachments: Vec<Attachment>,
}

pub struct Comment {
    pub author: String,
    pub body: String,
    pub created_at: u64,
}

pub struct Attachment {
    pub filename: String,
    pub path: PathBuf,
    pub schema: String,
}

pub enum Status {
    Todo,
    InProgress,
    Done,
    Failed,
}

pub enum TicketType {
    Task,
}

pub enum TicketError {
    NotFound,
    InvalidTransition,
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
        let _ = (summary, description, r#type, reporter);
        todo!()
    }

    pub fn get(&self, key: &str) -> Option<&Ticket> {
        let _ = key;
        todo!()
    }

    pub fn update_status(&mut self, key: &str, status: Status) -> Result<(), TicketError> {
        let _ = (key, status);
        todo!()
    }

    pub fn assign(&mut self, key: &str, assignee: String) -> Result<(), TicketError> {
        let _ = (key, assignee);
        todo!()
    }

    pub fn list_by_assignee(&self, assignee: &str) -> Vec<&Ticket> {
        let _ = assignee;
        todo!()
    }

    pub fn list_by_status(&self, status: Status) -> Vec<&Ticket> {
        let _ = status;
        todo!()
    }

    pub fn search(&self, query: &str) -> Vec<&Ticket> {
        let _ = query;
        todo!()
    }

    pub fn add_comment(
        &mut self,
        key: &str,
        author: String,
        body: String,
    ) -> Result<(), TicketError> {
        let _ = (key, author, body);
        todo!()
    }

    pub fn add_attachment(
        &mut self,
        key: &str,
        attachment: Attachment,
    ) -> Result<(), TicketError> {
        let _ = (key, attachment);
        todo!()
    }
}

impl fmt::Display for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "ticket not found"),
            Self::InvalidTransition => write!(f, "invalid status transition"),
        }
    }
}

impl fmt::Debug for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "NotFound"),
            Self::InvalidTransition => write!(f, "InvalidTransition"),
        }
    }
}

impl std::error::Error for TicketError {}
