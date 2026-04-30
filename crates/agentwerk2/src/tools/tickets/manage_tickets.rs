//! Read + mutate access to the surrounding ticket queue. Union of
//! `ReadTicketsTool` and `WriteTicketsTool`.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use crate::providers::ProviderResult;

use super::super::tool::{ToolContext, ToolLike, ToolResult};
use super::super::tool_file::ToolFile;
use super::{dispatch, READ_ACTIONS, WRITE_ACTIONS};

/// All eight ticket actions in one tool.
pub struct ManageTicketsTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("manage_tickets.tool.json")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

fn all_actions() -> &'static [&'static str] {
    static ACTIONS: OnceLock<Vec<&'static str>> = OnceLock::new();
    ACTIONS
        .get_or_init(|| {
            let mut v = Vec::with_capacity(READ_ACTIONS.len() + WRITE_ACTIONS.len());
            v.extend_from_slice(READ_ACTIONS);
            v.extend_from_slice(WRITE_ACTIONS);
            v
        })
        .as_slice()
}

impl ToolLike for ManageTicketsTool {
    fn name(&self) -> &str {
        &tool_file().name
    }

    fn description(&self) -> &str {
        description()
    }

    fn input_schema(&self) -> Value {
        tool_file().input_schema.clone()
    }

    fn is_read_only(&self) -> bool {
        tool_file().read_only
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
        Box::pin(async move { Ok(dispatch(input, ctx, all_actions())) })
    }
}
