use crate::agent::output::Status;
use crate::provider::types::TokenUsage;

#[derive(Debug, Clone)]
pub struct Event {
    pub agent_name: String,
    pub kind: EventKind,
}

impl Event {
    pub(crate) fn new(agent_name: impl Into<String>, kind: EventKind) -> Self {
        Self { agent_name: agent_name.into(), kind }
    }
}

#[derive(Debug, Clone)]
pub enum EventKind {
    /// Agent begins execution. `description` carries the LLM-supplied label when
    /// an agent was spawned via `SpawnAgentTool`; `None` for root runs.
    AgentStart { description: Option<String> },
    AgentEnd { turns: u32, status: Status },
    TurnStart { turn: u32 },
    TurnEnd { turn: u32 },
    ToolCallStart { tool_name: String, call_id: String, input: serde_json::Value },
    ToolCallEnd { tool_name: String, call_id: String, output: String, is_error: bool },
    TokenUsage { model: String, usage: TokenUsage },
    ResponseTextChunk { content: String },
    RequestStart { model: String },
    RequestEnd { model: String },
    TurnPaused { turn: u32 },
    TurnResumed { turn: u32, continuation: u32 },
}
