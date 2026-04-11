use crate::message::Usage;

#[derive(Debug, Clone)]
pub enum Event {
    TurnStart { agent: String, turn: u32 },
    Text { agent: String, text: String },
    ToolStart { agent: String, tool: String, id: String, input: serde_json::Value },
    ToolEnd { agent: String, tool: String, id: String, result: String, is_error: bool },
    Usage { agent: String, model: String, usage: Usage },
    AgentStart { agent: String },
    AgentEnd { agent: String, turns: u32 },
    Error { agent: String, error: String },
}
