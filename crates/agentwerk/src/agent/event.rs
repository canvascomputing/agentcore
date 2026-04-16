use crate::provider::types::TokenUsage;

#[derive(Debug, Clone)]
pub enum Event {
    AgentStart { agent_name: String },
    AgentEnd { agent_name: String, turns: u32 },
    TurnStart { agent_name: String, turn: u32 },
    TurnEnd { agent_name: String, turn: u32 },
    ToolCallStart { agent_name: String, tool_name: String, call_id: String, input: serde_json::Value },
    ToolCallEnd { agent_name: String, tool_name: String, call_id: String, output: String, is_error: bool },
    TokenUsage { agent_name: String, model: String, usage: TokenUsage },
    ResponseTextChunk { agent_name: String, content: String },
    RequestStart { agent_name: String, model: String },
    RequestEnd { agent_name: String, model: String },
}
