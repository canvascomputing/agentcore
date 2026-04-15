//! Line-buffered parser for LLM streaming responses.
//!
//! LLM providers stream responses using the Server-Sent Events (SSE) protocol
//! (<https://html.spec.whatwg.org/multipage/server-sent-events.html>).
//! Each event is a `data: {json}\n` line, with `data: [DONE]\n` signaling the end.
//!
//! The parser buffers incoming byte chunks (which may arrive at arbitrary boundaries),
//! extracts complete lines, and yields parsed JSON events. Non-data lines (comments,
//! event types) and malformed JSON are silently skipped.
//!
//! Used by `AnthropicProvider` and `OpenAiProvider` in their streaming implementations.

use serde_json::Value;

/// A parsed stream event.
pub(crate) enum SseEvent {
    Data(Value),
    Done,
}

/// Line-buffered SSE parser. Feed raw bytes via `push()`, get parsed events back.
pub(crate) struct StreamParser {
    buffer: String,
}

impl StreamParser {
    pub(crate) fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Feed a chunk of bytes and return all complete SSE events found.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));

        let mut events = Vec::new();
        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].trim().to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                events.push(SseEvent::Done);
                continue;
            }
            if let Ok(json) = serde_json::from_str::<Value>(data) {
                events.push(SseEvent::Data(json));
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_data_line() {
        let mut parser = StreamParser::new();
        let events = parser.push(b"data: {\"type\":\"ping\"}\n\n");
        assert_eq!(events.len(), 1);
        match &events[0] {
            SseEvent::Data(v) => assert_eq!(v["type"], "ping"),
            SseEvent::Done => panic!("Expected Data"),
        }
    }

    #[test]
    fn parse_done_sentinel() {
        let mut parser = StreamParser::new();
        let events = parser.push(b"data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SseEvent::Done));
    }

    #[test]
    fn ignore_non_data_lines() {
        let mut parser = StreamParser::new();
        let events = parser.push(b"event: message_start\n: comment\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn buffer_split_lines() {
        let mut parser = StreamParser::new();

        let events = parser.push(b"data: {\"type\":\"pi");
        assert!(events.is_empty());

        let events = parser.push(b"ng\"}\n\n");
        assert_eq!(events.len(), 1);
        match &events[0] {
            SseEvent::Data(v) => assert_eq!(v["type"], "ping"),
            SseEvent::Done => panic!("Expected Data"),
        }
    }

    #[test]
    fn burst_events() {
        let mut parser = StreamParser::new();
        let chunk = b"data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n";
        let events = parser.push(chunk);
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], SseEvent::Data(v) if v["a"] == 1));
        assert!(matches!(&events[1], SseEvent::Data(v) if v["a"] == 2));
        assert!(matches!(events[2], SseEvent::Done));
    }

    #[test]
    fn skip_malformed_json() {
        let mut parser = StreamParser::new();
        let events = parser.push(b"data: not-json\ndata: {\"ok\":true}\n\n");
        assert_eq!(events.len(), 1);
        match &events[0] {
            SseEvent::Data(v) => assert_eq!(v["ok"], true),
            SseEvent::Done => panic!("Expected Data"),
        }
    }
}
