//! Run-time stats. One [`Stats`] struct of atomic counters records every
//! observable event and exposes inherent read accessors. Each domain
//! interacts with its own write-only protocol — [`LoopStats`] for the
//! agent loop, [`TicketStats`] for the ticket system — so a domain
//! cannot reach another domain's events. The wiring is internal: the
//! caller never sees `Stats` at construction time, only afterwards
//! through `TicketSystem::stats()`.
//!
//! Lock-free for counter increments; readers do one atomic load per
//! call.

use std::sync::atomic::{AtomicU64, Ordering};

/// Recorder protocol for the agent loop. Each agent holds an
/// `Arc<dyn LoopStats + Send + Sync>` and reports loop events through
/// it. Write-only; reads happen on `Stats` directly.
pub trait LoopStats: Send + Sync {
    fn record_step(&self);
    fn record_request(&self, input_tokens: u64, output_tokens: u64);
    fn record_tool_call(&self);
    fn record_error(&self);
}

/// Recorder protocol for the ticket system. The ticket system holds an
/// `Arc<Stats>` directly but only exercises these methods, so by
/// convention the ticket-domain surface is narrow.
pub(crate) trait TicketStats: Send + Sync {
    fn record_created(&self);
    fn record_done(&self);
    fn record_failed(&self);
}

/// Run-wide counters. Implements every recorder protocol; exposes
/// inherent read methods for the caller to consume after a run.
pub struct Stats {
    steps: AtomicU64,
    requests: AtomicU64,
    tool_calls: AtomicU64,
    errors: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    tickets_created: AtomicU64,
    tickets_done: AtomicU64,
    tickets_failed: AtomicU64,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            steps: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            tool_calls: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
            tickets_created: AtomicU64::new(0),
            tickets_done: AtomicU64::new(0),
            tickets_failed: AtomicU64::new(0),
        }
    }

    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    pub fn requests(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    pub fn tool_calls(&self) -> u64 {
        self.tool_calls.load(Ordering::Relaxed)
    }

    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    pub fn input_tokens(&self) -> u64 {
        self.input_tokens.load(Ordering::Relaxed)
    }

    pub fn output_tokens(&self) -> u64 {
        self.output_tokens.load(Ordering::Relaxed)
    }

    pub fn tickets_created(&self) -> u64 {
        self.tickets_created.load(Ordering::Relaxed)
    }

    pub fn tickets_done(&self) -> u64 {
        self.tickets_done.load(Ordering::Relaxed)
    }

    pub fn tickets_failed(&self) -> u64 {
        self.tickets_failed.load(Ordering::Relaxed)
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopStats for Stats {
    fn record_step(&self) {
        self.steps.fetch_add(1, Ordering::Relaxed);
    }

    fn record_request(&self, input_tokens: u64, output_tokens: u64) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.input_tokens.fetch_add(input_tokens, Ordering::Relaxed);
        self.output_tokens.fetch_add(output_tokens, Ordering::Relaxed);
    }

    fn record_tool_call(&self) {
        self.tool_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }
}

impl TicketStats for Stats {
    fn record_created(&self) {
        self.tickets_created.fetch_add(1, Ordering::Relaxed);
    }

    fn record_done(&self) {
        self.tickets_done.fetch_add(1, Ordering::Relaxed);
    }

    fn record_failed(&self) {
        self.tickets_failed.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_stats_are_zero() {
        let s = Stats::new();
        assert_eq!(s.steps(), 0);
        assert_eq!(s.requests(), 0);
        assert_eq!(s.tool_calls(), 0);
        assert_eq!(s.errors(), 0);
        assert_eq!(s.input_tokens(), 0);
        assert_eq!(s.output_tokens(), 0);
        assert_eq!(s.tickets_created(), 0);
        assert_eq!(s.tickets_done(), 0);
        assert_eq!(s.tickets_failed(), 0);
    }

    #[test]
    fn loop_stats_writes_show_up_in_reads() {
        let s = Stats::new();
        s.record_step();
        s.record_step();
        s.record_request(10, 5);
        s.record_request(2, 1);
        s.record_tool_call();
        s.record_error();

        assert_eq!(s.steps(), 2);
        assert_eq!(s.requests(), 2);
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.errors(), 1);
        assert_eq!(s.input_tokens(), 12);
        assert_eq!(s.output_tokens(), 6);
    }

    #[test]
    fn ticket_stats_writes_show_up_in_reads() {
        let s = Stats::new();
        s.record_created();
        s.record_created();
        s.record_done();
        s.record_failed();

        assert_eq!(s.tickets_created(), 2);
        assert_eq!(s.tickets_done(), 1);
        assert_eq!(s.tickets_failed(), 1);
    }

}
