//! Run-time stats. One [`Stats`] struct of atomic counters records every
//! observable event and exposes inherent read accessors. Each domain
//! interacts with its own write-only protocol — [`LoopStats`] for the
//! agent loop, `TicketStats` for the ticket system — so a domain
//! cannot reach another domain's events. The wiring is internal: the
//! caller never sees `Stats` at construction time, only afterwards
//! through `TicketSystem::stats()`.
//!
//! Lock-free for counter increments; readers do one atomic load per
//! call.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
    /// First call wins (CAS into `started_at`); later calls are no-ops.
    fn record_started(&self, when: u64);
    /// Adds `ticket_duration.as_secs()` and `work_duration.as_secs()` to
    /// the corresponding atomic sums.
    fn record_done(&self, ticket_duration: Duration, work_duration: Duration);
    fn record_failed(&self, ticket_duration: Duration, work_duration: Duration);
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

    /// Run-start wall clock (millis since epoch). 0 = unset; first
    /// `record_started` wins via CAS.
    started_at: AtomicU64,
    /// Run-end wall clock (millis since epoch). 0 = still running;
    /// `mark_finished` stamps it when the watcher fires.
    finished_at: AtomicU64,
    /// Sum of finished tickets' creation→terminal durations, seconds.
    total_ticket_duration: AtomicU64,
    /// Sum of finished tickets' started→terminal durations, seconds.
    /// With concurrent agents this can exceed the run's wall-clock
    /// duration.
    total_work_duration: AtomicU64,
    /// Lazy-init map of nested counter slices keyed by ticket label.
    /// Always empty on a slice itself; populated only on the run-wide
    /// `Stats` owned by `TicketSystem`.
    label_stats: Mutex<HashMap<String, Arc<Stats>>>,
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
            started_at: AtomicU64::new(0),
            finished_at: AtomicU64::new(0),
            total_ticket_duration: AtomicU64::new(0),
            total_work_duration: AtomicU64::new(0),
            label_stats: Mutex::new(HashMap::new()),
        }
    }

    /// Live counters scoped to one ticket label. Lazy-init on first
    /// access; subsequent calls return the same `Arc`. Reads use the
    /// same accessors as the run-wide `Stats`; `run_duration()` is
    /// always `None` on a slice (run wall-clock stays global).
    pub fn stats_for_label(&self, label: &str) -> Arc<Stats> {
        let mut map = self.label_stats.lock().unwrap();
        map.entry(label.to_string())
            .or_insert_with(|| Arc::new(Stats::new()))
            .clone()
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

    /// Wall-clock duration from the first `record_started` call to
    /// the run-watcher firing (`mark_finished`). `None` while the run
    /// hasn't started, or while it's still running.
    pub fn run_duration(&self) -> Option<Duration> {
        let s = self.started_at.load(Ordering::Relaxed);
        let f = self.finished_at.load(Ordering::Relaxed);
        if s == 0 || f == 0 || f < s {
            None
        } else {
            Some(Duration::from_millis(f - s))
        }
    }

    /// `tickets_done / (tickets_done + tickets_failed)`. `None` when
    /// no ticket has reached a terminal state.
    pub fn success_rate(&self) -> Option<f64> {
        let done = self.tickets_done.load(Ordering::Relaxed);
        let failed = self.tickets_failed.load(Ordering::Relaxed);
        let total = done + failed;
        if total == 0 {
            None
        } else {
            Some(done as f64 / total as f64)
        }
    }

    /// Sum of finished tickets' creation→terminal spans.
    pub fn total_ticket_duration(&self) -> Duration {
        Duration::from_secs(self.total_ticket_duration.load(Ordering::Relaxed))
    }

    /// Mean of finished tickets' creation→terminal spans. `None`
    /// while no ticket has finished.
    pub fn avg_ticket_duration(&self) -> Option<Duration> {
        let n =
            self.tickets_done.load(Ordering::Relaxed) + self.tickets_failed.load(Ordering::Relaxed);
        if n == 0 {
            None
        } else {
            let secs = self.total_ticket_duration.load(Ordering::Relaxed);
            Some(Duration::from_secs(secs / n))
        }
    }

    /// Sum of finished tickets' started→terminal spans. With
    /// concurrent agents this aggregates work across all of them, so
    /// it can exceed `run_duration`.
    pub fn total_work_duration(&self) -> Duration {
        Duration::from_secs(self.total_work_duration.load(Ordering::Relaxed))
    }

    /// Stamp the run's finish wall-clock. Idempotent in practice
    /// (the watcher fires once); successive calls overwrite, which
    /// is fine.
    pub(crate) fn mark_finished(&self, when: u64) {
        self.finished_at.store(when, Ordering::Relaxed);
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
        self.output_tokens
            .fetch_add(output_tokens, Ordering::Relaxed);
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

    fn record_started(&self, when: u64) {
        // First call wins. Subsequent claims (Path A reclaim, late
        // bind) leave the original run-start untouched.
        let _ = self
            .started_at
            .compare_exchange(0, when, Ordering::Relaxed, Ordering::Relaxed);
    }

    fn record_done(&self, ticket_duration: Duration, work_duration: Duration) {
        self.tickets_done.fetch_add(1, Ordering::Relaxed);
        self.total_ticket_duration
            .fetch_add(ticket_duration.as_secs(), Ordering::Relaxed);
        self.total_work_duration
            .fetch_add(work_duration.as_secs(), Ordering::Relaxed);
    }

    fn record_failed(&self, ticket_duration: Duration, work_duration: Duration) {
        self.tickets_failed.fetch_add(1, Ordering::Relaxed);
        self.total_ticket_duration
            .fetch_add(ticket_duration.as_secs(), Ordering::Relaxed);
        self.total_work_duration
            .fetch_add(work_duration.as_secs(), Ordering::Relaxed);
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
        assert_eq!(s.total_ticket_duration(), Duration::ZERO);
        assert_eq!(s.total_work_duration(), Duration::ZERO);
        assert!(s.run_duration().is_none());
        assert!(s.avg_ticket_duration().is_none());
        assert!(s.success_rate().is_none());
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
        s.record_done(Duration::from_secs(3), Duration::from_secs(2));
        s.record_failed(Duration::from_secs(5), Duration::from_secs(4));

        assert_eq!(s.tickets_created(), 2);
        assert_eq!(s.tickets_done(), 1);
        assert_eq!(s.tickets_failed(), 1);
        assert_eq!(s.total_ticket_duration(), Duration::from_secs(8));
        assert_eq!(s.total_work_duration(), Duration::from_secs(6));
    }

    #[test]
    fn record_started_first_call_wins() {
        let s = Stats::new();
        s.record_started(1_000);
        s.record_started(2_000);
        s.record_started(3_000);
        // run_duration needs both started + finished:
        s.mark_finished(4_500);
        assert_eq!(s.run_duration(), Some(Duration::from_millis(3500)));
    }

    #[test]
    fn run_duration_none_until_finished() {
        let s = Stats::new();
        assert!(s.run_duration().is_none());
        s.record_started(1_000);
        assert!(s.run_duration().is_none()); // not finished yet
        s.mark_finished(2_500);
        assert_eq!(s.run_duration(), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn success_rate_done_failed_mix() {
        let s = Stats::new();
        s.record_done(Duration::from_secs(1), Duration::from_secs(1));
        s.record_done(Duration::from_secs(2), Duration::from_secs(2));
        s.record_failed(Duration::from_secs(3), Duration::from_secs(3));
        let rate = s.success_rate().unwrap();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9, "rate = {rate}");
    }

    #[test]
    fn success_rate_none_when_nothing_finished() {
        let s = Stats::new();
        assert!(s.success_rate().is_none());
    }

    #[test]
    fn avg_ticket_duration_is_arithmetic_mean() {
        let s = Stats::new();
        s.record_done(Duration::from_secs(2), Duration::from_secs(2));
        s.record_failed(Duration::from_secs(4), Duration::from_secs(4));
        assert_eq!(s.avg_ticket_duration(), Some(Duration::from_secs(3)));
    }

    #[test]
    fn stats_for_label_returns_same_slice_on_repeat_access() {
        let s = Stats::new();
        let a = s.stats_for_label("scan");
        let b = s.stats_for_label("scan");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn stats_for_label_slice_records_independently() {
        let s = Stats::new();
        let slice = s.stats_for_label("scan");
        slice.record_step();
        slice.record_request(10, 5);
        assert_eq!(slice.steps(), 1);
        assert_eq!(slice.input_tokens(), 10);
        assert_eq!(slice.output_tokens(), 5);
        assert_eq!(s.steps(), 0);
        assert_eq!(s.input_tokens(), 0);
    }

    #[test]
    fn stats_for_label_slice_run_duration_is_none() {
        let s = Stats::new();
        let slice = s.stats_for_label("scan");
        slice.record_done(Duration::from_secs(2), Duration::from_secs(1));
        assert!(slice.run_duration().is_none());
        assert_eq!(slice.tickets_done(), 1);
    }

    #[test]
    fn total_work_duration_can_exceed_run_duration_with_concurrency() {
        // Two tickets, each 5s of work, finished in a 6s window —
        // models 2 agents working in parallel.
        let s = Stats::new();
        s.record_started(1_000);
        s.record_done(Duration::from_secs(5), Duration::from_secs(5));
        s.record_done(Duration::from_secs(5), Duration::from_secs(5));
        s.mark_finished(7_000);
        assert_eq!(s.run_duration(), Some(Duration::from_secs(6)));
        assert_eq!(s.total_work_duration(), Duration::from_secs(10));
    }
}
