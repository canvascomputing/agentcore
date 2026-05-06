//! Handle returned from [`Runnable::run`]. Owns the background tokio
//! task driving the agent loop, the interrupt signal, and a strong
//! reference to the ticket system. Forwards task / result accessors to
//! the system through `Deref`, so a single binding (typically named
//! `agent` or `tickets`) carries both the queue surface and the
//! lifecycle methods.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;

use super::tickets::{now_millis, pending_count, policy_violated, TicketSystem};

/// In-flight run handle. Created by [`Runnable::run`]. Holds the
/// background tokio task driving the agent loop and the interrupt
/// signal that stops it.
pub struct Running {
    system: Arc<TicketSystem>,
    signal: Arc<AtomicBool>,
    join: JoinHandle<()>,
}

impl Running {
    pub(crate) fn new(
        system: Arc<TicketSystem>,
        signal: Arc<AtomicBool>,
        join: JoinHandle<()>,
    ) -> Self {
        Self {
            system,
            signal,
            join,
        }
    }

    /// The interrupt signal the run watches. Cloning lets a caller
    /// share the same flag with an external ctrl-c handler or peer
    /// subsystem.
    pub fn signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.signal)
    }

    /// Flip the interrupt signal. The run exits as soon as the
    /// in-flight ticket finishes; queued tickets are abandoned. Pair
    /// with [`Running::join`] to wait for it to actually exit.
    pub fn stop(&self) {
        self.signal.store(true, Ordering::Relaxed);
    }

    /// Wait for the run to finish. Returns when [`Running::stop`] has
    /// been called (or any other path flipped the signal) and every
    /// per-agent task has finished.
    pub async fn join(self) {
        let _ = self.join.await;
    }

    /// Abort the background task without waiting for graceful
    /// teardown. Use only on terminal-cancel paths (e.g. ctrl-c
    /// immediately followed by `std::process::exit`).
    pub fn abort(self) {
        self.join.abort();
    }

    /// Wait for the queue to drain, then stop and join. Polls every
    /// 20 ms; exits when `pending_count == 0`, a policy trips, or
    /// `max_time` elapses. Returns the most recent `Done` ticket's
    /// `result`, or an empty string when none settled.
    pub async fn run_dry(self) -> String {
        let started = Instant::now();
        let policies = self.system.policies();
        loop {
            tokio::time::sleep(Duration::from_millis(20)).await;
            if self.signal.load(Ordering::Relaxed) {
                break;
            }
            if policy_violated(&policies, &self.system.stats) {
                self.signal.store(true, Ordering::Relaxed);
                break;
            }
            if let Some(limit) = policies.max_time {
                if started.elapsed() >= limit {
                    self.signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
            if pending_count(&self.system) == 0 {
                self.signal.store(true, Ordering::Relaxed);
                break;
            }
        }
        let _ = self.join.await;
        self.system.stats.mark_finished(now_millis());
        self.system.last_done_result()
    }
}

impl Deref for Running {
    type Target = TicketSystem;
    fn deref(&self) -> &TicketSystem {
        &self.system
    }
}
