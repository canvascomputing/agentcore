//! Retry strategy for transient provider errors. Ported from the
//! pre-migration `crates/agentwerk/src/util.rs` (visible at
//! `git show aa725dc^:crates/agentwerk/src/util.rs`). The agent loop
//! holds an [`ExponentialRetry`] by value and invokes it through the
//! [`Retry`] trait.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Retry strategy: total attempt budget and delay between attempts.
/// Concrete impls live in this module; call sites hold the strategy
/// by value and invoke it through the trait.
pub(crate) trait Retry {
    fn max_attempts(&self) -> u32;

    /// Delay between attempt `attempt` and `attempt + 1` (0-indexed).
    /// `server_hint` (e.g. HTTP `Retry-After`) takes precedence when
    /// honoured; impls that ignore the hint document that choice.
    fn delay(&self, attempt: u32, server_hint: Option<Duration>) -> Duration;
}

/// Cap on per-attempt backoff so exponential growth doesn't run away
/// past a few attempts. Matches claude-code's `maxDelayMs` default.
const MAX_RETRY_DELAY: Duration = Duration::from_millis(32_000);

/// Exponential backoff `base_delay * 2^attempt`, clamped at 32 s,
/// extended by additive jitter in `[0, 25%]` of the clamped value. A
/// `server_hint` bypasses the cap and jitter: the server is explicit
/// about what it wants.
pub(crate) struct ExponentialRetry {
    pub base_delay: Duration,
    pub max_attempts: u32,
}

impl Retry for ExponentialRetry {
    fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    fn delay(&self, attempt: u32, server_hint: Option<Duration>) -> Duration {
        if let Some(hint) = server_hint {
            return hint;
        }

        let base_ms = self.base_delay.as_millis() as u64;
        let exponential_ms = base_ms
            .saturating_mul(1u64 << attempt.min(31))
            .min(MAX_RETRY_DELAY.as_millis() as u64);
        let jitter_range = exponential_ms / 4;

        if jitter_range == 0 {
            return Duration::from_millis(exponential_ms);
        }

        let entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        let jitter_offset = entropy % jitter_range;

        Duration::from_millis(exponential_ms.saturating_add(jitter_offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(base_ms: u64) -> ExponentialRetry {
        ExponentialRetry {
            base_delay: Duration::from_millis(base_ms),
            max_attempts: 10,
        }
    }

    #[test]
    fn exponential_backoff() {
        let policy = policy(1000);
        for attempt in 0..3 {
            let delay = policy.delay(attempt, None);
            let expected_base_ms = 1000u64 * (1u64 << attempt);
            let jitter_range_ms = expected_base_ms / 4;
            let delay_ms = delay.as_millis() as u64;
            assert!(delay_ms >= expected_base_ms);
            assert!(delay_ms <= expected_base_ms + jitter_range_ms);
        }
    }

    #[test]
    fn respects_retry_delay() {
        let delay = policy(1000).delay(0, Some(Duration::from_millis(5000)));
        assert_eq!(delay, Duration::from_millis(5000));
    }

    #[test]
    fn caps_at_max_delay() {
        let delay = policy(1000).delay(20, None);
        let max_ms = MAX_RETRY_DELAY.as_millis() as u64;
        let jitter_range_ms = max_ms / 4;
        let delay_ms = delay.as_millis() as u64;
        assert!(delay_ms >= max_ms);
        assert!(delay_ms <= max_ms + jitter_range_ms);
    }

    #[test]
    fn saturates_instead_of_overflow() {
        let _delay = policy(u64::MAX).delay(10, None);
    }
}
