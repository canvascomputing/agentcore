//! Shared backoff policy for transient provider failures — one implementation, every provider waits the same way.

use std::time::Duration;

/// Cap on the per-attempt backoff so exponential growth doesn't run away past
/// a few attempts. Matches claude-code's `maxDelayMs` default.
pub(crate) const MAX_DELAY: Duration = Duration::from_millis(32_000);

/// Compute the delay before the next retry attempt.
///
/// Exponential backoff `base_delay * 2^attempt`, clamped at [`MAX_DELAY`],
/// then extended by additive jitter in `[0, 25%]` of the clamped value. If
/// the server provides a `retry_delay` hint, that value takes
/// precedence and bypasses the cap (the server is explicit about what it
/// wants).
pub(crate) fn compute_delay(
    base_delay: Duration,
    attempt: u32,
    retry_delay: Option<Duration>,
) -> Duration {
    if let Some(server_delay) = retry_delay {
        return server_delay;
    }

    let base_ms = base_delay.as_millis() as u64;
    let exponential_ms = base_ms
        .saturating_mul(1u64 << attempt.min(31))
        .min(MAX_DELAY.as_millis() as u64);
    let jitter_range = exponential_ms / 4;

    if jitter_range == 0 {
        return Duration::from_millis(exponential_ms);
    }

    let entropy = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let jitter_offset = entropy % jitter_range;

    Duration::from_millis(exponential_ms.saturating_add(jitter_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_backoff() {
        for attempt in 0..3 {
            let delay = compute_delay(Duration::from_millis(1000), attempt, None);
            let expected_base_ms = 1000u64 * (1u64 << attempt);
            let jitter_range_ms = expected_base_ms / 4;
            let delay_ms = delay.as_millis() as u64;
            assert!(delay_ms >= expected_base_ms);
            assert!(delay_ms <= expected_base_ms + jitter_range_ms);
        }
    }

    #[test]
    fn respects_retry_delay() {
        let delay = compute_delay(
            Duration::from_millis(1000),
            0,
            Some(Duration::from_millis(5000)),
        );
        assert_eq!(delay, Duration::from_millis(5000));
    }

    #[test]
    fn caps_at_max_delay() {
        let delay = compute_delay(Duration::from_millis(1000), 20, None);
        let max_ms = MAX_DELAY.as_millis() as u64;
        let jitter_range_ms = max_ms / 4;
        let delay_ms = delay.as_millis() as u64;
        assert!(delay_ms >= max_ms);
        assert!(delay_ms <= max_ms + jitter_range_ms);
    }

    #[test]
    fn saturates_instead_of_overflow() {
        let _delay = compute_delay(Duration::from_millis(u64::MAX), 10, None);
    }
}
