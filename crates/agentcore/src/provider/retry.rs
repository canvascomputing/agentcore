/// Default number of retries for transient API errors.
pub const DEFAULT_MAX_REQUEST_RETRIES: u32 = 3;

/// Default base delay in milliseconds for exponential backoff.
pub const DEFAULT_BACKOFF_MS: u64 = 10_000;

/// Compute the delay before the next retry attempt.
///
/// Uses exponential backoff: `backoff_ms * 2^attempt`, with ±20% jitter.
/// If the server provides a `retry_after_ms` hint, that value takes precedence.
pub fn compute_delay(backoff_ms: u64, attempt: u32, retry_after_ms: Option<u64>) -> u64 {
    if let Some(server_delay) = retry_after_ms {
        return server_delay;
    }

    let exponential = backoff_ms.saturating_mul(1u64 << attempt.min(31));
    let jitter_range = exponential / 5; // ±20%

    if jitter_range == 0 {
        return exponential;
    }

    let entropy = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let jitter_offset = entropy % (jitter_range * 2);

    exponential.saturating_sub(jitter_range).saturating_add(jitter_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_backoff() {
        for attempt in 0..3 {
            let delay = compute_delay(1000, attempt, None);
            let expected_base = 1000u64 * (1u64 << attempt);
            let jitter_range = expected_base / 5;
            assert!(delay >= expected_base - jitter_range);
            assert!(delay <= expected_base + jitter_range);
        }
    }

    #[test]
    fn respects_retry_after() {
        let delay = compute_delay(1000, 0, Some(5000));
        assert_eq!(delay, 5000);
    }

    #[test]
    fn saturates_instead_of_overflow() {
        let _delay = compute_delay(u64::MAX, 10, None);
    }
}
