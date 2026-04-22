//! Small internal helpers shared across the crate — name generation, date formatting, and shell invocation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{AgenticError, Result};

/// Resolves when the cancel flag flips to true, polling every 100 ms. Pair
/// with `tokio::select!` to abort any mid-flight work: the loser branch is
/// dropped, which cascades to dropped HTTP futures (reqwest aborts the
/// connection) and dropped child processes (if `kill_on_drop(true)` is set).
pub(crate) async fn wait_for_cancel(cancel: &Arc<AtomicBool>) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Sleep for `duration`, but bail as soon as `cancel` trips. Returns
/// `Err(AgenticError::Aborted)` if cancel fired, `Ok(())` if the full
/// duration elapsed. Used by retry backoff to stay responsive to Ctrl-C.
pub(crate) async fn cancellable_sleep(duration: Duration, cancel: &Arc<AtomicBool>) -> Result<()> {
    tokio::select! {
        biased;
        _ = wait_for_cancel(cancel) => Err(AgenticError::Aborted),
        _ = tokio::time::sleep(duration) => Ok(()),
    }
}

pub(crate) fn generate_agent_name(name: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{name}_{nanos}")
}

pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Today's date as `YYYY-MM-DD`, via the civil-from-days algorithm.
/// http://howardhinnant.github.io/date_algorithms.html
pub(crate) fn format_current_date() -> String {
    let epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = epoch_secs / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}")
}
