//! Policy bundle and the `PolicyConform` trait. The loop reads policy
//! decisions through this trait; `TicketSystem` is the (currently only)
//! implementor.

use std::time::Duration;

/// Policy bundle consumed by the loop: caps and retry tuning.
#[derive(Clone, Debug)]
pub struct Policies {
    pub max_steps: Option<u32>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub max_request_tokens: Option<u32>,
    pub max_schema_retries: Option<u32>,
    pub max_request_retries: u32,
    pub request_retry_delay: Duration,
}

impl Policies {
    pub const DEFAULT_MAX_SCHEMA_RETRIES: u32 = 10;
    pub const DEFAULT_MAX_REQUEST_RETRIES: u32 = 10;
    pub const DEFAULT_REQUEST_RETRY_DELAY: Duration = Duration::from_millis(500);
}

impl Default for Policies {
    fn default() -> Self {
        Self {
            max_steps: None,
            max_input_tokens: None,
            max_output_tokens: None,
            max_request_tokens: None,
            max_schema_retries: Some(Self::DEFAULT_MAX_SCHEMA_RETRIES),
            max_request_retries: Self::DEFAULT_MAX_REQUEST_RETRIES,
            request_retry_delay: Self::DEFAULT_REQUEST_RETRY_DELAY,
        }
    }
}

/// Anything the loop can ask for policy decisions. Implemented by
/// `TicketSystem`; future per-agent overrides could implement it too.
pub trait PolicyConform {
    fn policies(&self) -> &Policies;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_values() {
        let p = Policies::default();
        assert_eq!(p.max_steps, None);
        assert_eq!(p.max_input_tokens, None);
        assert_eq!(p.max_output_tokens, None);
        assert_eq!(p.max_request_tokens, None);
        assert_eq!(p.max_schema_retries, Some(10));
        assert_eq!(p.max_request_retries, 10);
        assert_eq!(p.request_retry_delay, Duration::from_millis(500));
    }

    #[test]
    fn default_max_schema_retries_constant_is_ten() {
        assert_eq!(Policies::DEFAULT_MAX_SCHEMA_RETRIES, 10);
    }

    #[test]
    fn default_max_request_retries_constant_is_ten() {
        assert_eq!(Policies::DEFAULT_MAX_REQUEST_RETRIES, 10);
    }

    #[test]
    fn default_request_retry_delay_constant_is_500ms() {
        assert_eq!(
            Policies::DEFAULT_REQUEST_RETRY_DELAY,
            Duration::from_millis(500)
        );
    }
}
