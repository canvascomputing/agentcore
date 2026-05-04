//! Policy bundle: caps and retry tuning the loop reads through the
//! ticket system state.

use std::time::Duration;

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
}
