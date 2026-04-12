use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::types::TokenUsage;

#[derive(Debug, Clone)]
pub struct ModelCosts {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_usd: f64,
    pub request_count: u64,
}

#[derive(Clone)]
pub struct CostTracker {
    inner: Arc<Mutex<CostTrackerInner>>,
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CostTrackerInner {
                pricing: default_pricing(),
                usage: HashMap::new(),
                total_tool_calls: 0,
            })),
        }
    }

    pub fn model_pricing(&self, model: &str, costs: ModelCosts) {
        self.inner.lock().unwrap().pricing.insert(model.to_string(), costs);
    }

    pub fn record_usage(&self, model: &str, usage: &TokenUsage) {
        let mut inner = self.inner.lock().unwrap();

        let cost = inner.pricing.get(model).map_or(0.0, |p| {
            (usage.input_tokens as f64 * p.input_per_million
                + usage.output_tokens as f64 * p.output_per_million
                + usage.cache_read_input_tokens as f64 * p.cache_read_per_million
                + usage.cache_creation_input_tokens as f64 * p.cache_write_per_million)
                / 1_000_000.0
        });

        let entry = inner.usage.entry(model.to_string()).or_default();
        entry.input_tokens += usage.input_tokens;
        entry.output_tokens += usage.output_tokens;
        entry.cache_read_tokens += usage.cache_read_input_tokens;
        entry.cache_write_tokens += usage.cache_creation_input_tokens;
        entry.cost_usd += cost;
        entry.request_count += 1;
    }

    pub fn record_tool_calls(&self, count: u64) {
        self.inner.lock().unwrap().total_tool_calls += count;
    }

    pub fn total_cost_usd(&self) -> f64 {
        self.inner.lock().unwrap().usage.values().map(|u| u.cost_usd).sum()
    }

    pub fn total_requests(&self) -> u64 {
        self.inner.lock().unwrap().usage.values().map(|u| u.request_count).sum()
    }

    pub fn total_tool_calls(&self) -> u64 {
        self.inner.lock().unwrap().total_tool_calls
    }

    pub fn model_usage(&self) -> HashMap<String, ModelUsage> {
        self.inner.lock().unwrap().usage.clone()
    }

    pub fn summary(&self) -> String {
        let inner = self.inner.lock().unwrap();
        let total_cost: f64 = inner.usage.values().map(|u| u.cost_usd).sum();
        let mut result = format!("Total cost:            ${total_cost:.4}\n");

        let mut models: Vec<_> = inner.usage.iter().collect();
        models.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (model, usage) in models {
            result.push_str(&format!(
                "{model}:  {} input, {} output, {} cache read (${:.4})\n",
                format_tokens(usage.input_tokens),
                format_tokens(usage.output_tokens),
                format_tokens(usage.cache_read_tokens),
                usage.cost_usd,
            ));
        }
        result
    }
}

#[derive(Debug)]
struct CostTrackerInner {
    pricing: HashMap<String, ModelCosts>,
    usage: HashMap<String, ModelUsage>,
    total_tool_calls: u64,
}

fn default_pricing() -> HashMap<String, ModelCosts> {
    HashMap::from([
        ("claude-haiku-4-5-20251001".into(), ModelCosts {
            input_per_million: 0.80,
            output_per_million: 4.0,
            cache_read_per_million: 0.08,
            cache_write_per_million: 1.0,
        }),
        ("claude-sonnet-4-20250514".into(), ModelCosts {
            input_per_million: 3.0,
            output_per_million: 15.0,
            cache_read_per_million: 0.30,
            cache_write_per_million: 3.75,
        }),
        ("claude-opus-4-20250514".into(), ModelCosts {
            input_per_million: 15.0,
            output_per_million: 75.0,
            cache_read_per_million: 1.50,
            cache_write_per_million: 18.75,
        }),
        ("mistral-large-latest".into(), ModelCosts {
            input_per_million: 2.0,
            output_per_million: 6.0,
            cache_read_per_million: 0.0,
            cache_write_per_million: 0.0,
        }),
        ("mistral-small-latest".into(), ModelCosts {
            input_per_million: 0.10,
            output_per_million: 0.30,
            cache_read_per_million: 0.0,
            cache_write_per_million: 0.0,
        }),
        ("codestral-latest".into(), ModelCosts {
            input_per_million: 0.30,
            output_per_million: 0.90,
            cache_read_per_million: 0.0,
            cache_write_per_million: 0.0,
        }),
        ("mistral-medium-2508".into(), ModelCosts {
            input_per_million: 2.75,
            output_per_million: 8.10,
            cache_read_per_million: 0.0,
            cache_write_per_million: 0.0,
        }),
    ])
}

fn format_tokens(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_zero_cost() {
        let tracker = CostTracker::new();
        assert_eq!(tracker.total_cost_usd(), 0.0);
        assert_eq!(tracker.total_requests(), 0);
        assert_eq!(tracker.total_tool_calls(), 0);
    }

    #[test]
    fn record_usage_accumulates() {
        let tracker = CostTracker::new();
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        tracker.record_usage("claude-sonnet-4-20250514", &usage);
        tracker.record_usage("claude-sonnet-4-20250514", &usage);

        let model_usage = tracker.model_usage();
        let sonnet = &model_usage["claude-sonnet-4-20250514"];
        assert_eq!(sonnet.input_tokens, 2000);
        assert_eq!(sonnet.output_tokens, 1000);
        assert_eq!(sonnet.request_count, 2);
    }

    #[test]
    fn multiple_models_tracked_separately() {
        let tracker = CostTracker::new();
        tracker.record_usage(
            "claude-sonnet-4-20250514",
            &TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
        );
        tracker.record_usage(
            "claude-opus-4-20250514",
            &TokenUsage {
                input_tokens: 200,
                output_tokens: 100,
                ..Default::default()
            },
        );

        let usage = tracker.model_usage();
        assert_eq!(usage.len(), 2);
        assert_eq!(usage["claude-sonnet-4-20250514"].input_tokens, 100);
        assert_eq!(usage["claude-opus-4-20250514"].input_tokens, 200);
    }

    #[test]
    fn custom_pricing_applied() {
        let tracker = CostTracker::new();
        tracker.model_pricing(
            "custom-model",
            ModelCosts {
                input_per_million: 1.0,
                output_per_million: 0.0,
                cache_read_per_million: 0.0,
                cache_write_per_million: 0.0,
            },
        );
        tracker.record_usage(
            "custom-model",
            &TokenUsage {
                input_tokens: 1_000_000,
                output_tokens: 0,
                ..Default::default()
            },
        );
        let cost = tracker.total_cost_usd();
        assert!((cost - 1.0).abs() < 0.0001, "Expected $1.00, got ${cost}");
    }

    #[test]
    fn tool_calls_tracked() {
        let tracker = CostTracker::new();
        tracker.record_tool_calls(5);
        tracker.record_tool_calls(3);
        assert_eq!(tracker.total_tool_calls(), 8);
    }

    #[test]
    fn summary_contains_model_name() {
        let tracker = CostTracker::new();
        tracker.record_usage(
            "claude-sonnet-4-20250514",
            &TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
        );
        let summary = tracker.summary();
        assert!(summary.contains("claude-sonnet-4-20250514"));
    }

    #[tokio::test]
    async fn concurrent_recording_thread_safe() {
        let tracker = CostTracker::new();
        let mut handles = Vec::new();

        for _ in 0..10 {
            let t = tracker.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    t.record_usage(
                        "claude-sonnet-4-20250514",
                        &TokenUsage {
                            input_tokens: 1,
                            output_tokens: 1,
                            ..Default::default()
                        },
                    );
                }
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(tracker.total_requests(), 1000);
        let usage = tracker.model_usage();
        assert_eq!(usage["claude-sonnet-4-20250514"].input_tokens, 1000);
    }
}
