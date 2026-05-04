//! Integration tests that hit a live LLM provider.
//! Run with provider env vars set (e.g. `ANTHROPIC_API_KEY` + `MODEL`).

#[path = "integration/common.rs"]
mod common;

#[path = "integration/bash_usage.rs"]
mod bash_usage;

#[path = "integration/file_exploration.rs"]
mod file_exploration;
