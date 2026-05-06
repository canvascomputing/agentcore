//! Integration tests that hit a live LLM provider.
//! Run with provider env vars set (e.g. `ANTHROPIC_API_KEY` + `MODEL`).

#[path = "integration/common.rs"]
mod common;

#[path = "integration/bash_usage.rs"]
mod bash_usage;

#[path = "integration/edit_file_replaces_content.rs"]
mod edit_file_replaces_content;

#[path = "integration/file_exploration.rs"]
mod file_exploration;

#[path = "integration/glob_finds_nested_files.rs"]
mod glob_finds_nested_files;

#[path = "integration/grep_finds_code_pattern.rs"]
mod grep_finds_code_pattern;

#[path = "integration/list_directory_enumerates_entries.rs"]
mod list_directory_enumerates_entries;

#[path = "integration/traces_call_path_across_files.rs"]
mod traces_call_path_across_files;

#[path = "integration/write_file_creates_file.rs"]
mod write_file_creates_file;
