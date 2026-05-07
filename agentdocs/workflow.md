# Workflow

Commands used to build, test, release, and run example agents.

## Build

**Every build MUST run with `-D warnings`.**

- `make` compiles the crate.
- `make fmt` formats the code.
- `make clean` removes build artefacts.
- Any warning fails the build.

## Test

**Test layout and writing rules live in [testing.md](testing.md).**

- `make test` runs `cargo test --workspace --lib` (every crate's inline `#[cfg(test)] mod tests`).
- `make test_integration` runs the live-provider tests bundled by `tests/integration.rs`.

## Release

**`make bump` runs the full release step in one command.**

- `make bump` runs tests, bumps the patch version, commits, and tags.
- `make bump part=minor` bumps the minor version.
- `make bump part=major` bumps the major version.
- Push the new tag with `git push --tags`.

## Use cases

**Example agents live in a separate crate and run through `make use_case`.**

- Source is in `crates/use-cases/src/`.
- Run an example with `make use_case name=<name>`.
- `terminal-repl` is a per-turn interactive chat with streaming output.
- `divide-and-conquer` partitions an arithmetic problem across worker agents sharing one ticket queue.
- `deep-research` is a two-phase research pipeline with web search (requires `BRAVE_API_KEY`).
