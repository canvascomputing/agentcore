# Workflow

Commands used to build, test, release, and run example agents.

## 1. Build

**Every build runs with `-D warnings`.**

- `make` compiles the crate.
- `make fmt` formats the code.
- `make clean` removes build artefacts.
- Any warning fails the build.

## 2. Test

**Unit tests use a mock provider; integration tests use a real one.**

- `make test` runs unit tests bundled by `tests/unit.rs`.
- `make test_integration` runs integration tests bundled by `tests/integration.rs`.
- Shared integration helpers live in `tests/integration/common.rs`.
- Inline tests live in each module as `#[cfg(test)] mod tests`.
- `MockProvider` and `TestHarness` are in `testutil.rs`.

## 3. Release

**`make bump` runs the full release step in one command.**

- `make bump` runs tests, bumps the patch version, commits, and tags.
- `make bump part=minor` bumps the minor version.
- `make bump part=major` bumps the major version.
- Push the new tag with `git push --tags`.

## 4. Use cases

**Example agents live in a separate crate and run through `make use_case`.**

- Source is in `crates/use-cases/src/`.
- Run an example with `make use_case name=<name>`.
- `project_scanner` scans a project directory.
- `deep_research` runs multi-agent research with web search.
