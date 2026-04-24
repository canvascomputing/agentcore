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

- `make test` runs unit tests bundled by `tests/unit.rs`.
- `make test_integration` runs integration tests bundled by `tests/integration.rs`.

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
- `project_scanner` scans a project directory.
- `deep_research` runs multi-agent research with web search.
