# Testing

How tests are organized and written. Commands used to run them live in [workflow.md](workflow.md).

## Layers

**Three layers: unit, integration, inline.**

- `tests/unit/` uses `MockProvider`; bundled by `tests/unit.rs`.
- `tests/integration/` uses a real provider; bundled by `tests/integration.rs`.
- Inline `#[cfg(test)] mod tests` lives next to the code it covers.
- `MockProvider` and `TestHarness` live in `testutil.rs`.
- Shared integration helpers live in `tests/integration/common.rs`.

## Purpose

**One test, one observable behavior.**

- A test exists because a single contract would otherwise go undemonstrated.
- A failure points to one cause: no grab-bag assertions across unrelated concerns.
- A sibling that already covers the same behavior with different inputs is merged or removed.
- Behaviour is tested at the layer where it lives: unit, integration, or inline.

## Naming

**The name states the behavior, not the method called.**

- Accepted: `rejects_submit_when_cart_is_empty`, `deposit_increases_balance`.
- Rejected: `test_submit`, `test_submit_works`, `test_balance`.
- The body verifies what the name claims, with no surprise assertions.
- The name is the first line of the documentation the test provides.

## API focus

**Tests exercise the public surface the way callers hold it.**

- Call the public entry point; do not poke at private fields, patched internals, or field assignments.
- Mock at trust boundaries (network, clock, disk), never at the subject under test.
- Assert observable outcomes, not call logs or the order internal methods ran in.
- The arrange/act/assert shape mirrors how a real caller would use the API.

## State transitions

**Actions and the resulting state MUST be visible through the public API.**

- Build starting state by calling real actions, not by field assignment that bypasses invariants.
- Read resulting state back through a public query, not by peeking at private fields.
- Assert both starting and final state so the transition is shown, not implied.
- Cover illegal transitions and verify state is unchanged after a rejection.
- One transition per test so a failure locates the exact broken action.

## Clarity

**Setup is hidden. Intent is highlighted.**

- Push scaffolding into factories, builders, and fixtures so the body reads as a short story.
- Name literals that carry meaning: `EXPIRED_COUPON`, not `42`; `ADMIN_USER`, not `"foo"`.
- Keep the act step a single visible line; do not bury it in setup.
- Comments are justified only to pin an architectural invariant the test guards.

## Coverage shape

**Every public operation has a test that demonstrates intended usage.**

- Error cases, edge conditions, and boundaries sit at the same interface level as the happy path.
- Overlapping cases that exercise the same branch with trivial input changes are merged.
- A missing behavior is added before a duplicate case is kept for symmetry.
- IMPORTANT: a public method with no test is a documentation gap, not just a coverage gap.
