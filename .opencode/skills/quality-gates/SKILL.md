---
name: quality-gates
description: Use before integrating a task. Defines the hard gate that must pass (tests + style) and how to interpret its result. References the CI workflow and the pre-commit config for the exact commands.
---

# Quality Gates

The hard gate a task must pass before integration. It has exactly two terminal
states: **PASS** or **BLOCKED**. There is no "pass with exceptions". The same
gate is run again over the whole `main...$INTEGRATION` diff immediately before
the final pull request to `main`.

## What the gate is

The gate is the combination of the test suite and the style/lint checks:

- **Tests** — the per-file `pytest -v` invocations in
  `.github/workflows/install_and_test.yml` (see the `run-tests` skill). 0
  failures, 0 skipped.
- **Style/lint** — the hooks in `.pre-commit-config.yaml`: `black` (rev pinned
  there) and `pylint -rn -sn --rcfile=.pylintrc`.

There is **no docs build gate** — this repository has no Sphinx site; the
documentation surface is `README.md` plus `examples/`, checked by review, not
by a build. There is also no type-check step and no coverage threshold.

## Procedure

1. Install the package and extras (`pip install -e .`,
   `pip install -e ".[examples]"`), then run the tests. 0 failures, 0 skipped.
2. Run `pre-commit run --all-files` (or `black` and `pylint -rn -sn
   --rcfile=.pylintrc` directly). No errors.
3. Interpret the result:
   - All clean → **PASS**. Proceed to merge into the integration branch.
   - Any failure → **BLOCKED**.

## On BLOCKED

- STOP. Do not merge. Do not mark the task done.
- Do **not** assess whether a failure is pre-existing or unrelated to your
  changes — fix it regardless.
- Do **not** weaken, skip, or comment out failing tests to make the suite green
  (see the Blocked Work Protocol in the `subtask-loop` skill).
- Fix every failure and re-run until **PASS**.
