---
name: code-review
description: Use after all subtasks are committed and before integrating a task. Perform a whole-repo review of the task's changes against the integration branch, iterate to zero findings.
---

# Code Review

Run after all subtasks are committed, before the quality gate and integration
into the integration branch. Review the entire repo's diff for the task and
iterate until there are zero findings.

## Procedure

1. Determine the task's changes against the integration branch
   (`INTEGRATION=$(git config --get graddft.integrationBranch || echo agent_settings)`):
   `git diff --stat "$INTEGRATION"...HEAD` and `git diff "$INTEGRATION"...HEAD`.
2. Review against the checklist below.
3. For every finding, fix it in a follow-up commit on the same feature branch,
   using the commit format from the `subtask-loop` skill.
4. Repeat until zero findings.

## Checklist

- [ ] **Correctness** — every clause of the task issue (#N) is traceable to
      committed code.
- [ ] **Tests** — new code is covered by `tests/unit/` or
      `tests/integration/`; no test was weakened or skipped; the suite passes
      (see `run-tests`).
- [ ] **Docs** — the documentation surface (`README.md`, `examples/`,
      docstrings) was updated per the mapping in the `planning` skill, and every
      path, file name, and API name it mentions exists.
- [ ] **Duplication** — no copy-pasted or near-identical logic; shared helpers
      belong in `grad_dft/utils/`.
- [ ] **Style** — `black` and `pylint` (per `.pre-commit-config.yaml`) pass
      clean; types come from `grad_dft/utils/types.py` where applicable.
- [ ] **Dead code** — no unused imports, functions, or debug artifacts.
- [ ] **Scope** — changes touch only what the task requires; no unrelated edits.

Zero findings means all boxes pass. Do not merge with known findings.

## Aggregate final review (stacked mode)

Immediately before the final pull request to `main`, review the whole
`git diff main...$INTEGRATION` (all tasks accumulated on the integration
branch) with the same checklist, and re-run the quality gate. This catches
interactions between tasks that per-task reviews cannot see.
