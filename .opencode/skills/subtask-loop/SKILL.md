---
name: subtask-loop
description: Use when executing an atomic subtask from the detailed plan: implement -> test -> document -> pre-commit checks -> quality checks -> commit -> mark done. Covers the full execution cycle, commit format, blocked work protocol, and completion tracking.
---

# Subtask Execution Loop

Each atomic subtask from `tasks/detailed_plan.md` is executed as a
self-contained cycle.

## Cycle Steps

Execute these steps in order. **Do not skip steps. Do not proceed past a step
until it is verified complete.**

## Documentation-Only Subtasks

When a subtask modifies only `.md`/`.ipynb` files (no `.py` files), the
following cycle steps are adapted:

| Step | Action |
|------|--------|
| 1. Implement | Write documentation |
| 2. Write Tests | **Skip** — no code to test |
| 3. Update Docs | The implementation IS the documentation; verify README links and referenced example paths exist |
| 4. Pre-Commit Check | **Skip** — the `black`/`pylint` hooks in `.pre-commit-config.yaml` only match Python files |
| 5. Code Quality Checks | **Skip** — no code to check |
| 6. Commit | **Follow exactly** — one commit per subtask, single SN identifier |
| 7. Mark Done | **Follow exactly** |

The absence of code changes **never** justifies batching multiple subtasks
into a single commit.

### 1. Implement

Write the code specified in the subtask's **Code** section. Follow the existing
structure and conventions of the module being changed (Apache-2.0 header,
import grouping, `grad_dft/utils/types.py` aliases).

### 2. Write Tests

Write the tests specified in the subtask's **Tests** section. See the
`run-tests` skill for how to run them. Unit tests live in `tests/unit/`,
integration tests in `tests/integration/molecules/` or
`tests/integration/solids/`.

### 3. Update Documentation

Map the source changes onto the documentation surface (`README.md`,
`examples/`, docstrings) exactly as defined in the **Documentation surface**
section of the `planning` skill — that mapping is the single source of truth.

**Hard gate — this step is not complete until:**

- [ ] The subtask's **Docs** section items are done, or an explicit
      "No user-facing change needed" justification is in the plan
- [ ] Every README/example path, file name, and function name mentioned in the
      updated docs actually exists (`glob`/`grep` to confirm) — a stale
      example path is a documentation defect

### 4. Pre-Commit Check

Run `pre-commit run --all-files` (or `black` and
`pylint -rn -sn --rcfile=.pylintrc` directly, per
`.pre-commit-config.yaml`) and fix everything it reports. Do not commit until it
passes.

### 5. Code Quality Checks

- **Duplication check**: scan for accidental code duplication (same logic under
  different names, copy-pasted blocks). Consolidate into
  `grad_dft/utils/` if found.
- **Genericity check**: verify new code reuses existing abstractions and the
  shared types from `grad_dft/utils/types.py` instead of bespoke
  re-implementations.
- **Equivalence test check**: if the subtask is a variant of an existing
  algorithm, ensure an equivalence test exists comparing it to the reference
  implementation (e.g., a jitted path against the eager path, a new functional
  against the PySCF reference energy).
- **Separation check**: verify library modules (`grad_dft/`) do not contain
  plotting, file I/O, or ad-hoc string generation that belongs in
  `examples/advanced_scripts/`.

### 6. Commit

**Commit format** (the single source of truth for commit format in this
repository):

- Subject line: `(<issue>-S<n>): <imperative summary>` — e.g.
  `(42-S2): Reuse `chunk` for the two-electron integrals`. This exact prefix is
  what task-completion detection greps for, so never reword it.
- Body: explain **why** the change was made and what was rejected, not a
  restatement of the diff. Reference the subtask and any design decision
  recorded in `## Design Notes`.
- The **last** subtask of a task adds `Closes #<issue>` as a standalone line
  (the task's own issue). The last task of a batch also adds `Closes #<hub>`.

Before committing, verify `git status` shows only the files this subtask
touches, and that the work is not left uncommitted across subtasks.

### 7. Mark Completed

Mark the subtask as completed in `tasks/detailed_plan.md` (`[done]` suffix +
commit hash on the subtask heading, and tick its cycle steps in the same
plan), then mirror the file onto the plan comment of the task issue: find the
comment by its `<!-- detailed-plan -->` first line and PATCH its body to the
current file content:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
CID=$(gh api "repos/$REPO/issues/<N>/comments" \
  --jq '[.[] | select(.body | startswith("<!-- detailed-plan -->"))] | last.id')
{ echo '<!-- detailed-plan -->'; cat tasks/detailed_plan.md; } > /tmp/plan_comment.md
gh api -X PATCH "repos/$REPO/issues/comments/$CID" \
  --input <(jq -Rs '{body: .}' /tmp/plan_comment.md)
```

The local file is the single source of truth; the comment is a pure mirror.
The per-subtask cycle checklist required by "Per-Subtask Execution Tracking"
below lives in this same plan file — there is no separate tracker.

If at any point in steps 1–7 you hit an unresolvable problem that prevents 100%
completion, **STOP the cycle immediately** and follow the Blocked Work Protocol
below. Do NOT attempt to "complete" the subtask with partial results, reverted
work, or known limitations. Do NOT proceed to the next subtask.

## Subtask Outcome

A subtask has exactly two valid outcomes:

- **Resolved**: implemented, tested, docs updated, committed. Record the commit
  hash in `tasks/detailed_plan.md`.
- **Blocked**: an algorithmic or design problem prevents 100% completion. Do NOT
  commit partial work. Do NOT proceed to the next subtask. Follow the Blocked
  Work Protocol.

There is no third state. "Reverted and left as a known limitation" is not a
valid outcome — it means the subtask is blocked. Report it.

Never silently skip a subtask. If a subtask was attempted, reverted, and its
planned changes were not committed, the subtask is incomplete. Do not mark it
done. Do not proceed. Report it as blocked.

## Per-Subtask Execution Tracking

Track the cycle steps as a checklist inside the subtask's own section of
`tasks/detailed_plan.md`, e.g. for S1:

```
- [x] S1: Implement
- [x] S1: Write tests
- [x] S1: Update documentation
- [x] S1: Pre-Commit Check (black + pylint)
- [x] S1: Quality checks
- [x] S1: Commit (<hash>)
```

**No subtask may be committed with any step still unticked.**

## Multi-Subtask Discipline

When a task has multiple subtasks (S1, S2, S3, ...), execute them **strictly
sequentially**:

1. Complete all cycle steps for S1 (Implement → Tests → Docs → Pre-Commit Check
   → Quality → Commit → Mark Done).
2. Only after S1 is committed, start S2.
3. Never tick multiple subtasks' steps before committing each individually.

A plan listing "S1: Implement [x], S2: Implement [x], S1: Write tests [x]"
indicates skipped commits — each subtask must be fully committed before the next
begins.

## Blocked Work Protocol

If you encounter an algorithmic problem that you cannot resolve to 100%
correctness, **STOP**. Do not commit. Do not merge. Do not comment out or weaken
failing tests to make the suite green. Instead:

1. Stay on the feature branch.
2. Report the problem concretely to the user:
   - Which tests fail and why.
   - What algorithmic gap exists (e.g., "gradients are missing for the
     range-separated Hartree-Fock term in `grad_dft/external/_hf_density.py`").
   - What you've tried and what remains unresolved.
3. Ask the user for guidance: additional subtasks, algorithmic hints, descoping,
   or splitting the task.
4. **Transfer user guidance to the task** per the `user-guidance-transfer`
   skill — post it verbatim as a comment on the task issue.
5. Append a `## Design Notes` section to `tasks/detailed_plan.md`. See the
   `planning` skill for the full template. Minimum required content:

   - **Correct Design**: algorithmic design as confirmed by the user —
     ansatz, loss decomposition, invariants. Quote the user's design guidance
     verbatim where available.
   - **Blocked Subtasks**: which subtasks are blocked and why.
   - **Root Causes**: why each failure occurs, with concrete examples. Every
     limitation MUST be traceable to a concrete input, a concrete location in
     the data structure, and a concrete execution path in the code. Never write
     vague descriptions.
   - **Approaches Tried**: what was attempted and why it didn't fully work.
   - **Remaining Work**: concrete, actionable items (e.g., "propagate the
     density matrix through the SCF loop by adding a field to the `Molecule`
     dataclass") — not vague goals.
   - **Skipped Tests**: list any tests skipped and the reason.

     Commit this summary so the plan serves as a persistent design record for
     future task refinement.
6. Post the block report (the `## Design Notes` content) as a comment on the
   task issue, so the block is visible in the task record.

## Task Completion Verification

This section is the **single source of truth** for what "done" means. Before
a task is considered done — its last subtask's commit carries `Closes #<N>`
(the task's own issue) and is merged to the integration branch
(`git config --get graddft.integrationBranch || echo agent_settings`) —
verify:

- [ ] Every clause in the task description is traceable to implemented and
      committed code.
- [ ] No subtask was reverted without resolution.
- [ ] No requirement was silently skipped or deferred.
- [ ] All tests pass (0 failures, 0 skipped).
- [ ] All quality gates pass (see the `quality-gates` skill).
- [ ] If any of the above fails, the task is NOT done — it is blocked. Follow
      the Blocked Work Protocol. Do NOT close it with known unresolved
      limitations.

Partial completion is not completion. "All passing tests are for the parts I
did" does not mean the task is done if other parts were reverted.

## Marking Complete

A task is complete when its last subtask's commit carries `Closes #<N>` (the
task's own issue number) as a standalone line and that commit is on the
integration branch. The closing keyword takes effect when the commit reaches
`main`: because work is stacked on the integration branch, the issue stays open
until the final pull request to `main` (opened on the user's explicit request)
lands. Never edit the issue body — it is user-authored and immutable.

Commit hashes recorded in `tasks/detailed_plan.md` may change after an
integration-branch rebase; the `(<issue>-S<n>)` commit subject is the stable
key.

Complete means COMPLETE: every requirement met, every test passing, every
edge case handled. Never close a task with known failures or unresolved
limitations.
