---
name: planning
description: Use when planning tasks: creating global plans across multiple tasks, decomposing a task into atomic subtasks, or authoring new task descriptions. Covers multi-task planning, detailed plan format, atomic subtask requirements, documentation surface mapping, and task authoring guidelines.
---

# Planning

## Multi-task Planning

When the user asks to work on a **set of related tasks**, do NOT jump directly
into implementation. First set up the batch, then plan it:

1. Create a **hub** issue (label `hub`) for the batch and one issue per task
   (label `task`), and attach each task issue to the hub as a GitHub
   **sub-issue** via the REST API (`POST repos/$REPO/issues/<HUB>/sub_issues`
   with the task's database `id`); a link in the hub body is **not** a
   sub-issue. The exact commands live in the `workflow-management` skill.
2. Create the high-level global plan in `tasks/global_plan.md`, naming the hub
   and listing each task with its issue number, then mirror it to the hub as a
   comment whose first line is `<!-- global-plan -->`.

The global plan must:

- Name the **hub** issue it belongs to.
- List all tasks to be done with their IDs, issue numbers (`#N`), brief
  descriptions, and a **Status** (`[done #N]` once the task is integrated).
- Record the integration branch the batch targets (`agent_settings` by
  default; see the resolution command in the `workflow-management` skill).
- Identify dependencies between tasks (which must be done before which).
- Identify potential conflicts or overlapping changes (e.g., two tasks
  modifying the same file).
- Identify shared infrastructure that multiple tasks need, and place it in a
  single shared module instead of duplicating it across tasks. The usual home
  for it is `grad_dft/utils/` — type aliases in `grad_dft/utils/types.py`,
  array helpers in `grad_dft/utils/utils.py`, memory helpers in
  `grad_dft/utils/chunk.py`.
- Identify existing reusable abstractions (see "Reuse analysis" below) that new
  tasks should use rather than reinvent.
- Keep plotting, reporting, and experiment-driver code out of `grad_dft/`; it
  belongs in `examples/` (`advanced_scripts/` for scripts,
  `article_experiments/` for paper reproductions) so rendering and library
  logic stay in separate modules from the start.
- Propose an execution order that minimizes rework and avoids conflicts.
- Align tasks with the project architecture: `README.md` describes the
  intended usage flow (`Molecule`/`Solid` -> energy densities and coefficients
  -> `NeuralFunctional` -> training), and the tests under `tests/unit/` and
  `tests/integration/` show how each layer is exercised.

After the global plan is created and mirrored to the hub, proceed with the
normal working loop: one task at a time, a feature branch per task created from
the integration branch, a detailed plan in `tasks/detailed_plan.md` for each.
After each task integrates, mark it `[done #<N>]` in `tasks/global_plan.md` and
update the hub's `<!-- global-plan -->` comment. The local file is the single
source of truth; the hub comment is a pure mirror (see the `workflow-management`
skill).

## Detailed Plan (Atomic Subtasks)

`tasks/detailed_plan.md` MUST decompose the task into atomic subtasks. Each
subtask:

- Is small enough to complete in a single focused work session (typically
  30–90 minutes).
- Produces a compilable, testable increment — no partial implementations left
  uncommitted.
- Has a unique identifier (e.g., "S1", "S2") used in commit messages and plan
  tracking.

If a task is ambiguous or underspecified **during planning** (conflicting
requirements, unclear scope, missing constraints), do not guess. Ask the user
for clarification, then transfer the guidance to the task per the
`user-guidance-transfer` skill before proceeding with decomposition.

### Reuse analysis

Before writing the detailed plan, search the repository for functions, types,
and documentation sections that can be reused or generalized rather than
reimplemented: `grep`/`glob` across `grad_dft/`, `tests/`, and `examples/`,
starting from `grad_dft/utils/` and the neighbouring module of the code being
changed. Record what is reused in each subtask's **Code** section. Reusing
`grad_dft/utils/types.py` aliases (`Array`, `Key`, `PyTree`, `Scalar`, ...) in
new signatures counts as reuse.

### Subtask Format

Each subtask in `tasks/detailed_plan.md` MUST include these four sections. **A
subtask with missing Code, Tests, or Docs sections is incomplete and must not
be executed.**

```
### SN: <title>

**Code:** <files to create or modify, types and functions to add>
**Tests:** <test files to create or modify, specific test approaches:
           pytest unit test in tests/unit/, integration test in
           tests/integration/... , etc.>
**Docs:** <README.md sections, examples/ notebooks or scripts, and docstrings
         to create or update — see "Documentation surface" below.>

**Spec:**
- <detailed implementation specification>
```

Example:

```
### S1: Add a prediction test for the DM21 functional

**Code:** New `tests/integration/molecules/test_predict_DM21.py` (reuse the
          molecule construction and energy-comparison helpers from
          `tests/integration/molecules/test_predict_B88.py`)
**Tests:** The new file itself is the test: assert the neural functional
          predicts the reference DM21 energies within tolerance, and that the
          loss is finite over a few SCF iterations
**Docs:** No user-facing change needed (test-only); keep the README untouched

**Spec:**
- Reuse the `B3LYP_WITH_VWN5`-style PySCF configuration path ...
```

### Documentation surface

This repository has no Sphinx site. The documentation surface is:

- **`README.md`** — `Functionality`, `Install`, `Use example`, `Bibtex`.
  Update it when a task adds or changes public API, installation, or the
  intended usage flow.
- **`examples/`** — `basic_notebooks/` (first use of the library),
  `intermediate_notebooks/` (specific training/functional techniques),
  `advanced_scripts/` (scripts such as `examples/advanced_scripts/train_scf_loop.py`,
  plus `examples/advanced_scripts/readme.md`), `article_experiments/` (paper
  reproductions). Update or add a notebook/script when a task changes how the
  library is used.
- **Docstrings** — numpydoc-style docstrings on the public API in
  `grad_dft/`. New public functions, classes, and parameters get them; keep
  examples in docstrings consistent with the README snippets.

Use this mapping to fill each subtask's **Docs** section. A subtask whose change
is invisible in all three places writes "No user-facing change needed" and
explains why — it is never left blank.

### Granularity

If a subtask cannot be committed as a self-contained increment, it is too large
— split it further. The commit message format lives in the `subtask-loop`
skill; it is the single source of truth for commit format and procedure.

### Bounded uncommitted work

Uncommitted work on a feature branch must never exceed one atomic subtask. If a
session is interrupted, the loss is bounded to that single subtask.

## Post-Implementation Design Notes

When a task hits algorithmic limitations — partially completed, with skipped
tests or remaining work — append a `## Design Notes` section to
`tasks/detailed_plan.md`. This serves as persistent design knowledge for future
task refinement. The section must be structured as follows:

```
## Design Notes (discovered during implementation)

### <Topic Title>

Design rationale, coordinate spaces, invariants — as confirmed by the user.

### <Failure Topic Title>

- Root causes with concrete examples (e.g., "for input `mol` with 2 basis
  functions, the generalized eigenvalue problem returns 0 occupied states
  because...")
- What was attempted and why it didn't fully work
- Remaining work: concrete, actionable items
- Skipped tests: list them and the reason
```

Requirements:

- Every algorithmic limitation MUST be traceable to a concrete input, a
  concrete location in the data structure, and a concrete execution path in
  the code.
- Never write vague descriptions like "some gradients are missing" — specify
  which input, which state/range, and which path should produce them.
- Remaining work items must be actionable (e.g., "propagate the electron count
  through the SCF loop by adding it to the `Molecule` dataclass") — not vague
  goals.
- If the user provided design guidance (e.g., ansatz choice, loss
  decomposition), record it verbatim in the `### <Topic>` section as the
  authoritative reference.

## Task Authoring Guidelines

When creating a new task issue (`gh issue create`, label `task`), follow
these rules (they apply to the issue body):

- **Specify output format upfront**. If the task produces artifacts (plots,
  tables, checkpoint files, trained models), include the exact layout, units,
  and formatting rules in the task description.
- **Keep tasks single-responsibility**. A task should do one thing. If it
  requires more than 5 sub-items or spans multiple unrelated concerns, split it
  into multiple tasks.
- **Specify equivalence requirements**. For any new algorithm variant,
  explicitly state "must produce results identical to X" (e.g., identical to
  the non-jitted implementation, or to the PySCF reference energy) so
  equivalence tests are built from the start.
- **Specify type genericity**. If a module must handle arbitrary inputs, state
  it explicitly (e.g., "generic over the `Molecule` and `Solid` classes", not
  molecules only).
- **Specify reuse expectations**. If the task builds on existing infrastructure
  (e.g., "reuse the array aliases from `grad_dft/utils/types.py`"), name the
  dependencies. This prevents reinvention.

## Task Completeness Verification

See the `subtask-loop` skill — it is the single source of truth for verifying
task completion before a task is considered done.
