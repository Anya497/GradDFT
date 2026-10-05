# Detailed Plan — #5 Support newer JAX

Integration branch: `agent_settings` (stacked on `main`).
Feature branch: `support_newer_jax`.

Problem: `from jax.config import config` was removed from JAX, so the molecule
test suite fails at collection on a current JAX (verified on jax/jaxlib 0.10.2).
`requirements.txt` also declares `jax>=0.4.14`, a floor the code cannot satisfy,
because `grad_dft/utils/chunk.py` imports `jax.extend.linear_util`.

B3LYP stays excluded from CI by design (see the issue); DM21 stays excluded
because it is broken independently of JAX (see the issue).

## Reuse analysis

- 12 of the 17 files that call `config.update("jax_enable_x64", True)` already
  use the supported import `from jax import config` (e.g.
  `tests/integration/molecules/test_Harris.py`, `tests/unit/test_eigenproblem.py`,
  `examples/advanced_scripts/test_constraints.py`). The 5 broken files are fixed
  to that existing convention; no new pattern is introduced.
- `grad_dft/utils/types.py` (`jax.config.x64_enabled`) and
  `grad_dft/utils/chunk.py` (`jax.api_util.argnums_partial`,
  `jax.extend.linear_util`) were probed on jax 0.10.2 and resolve, so the library
  needs no import surgery.

### S1: Replace the removed `jax.config` module import

**Code:** `from jax.config import config` -> `from jax import config` in
          `tests/integration/molecules/test_predict_B88.py`,
          `tests/integration/molecules/test_predict_B3LYP.py`,
          `tests/integration/molecules/test_predict_DM21.py`,
          `examples/advanced_scripts/train_scf_loop.py`,
          `examples/advanced_scripts/train_complex_model.py`
          (the two example scripts are the user-facing documentation surface and
          carry the identical defect; `B3LYP` and `DM21` are fixed for the same
          reason even though they stay out of CI)
**Tests:** `pytest --collect-only` over the whole `tests/` tree, before vs after.
          Criterion: the three `test_predict_*.py` modules stop erroring. Measured:
          7 collection errors / 60 tests collected before, 4 errors / 69 tests after.
          The 4 remaining errors are pre-existing, in files this subtask does not
          touch, and belong to S2 (see Design Notes). No test assertion changes.
**Docs:** No user-facing change needed — no API, install, or usage-flow change;
          the example scripts keep their current behaviour, they just stop raising
          `ModuleNotFoundError` at import.

**Spec:**
- One-line change per file; `config.update(...)` calls stay exactly as they are.
- `jax_enable_x64` must still be set before any array is created (the comment
  "This only works on startup!" stays).
- Success criterion: `pytest --collect-only tests/` reports no collection errors
  for the three `test_predict_*.py` modules.

- [x] S1: Implement
- [x] S1: Write tests (collect-only before/after: 7 errors/60 tests -> 4/69)
- [x] S1: Update documentation (no user-facing change needed, justified above)
- [x] S1: Pre-Commit Check (black clean on all 5 files; pylint 124 -> 111,
      no new messages)
- [x] S1: Quality checks (reused the existing `from jax import config`
      convention; no tolerance or assertion touched; library modules untouched)
- [x] S1: Commit

### S2: Revalidate the suite on modern JAX and fix library breakage

**Code:** whatever `grad_dft/` changes the suite proves necessary. None is
          anticipated from the import probe; the diff is driven by test results
          only.
**Tests:** the full CI list from `.github/workflows/install_and_test.yml`, run with
          `PYSCF_CONFIG_FILE=.github/workflows/pyscf_conf.py`. 0 failures, 0
          skipped. `B3LYP` and `DM21` are not in that list and stay out.
**Docs:** No user-facing change needed unless a fix changes public behaviour.

**Spec:**
- No tolerance may be widened, and no assertion weakened or skipped, to make a
  test pass. A failure that cannot be fixed within the JAX update is a block, not
  a licence to adjust the test.
- The B88 jitted-vs-non-jitted equivalence test and the energy-vs-PySCF
  comparisons are the numerical guard: they must pass unchanged.

### S3: Make the declared JAX bound honest

**Code:** `requirements.txt` — raise the `jax` and `jaxlib` floors from `>=0.4.14`
          to `>=0.4.16`, the oldest release that provides the
          `jax.extend.linear_util` module `grad_dft/utils/chunk.py` imports.
**Tests:** none — a dependency bound. Verified against the published artifacts:
          `jax/extend/linear_util.py` is absent in the 0.4.14 and 0.4.15 sdists and
          present from 0.4.16 on.
**Docs:** No user-facing change needed — `README.md` states no JAX version.

**Spec:**
- `jax` and `jaxlib` move together; they are released in lockstep and pip does
  not resolve them independently in practice.
- Keep the change to those two lines; do not touch unrelated pins.
- Do not raise the floor further: CI still runs Python 3.9, so the floor must stay
  within what JAX supports there.

## Design Notes (discovered during implementation)

### The style hooks have never been enforced on this repository

`.pre-commit-config.yaml` pins `black` 22.3.0 and runs `pylint -rn -sn
--rcfile=.pylintrc`, but **51 of the 59 tracked Python files fail `black
--check`**, and there is no lint job in `.github/workflows/`. The hook was
configured and never run over the codebase. Two consequences for this task:

- `black` was applied to the five files this task touches (it also cleaned up 13
  pre-existing findings there: trailing whitespace, a missing final newline,
  collapsible imports). Pylint on those files went from 124 to 111 messages, and
  a normalized before/after diff shows **no genuinely new message** — the four
  apparent additions are the pre-existing `wrong-import-order` violation on the
  same import line, re-rendered with a different module string
  (`jax.config.config` before, `jax.config` after).
- Reformatting the whole repository is deliberately **not** done here. It would
  bury a JAX compatibility fix under ~50 files of churn. It belongs in its own
  task, together with adding a lint job to CI so the hook cannot rot again.

`black` 22.3.0 also cannot be run over multiple files on this interpreter: it
calls `asyncio.get_event_loop()`, which raises under the installed `uvloop`
policy on Python 3.12. Single-file invocations work.

### Root causes behind the 4 remaining collection errors

Both are API removals, unrelated to `jax.config`, and both are S2's work:

1. **`jnp.clip(..., a_min=..., a_max=...)` was removed from JAX.** 24 call sites:
   19 in `grad_dft/` (`functional.py`, `popular_functionals.py`, `train.py`) and
   5 in `tests/`. The replacement keywords are `min=` / `max=`. This is what
   breaks `tests/integration/molecules/test_training.py` at import, from the
   `coefficient_inputs` helper at line 98.
2. **`PySCF`'s `get_init_guess()` no longer accepts `max_cycle`**, breaking
   `tests/integration/solids/test_training.py` at import.

The two remaining `solids/` collection errors are order-dependent: those files
collect cleanly on their own (18 tests) and only fail in a full-tree run, so the
cause is import-time state shared across the `solids` test modules rather than
the files themselves. S2 must determine it; do not assume it is fixed by 1 and 2.

### `jax.config` as a module is gone; the package attribute is not

`from jax.config import config` fails on jax 0.10.2 because the submodule was
removed, while `jax.config` remains reachable as an attribute of the `jax`
package. Hence `from jax import config` works and `jax.config.x64_enabled` in
`grad_dft/utils/types.py` still resolves. This is why the fix is an import-line
change only, with no call-site changes.

### The old floor was never satisfiable

`requirements.txt` declared `jax>=0.4.14` while `grad_dft/utils/chunk.py` imports
`jax.extend.linear_util`. Inspecting the published sdists shows
`jax/extend/linear_util.py` is **absent in 0.4.14 and 0.4.15** and **present from
0.4.16** onwards (in 0.4.16 it is the real module, re-exporting
`jax._src.linear_util`; `jax/extend/__init__.py` alone already exists in 0.4.14,
which is why the breakage is easy to miss). An installation that resolved to the
oldest allowed JAX would fail at import of `grad_dft`, so the floor is
documentation of a real constraint, not a policy choice. The fix is `>=0.4.16`
for both `jax` and `jaxlib`, which must move together.

This floor is deliberately kept low rather than raised to a modern release: CI
still runs Python 3.9, and JAX releases that support 3.9 stop well before the
current ones. "Newer JAX" therefore means the code must work across the whole
range the matrix resolves to, which is exactly what the `from jax import config`
change achieves (valid on both ends).