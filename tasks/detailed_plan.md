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

**Code:** driven by test results, in four groups.
          1. `jnp.clip(..., a_min=/a_max=)` → `min=/max=` at 28 sites: 24 in
             `grad_dft/{functional,popular_functionals,train}.py`, 4 in
             `tests/integration/{molecules,solids}/test_training.py`.
          2. `mf.kernel(max_cycle=N)` → `mf.max_cycle = N; mf.kernel()` at 5
             sites in the same two test modules. `max_cycle` is an attribute on
             the PySCF class; passing it as a keyword leaks into
             `get_init_guess(**kwargs)` and raises `TypeError`.
          3. `tests/**/__init__.py` (5 new files). `test_training.py`,
             `test_non_xc_energy.py` and `test_functional_implementations.py`
             each exist twice, once under `molecules/` and once under `solids/`.
             Without packages, pytest imports them under one basename and errors
             with "import file mismatch" in any run that sees both. CI hid this by
             invoking one file per process; the tree could not be validated as a
             whole.
          4. Notebooks: 3 × the S1 import fix, 3 × the `a_min` rename.
**Tests:** the full CI list from `.github/workflows/install_and_test.yml`, run with
          `PYSCF_CONFIG_FILE=.github/workflows/pyscf_conf.py`, one file per
          invocation exactly as CI does. 0 failures, 0 skipped. `B3LYP` and `DM21`
          are not in that list and stay out.
**Docs:** No user-facing change needed — the fixed notebooks keep their behaviour,
          they stop raising at their first cell.

**Spec:**
- No tolerance may be widened, and no assertion weakened or skipped, to make a
  test pass. A failure that cannot be fixed within the JAX update is a block, not
  a licence to adjust the test.
- The B88 jitted-vs-non-jitted equivalence test and the energy-vs-PySCF
  comparisons are the numerical guard: they must pass unchanged.

- [x] S2: Implement (items 1-4 above)
- [x] S2: Write tests (`pytest --collect-only tests/`: 4 errors / 69 tests ->
      **0 errors / 127 tests**)
- [x] S2: Update documentation (no user-facing change needed, justified above)
- [x] S2: Pre-Commit Check (pylint 197 -> 186 messages on the 7 touched modules,
      11 `E1123 unexpected-keyword-arg 'a_min'` errors eliminated and no genuinely
      new message; `black` deliberately **not** applied here — see Design Notes)
- [x] S2: Quality checks (existing conventions reused; no tolerance or assertion
      touched)
- [ ] S2: Commit
- [x] S2: Quality gate — **9 of 10 CI files green**; `test_dm21` blocked, see
      "DM21 is blocked by two independent upstream breaks" below

### S3: Make the declared JAX bound honest

**Code:** `requirements.txt` — raise the `jax` and `jaxlib` floors from `>=0.4.14`
          to `>=0.4.27`.
**Tests:** none — a dependency bound. Verified against the published artifacts, see
          Design Notes ("The true floor is 0.4.27, not 0.4.16").
**Docs:** No user-facing change needed — `README.md` states no JAX version.

**Spec:**
- `jax` and `jaxlib` move together; they are released in lockstep and pip does
  not resolve them independently in practice.
- Keep the change to those two lines; do not touch unrelated pins.
- Do not raise the floor further: CI still runs Python 3.9, and the last JAX
  release supporting 3.9 is 0.4.30, so a floor above 0.4.30 would make the 3.9 job
  uninstallable.

- [ ] S3: Implement
- [ ] S3: Write tests (n/a — dependency bound)
- [ ] S3: Update documentation (n/a)
- [ ] S3: Pre-Commit Check
- [ ] S3: Quality checks
- [ ] S3: Commit

## Design Notes (discovered during implementation)

### The style hooks have never been enforced on this repository

`.pre-commit-config.yaml` pins `black` 22.3.0 and runs `pylint -rn -sn
--rcfile=.pylintrc`, but **46 of the 64 tracked Python files fail `black
--check`** (40 of 53 first-party, plus 6 of 11 under `grad_dft/external/`), and
there is no lint job in `.github/workflows/` — the single `build` job installs and
runs `pytest` and never invokes `pre-commit`. The hook was configured and never run
over the codebase. Two consequences for this task:

- `black` was applied to the five files S1 touches (it also cleaned up 13
  pre-existing findings there: trailing whitespace, a missing final newline,
  collapsible imports). Pylint on those files went from 124 to 111 messages, and
  a normalized before/after diff shows **no genuinely new message** — the four
  apparent additions are the pre-existing `wrong-import-order` violation on the
  same import line, re-rendered with a different module string
  (`jax.config.config` before, `jax.config` after).
- `black` was **not** applied to S2's modules. Reformatting those 7 files costs
  ~737 changed lines against a 44-line compatibility fix — 17:1 noise that would
  make the change unreviewable and mix two unrelated concerns into one commit.
  Pylint was run there and shows the fix removing 11 real errors.
- Reformatting the whole repository is deliberately **not** done here. It would
  bury a JAX compatibility fix under ~50 files of churn, and `grad_dft/external/` is
  vendored verbatim from `deepmind-research`, so reformatting it also creates a
  local divergence from upstream. Both are filed as follow-up **#7**, together with
  adding a lint job to CI so the hook cannot rot again.

`black` 22.3.0 also cannot be run over multiple files on this interpreter: it
calls `asyncio.get_event_loop()`, which raises under the installed `uvloop`
policy on Python 3.12. Single-file invocations work.

### Root causes behind the 4 remaining collection errors

All three are API removals, unrelated to `jax.config`, and all are S2's work:

1. **`jnp.clip(..., a_min=..., a_max=...)` was removed from JAX.** 28 call sites:
   24 in `grad_dft/` (`functional.py`, `popular_functionals.py`, `train.py`) and
   4 in `tests/`. The replacement keywords are `min=` / `max=`. This is what
   breaks `tests/integration/molecules/test_training.py` at import, from the
   `coefficient_inputs` helper.
2. **`PySCF`'s `get_init_guess()` no longer accepts `max_cycle`**, breaking
   `tests/integration/solids/test_training.py` at import. `max_cycle` is an
   attribute on the SCF class, so the fix sets it rather than passing it.
3. **Duplicate test-module basenames.** `test_training.py`,
   `test_non_xc_energy.py` and `test_functional_implementations.py` each exist
   under both `molecules/` and `solids/`. Pytest derives a module name from the
   basename, so the second import collides with the first and raises "import file
   mismatch". This is why the `solids/` errors were order-dependent: each file
   collected alone, and only failed once both copies were collected. The
   `tests/**/__init__.py` files make each directory a package, so the derived
   module names become distinct.

Cause 3 was invisible to CI, which runs one file per `pytest` process and so
never has both copies in a single collection.

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
documentation of a real constraint, not a policy choice. `jax` and `jaxlib` are
released in lockstep and must move together.

### But 0.4.16 is not enough — the real floor is 0.4.27

`jax.extend.linear_util` sets the floor at 0.4.16, but S2's `min=` / `max=`
rewrite sets a higher one. Reading `jnp.clip`'s published signature per release:

| release | `a_min` / `a_max` | `min` / `max` |
|---|---|---|
| ≤ 0.4.26 | yes | **no** |
| 0.4.27 – 0.4.30 | yes (deprecated) | yes |
| ≥ 0.4.31 | **removed** | yes |

Since S2 removes the last `a_min` / `a_max` call site, the declared floor must be
a release where `min` / `max` exist: **0.4.27**. Below that, the requirements
would advertise a JAX the code cannot run on. Above 0.4.30, CI's Python 3.9 job
becomes uninstallable. 0.4.27 is therefore both the lowest correct bound and the
lowest that keeps the existing CI matrix installable.

### DM21 is blocked by two independent upstream breaks

`tests/integration/molecules/test_functional_implementations.py::test_dm21` fails
on two counts, neither related to JAX. Verified by stashing this task's changes and
re-running on pristine `agent_settings`: it fails identically there, so this task
did not cause it and cannot be finished by finishing S2.

1. **TF-Hub 0.16 removed the TF1 `Module` API.** `hub.Module` raised
   `AttributeError` at `neural_numint.py:219`. `tensorflow_hub` 0.15.0 still ships
   it, but cannot be used as a fix: its `__init__.py` imports
   `tensorflow_hub.estimator`, which does `from tensorflow.compat.v1 import
   estimator`, and that submodule no longer exists in TF 2.21. Pinning TF-Hub
   below 0.16 therefore drags TF back too — the opposite of this task's goal, and
   it fails against the `tensorflow>=2.13.0` already declared.
2. **PySCF 2.13's `eval_xc_eff` no longer routes through `eval_xc`.** This is the
   real blocker. `numint.nr_uks` calls `ni.eval_xc_eff(...)`, which now calls
   `self.eval_xc1(...)` — a method `NeuralNumInt` does not override, so it falls
   through to `libxc`, which then tries to resolve the DM21 functional as a real
   libxc name and dies on `ValueError: cannot reshape array of size 240720 into
   shape (2,1,24072)`. `NeuralNumInt.eval_xc` is now dead code: **it is never
   called**.

A `hub.load`-based port was prototyped to prove point 1 is fixable in isolation
(`hub.load(path).signatures["default"]` with `$`→`_` keyword mangling, and the
signature's `(batch, 1)` shapes match the code's `features` dict exactly). It
does clear break 1, and then break 2 is reached. It was reverted rather than
committed: a partial port of vendored third-party numerics that still leaves the
test red, and that changes the failure message from the clear
`AttributeError: hub.Module` to a misleading reshape error, is worse than leaving
the file pristine.

Fixing break 2 means reimplementing PySCF's derivative-tensor contract for a
neural meta-GGA inside vendored DeepMind code — reshaping `eval_xc`'s
`(vrho, vsigma, vlapl, vtau)` outputs into the `(spin, ncomp, ngrids)` layout
`nr_uks` indexes. That is research-grade numerics work, unrelated to JAX, and it
belongs in its own task rather than inside a JAX compatibility update.

**Decision:** the user chose to keep it out of this task and file it separately.
Follow-up issue: **#6**. The gate is therefore reported as BLOCKED on file 4, not
quietly reported green.

## Gate status

Per-file `pytest -v` invocations from `.github/workflows/install_and_test.yml`,
with `PYSCF_CONFIG_FILE=.github/workflows/pyscf_conf.py`:

| # | file | result |
|---|---|---|
| 1 | `tests/unit/test_eigenproblem.py` | 2 passed |
| 2 | `tests/unit/test_loss.py` | 3 passed |
| 3 | `tests/integration/molecules/test_non_xc_energy.py` | 35 passed |
| 4 | `tests/integration/molecules/test_functional_implementations.py` | 12 passed, **2 failed** (`test_dm21`) |
| 5 | `tests/integration/molecules/test_Harris.py` | 6 passed |
| 6 | `tests/integration/molecules/test_predict_B88.py` | 4 passed |
| 7 | `tests/integration/molecules/test_training.py` | 26 passed |
| 8 | `tests/integration/solids/test_training.py` | 14 passed |
| 9 | `tests/integration/solids/test_non_xc_energy.py` | 10 passed |
| 10 | `tests/integration/solids/test_functional_implementations.py` | 8 passed |

**BLOCKED** on file 4, per the rule "do not assess whether a failure is
pre-existing — fix it regardless". The fix is out of scope for this task and needs
a decision; see the DM21 section above.