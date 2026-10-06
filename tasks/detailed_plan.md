# Detailed Plan — #10 CI collection fails: tensorflow-hub imports pkg_resources, removed in setuptools >= 81

Integration branch: `agent_settings` (stacked on `main`).
Feature branch: `fix_ci_pkg_resources` (created from the rebased integration branch).

Problem: CI (PR #8 head `agent_settings` @ 840e2d8, Python 3.12 / ubuntu-latest) fails at
collection with `ModuleNotFoundError: No module named 'pkg_resources'`. A fresh
`pip install -e .` resolves `setuptools` (an unpinned requirement of
`pyscf>=2.3.0`) to 84.0.0, which removed `pkg_resources`; `tensorflow-hub
0.16.1` (the newest release, and the floor `>=0.16.0` we set in #6) still does
`from pkg_resources import parse_version` inside `_ensure_tf_install()` at
import time, so any `import tensorflow_hub` — reached via the vendored
`neural_numint.py` — crashes during test collection.

The local environment (setuptools 81.0.0, `pkg_resources` still shipped) could
not reproduce this, which is why the #6 gate passed while CI failed. Verified
against the real CI env: installing `setuptools==84.0.0` removes
`pkg_resources` locally and reproduces the exact failure.

## Reuse analysis

- `pkg_resources` is referenced in `tensorflow_hub` exactly once at import time
  (`tensorflow_hub/__init__.py:61`, `parse_version` used only to compare
  `tf.__version__ >= 1.15.0`). No other package in our import chain uses it.
- The vendored `neural_numint.py` is the only repo module that imports
  `tensorflow_hub`; it is already patched by #6 (hub.load port), so the shim
  fits the established "surgical edits to the vendored file" precedent.
- No newer `tensorflow-hub` release exists (0.16.1 is latest; pip resolved it
  in CI). Pinning `setuptools<81` in `requirements.txt` is rejected: it fixes
  only this symbol via a build-tool downgrade, adds a dependency-resolution
  constraint that rots, and does not future-proof the import.

## Subtasks

### S1: Shim `pkg_resources.parse_version` on the tensorflow-hub import path

**Code:** in
`grad_dft/external/density_functional_approximation_dm21/density_functional_approximation_dm21/neural_numint.py`,
immediately before `import tensorflow_hub as hub`:

```python
try:
    import pkg_resources  # pylint: disable=unused-import
except ModuleNotFoundError:  # setuptools >= 81 removed pkg_resources
    # tensorflow_hub.__init__ still imports pkg_resources.parse_version to
    # check the installed tensorflow version. Provide the one symbol it needs
    # when pkg_resources is no longer part of the environment.
    import sys as _sys
    from types import SimpleNamespace as _SimpleNamespace

    def _parse_version(version):
        """Returns a comparable key for a dotted numeric version string."""
        return tuple(int(part) for part in version.split(".") if part.isdigit())

    _sys.modules.setdefault(
        "pkg_resources", _SimpleNamespace(parse_version=_parse_version)
    )

import tensorflow_hub as hub
```

Notes:
- `parse_version` semantics: `pkg_resources.parse_version("2.21.0")` returns a
  comparable version key; a dotted-numeric `int` tuple compares identically for
  the only use (`tf.__version__` vs `"1.15.0"`, both clean release tags).
- No external deps (e.g. `packaging`) added; zero-dependency and
  setuptools-version-proof.
- `_sys.modules.setdefault` keeps a pre-existing `pkg_resources` untouched and
  only fills the gap.

**Tests:** reproduce the CI failure and the fix:
- `/tmp/opencode/repro_pkg_resources.py` (a `sys.meta_path` finder that raises
  `ModuleNotFoundError` for `pkg_resources`, then imports `grad_dft.external`):
  without the shim → the CI error; with it → `IMPORT OK`.
- Real-env check: `pip install setuptools==84.0.0` (removes `pkg_resources`,
  matching CI) → `pytest -v tests/unit/test_eigenproblem.py` passes.
- Full CI suite green under setuptools 84.
- `pylint -rn -sn --rcfile=.pylintrc` on the vendored file: symbol multiset
  identical to subtask parent (no new messages — verified: the shim adds none).
- `black`: vendored file not reformatted (same precedent as #6; baseline debt
  unchanged).

**Docs:** no user-facing change — nothing in README/examples describes the
vendored import internals. The shim comment records the reason.

**Quality checks:** no duplication (the shim lives on the single import path);
no tolerance or assertion touched; scope limited to the import shim.

**Commit:** `(#10-S1): shim pkg_resources.parse_version for tensorflow-hub imports`
plus a standalone `Closes #10`.

- [x] S1: Implement (stdlib-only shim before `import tensorflow_hub`; comment
      documents the setuptools >= 81 removal)
- [x] S1: Write tests (meta-path repro fails pre-fix / passes post-fix; real
      `setuptools==84.0.0` env: `test_eigenproblem.py` 2 passed; full CI suite
      123 passed)
- [x] S1: Pre-Commit Check (pylint vs parent: symbol multiset identical; black:
      vendored file untouched, no new debt)
- [x] S1: Quality checks (no duplication — single import path; zero new deps;
      no tolerance/assertion touched)
- [x] S1: Commit (Closes #10)

## Acceptance criteria mapping

| Criterion | Where |
|---|---|
| CI collection error gone (`test_eigenproblem.py` collects on a fresh 3.12 install) | S1: shim before `import tensorflow_hub` |
| No new lint/debt | S1: pylint multiset identical; black untouched (vendored) |
| Robust to future setuptools | S1: shim depends only on stdlib; `setuptools<81` pin rejected |