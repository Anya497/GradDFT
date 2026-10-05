---
name: run-tests
description: Use when running the GradDFT test suite (pytest). Thin pointer to .github/workflows/install_and_test.yml, which holds the exact commands and working directories, plus the machine-specific pitfalls.
---

# Run tests

The test pipeline and the exact commands live in the CI workflow steps in
`.github/workflows/install_and_test.yml`. Read that workflow before running
anything — do not reconstruct the command list from memory.

## Setup

The suite needs the package and its example extras installed:

```bash
pip install -e .
pip install -e ".[examples]"
```

## Commands

CI runs `pytest -v <file>` from the repository root, once per test file (two
unit files, then the molecules and solids integration files). Run the same
invocations locally so local and CI results are comparable. Example:

```bash
PYSCF_CONFIG_FILE=.github/workflows/pyscf_conf.py \
  pytest -v tests/unit/test_eigenproblem.py
```

## Notes

- `PYSCF_CONFIG_FILE` must point at `.github/workflows/pyscf_conf.py` for the
  functional-prediction tests; CI sets it as a job-level `env` var. Without it
  PySCF builds functionals with its own defaults and results differ from CI.
- `tests/unit/` is fast; `tests/integration/` compiles SCF loops and is slow.
  Prefer the unit file that covers your change while iterating, then run the
  full CI list before calling a subtask done.
- No coverage gate is configured; the suite is expected to report 0 failures
  and 0 skipped.
