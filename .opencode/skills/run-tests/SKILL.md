---
name: run-tests
description: Use when running the GradDFT test suite (pytest). Thin pointer to .github/workflows/install_and_test.yaml, which holds the exact command and working directories, plus the machine-specific pitfalls.
---

# Run tests

What the test pipeline is and where the exact commands live (the CI workflow
steps) are in `.github/workflows/install_and_test.yaml`. Read that workflow before
running anything.

## Commands

CI runs pytest from root directory. Example:

    pytest -v tests/unit/test_eigenproblem.py

## Notes

- No coverage gate is configured; the suite is expected to report 0 failures
  and 0 skipped.
