---
name: verify-fix-loop
description: Methodology for rapidly diagnosing, reproducing, and fixing test failures without regressions.
---
# Verify-Fix-Loop

1. Reproduce failing test with `-x -q --tb=short`.
2. Classify:
   - import/env
   - logic error
   - fixture/mock mismatch
   - nondeterminism
   - missing model / weights
3. Fix the source code, never the assertions, unless test assertion contradicts spec.
4. Use `pytest.approx` for floating point comparisons.
5. Missing model -> route cleanly to fallback and add a fallback test.
6. Rerun failing test, then all phase tests, then full test suite.
7. Log error signature and resolution in `PROGRESS.md`.
