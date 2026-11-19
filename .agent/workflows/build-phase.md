# Build Phase Workflow

This workflow drives each implementation phase P0 through P9:

1. **PLAN**: Inspect target requirements, check current `PROGRESS.md`, define minimal concrete files.
2. **BUILD**: Write complete code files. No placeholders or incomplete implementations.
3. **VERIFY**: Run phase-specific verification command.
4. **JUDGE**:
   - IF PASS: Proceed to step 6.
   - IF FAIL: Proceed to step 5.
5. **FIX**: Diagnose failure root cause, apply minimal fix, re-verify. Max 5 iterations before logging BLOCKED.
6. **COMMIT**: Stage changes, commit with `P<n>: <summary>`, update `PROGRESS.md`, run full regression test suite.
