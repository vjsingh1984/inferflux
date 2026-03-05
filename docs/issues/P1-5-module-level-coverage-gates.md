# [P1-5] Module-Level Coverage Gates for High-Risk Paths

Priority: P1
Owner: QA
Effort: 0.75 eng-week
Risk: Low
Dependencies: None
Labels: qa, ci, coverage, quality-gate

## Problem
Global coverage thresholds alone do not protect critical runtime and scheduler modules from silent test erosion.

## Scope Files
- `.codecov.yml`
- `.github/workflows/ci.yml`
- `docs/DeveloperGuide.md`
- `docs/ReleaseProcess.md`

## Test Plan
1. Add path-based coverage status rules for high-risk modules.
2. Ensure CI logs surface module coverage pass/fail explicitly.
3. Validate PR behavior with synthetic low-coverage change (dry run on branch).
4. Confirm no false positives on untouched paths.

## Acceptance Checklist
- [ ] Coverage policy includes module-level gates for `runtime/backends/cuda/**` and `scheduler/**`.
- [ ] CI output shows explicit module coverage status lines.
- [ ] Failing module coverage blocks merge on protected branches.
- [ ] Developer docs explain how to run and interpret coverage checks.
