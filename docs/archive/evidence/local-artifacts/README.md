# Local Artifacts

This directory is the sink for local benchmark outputs, profiler captures,
scratch logs, and other generated evidence that should not live at the
repository root for OSS release hygiene.

Rules:

- Put generated runs under `generated/`.
- Do not reference files in `generated/` from canonical docs.
- Promote only curated evidence into a dated tracked document under
  `docs/archive/evidence/` if it materially supports a release claim.
