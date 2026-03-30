# Security Policy

## Supported Release Surface

| Area | Status |
|---|---|
| Current default branch | Supported |
| Historical snapshots and archived benchmark artifacts | Not supported for security fixes |
| Local benchmark scripts and experimental kernel toggles | Best effort only |

## Reporting a Vulnerability

Please report security issues privately to the maintainers instead of opening a public issue.

When reporting, include:
- affected version or commit
- reproduction steps
- impact assessment
- any suggested mitigation

## Response Goals

| Stage | Target |
|---|---|
| Initial acknowledgment | 5 business days |
| Triage | 10 business days |
| Fix or mitigation plan | Best effort after triage |

## Scope Notes

- Authentication, admin APIs, policy enforcement, and audit paths are high-priority review areas.
- Unsafe release claims or insecure default configs may also be treated as security issues when they materially mislead operators.
