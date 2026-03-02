# Release & Pre-Release Pipeline

InferFlux now has a dedicated GitHub Actions workflow (`.github/workflows/release.yml`) that attaches to every successful `CI` run:

1. **Pre-release verification (CI on `main`):** once the primary `CI` workflow finishes successfully on `main`, `release.yml` is triggered via `workflow_run`. Packaging jobs (Linux, macOS, Windows) run exactly once using the already-tested commit, and we upload the resulting installers as workflow artifacts for smoke testing.
2. **Tagged release (`vX.Y.Z`):** CI also runs for tag pushes, which triggers the same packaging jobs. After all artifacts (including the Homebrew formula and winget manifest) are produced, the workflow creates a GitHub Release and publishes the exact artifacts that were validated in pre-release. No redundant builds occur between pre-release and release.

## Trigger matrix
| Trigger (via `workflow_run`) | Jobs | Output |
|------------------------------|------|--------|
| `CI` success on `main`       | `linux-packages`, `mac-packages`, `windows-packages`, `manifests` | TGZ/DEB/RPM + PKG/DMG + MSI/ZIP artifacts plus rendered Homebrew/Winget manifests (pre-release only). |
| `CI` success on `v*.*.*` tag | Same jobs + `create-release` | Publishes a GitHub Release that reuses the pre-built artifacts (Linux/Mac/Windows installers + manifests). |

## Artifact names
- Linux: `inferflux-<version>-Linux.tar.gz`, `.deb`, `.rpm`
- macOS: `inferflux-<version>-Darwin.tar.gz`, `.pkg`, `.dmg`
- Windows: `inferflux-<version>-win64.msi`, `.zip`
- Manifests: `homebrew/inferflux.rb`, `winget/inferencial.inferflux.yaml` (rendered with the correct version, URLs, and SHA256 values)

## Manual promotion flow
1. Merge changes into `main` — the pre-release jobs produce installers and manifests for smoke testing (no GitHub Release is created yet).
2. Verify installers locally (see `docs/Installer.md` for commands).
3. Tag the desired commit (`git tag v0.x.x && git push origin v0.x.x`).
4. The tagged workflow reuses the exact artifacts produced by the packaging jobs and publishes a GitHub Release automatically (Linux/Mac/Windows installers plus Homebrew/Winget manifests).
