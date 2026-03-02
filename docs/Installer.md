# Installer & Packaging Notes

InferFlux now produces native installers across every major desktop/server platform using `cmake --install` + CPack—no Docker images are required for these flows.

## Build once, package everywhere

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_WEBUI=ON
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
cmake --install build --prefix build/install # optional staging dir
```

All installers are emitted via `cpack --config build/CPackConfig.cmake -G <generator>`. The sections below call out the generator and any platform-specific metadata.

---

## Homebrew (macOS)

- Formula lives in `installers/homebrew/inferflux.rb`. The release pipeline renders a versioned copy (sha256 + URL) and uploads it alongside the other artifacts so you can `brew tap inferencial/cli` without editing the template by hand.

## macOS `.pkg` / `.dmg`

- Instructions live in `installers/macos/README.md`.
- Build commands:
  ```bash
  cpack --config build/CPackConfig.cmake -G productbuild
  cpack --config build/CPackConfig.cmake -G DragNDrop
  ```
- Notarize/sign the `.pkg` and `.dmg`, then upload them to GitHub Releases (the release workflow already does this for tagged builds).
- MLX backend: rebuild with `-DENABLE_MLX=ON` before running `cpack` if you need the experimental MLX backend available to end users.

---

## Ubuntu / Debian (.deb)

- See `installers/deb/README.md` for the Launchpad/apt metadata template.
- Build commands:
  ```bash
  cpack --config build/CPackConfig.cmake -G DEB
  sudo apt install ./inferflux-*.deb
  ```
- The generated `control` file already depends on `libssl3` and `libyaml-cpp0.7`.

## Red Hat / Fedora (.rpm)

- See `installers/rpm/README.md` for the SPEC snippet used by Copr/OBS.
- Build commands:
  ```bash
  cpack --config build/CPackConfig.cmake -G RPM
  sudo rpm -i inferflux-*.rpm
  ```

---

## Windows (MSI + winget)

- `installers/windows/README.md` documents the WiX-based workflow.
- Build commands (WiX installed and on PATH):
  ```powershell
  cpack --config build/CPackConfig.cmake -G WIX
  ```
- The release pipeline renders `installers/winget/inferencial.inferflux.yaml` with the correct version, download URL, and SHA256. Submit that manifest to `microsoft/winget-pkgs` once the GitHub Release is live.

---

## Manual tarball (fallback)

If you need a raw tarball (for GitHub Releases or mirrors):

```bash
cmake --install build --prefix dist
tar -czf inferflux-$(git describe --tags --always).tar.gz -C dist .
```

---

## (Optional) Containers

Docker images remain available under `docker/` for environments that still prefer containers, but the official “Ease of Setup” score now hinges on the package-native flows above rather than Docker.

---

## CI-driven releases

See `docs/ReleaseProcess.md` for the GitHub Actions workflow that runs these packaging steps on every push to `main` (pre-release validation) and again when a tag matching `v*.*.*` is pushed (publishes a GitHub Release with every installer attached).
