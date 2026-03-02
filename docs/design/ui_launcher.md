# UI Launcher & Embedded WebUI Design Brief

> Goal: deliver an Ollama-class local experience without sacrificing InferFlux’s server-centric roadmap. Scope covers the desktop launcher (litehtml-based WebView), `/ui` SPA, installer quickstart, and CLI polish.

## Objectives

1. **Single-binary launcher** — ship `inferfluxd` with an optional embedded WebView (litehtml/Ultralight). Users double-click or run `inferfluxd --ui` and get a native window plus the REST API.
2. **Shared `/ui` SPA** — serve the same React/Svelte bundle over HTTP so remote operators can open `http://host:8080/ui` in a browser.
3. **Quickstart workflow** — `inferctl quickstart llama3` writes config/server.yaml, pulls the model, and launches the UI.
4. **Installers & CLI polish** — brew/winget/tarball packages with shell completions; CLI gains `chat`, `serve`, `profile` commands.

## Task Breakdown

### 1. WebUI + litehtml Shell
- [ ] Add `ENABLE_WEBUI` CMake option (default OFF). When ON:
  - Embed litehtml (or Ultralight) into `inferfluxd`.
  - Serve static assets from `resources/ui/` via `/ui` and load them in the WebView.
- [ ] Implement `--ui` flag that opens a native window hosting the WebView; expose reload/devtools shortcuts for development.
- [ ] SPA requirements:
  - Chat playground (model selection, streaming output).
  - Metrics dashboard (Prometheus snapshot, health status).
  - Settings page (config editor, model pull button).

### 2. Installer & Quickstart
- [ ] Brew formula / winget manifest / tarball with shell completions.
- [ ] `inferctl quickstart <model>`:
  - Detect GPU vs CPU profile.
  - Write config (port, auth, storage paths).
  - Pull model via `inferctl pull`.
  - Launch `inferfluxd --ui`.
- [ ] Document Docker one-liner (`docker run inferflux --ui`).

### 3. CLI Enhancements
- [ ] `inferctl serve --profile cpu-laptop` (reads preset config).
- [ ] `inferctl chat --model <id>` (simple REPL, uses same API as UI).
- [ ] `inferctl models list/pull/remove` parity with `/v1/models`.

### 4. Docs & Support
- [ ] Update Quickstart guide with screenshots of the new UI + installer steps.
- [ ] Add troubleshooting FAQ (port conflicts, missing GPU drivers, auth errors).
- [ ] Add telemetry docs (what the UI reports locally, opt-in/out).

## Open Questions
- Hosting strategy (single binary vs sidecar process?) – decision pending.
- Authentication UX: should the UI prompt for API key or auto-inject local token?
- Packaging assets: bundle SPA into the binary vs load from disk?

Keep this document updated as tasks land; cross-link from the Ease of Local Setup scorecard entry.
