# Troubleshooting

Common issues when running InferFlux locally.

## WebUI issues

- **Blank page:** the binary must be built with `-DENABLE_WEBUI=ON` and started via `inferctl serve` (which passes `--ui`). Rebuild and relaunch.
- **Stale API key:** click “Save” after updating the API key field, or clear localStorage in the browser.
- **Model list empty:** ensure at least one model is loaded (`inferctl admin models --list`). The WebUI calls `/v1/models` using the configured API key.

## CLI quickstart/serve

- **Config not found:** `inferctl serve` defaults to `~/.inferflux/config.yaml`. Override via `--config` if you generated multiple profiles.
- **API key errors:** set `INFERCTL_API_KEY=<key>` or pass `--api-key` to each command (guardrails/models admin require it).
- **Model not found at launch:** `inferctl pull <repo>` downloads to `~/.inferflux/models/<owner>/<file.gguf>` — ensure that file exists (the command prints the resolved path) and matches the path inside `~/.inferflux/config.yaml`.
- **Metal command-queue failure on macOS:** export `GGML_METAL_DISABLE=1` before running `inferctl serve` to force pure CPU mode. The quickstart config already pins the server to `127.0.0.1`, so CLI/WebUI still work once the CPU backend loads.
- **Backend mismatch:** If you scaffolded `backend: mlx` (or `cuda`/`rocm`) but the binary was compiled without the corresponding `-DENABLE_*` flag, InferFlux falls back to CPU and logs a warning. Rebuild with `cmake -DENABLE_MLX=ON` (or the relevant flag) before loading the model.

## Docker deployment

- **Build:** `docker build -t inferflux:latest -f docker/Dockerfile .`
- **Run (with UI):** `docker run --rm -p 8080:8080 inferflux:latest --ui`
- **Compose:** `docker compose -f docker/docker-compose.yaml up`
- **Volume permissions:** map `~/.inferflux` into the container via `-v ~/.inferflux:/app/config` for persistent models/config.

## Installer targets (coming soon)

- **Homebrew tap:** `brew tap inferencial/cli` then `brew install inferflux` once the tap is published.
- **winget:** manifests under `installers/winget` describe the package layout; submit after tagging a release.
