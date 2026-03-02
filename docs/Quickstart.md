# InferFlux Quickstart

This guide shows how to bootstrap a local InferFlux install with the new CLI workflow and embedded WebUI.

## 1. Install (build from source for now)

```bash
git clone https://github.com/inferencial/InferFlux.git
cd InferFlux
cmake -S . -B build -DENABLE_WEBUI=ON
cmake --build build -j$(nproc)
```

> Packaged installers (Homebrew/winget/tarball) are coming soon; for now build from source.

## 2. Bootstrap config and model

```bash
# Scaffold ~/.inferflux/config.yaml + models directory (choose backend)
./build/inferctl quickstart meta-llama/Meta-Llama-3-8B-Instruct --profile cpu-laptop --backend cpu
# For Apple MLX test builds (after rebuilding with -DENABLE_MLX=ON):
# ./build/inferctl quickstart meta-llama/Meta-Llama-3-8B-Instruct --backend mlx

# Download the requested model (HuggingFace GGUF)
./build/inferctl pull meta-llama/Meta-Llama-3-8B-Instruct
```

The quickstart command resolves the best GGUF artifact, writes the config to point at `~/.inferflux/models/<repo>/<file.gguf>`, and seeds the default API key (`dev-key-123`) with `generate/read/admin` scopes so the WebUI’s model-management buttons work immediately. `inferctl pull` then downloads the same artifact into that directory (safe to rerun; it prints the full path on success).

## 3. Launch the server with WebUI

```bash
# Automatically runs `inferfluxd --config ... --ui`
./build/inferctl serve --config ~/.inferflux/config.yaml

# Apple Silicon: force CPU mode if Metal init fails
GGML_METAL_DISABLE=1 ./build/inferctl serve --config ~/.inferflux/config.yaml
```

Navigate to `http://localhost:8080/ui` for the litehtml-based WebUI (chat playground, model list, output view).

## 4. CLI usage

- `./build/inferctl completion --prompt "Hello" --model llama3`
- `./build/inferctl chat --message "user:Who are you?" --interactive`
- `./build/inferctl status`
- `./build/inferctl admin models --list --api-key dev-key-123`
- `./build/inferctl admin models --load ~/.inferflux/models/<repo>/<file.gguf> --id local --default --api-key dev-key-123` (verifies the admin flow the WebUI uses)

## 5. Docker

- Build: `docker build -t inferflux:latest -f docker/Dockerfile .`
- Run: `docker run --rm -p 8080:8080 inferflux:latest --ui`
- Compose: `docker compose -f docker/docker-compose.yaml up`
- Publish: `PUSH=1 DOCKER_REPO=inferencial/inferflux scripts/docker/publish.sh`

## Troubleshooting

See `docs/Troubleshooting.md` for WebUI, CLI, and Docker tips.
