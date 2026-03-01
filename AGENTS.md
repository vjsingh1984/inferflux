# Repository Guidelines

## Project Structure & Module Organization
InferFlux production code spans `runtime/` (device backends, paged KV, speculative decoding), `server/` (HTTP/auth/metrics), plus `scheduler/`, `policy/`, and `model/`, while `cli/` hosts the `inferctl` client. Deployment tooling sits in `docker/`, `charts/`, and `scripts/`; configs live under `config/`, docs under `docs/`, and tests in `tests/unit` with scenario scaffolds in `tests/integration`. Treat `external/` as pinned vendor drops managed through CMake.

## Build, Test, and Development Commands
- `./scripts/build.sh` configures CMake into `build/` and compiles `inferfluxd` plus `inferctl` with the CUDA/ROCm/MPS toggles declared at the top of `CMakeLists.txt`.
- `cmake -S . -B build && cmake --build build -j` is the fastest incremental loop when iterating on a single component.
- `./scripts/run_dev.sh --config config/server.yaml` starts the dev server with sample API keys and guardrail knobs.
- `./build/inferctl chat --message 'user:Hello' --api-key dev-key-123 --stream` (or `... completion`) verifies the OpenAI-style endpoints once `INFERFLUX_MODEL_PATH` points at a GGUF.

## Coding Style & Naming Conventions
We target C++17, keep headers beside their `.cpp` implementations, and rely on RAII plus `std::unique_ptr` for ownership. Run `clang-format` (2-space indent, sorted includes) on touched files. File names and free functions use snake_case, public types adopt PascalCase (`SpeculativeDecoder`), constants start with `k` (`kLRU`), and member fields end in `_`. Keep helpers inside anonymous namespaces and ensure everything lives in the `inferflux` namespace.

## Testing Guidelines
Catch2-based unit tests (see `tests/unit/test_tokenizer.cpp`) run through `ctest --test-dir build --output-on-failure`. Integration smoke tests for SSE, guardrails, and rate limiting run via `ctest -R IntegrationSSE --output-on-failure` and require `INFERFLUX_MODEL_PATH` plus `INFERCTL_API_KEY`. Name `TEST_CASE`s after observable behaviors, keep fixtures deterministic in `tests/data/`, and attach CLI or HTTP transcripts whenever you touch user-visible flows.

## Commit & Pull Request Guidelines
History favors short imperative subjects such as `Add Metal/MPS and BLAS acceleration toggles`; keep the first line under ~72 characters, mention the subsystem, and explain the rationale in the body. Each PR should link a tracking issue, summarize config/env changes, attach `ctest` (or equivalent manual) output, and update `README.md`, `docs/`, or deployment assets when knobs move. Include screenshots or curl transcripts for API changes.

## Security & Configuration Tips
Never commit real API keys or passphrases into `config/`; rely on env vars such as `INFERFLUX_POLICY_PASSPHRASE`, `INFERCTL_API_KEY`, and `INFERFLUX_RATE_LIMIT_PER_MINUTE`. When modifying guardrail, auth, or audit paths (`policy/`, `server/auth/`, `server/logging/`), confirm `logs/audit.log` remains writable, document RBAC impacts, and describe rollback steps in the PR.
