# Contributing to InferFlux

Thanks for contributing.

## Scope

| Area | Expectation |
|---|---|
| Bug fixes | Include a focused reproduction or test when practical |
| User-visible changes | Update canonical docs in `README.md` or `docs/` |
| Runtime/backend changes | Prefer measured evidence over intuition-only tuning |
| Release-facing claims | Keep benchmark statements narrow and source-aligned |

## Development Loop

1. Build the repo.
2. Run the smallest relevant test set.
3. Update docs if behavior or claims changed.
4. Keep changes scoped and reviewable.

Useful commands:

```bash
./scripts/build.sh
ctest --test-dir build --output-on-failure --timeout 90
python3 scripts/check_docs_contract.py
```

## Coding Expectations

| Topic | Expectation |
|---|---|
| Language | C++17 |
| Style | `clang-format`, 2-space indentation, sorted includes |
| Naming | snake_case for files/functions, PascalCase for public types |
| Ownership | Prefer RAII and `std::unique_ptr` |
| Docs | Keep release-facing docs concise and code-aligned |

## Benchmark and Performance Changes

- Do not promote a microbenchmark win as a serving win without runtime evidence.
- Distinguish `llama_cpp_cuda` benchmark claims from `inferflux_cuda` progress notes.
- Keep local benchmark artifacts out of commits.

## Pull Request Checklist

- [ ] Build succeeds
- [ ] Relevant tests pass
- [ ] Canonical docs updated if behavior or public claims changed
- [ ] Local benchmark/profiling artifacts are not included
- [ ] Benchmark claims are tied to a reproducible harness or documented evidence

## Security

Please do not open public issues for suspected security vulnerabilities. See [SECURITY.md](SECURITY.md).
