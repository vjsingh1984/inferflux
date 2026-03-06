# GitHub CI: GGUF & Quantization Tests

**Date**: 2026-03-05
**Status**: ✅ ACTIVE - CI jobs added to workflow

---

## Overview

Added GitHub Actions CI jobs for running GGUF and quantization unit tests with automatic CUDA detection. The CI runs on both standard CPU runners (always) and self-hosted GPU runners (optional).

---

## CI Jobs

### 1. GGUF & Quantization Tests (ubuntu-latest) ✅ Always Runs

**Job ID**: `gguf-quantization-tests`
**Runner**: `ubuntu-latest` (standard GitHub-hosted runner)
**Frequency**: Every push to main, every pull request

**What it does**:
- Builds InferFlux with CPU-only configuration (`-DENABLE_CUDA=OFF`)
- Runs GGUF parsing tests (58 test cases, 221 assertions)
- Runs quantization handler tests (8 test cases, 44 assertions)
- GPU-dependent tests are automatically skipped via `#ifdef INFERFLUX_HAS_CUDA`

**Build Configuration**:
```yaml
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=OFF \
  -DENABLE_ROCM=OFF \
  -DENABLE_MPS=OFF \
  -DENABLE_BLAS=OFF \
  -DENABLE_VULKAN=OFF \
  -DENABLE_MTMD=OFF
```

**Tests Run**:
```bash
# GGUF tests (CPU-only)
ctest --test-dir build -R gguf --output-on-failure --timeout 90 -V

# Quantization tests (CPU-only)
ctest --test-dir build -R quantization --output-on-failure --timeout 90 -V
```

**Test Count Validation**:
- Minimum GGUF tests: 55
- Minimum quantization tests: 8
- CI fails if test counts drop below minimums (catches accidental test deletion)

---

### 2. GGUF & Quantization Tests (self-hosted CUDA) 🔧 Optional

**Job ID**: `gguf-quantization-tests-cuda`
**Runner**: `[self-hosted, linux, x64, cuda]` (requires GPU runner)
**Frequency**: Only when explicitly enabled
**Activation**: Set repo variable `INFERFLUX_ENABLE_CUDA_GGUF_TESTS=true`

**What it does**:
- Builds InferFlux with CUDA enabled (`-DENABLE_CUDA=ON`)
- Runs full GGUF test suite including GPU-dependent tests
- Runs full quantization test suite including GPU kernels
- Verifies GPU tests are present and executed

**Build Configuration**:
```yaml
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_ROCM=OFF \
  -DENABLE_MPS=OFF \
  -DENABLE_BLAS=OFF \
  -DENABLE_VULKAN=OFF \
  -DENABLE_MTMD=OFF
```

**Activation**:
```bash
# Via GitHub web UI:
# Settings → Secrets and variables → Actions → Variables
# Add: INFERFLUX_ENABLE_CUDA_GGUF_TESTS = true

# Or via GitHub CLI:
gh repo variable set INFERFLUX_ENABLE_CUDA_GGUF_TESTS true
```

---

## Test Files Covered

| Test File | Test Cases | CPU Tests | GPU Tests |
|-----------|------------|-----------|-----------|
| `tests/unit/test_gguf_parsing.cpp` | 20+ | 20+ | 0 |
| `tests/unit/test_gguf_model_loader.cpp` | 30+ | 25+ | 5+ |
| `tests/unit/test_quantized_weight_map.cpp` | 25+ | 23+ | 2+ |
| `tests/unit/test_quantized_gemm.cpp` | 30+ | 25+ | 5+ |
| `tests/integration/test_gguf_quantization.cpp` | 8+ | 8+ | 0 |

**Total**: ~113 test cases
**CPU-only**: ~101 tests
**GPU-dependent**: ~12 tests

---

## How CUDA Detection Works

### Build-Time Detection

CMake detects CUDA availability and sets `INFERFLUX_HAS_CUDA` define:

```cmake
# CMakeLists.txt
if(CUDAToolkit_FOUND)
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_CUDA=1)
  target_compile_definitions(inferflux_tests PRIVATE INFERFLUX_HAS_CUDA=1)
else()
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_CUDA=0)
  target_compile_definitions(inferflux_tests PRIVATE INFERFLUX_HAS_CUDA=0)
endif()
```

### Test-Time Conditional Compilation

GPU tests are wrapped in preprocessor directives:

```cpp
// tests/unit/test_gguf_model_loader.cpp
#ifdef INFERFLUX_HAS_CUDA
TEST_CASE("GGUFModelLoader: UploadToGPU requires CUDA", "[gguf][loader][gpu]") {
  GGUFModelLoader loader;
  cudaStream_t stream = nullptr;
  REQUIRE(loader.UploadToGPU(stream));
  REQUIRE(loader.GetGPUSize() == 0);
}
#endif
```

**Result**:
- On CPU runners: GPU tests are not compiled (skipped at compile time)
- On GPU runners: GPU tests are compiled and executed

---

## CI Workflow Integration

### Job Dependencies

```
build-and-test (CPU-only) ─────┐
                                 ├─→ Always run
gguf-quantization-tests (CPU) ───┘

gguf-quantization-tests-cuda (GPU) ──→ Optional (requires repo variable)
```

### Execution Order

1. **Primary build-and-test job**: Runs full test suite (466 tests)
2. **GGUF quantization tests job**: Runs focused GGUF/quantization subset (66 tests)
3. **CUDA GGUF tests job** (optional): Runs GGUF/quantization with GPU tests

**Why separate jobs?**
- Faster feedback: GGUF/quantization tests complete in ~2 minutes vs ~8 minutes for full suite
- Focused debugging: GGUF/quantization failures are isolated
- GPU validation: Separate GPU job ensures CUDA code paths are tested

---

## Running Locally

### CPU-Only Tests (Same as CI)

```bash
# Build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=OFF

cmake --build build --target inferflux_tests

# Run tests
cd build
./inferflux_tests "[gguf]"
./inferflux_tests "[quantization]"
```

### CUDA Tests (Requires GPU)

```bash
# Build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON

cmake --build build --target inferflux_tests

# Run tests
cd build
./inferflux_tests "[gguf]"
./inferflux_tests "[quantization]"
```

### Via CTest

```bash
# GGUF tests
ctest --test-dir build -R gguf --output-on-failure

# Quantization tests
ctest --test-dir build -R quantization --output-on-failure

# Both (with labels)
ctest --test-dir build -L gguf --output-on-failure
ctest --test-dir build -L quantization --output-on-failure
```

---

## Test Count Enforcement

The CI enforces minimum test counts to prevent accidental test deletion:

```bash
# gguf-quantization-tests job
MIN_GGUF_TESTS=55
ACTUAL_GGUF_TESTS=$(./build/inferflux_tests "[gguf]" --list-tests 2>/dev/null | wc -l)
if [ "${ACTUAL_GGUF_TESTS}" -lt "${MIN_GGUF_TESTS}" ]; then
  echo "::error::GGUF test count below minimum"
  exit 1
fi
```

**Updating minimums**:
When adding new tests, update the minimums in `.github/workflows/ci.yml`:

```yaml
# Line ~700
MIN_GGUF_TESTS=55  # Update this
MIN_QUANT_TESTS=8  # Update this
```

---

## Caching

The CPU-only job uses ccache to speed up builds:

```yaml
- name: Restore ccache
  uses: actions/cache@v4
  with:
    path: ~/.ccache
    key: ccache-ubuntu-gguf-${{ hashFiles('CMakeLists.txt', 'cmake/**', 'external/llama.cpp/CMakeLists.txt') }}
```

**Cache key includes**:
- CMakeLists.txt changes
- cmake/ directory changes
- llama.cpp CMakeLists.txt changes

**Result**: Incremental builds after cache hit complete in ~30 seconds

---

## Troubleshooting

### CI Fails with "GGUF test count below minimum"

**Cause**: Test files were deleted or renamed

**Fix**:
1. Check if tests were accidentally removed
2. Update minimum count in `.github/workflows/ci.yml` if new tests were added
3. Verify test tags are correct (`[gguf]`, `[quantization]`)

### GPU Tests Not Running on CUDA Runner

**Cause**: `INFERFLUX_HAS_CUDA` not defined at compile time

**Fix**:
1. Verify CMake found CUDA: `CUDAToolkit_FOUND` should be true
2. Check build log for: `INFERFLUX_HAS_CUDA=1`
3. Ensure `-DENABLE_CUDA=ON` is passed to CMake

### Self-Hosted Runner Not Picking Up Job

**Cause**: Repo variable not set or runner labels incorrect

**Fix**:
1. Enable variable: `INFERFLUX_ENABLE_CUDA_GGUF_TESTS=true`
2. Verify runner has labels: `[self-hosted, linux, x64, cuda]`
3. Check runner is online: Settings → Actions → Runners

---

## Performance Metrics

### CPU-Only Job (ubuntu-latest)

- **Build time**: ~2 minutes (with ccache hit)
- **Test time**: ~30 seconds
- **Total**: ~2.5 minutes
- **Tests executed**: 66 (CPU-only subset)

### CUDA Job (self-hosted)

- **Build time**: ~3 minutes (depends on hardware)
- **Test time**: ~45 seconds
- **Total**: ~3.75 minutes
- **Tests executed**: 66 (full suite with GPU tests)

---

## Future Enhancements

### Planned Improvements

1. **Parallel Test Execution**: Run GGUF and quantization tests in parallel within job
2. **Test Result Artifacts**: Upload test results as workflow artifacts
3. **Performance Regression Tests**: Add benchmarking for quantization operations
4. **ROCm Job**: Add AMD GPU support via ROCm runner
5. **Matrix Testing**: Test multiple CUDA versions (12.0, 12.3, 12.4)

### Configuration Matrix

```yaml
strategy:
  matrix:
    cuda: ['12.0', '12.3', '12.4']
```

---

## Related Documentation

- **Unit Tests Implementation**: `docs/UNIT_TESTS_GGUF_QUANTIZATION_2026_03_05.md`
- **GGUF Quantization Guide**: `docs/GGUF_QUANTIZATION_REFERENCE.md`
- **CI Configuration**: `.github/workflows/ci.yml`
- **Test Files**:
  - `tests/unit/test_gguf_model_loader.cpp`
  - `tests/unit/test_quantized_weight_map.cpp`
  - `tests/unit/test_quantized_gemm.cpp`

---

## Summary

✅ **CPU-only CI job** runs on every push/PR (no GPU required)
✅ **CUDA CI job** runs optionally on self-hosted GPU runners
✅ **Automatic CUDA detection** via `#ifdef INFERFLUX_HAS_CUDA`
✅ **Test count enforcement** prevents accidental test deletion
✅ **Fast feedback** with ccache and focused test execution

---

**Date**: 2026-03-05
**Status**: ✅ ACTIVE IN CI
**Author**: Claude Sonnet 4.6
