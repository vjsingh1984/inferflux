# [P0-4] GPU KV Page Allocator + Prefix Reuse Bridge

Priority: P0
Owner: Runtime
Effort: 2.5 eng-weeks
Risk: High
Dependencies: P0-3
Labels: runtime, kv-cache, prefix-cache, cuda

## Problem
GPU KV page reuse is incomplete, and native prefix copy still includes a no-op path, limiting reuse and efficiency.

## Scope Files
- `runtime/backends/cuda/native/kv_cache_gpu.cu`
- `runtime/backends/cuda/native/kv_cache_gpu.h`
- `runtime/backends/cuda/native_kernel_executor.cpp`
- `runtime/backends/cuda/native_kernel_executor.h`
- `runtime/kv_cache/paged_kv_cache.cpp`
- `runtime/kv_cache/paged_kv_cache.h`
- `runtime/prefix_cache/radix_prefix_cache.cpp`
- `tests/unit/test_gpu_kv_cache.cpp`
- `tests/unit/test_paged_kv_cache.cpp`
- `tests/unit/test_radix_prefix_cache.cpp`

## Test Plan
1. Add unit tests for GPU page alloc/free/refcount lifecycle.
2. Add unit tests for prefix donation/reuse across requests with block-accounting assertions.
3. Add regression test for duplicate block-ID prevention in reuse paths.
4. Run GPU KV, paged KV, and radix prefix cache suites.

## Acceptance Checklist
- [ ] `NativeCopySequencePrefix` is fully implemented (no no-op path for supported flow).
- [ ] KV page accounting remains balanced under stress.
- [ ] Prefix cache reuse does not duplicate block IDs.
- [ ] Reuse metrics are exported and validated in tests.
- [ ] Eviction paths remain safe and deterministic.
