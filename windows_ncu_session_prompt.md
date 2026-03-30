# Windows Native CUDA Profiling Handoff

Branch: `distributed-runtime-bminus-foundation`

Goal: continue native CUDA throughput work from an elevated Windows shell and
use Windows `ncu` to profile the real fused FFN hotspot on a Windows CUDA
binary.

Current state:
- Harness fixes are already in tree:
  - `scripts/benchmark_request_driver.py`
  - `scripts/benchmark_multi_backend_comparison.sh`
  - `scripts/profile_backend.sh`
  - `scripts/profile_backend_ncu.sh`
  - `tests/integration/benchmark_request_driver_test.py`
- P5 gate/up epilogue correctness fixes are in tree:
  - `runtime/backends/cuda/native/llama_forward.h`
  - `runtime/backends/cuda/native/transformer_forward.cu`
  - `tests/unit/test_fused_kernels.cpp`
- New microbenchmark is in tree:
  - `tests/unit/benchmark_fused_gate_up_q81.cu`
  - `CMakeLists.txt` registers target `benchmark_fused_gate_up_q81`

What the new benchmark showed on WSL2:
- `M=1`: fused gate/up was about `1.27x` faster than live grouped pair + `SiluMul`
- `M=2`: about `1.18x`
- `M=4`: about `1.06x`
- `M=8`: about `1.03x`
- The Q8 epilogue variant was roughly breakeven or slightly worse

Interpretation:
- The fused Q4_K gate/up kernel is a real win for singleton/small decode.
- At `M=4/8`, which matters for concurrent serving, the gain is only low single digits.
- The remaining gap to `llama.cpp` will not be closed by “having fusion” alone.
- Next likely targets after Windows `ncu`:
  1. inspect `inferflux_mmvq_q4k_fused_gate_up_silu<4>` with native Windows `ncu`
  2. profile the disabled Q6_K vectorized MMVQ path
  3. compare against the live grouped `q8_1_group_mmq3` FFN path

Recommended Windows steps:

1. Open an elevated PowerShell in `C:\Users\vjsin\code\inferflux`

2. Regenerate the Windows build tree so the new benchmark target exists:

```powershell
& 'C:\Program Files\CMake\bin\cmake.exe' `
  -S C:\Users\vjsin\code\inferflux `
  -B C:\Users\vjsin\code\inferflux\build-cuda-opt `
  -G "Visual Studio 17 2022" `
  -A x64
```

3. Build the new benchmark in `Release`:

```powershell
& 'C:\Program Files\CMake\bin\cmake.exe' `
  --build C:\Users\vjsin\code\inferflux\build-cuda-opt `
  --config Release `
  --target benchmark_fused_gate_up_q81
```

4. Sanity-check the benchmark:

```powershell
.\build-cuda-opt\Release\benchmark_fused_gate_up_q81.exe 4
```

5. Profile the fused kernel directly with Windows `ncu`:

```powershell
& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ncu.exe' `
  --set basic `
  --kernel-name "regex:inferflux_mmvq_q4k_fused_gate_up_silu.*" `
  --launch-count 1 `
  --target-processes all `
  .\build-cuda-opt\Release\benchmark_fused_gate_up_q81.exe 4
```

6. If Windows `ncu` works, inspect:
- achieved occupancy
- register pressure
- SM vs memory utilization
- whether the fused kernel is compute-bound or launch/latency bound

7. If the fused kernel still looks healthy and only offers ~5-6% at `M=4`,
shift the next engineering pass to:
- `runtime/backends/cuda/native/fused_quant_gemm.cu`
- `runtime/backends/cuda/native/kernels/mmvq.cuh`
- especially the Q6_K paths and the live grouped FFN path used at `M=4/8`
