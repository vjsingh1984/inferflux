# CUDA 12.1 Installation Guide for InferFlux

## Quick Installation Steps

### 1. Uninstall CUDA 13.2 (if present)
1. Open Windows Settings → Apps → Installed apps
2. Search for "CUDA"
3. Uninstall all CUDA 13.2 components:
   - NVIDIA CUDA Toolkit 13.2
   - NVIDIA CUDA Runtime 13.2
   - NVIDIA CUDA Development 13.2
   - Any other CUDA 13.2 components

### 2. Download CUDA 12.1
From the browser page you have open:
- **URL**: https://developer.nvidia.com/cuda-12-1-0-download-archive
- Select:
  - Operating System: Windows
  - Architecture: x86_64
  - Version: 10 or 11
  - Installer Type: exe (local)

### 3. Install CUDA 12.1
1. Run the downloaded installer
2. Choose "Express Installation" (recommended)
3. After installation, **reboot your computer** (critical for CUDA_PATH to be set)

### 4. Verify Installation
Open Command Prompt and check:
```cmd
echo %CUDA_PATH%
REM Should output: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

nvcc --version
REM Should show: Cuda compilation tools, release 12.1
```

### 5. Build InferFlux
Run the new build script:
```cmd
cd C:\Users\vjsin\code\inferflux
build_with_cuda_12.bat
```

## Expected Build Output

Successful build should show:
- CMake configuration succeeds without CUDA integration errors
- All targets compile successfully
- Binaries in `build-cuda-12\Release\`:
  - `inferfluxd.exe` (~50 MB)
  - `inferctl.exe` (~10 MB)
  - `inferflux_tests.exe` (~30 MB)

## Troubleshooting

### "CUDA 12.1.targets not found"
**Cause**: Visual Studio integration not installed
**Fix**: Re-run CUDA installer and choose "Custom Installation", ensure "Visual Studio Integration" is checked

### "CUDA_PATH not set"
**Cause**: Need to reboot after CUDA installation
**Fix**: Restart Windows, or set manually:
```cmd
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" /M
```

### "Cannot open include file 'cuda_fp16.h'"
**Cause**: CUDA not in PATH
**Fix**: Use the provided build script (sets PATH automatically)

## Performance Validation

After successful build, run the throughput benchmark:
```cmd
bash scripts/benchmark.sh throughput-gate
```

**Expected results with Phase 1+2 optimizations**:
- Baseline (old): 75.5 tok/s
- Target (new): 108+ tok/s (≥0.9x parity with llama.cpp)
- Improvement: 1.43x speedup

## Code Changes Applied

### Phase 1: Memory Access Optimization
- Added `__ldg()` cache hints to all weight loads in MMVQ kernels
- Replaced `memcpy()` with vectorized loads for activations
- **Expected impact**: 1.3-1.5x speedup

### Phase 2: Adaptive Thread Configuration
- Implemented llama.cpp-style adaptive thread policy
- Reduces threads from 128 to 64 for ncols=5-8
- **Expected impact**: 1.15-1.25x additional speedup

### Total Expected Improvement
- **Conservative**: 75.5 → 98 tok/s (1.3x, 0.82x parity)
- **Target**: 75.5 → 113 tok/s (1.5x, 0.94x parity)
- **Optimistic**: 75.5 → 119 tok/s (1.58x, 0.99x parity)

## Next Steps After Build

1. **Run unit tests** (no model required):
   ```cmd
   build-cuda-12\Release\inferflux_tests.exe "[native]"
   ```

2. **Run throughput benchmark** (requires model):
   ```cmd
   INFERFLUX_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf ^
   bash scripts/benchmark.sh throughput-gate
   ```

3. **Profile with NCU** (if needed):
   ```cmd
   ncu --target-processes all --set full ^
       -k "inferflux_mmvq" ^
       --launch-skip 200 --launch-count 20 ^
       -o profiling_results/ncu_mmvq_after_opt ^
       build-cuda-12\Release\inferfluxd.exe --config config/server.cuda.yaml
   ```

4. **If ≥0.9x achieved**: Phase 3 (kernel fusion) can be skipped

5. **If <0.9x after Phase 2**: Implement fused ResidualAdd+RmsNorm kernel
