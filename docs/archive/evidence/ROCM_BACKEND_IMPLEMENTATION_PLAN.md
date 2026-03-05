# ROCm Backend Implementation Plan

## 🎯 Executive Summary

**Goal:** Build native ROCm backend for InferFlux to support AMD GPUs (MI200, MI250X, MI300X) with FlashAttention optimization.

**Status:** ✅ FEASIBLE - llama.cpp already has HIP/ROCm + FlashAttention support!

**Key Insight:** llama.cpp's `ggml-hip` backend reuses CUDA FlashAttention kernels, so FA2 works on AMD GPUs too!

---

## 📊 Current State Analysis

### ✅ What Already Exists

#### **llama.cpp ROCm Support**
```cpp
// external/llama.cpp/ggml/src/ggml-hip/CMakeLists.txt
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

// Reuses CUDA FlashAttention kernels!
file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
file(GLOB SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
file(GLOB SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")

// FlashAttention enabled by default
if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()
```

#### **InferFlux Infrastructure**
- ✅ `ENABLE_ROCM` option in CMakeLists.txt
- ✅ Backend factory with priority chain
- ✅ LlamaBackendTraits for different targets
- ✅ Plugin interface for backends
- ✅ Metrics infrastructure

### ❌ What's Missing

#### **InferFlux ROCm Backend Implementation**
- ❌ `runtime/backends/rocm/rocm_backend.cpp`
- ❌ `runtime/backends/rocm/rocm_backend.h`
- ❌ `runtime/backends/rocm/rocm_device_context.cpp`
- ❌ Backend factory ROCm target support
- ❌ ROCm-specific FlashAttention metrics
- ❌ HIP device context implementation

---

## 🏗️ Implementation Plan

### Phase 1: Core ROCm Backend (Week 1-2)

#### **Step 1: Create ROCm Backend Structure**

```cpp
// runtime/backends/rocm/rocm_backend.h
#pragma once
#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/device_context.h"
#include <memory>

namespace inferflux {

class RocmBackend : public LlamaCPUBackend {
public:
  RocmBackend();
  ~RocmBackend() override;

  bool LoadModel(const std::filesystem::path& model_path,
                 const LlamaBackendConfig& config) override;

  std::string GetBackendType() const override { return "rocm"; }
  bool IsReady() const override;

  // FlashAttention support for ROCm
  bool SupportsFlashAttention() const;
  std::string GetFlashAttentionVersion() const;

private:
  bool hip_initialized_{false};
  std::string selected_attention_kernel_{"standard"};
  int hip_device_{0};
};

} // namespace inferflux
```

#### **Step 2: Implement HIP Device Context**

```cpp
// runtime/backends/rocm/rocm_device_context.h
#pragma once
#include "runtime/device_context.h"
#include <memory>

namespace inferflux {

class RocmDeviceContext : public DeviceContext {
public:
  RocmDeviceContext(int device_id = 0);
  ~RocmDeviceContext() override;

  void* Allocate(size_t size, size_t alignment = 0) override;
  void Free(void* ptr) override;
  void CopyToDevice(void* dst, const void* src, size_t size) override;
  void CopyFromDevice(void* dst, const void* src, size_t size) override;

  bool IsAvailable() const override;
  std::string GetDeviceName() const override;
  size_t GetTotalMemory() const override;
  size_t GetFreeMemory() const override;

private:
  int device_id_;
  std::string device_name_;
  size_t total_memory_;
};

} // namespace inferflux
```

#### **Step 3: Update Backend Factory**

```cpp
// runtime/backends/backend_factory.cpp
#ifdef INFERFLUX_HAS_ROCM
#include "runtime/backends/rocm/rocm_backend.h"

BackendFactoryResult CreateRocmBackend(...) {
  // Check if ROCm is available
  if (!hipDeviceGetCount(&device_count)) {
    // Try to load ROCm
    return CreateRocmBackendInternal();
  }
  return CpuFallback("ROCm not available");
}
#endif
```

---

### Phase 2: FlashAttention for ROCm (Week 3-4)

#### **Step 4: Detect ROCm FlashAttention Support**

```cpp
// runtime/backends/rocm/rocm_backend.cpp
bool RocmBackend::LoadModel(...) {
#ifdef INFERFLUX_HAS_ROCM
  // Query HIP device properties
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, hip_device_);

  // Check if FlashAttention is supported
  // AMD GPUs support FA2-style optimization
  bool supports_fa = (prop.major >= 9);  // GFX9+ (MI200, MI250X, MI300X)

  if (supports_fa && config.use_flash_attention) {
    selected_attention_kernel_ = "fa2";  // ROCm uses FA2 algorithms
    GlobalMetrics().SetCudaAttentionKernel("fa2");
    GlobalMetrics().RecordFlashAttentionRequest("fa2");
  }
#endif
}
```

#### **Step 5: ROCm-Specific Metrics**

```cpp
// server/metrics/metrics.h
// Add ROCm-specific metrics

// ROCm backend metrics
void RecordRocmKernelSelection(const std::string& kernel);
void RecordRocmFlashAttentionExecution(double duration_ms, int tokens);
void SetRocmMemoryUsageMB(double memory_mb);
void RecordRocmDeviceProperties(int device_id, const std::string& arch);
```

---

### Phase 3: Build System Integration (Week 5-6)

#### **Step 6: Update CMakeLists.txt**

```cmake
# CMakeLists.txt
option(ENABLE_ROCM "Enable ROCm runtime" ON)

if(ENABLE_ROCM)
  find_package(hip 6.1 QUIET)
  if(hip_FOUND)
    message(STATUS "ROCm/HIP found: enabling ROCm build path")
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_ROCM=1)
    target_link_libraries(inferflux_core PUBLIC hip::host roc::rocblas roc::hipblas)

    # Add ROCm backend sources
    target_sources(inferflux_core PRIVATE
      runtime/backends/rocm/rocm_backend.cpp
      runtime/backends/rocm/rocm_backend.h
      runtime/backends/rocm/rocm_device_context.cpp
      runtime/backends/rocm/rocm_device_context.h
    )
  else()
    message(WARNING "ENABLE_ROCM=ON but ROCm toolkit not found")
    set(ENABLE_ROCM OFF)
  endif()
endif()
```

#### **Step 7: Update LlamaBackendTraits**

```cpp
// runtime/backends/llama/llama_backend_traits.cpp
#ifdef INFERFLUX_HAS_ROCM
LlamaBackendTarget GetBackendTarget(const BackendConfig& config) {
  if (config.backend == "rocm") {
    return LlamaBackendTarget::kRocm;
  }
  // ... existing logic
}

bool SupportsFlashAttention(LlamaBackendTarget target) {
  switch (target) {
    case LlamaBackendTarget::kCuda:
      return true;  // FA2/FA3
    case LlamaBackendTarget::kRocm:
      return true;  // FA2 (via HIP)
    default:
      return false;
  }
}
```

---

### Phase 4: Testing & Validation (Week 7-8)

#### **Step 8: Create ROCm Test Suite**

```cpp
// tests/unit/test_rocm_backend.cpp
TEST_CASE("ROCm backend detection", "[rocm]") {
  #ifdef INFERFLUX_HAS_ROCM
  REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
  REQUIRE(device_count > 0);
  #endif
}

TEST_CASE("ROCm FlashAttention support", "[rocm][flash]") {
  // Test that ROCm backend reports FA support
  RocmBackend backend;
  backend.LoadModel(model_path, config);

  REQUIRE(backend.SupportsFlashAttention());
  REQUIRE(backend.GetFlashAttentionVersion() == "fa2");
}
```

#### **Step 9: Integration Test**

```bash
# scripts/test_rocm_flash_attention.sh
#!/bin/bash

# 1. Build with ROCm
cmake -DENABLE_ROCM=ON -B build
cmake --build build -j

# 2. Test on AMD GPU (requires AMD hardware)
./build/inferfluxd --config config/server.rocm.yaml

# 3. Make request
curl -X POST http://localhost:8080/v1/completions \
  -H "Authorization: Bearer dev-key-123" \
  -d '{"prompt": "Test", "max_tokens": 50, "model": "tinyllama"}'

# 4. Check metrics
curl -s http://localhost:8080/metrics \
  -H "Authorization: Bearer dev-key-123" | grep flash_attention
```

---

## 🔧 Development Environment Setup

### Option 1: Local AMD GPU (Best for Development)
```bash
# Install ROCm on Ubuntu
wget https://repo.radeon.com/amdgpu-install/ubuntu/22.04/amdgpu-install_5.4.50400-1_all.deb
sudo dpkg -i amdgpu-install_5.4.50400-1_all.deb

# Install HIP, rocBLAS, hipBLAS
amdgpu-install --usecase=rocm,hip --no-dkms

# Set environment
export ROCm_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Build InferFlux with ROCm
cmake -DENABLE_ROCM=ON -S . -B build
cmake --build build --target inferflux_core -j
```

### Option 2: Cloud AMD GPU (Vast.ai)
```bash
# Rent AMD GPU on Vast.ai
# Search for: MI300X, MI250X, or RX 7900 XTX
# Filter: ROCm pre-installed

# Connect to instance
ssh user@vast.ai-instance

# Clone and build
git clone https://github.com/your-repo/inferflux.git
cd inferflux
cmake -DENABLE_ROCM=ON -B build
cmake --build build -j

# Run tests
./build/inferfluxd --config config/server.rocm.yaml
```

---

## 📊 Expected Results

### AMD GPU Support Matrix

| GPU | Architecture | FA2 Support | Expected Performance |
|-----|--------------|-------------|---------------------|
| MI200 | GFX90A | ✅ Yes | 200-400 tok/s (est.) |
| MI250X | GFX90A | ✅ Yes | 300-500 tok/s (est.) |
| MI300X | GFX942 | ✅ Yes | 400-600 tok/s (est.) |
| RX 7900 XTX | GFX1100 | ⚠️ Maybe | 100-200 tok/s (est.) |

### ROCm vs CUDA Performance

| Feature | CUDA (H100) | ROCm (MI300X) | Ratio |
|---------|-------------|----------------|-------|
| FlashAttention-2 | ✅ Native | ✅ Native | ~0.9x |
| FlashAttention-3 | ✅ Native | ❌ N/A (Hopper-only) | - |
| Memory Bandwidth | 2 TB/s | 1.5 TB/s | ~0.75x |
| Compute (FP16) | 60 TFLOPS | 80 TFLOPS (MHA) | ~1.3x |

**Key Insight:** MI300X can be competitive with H100 for certain workloads due to higher compute density!

---

## 🎯 Success Criteria

### Phase 1: Core Backend
- [ ] RocmBackend class implemented
- [ ] HipDeviceContext functional
- [ ] Backend factory supports ROCm
- [ ] Model loading works on AMD GPU

### Phase 2: FlashAttention
- [ ] FA2 support detected and enabled
- [ ] Metrics track ROCm FA usage
- [ ] Performance benchmarks pass

### Phase 3: Integration
- [ ] CMake build system updated
- [ ] LlamaBackendTraits supports ROCm
- [ ] Config files updated
- [ ] Tests passing

### Phase 4: Validation
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Known limitations documented

---

## 📝 Implementation Order

### Week 1-2: Core Infrastructure
1. Create RocmBackend class
2. Implement HipDeviceContext
3. Update backend factory
4. Basic model loading

### Week 3-4: FlashAttention
5. Detect FA support
6. Implement ROCm metrics
7. Test FA2 on AMD hardware

### Week 5-6: Build System
8. Update CMakeLists.txt
9. Configure llama.cpp for ROCm
10. Integration testing

### Week 7-8: Testing & Docs
11. Unit tests
12. Integration tests
13. Documentation
14. Benchmarking

---

## 🚀 Quick Start (Once Complete)

```bash
# Build with ROCm support
cmake -DENABLE_ROCM=ON -B build
cmake --build build -j

# Run on AMD GPU
./build/inferfluxd --config config/server.rocm.yaml

# Test FlashAttention
curl -X POST http://localhost:8080/v1/completions \
  -H "Authorization: Bearer dev-key-123" \
  -d '{"prompt": "Hello AMD!", "max_tokens": 50, "model": "tinyllama"}'

# Check metrics
curl -s http://localhost:8080/metrics \
  -H "Authorization: Bearer dev-key-123" | grep flash_attention
```

---

## 💡 Key Insights

### 1. FlashAttention Works on AMD GPUs!
- llama.cpp reuses CUDA FA2 kernels for HIP
- AMD MI200/MI250X/MI300X support FA2
- Performance should be competitive

### 2. ROCm is More Than Just "AMD CUDA"
- Different architecture (GFX9 vs Ampere/Ada)
- Different optimization strategies
- Need AMD-specific tuning

### 3. Backend Parity is Achievable
- Same llama.cpp foundation
- Same FlashAttention algorithms
- Similar metrics infrastructure

### 4. Testing Requires AMD Hardware
- No easy emulation/translation
- Vast.ai is the most cost-effective option
- MI300X recommended for best performance

---

## 📚 Resources

### Documentation
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/HIP_Language_Usage_Guide.html)
- [llama.cpp ROCm Support](https://github.com/ggerganov/llama.cpp)

### Hardware Providers
- [Vast.ai AMD GPUs](https://vast.ai/search?gpu=MI300X)
- [Lambda Labs](https://lambdalabs.com/)
- [AWS g5ad instances](https://aws.amazon.com/ec2/instance-types/g5/)

### FlashAttention on AMD
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [AMD MI300X Architecture](https://www.amd.com/en/products/accelerators/instinct-mi300/)
- [ROCm Support Matrix](https://rocm.docs.amd.com/

---

**Status:** ✅ READY TO IMPLEMENT
**Estimated Timeline:** 6-8 weeks
**Blocking Issue:** Need AMD GPU access for testing
**Recommendation:** Use Vast.ai for cost-effective AMD GPU rental
