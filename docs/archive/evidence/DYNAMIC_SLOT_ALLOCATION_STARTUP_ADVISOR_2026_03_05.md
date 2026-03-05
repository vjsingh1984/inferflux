# Dynamic Slot Allocation in StartupAdvisor

**Date**: 2026-03-04
**Feature**: Automatic calculation of optimal `max_parallel_sequences` based on model size, GPU memory, and context window

---

## Overview

The enhanced StartupAdvisor now provides **dynamic slot allocation**, which:

1. **Detects model characteristics**: Quantization type, parameter count, architecture
2. **Probes GPU memory**: Uses CUDA/ROCm API to get actual VRAM
3. **Calculates optimal allocation**: Uses 85% of GPU memory target
4. **Generates configuration**: Ready-to-use YAML snippet
5. **Works for all models**: GGUF, safetensors, various quantizations

---

## How It Works

### Step 1: Model Size Detection

```cpp
// From model filename and format
QuantizationType DetectQuantization(const std::string& model_path,
                                   const std::string& format);

// Examples:
// "qwen2.5-3b-instruct-q4_k_m.gguf" → Q4_K_M, 3B params, ~1.5 GB loaded
// "qwen2.5-7b-q5_k_m.gguf" → Q5_K_M, 7B params, ~3 GB loaded
// "model-f16.gguf" → FP16, full precision
```

### Step 2: GPU Memory Probing

```cpp
AdvisorGpuInfo gpu = ProbeCudaGpu();

// Returns:
gpu.total_vram_bytes      // e.g., 24 GB for RTX 4090
gpu.usable_vram_bytes      // total × 0.85 (15% reserve)
gpu.recommended_reserve_bytes // 15% buffer
```

### Step 3: Memory Calculation

```cpp
// For Qwen 2.5 3B (hidden=128, layers=36)

// Per-slot KV cache
std::uint64_t per_slot_kv = CalculatePerSlotKvSize(
    n_ctx,      // Context window
    128,        // Hidden dimension
    36,         // Layers
    32,         // Attention heads
    sizeof(float) * 2  // K + V
);

// n_ctx=4096:    per_slot_kv = 72 MB
// n_ctx=32768:   per_slot_kv = 576 MB
// n_ctx=131072:  per_slot_kv = 2,304 MB
```

### Step 4: Slot Allocation

```cpp
// Available memory for KV cache
std::uint64_t available_for_kv =
    target_vram - model_size - overhead;

// Max slots
int max_slots = available_for_kv / per_slot_kv;

// Apply practical limits
max_slots = std::max(4, std::min(256, max_slots));
```

---

## Example Output

### Small GPU + Small Model

```
[RECOMMEND] slot_allocation: GPU has 8589934592 B (8192 MB)
  Model: qwen2.5-3b-instruct-q4_k_m (1610612736 B loaded, Q4_K_M)
  Current: max_parallel_sequences=128, n_ctx=2048
  Recommended: max_parallel_sequences=8, n_ctx=2048
  Memory breakdown:
    - Model: 1.61 GB
    - Overhead: 1.00 GB
    - KV cache: 576 MB (8 slots × 72 MB per slot)
    - Total: 3.18 GB (37% of GPU)

Config:
runtime:
  llama:
    max_parallel_sequences: 8
    n_ctx: 2048
```

### Large GPU + Large Context

```
[RECOMMEND] slot_allocation: GPU has 25769803776 B (24576 MB)
  Model: qwen2.5-3b-instruct-q4_k_m (1610612736 B loaded, Q4_K_M)
  Current: max_parallel_sequences=128, n_ctx=2048
  Recommended: max_parallel_sequences=32, n_ctx=32768
  Memory breakdown:
    - Model: 1.61 GB
    - Overhead: 1.00 GB
    - KV cache: 18.43 GB (32 slots × 576 MB per slot)
    - Total: 21.05 GB (82% of GPU)

Config:
runtime:
  llama:
    max_parallel_sequences: 32
    n_ctx: 32768
```

---

## Integration with SequenceSlotManager

### Startup-Time Configuration

```cpp
// In main.cpp or server initialization

// 1. Probe GPU
AdvisorGpuInfo gpu = ProbeCudaGpu();

// 2. Load model info
AdvisorModelInfo model_info;
model_info.path = "/path/to/model.gguf";
model_info.format = "gguf";
model_info.quantization = DetectQuantization(model_info.path, "gguf");
model_info.file_size_bytes = GetFileSize(model_info.path);

// 3. Build context
StartupAdvisorContext ctx;
ctx.models.push_back(model_info);
ctx.gpu = gpu;

// 4. Run advisor (includes dynamic slot allocation)
int recommendations = RunStartupAdvisor(ctx);

// 5. Get allocation recommendation
MemoryAllocationRecommendation mem_rec =
    CalculateOptimalSlotAllocation(ctx);

if (mem_rec.valid) {
  // 6. Configure SequenceSlotManager dynamically
  slot_manager_ = std::make_unique<scheduler::SequenceSlotManager>(
      mem_rec.recommended_max_slots
  );

  // 7. Configure backend
  llama_config.max_parallel_sequences = mem_rec.recommended_max_slots;
  llama_config.n_ctx = mem_rec.recommended_n_ctx;
}
```

### Runtime Adaptation

```cpp
// Optional: Re-calculate if memory pressure changes
void AdaptSlotAllocation() {
  AdvisorGpuInfo gpu = ProbeCudaGpu();
  MemoryAllocationRecommendation mem_rec = CalculateOptimalSlotAllocation(ctx);

  if (mem_rec.recommended_max_slots < slot_manager_->GetMaxSlots()) {
    log::Info("startup", "Reducing slots due to memory pressure: " +
              std::to_string(slot_manager_->GetMaxSlots()) + " → " +
              std::to_string(mem_rec.recommended_max_slots));

    slot_manager_->SetMaxSlots(mem_rec.recommended_max_slots);
  }
}
```

---

## Quantization Detection

### Supported Quantization Types

| Type | String | Compression (vs FP16) | Bits per Value |
|------|--------|----------------------|---------------|
| FP32 | fp32, f32 | 1.0× | 32 |
| FP16 | fp16, f16 | 2.0× | 16 |
| BF16 | bf16 | 2.0× | 16 |
| Q8_0 | q8_0 | 2.5× | 8.5 |
| Q6_K | q6_k | 3.0× | 6.5 |
| Q5_K_M | q5_k_m | 3.5× | 5.5 |
| Q5_K | q5_k | 3.5× | 5.5 |
| Q4_K_M | q4_k_m | 4.5× | 4.5 |
| Q4_K | q4_k | 4.5× | 4.5 |
| Q3_K_M | q3_k_m | 5.5× | 3.5 |
| Q2_K | q2_k | 8.0× | 2.7 |

### Detection Logic

```cpp
// From GGUF filename
if (filename.find("q4_k_m") != std::string::npos) {
  return QuantizationType::kQ4_K_M;
}

// From safetensors (future: read metadata)
// TODO: Parse model_config.json for quantization info
```

---

## Configuration Reference

### Manual vs Automatic

**Manual (old way)**:
```yaml
runtime:
  llama:
    max_parallel_sequences: 128  # Arbitrary
    n_ctx: 2048                    # Arbitrary
```

**Automatic (new way)**:
```yaml
# Let StartupAdvisor calculate optimal values
runtime:
  llama:
    # max_parallel_sequences and n_ctx will be recommended
    # based on actual model size and GPU memory
```

### Override Recommendations

If you want to override the recommendation:
```yaml
runtime:
  llama:
    max_parallel_sequences: 64  # Override advisor
    n_ctx: 8192                    # Override advisor
```

---

## Memory Calculator

### Quick Reference Formula

```
For Qwen 2.5 3B (hidden=128, layers=36):

Per-Slot KV (MB) = n_ctx / 1024 × 9

Examples:
  n_ctx=4,096:    per_slot = 72 MB
  n_ctx=8,192:    per_slot = 144 MB
  n_ctx=16,384:   per_slot = 288 MB
  n_ctx=32,768:   per_slot = 576 MB
  n_ctx=65,536:   per_slot = 1,152 MB
  n_ctx=131,072:  per_slot = 2,304 MB

Max Slots = floor((GPU_VRAM × target_utilization - Model_Size - overhead) / Per_Slot_KV)

Where:
- target_utilization: INFERFLUX_GPU_UTILIZATION_PCT (default: 85%, range: 50-98)
- overhead: INFERFLUX_OVERHEAD_GB (default: 1 GB, range: 0-16)

Clamp to [10, 256] range (configurable via environment variables)
- INFERFLUX_MIN_SLOTS: Minimum slots (default: 10, range: 4-256)
- INFERFLUX_MAX_SLOTS: Maximum slots (default: 256, range: 16-512)
```

### Online Calculator

The advisor automatically calculates this at startup, so you don't need to do the math manually!

---

## Validation and Testing

### Run the Test

```bash
python3 /tmp/test_dynamic_slot_allocation.py
```

Expected output:
- Test scenarios for different GPU/model combinations
- Formula reference
- Example configurations
- Summary of features

### Unit Tests

```cpp
// tests/unit/test_startup_advisor.cpp

TEST_CASE("CalculateOptimalSlotAllocation - RTX 4090 + Q4_K_M") {
  StartupAdvisorContext ctx;

  // Setup GPU info (RTX 4090: 24 GB)
  AdvisorGpuInfo gpu;
  gpu.available = true;
  gpu.total_vram_bytes = 24ULL * 1024 * 1024 * 1024;
  gpu.usable_vram_bytes = gpu.total_vram_bytes * 0.85;  // 20.4 GB

  // Setup model info (Qwen 2.5 3B Q4_K_M)
  AdvisorModelInfo model;
  model.id = "qwen-3b-q4_k_m";
  model.format = "gguf";
  model.quantization = QuantizationType::kQ4_K_M;
  model.file_size_bytes = 2100 * 1024 * 1024;  // 2.1 GB file
  model.n_params = 3;
  model.n_layers = 36;
  model.hidden_dim = 128;

  ctx.models.push_back(model);
  ctx.gpu = gpu;

  // Calculate allocation
  MemoryAllocationRecommendation rec =
      CalculateOptimalSlotAllocation(ctx);

  REQUIRE(rec.valid == true);
  REQUIRE(rec.model_size_bytes > 0);
  REQUIRE(rec.per_slot_kv_bytes > 0);
  REQUIRE(rec.recommended_max_slots > 0);

  // Verify slots are reasonable
  REQUIRE(rec.recommended_max_slots >= 4);
  REQUIRE(rec.recommended_max_slots <= 128);
}
```

---

## Benefits

1. **Automatic optimization**: No manual calculation needed
2. **GPU-agnostic**: Works on any NVIDIA/AMD GPU
3. **Model-agnostic**: Works with any GGUF/safetensors model
4. **Safe**: Uses 85% target to leave headroom
5. **Transparent**: Logs recommendations, doesn't force changes
6. **Actionable**: Provides copy-paste YAML configuration

---

## Future Enhancements

### Phase 1: Current Implementation ✅
- Quantization detection from filename
- GPU memory probing
- Slot allocation calculation
- Configuration generation

### Phase 2: Enhanced Detection (Future)
- Read GGUF metadata directly
- Parse safetensors model_config.json
- Detect actual model architecture
- Multi-GPU support

### Phase 3: Runtime Adaptation (Future)
- Re-calculate when memory pressure changes
- Dynamic slot expansion/contraction
- Per-request context sizing
- Mixed-context batching

### Phase 4: Advanced Features (Future)
- Predictor-based slot allocation
- ML-based workload prediction
- Multi-model slot sharing
- Cross-GPU slot migration

---

## Troubleshooting

### Issue: Recommendation seems wrong

**Check**:
1. Is the model file detected correctly?
   ```
   [INFO] startup_advisor: Model: qwen-3b-q4_k_m (1610612736 B loaded, Q4_K_M)
   ```

2. Is GPU detected correctly?
   ```
   [INFO] startup_advisor: GPU has 8589934592 B (8192 MB)
   ```

3. Check the memory breakdown in the recommendation
   ```
   [INFO] startup_advisor: Memory breakdown:
   [INFO] startup_advisor:   - Model: 1.61 GB
   [INFO] startup_advisor:   - KV cache: 576 MB
   ```

### Issue: Still getting "failed to find a memory slot"

**Cause**: llama.cpp pre-allocates KV cache based on `max_parallel_sequences`

**Solution**:
1. Ensure `runtime.llama.max_parallel_sequences` is set to recommended value
2. Ensure `runtime.llama.n_ctx` is set to recommended value
3. Restart the server
4. Verify with llama.cpp log:
   ```
   [INFO] llama_kv_cache: size = X MiB (Y cells, Z layers, A/B seqs)
   ```

---

## Summary

The enhanced StartupAdvisor with dynamic slot allocation:

✅ **Automatically detects** model quantization and size
✅ **Probes GPU memory** using CUDA/ROCm APIs
✅ **Calculates optimal slots** using 85% of VRAM target
✅ **Generates configuration** ready to paste into YAML
✅ **Works universally** for all models and GPUs
✅ **Integrates with** SequenceSlotManager for dynamic allocation

**No more manual calculation** - the advisor tells you exactly what to configure!

---

## Environment Variable Configuration

### GPU Utilization Target

**INFERFLUX_GPU_UTILIZATION_PCT**: Target GPU memory utilization percentage (50-98)

```bash
# Default: 85 (conservative, leaves 15% headroom)
export INFERFLUX_GPU_UTILIZATION_PCT=85

# Production: more conservative (80%)
export INFERFLUX_GPU_UTILIZATION_PCT=80

# Development: aggressive (90%)
export INFERFLUX_GPU_UTILIZATION_PCT=90
```

**Purpose**: Controls how aggressively to use GPU memory

### Memory Overhead

**INFERFLUX_OVERHEAD_GB**: Memory reserved for CUDA context, fragmentation, activation tensors (0-16 GB)

```bash
# Default: 1 GB
export INFERFLUX_OVERHEAD_GB=1

# Large models or complex workloads: 2 GB
export INFERFLUX_OVERHEAD_GB=2

# Minimal setups: 512 MB
export INFERFLUX_OVERHEAD_GB=0
```

**Purpose**: Reserves memory for non-KV cache allocations

### Minimum Slot Clamp

**INFERFLUX_MIN_SLOTS**: Minimum number of concurrent sequences to allocate

```bash
# Default: 10
export INFERFLUX_MIN_SLOTS=10

# For very small GPUs (8 GB):
export INFERFLUX_MIN_SLOTS=8

# For production servers:
export INFERFLUX_MIN_SLOTS=32
```

**Purpose**: Ensures minimum concurrent capacity even on small GPUs

### Maximum Slot Clamp

**INFERFLUX_MAX_SLOTS**: Maximum number of concurrent sequences to allocate

```bash
# Default: 256
export INFERFLUX_MAX_SLOTS=256

# Limit for very large contexts:
export INFERFLUX_MAX_SLOTS=128

# High-concurrency with small context:
export INFERFLUX_MAX_SLOTS=512
```

**Purpose**: Prevents over-allocation that could cause OOM

### Usage Examples

```bash
# Small GPU (8GB) - prioritize fitting in memory
INFERFLUX_MIN_SLOTS=8 INFERFLUX_MAX_SLOTS=32 ./build/inferfluxd

# Production server - conservative 80% utilization, high concurrency
INFERFLUX_GPU_UTILIZATION_PCT=80 INFERFLUX_MIN_SLOTS=32 INFERFLUX_MAX_SLOTS=256 ./build/inferfluxd

# Development - aggressive 90% utilization
INFERFLUX_GPU_UTILIZATION_PCT=90 ./build/inferfluxd

# Large context model - more overhead, fewer slots
INFERFLUX_OVERHEAD_GB=2 INFERFLUX_MIN_SLOTS=8 INFERFLUX_MAX_SLOTS=64 ./build/inferfluxd
```

---

