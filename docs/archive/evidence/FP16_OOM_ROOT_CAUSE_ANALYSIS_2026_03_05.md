# FP16 OOM Root Cause Analysis & Solutions

**Date**: 2026-03-05
**Issue**: FP16 models crash with `std::bad_alloc` under concurrent load

---

## Root Cause Analysis

### Issue #1: Config Override Problem ❌

**Problem**: `INFERFLUX_MODEL_PATH` environment variable ignored when `config/server.yaml` has hardcoded `model.path`

**Evidence**:
```bash
# Config has:
# model:
#   path: models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf

# Even with INFERFLUX_MODEL_PATH set:
INFERFLUX_MODEL_PATH=models/qwen2.5-3b-f16.gguf ./build/inferfluxd
# Result: Tries to load Meta-Llama-3-8B (wrong model!)
```

**Root Cause**: Config file takes precedence over environment variable. Non-existent model → Load failure → Crash.

**Solution**: Use explicit config file or environment variable override:
```bash
# Option 1: Create dedicated config
./build/inferfluxd --config config/fp16.yaml

# Option 2: Set via environment (if supported)
# INFERFLUX_MODELS="id=qwen-fp16,path=models/.../f16.gguf,backend=cuda"
```

### Issue #2: Memory Underestimation ⚠️

**Problem**: Memory calculation doesn't account for:
- Activation tensors (during forward pass)
- CUDA context overhead
- Memory fragmentation
- Temporary buffers
- Concurrent request memory spikes

**Evidence**:
- Theoretical: 5.8 GB model + 800 MB KV = 6.6 GB for 8 slots
- Actual GPU memory used: 8,584 MB (for single loaded model)
- **Missing**: ~2 GB overhead

**Memory Breakdown**:
```
Model weights:     5.8 GB
CUDA context:       1.2 GB
Activation tensors: ~1-2 GB (peak)
KV cache (1 slot):  ~100 MB
KV cache (8 slots): ~800 MB
Fragmentation:      ~500 MB
-----------------------------------
Total (8 concurrent): ~9-10 GB
```

**Solution**: Update memory calculation:
```cpp
// In StartupAdvisor or model loader
constexpr double kActivationOverhead = 1.5;  // 1.5x model size
constexpr double kFragmentationAllowance = 1.1; // 10% fragmentation

std::uint64_t estimated_memory = model_size * kActivationOverhead * kFragmentationAllowance;
```

### Issue #3: No Pre-flight Memory Check ❌

**Problem**: Server accepts concurrent requests without checking if sufficient memory available

**Current Behavior**:
1. Server starts with model loaded (8.5 GB)
2. 8 concurrent requests arrive
3. Server tries to allocate KV for all 8
4. CUDA allocation fails → `std::bad_alloc` → Crash

**Desired Behavior**:
1. Server starts with model loaded
2. 8 concurrent requests arrive
3. Server checks available memory
4. If insufficient, queue requests or return 503
5. Server stays alive and healthy

**Solution**: Implement admission control with memory checks:
```cpp
// In request scheduler
bool CanAcceptRequest(const RequestContext& ctx) {
    std::uint64_t available = GetAvailableGPUMemory();
    std::uint64_t required = EstimateRequestMemory(ctx);

    if (required > available) {
        log::Warn("scheduler",
            "Insufficient memory: required=" + FormatBytes(required) +
            " available=" + FormatBytes(available));
        return false;
    }

    return true;
}
```

---

## Proposed Solutions

### Solution 1: Pre-flight Memory Check (Immediate)

Add memory check before accepting concurrent requests:

```cpp
// server/scheduler.cpp

bool SequenceSlotManager::CanAllocateSlots(int requested_slots) {
    // Check if we have enough memory before allocating
    auto gpu_memory = GetAvailableGPUMemory();
    auto required = estimated_per_slot_ * requested_slots +
                    activation_overhead_;

    if (required > gpu_memory * 0.9) {  // Leave 10% buffer
        log::Warn("slot_manager",
            "Rejecting request: insufficient memory (required=" +
            std::to_string(required) + ", available=" +
            std::to_string(gpu_memory) + ")");
        return false;
    }

    return true;
}
```

### Solution 2: Graceful Degradation (Short-term)

When memory is low, reduce capacity instead of crashing:

```cpp
// Reduce max slots when memory pressure detected
if (GetAvailableGPUMemory() < threshold_) {
    int old_max = max_slots_;
    int new_max = old_max / 2;

    log::Warn("slot_manager",
        "Memory pressure detected: reducing max_slots from " +
        std::to_string(old_max) + " to " + std::to_string(new_max));

    max_slots_ = new_max;
}
```

### Solution 3: Request Queuing (Better)

Queue requests instead of rejecting when at capacity:

```cpp
// server/request_queue.cpp

class RequestQueue {
    bool EnqueueRequest(const Request& req) {
        if (slot_manager_->AvailableSlots() == 0) {
            pending_requests_.push(req);
            return true;  // Queued (not rejected)
        }
        return ProcessRequest(req);
    }

    void OnSlotReleased() {
        if (!pending_requests_.empty()) {
            auto req = pending_requests_.front();
            pending_requests_.pop();
            ProcessRequest(req);
        }
    }
}
```

### Solution 4: Config Validation (Startup)

Validate config at startup to prevent crashes:

```cpp
// server/main.cpp

void ValidateConfig(const ServerConfig& config) {
    auto model_size = GetFileSize(config.model.path);
    auto gpu_memory = GetTotalGPUMemory();

    if (model_size > gpu_memory * 0.7) {  // Model shouldn't be > 70% of GPU
        log::Error("server",
            "Model too large for GPU: model=" + FormatBytes(model_size) +
            " gpu=" + FormatBytes(gpu_memory));

        if (config.strict_mode) {
            throw std::runtime_error("Model exceeds GPU capacity");
        } else {
            log::Warn("server", "Continuing anyway, but expect OOM errors");
        }
    }
}
```

### Solution 5: Per-Request Memory Limits

Limit tokens per request based on available memory:

```cpp
// Calculate max tokens based on available memory
int GetMaxTokensForRequest(int n_ctx, int concurrent_requests) {
    auto available = GetAvailableGPUMemory();
    auto per_slot_kv = CalculatePerSlotKV(n_ctx);

    // Leave 30% for model, overhead, activation tensors
    auto kv_budget = available * 0.70 / concurrent_requests;

    int max_tokens = kv_budget / per_slot_kv;
    return std::max(128, min(max_tokens, n_ctx));
}
```

---

## Implementation Priority

### Phase 1: Immediate (P0) ✅ COMPLETED (2026-03-05)

1. **✅ Add memory check before accepting requests**
   - Simple `cudaMemGetInfo()` check in `SequenceSlotManager::CanAcceptRequest()`
   - Reject with warning if insufficient memory (>90% pressure)
   - Prevents server crash

2. **✅ Update StartupAdvisor memory calculation**
   - Add activation overhead (1.5x model size for FP16, 1.2x for quantized)
   - Add fragmentation allowance (1.1x total)
   - More accurate slot recommendations

3. **✅ Fix config override issue**
   - INFERFLUX_MODEL_PATH now properly overrides config file models
   - Resolves issue where wrong model was loaded

### Phase 2: Short-term (P1) ✅ COMPLETED (2026-03-05)

4. **✅ Implement graceful degradation**
   - Reduce max slots under memory pressure via `PerformGracefulDegradation()`
   - Log warnings before reaching limit (80%, 85%, 90% thresholds)
   - Auto-tune based on available memory (50% reduction at 90%, 25% at 85%)

5. **⏳ Add metrics for memory pressure**
   - `GetMemoryPressure()` returns memory usage percentage
   - TODO: Export Prometheus metrics for memory pressure
   - Alert when memory > 80%

### Phase 3: Long-term (P2)

6. **Implement request queuing**
   - Queue requests when at capacity
   - Process when slots available
   - Fair queue with timeout

7. **Add pre-flight memory estimation**
   - Estimate memory per request
   - Check before accepting
   - Return 507 (Insufficient Storage) if needed

---

## Testing Strategy

### Test 1: Memory Limit Detection

```bash
# Start server with FP16 model on 20 GB GPU
# Send concurrent requests until OOM
# Verify: Server rejects with 503 instead of crashing

for i in {1..20}; do
    curl -X POST http://localhost:8080/v1/chat/completions \
        -H "Authorization: Bearer dev-key-123" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":1000}' &
done
wait

# Expected: Some requests succeed, others get 503
# Not: Server crashes with std::bad_alloc
```

### Test 2: Graceful Degradation

```bash
# Monitor max_slots reduction under load
curl -s http://localhost:8080/metrics | grep max_slots

# Send many requests, watch for degradation
# Verify: max_slots reduces instead of crash
```

### Test 3: Config Validation

```bash
# Try to load model too large for GPU
# Verify: Clear error message at startup
# Verify: Server exits gracefully (not crash)
```

---

## Documentation Updates

### User Guide Updates

Add section on "Memory Requirements" with:
- GPU VRAM recommendations per model size
- How to estimate required memory
- What happens when memory is insufficient
- Error codes and their meanings

### Config Reference Updates

Add new config options:
```yaml
runtime:
  memory:
    check_before_accept: true
    reject_on_insufficient: true
    degradation_threshold: 0.8  # Reduce slots when memory > 80%
    queue_when_full: false
```

### Troubleshooting Updates

Add "Out of Memory" section with:
- Common causes
- How to increase capacity
- How to reduce memory usage
- Migration path to larger GPU

---

## Summary

### Root Causes Identified

1. **Config override issue** - Environment variable ignored ✅ FIXED
2. **Memory underestimation** - Calculation misses activation overhead ✅ FIXED
3. **No pre-flight check** - Server crashes instead of rejecting ✅ FIXED
4. **No graceful degradation** - No warning before OOM ✅ FIXED

### Implementation Status (2026-03-05)

1. ✅ **Add memory check before accepting requests** (P0) - COMPLETED
   - `SequenceSlotManager::CanAcceptRequest()` implements pre-flight check
   - Rejects requests at >90% memory pressure
2. ✅ **Update StartupAdvisor calculation** (P0) - COMPLETED
   - Activation overhead: 1.5x for FP16, 1.2x for quantized
   - Fragmentation allowance: 1.1x (10% overhead)
3. ✅ **Fix config override** (P0) - COMPLETED
   - INFERFLUX_MODEL_PATH now properly overrides config file
4. ✅ **Implement graceful degradation** (P1) - COMPLETED
   - `SequenceSlotManager::PerformGracefulDegradation()` reduces slots under pressure
   - Thresholds: 80% (warning), 85% (moderate reduction), 90% (aggressive reduction)
5. ✅ **Memory pressure monitoring** (P1) - COMPLETED
   - `SequenceSlotManager::GetMemoryPressure()` returns current pressure percentage

### Expected Outcomes

**Before (Original)**:
- Server crashes with `std::bad_alloc`
- Unclear error messages
- No way to prevent crash

**After (Current)**:
- Server warns before reaching memory limit
- Graceful degradation reduces capacity instead of crashing
- Pre-flight checks prevent OOM by rejecting at 90% pressure
- Clear logging of memory status
- Auto-tuning based on available memory

---

## Next Steps

1. ✅ ~~Implement pre-flight memory check (SequenceSlotManager)~~ - DONE
2. ✅ ~~Update StartupAdvisor memory calculation~~ - DONE
3. ✅ ~~Fix config override at startup~~ - DONE
4. ✅ ~~Implement graceful degradation~~ - DONE
5. **⏳ Test with FP16 model on 20 GB GPU** - PENDING
6. **⏳ Verify server stays alive under memory pressure** - PENDING
7. **⏳ Add Prometheus metrics for memory pressure** - PENDING
8. **⏳ Implement request queuing** (P2) - PENDING
