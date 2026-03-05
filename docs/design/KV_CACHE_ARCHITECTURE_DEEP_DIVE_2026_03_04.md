# KV Cache Architecture: Deep Dive on Slots, Memory, and Eviction

**Date**: 2026-03-04
**Context**: Understanding how context windows, slot limits, FlashAttention, and eviction strategies work together

---

## 1. Timeout-Based Eviction vs LRU: Current Implementation

### Current State: Simple Timeout (5 minutes)

```cpp
// sequence_slot_manager.cpp
bool SequenceSlot::IsIdle(std::chrono::milliseconds timeout) const {
  if (state != SequenceState::kDecoding) return true;
  auto idle = std::chrono::steady_clock::now() - last_access;
  return idle > timeout;
}
```

**Problem**: Pure timeout is "dumb" - a slot used 1000 times 5 minutes ago is treated the same as a slot used once 5 minutes ago.

**Better Approach: LRU + Idle Timeout**

```cpp
struct SequenceSlot {
  int slot_id;
  int64_t request_id;
  int sequence_id;
  SequenceState state;

  // For LRU
  std::chrono::steady_clock::time_point last_access;
  int access_count{0};           // NEW: Track usage frequency
  int token_count{0};            // NEW: Track total work done

  // Hybrid scoring
  double GetEvictionScore() const {
    auto idle_minutes = std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::steady_clock::now() - last_access).count();

    // Score = idle_time - (usage_factor × access_count)
    // Higher score = should be evicted first
    return static_cast<double>(idle_minutes) - (access_count * 0.1);
  }
};
```

**Recommended Hybrid Strategy**:

| Policy | When to Use | Formula |
|--------|------------|---------|
| **Timeout-only** | Simple workloads, low concurrency | `idle > 5min` |
| **LRU** | High contention, many short requests | Evict least recently used |
| **LFU (Least Frequently Used)** | Mixed workloads | Evict lowest access_count |
| **Hybrid** | Production | `(idle_time × 0.7) - (access_count × 0.3)` |

---

## 2. Tiered Lifecycle: LRU + Idle Timeout + CPU Offload

### Three-Tier KV Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEQUENCE SLOT MANAGER                        │
│                   (Orchestration Layer)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tier 0: GPU KV Cache (HOT) - 128 slots, 4.5 MB each           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Criteria: last_access < 30 seconds ago                  │    │
│  │ Eviction: LRU within this tier                          │    │
│  │ Size: 128 × 4.5 MB = 576 MB                             │    │
│  └────────────────────────────────────────────────────────┘    │
│                       ↓ Idle > 30s                               │
│  Tier 1: CPU RAM KV Cache (WARM) - Expandable                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Criteria: last_access < 5 minutes ago                   │    │
│  │ Eviction: LFU (access_count weighted)                   │    │
│  │ Size: Up to 16 GB (system RAM dependent)               │    │
│  └────────────────────────────────────────────────────────┘    │
│                       ↓ Idle > 5min                               │
│  Tier 2: Evicted (COLD) - Freed from memory                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Action: Call backend->FreeSequence()                    │    │
│  │ Next request: Must reload from disk                     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Status**:
- ✅ Tier 0: Implemented (128 GPU slots)
- ⚠️ Tier 1: Partially implemented (PagedKVCache exists but not integrated)
- ❌ Tier 2: NOT implemented (we mark as evicted but don't call FreeSequence)

**Critical Gap**: The current SequenceSlotManager marks slots as `kEvicted` but doesn't actually call `backend->FreeSequence()`. This means:
- llama.cpp internal KV cache still holds the memory
- "Evicted" slots aren't actually freed until the request completes naturally
- **This is a documented TODO in the code**

---

## 3. llama.cpp KV Cache: Private Implementation

### Can We Use Tiered Lifecycle with llama.cpp?

**Short Answer**: NO - Not with the current architecture.

**Why**:

1. **llama.cpp's KV cache is private and encapsulated**
   ```cpp
   // llama.cpp internal (NOT accessible from InferFlux)
   struct llama_kv_cache {
     void *data;  // Opaque pointer
     size_t size;
     int n_max_seq;  // Fixed at context creation time
   };
   ```

2. **InferFlux SequenceSlotManager is ABOVE llama.cpp**
   ```
   InferFlux Request
       ↓
   SequenceSlotManager (tracks which requests are active)
       ↓
   llama.cpp KV cache (internal, opaque)
       ↓
   GPU memory (actual KV data)
   ```

3. **We cannot directly move llama.cpp KV cache between GPU/CPU**
   - llama.cpp manages its own memory internally
   - `n_gpu_layers` controls which MODEL LAYERS are on GPU
   - KV cache ALWAYS follows the model context (GPU or CPU)
   - No API to serialize/deserialize individual KV slots

### What CAN We Do?

**Option A: Timeout-Based Request Eviction** (Current approach)
- Track request-level timeouts
- When request exceeds timeout, mark for cleanup
- Wait for request to complete naturally, then free slot
- **Limitation**: Doesn't actually free llama.cpp memory early

**Option B: Request Cancellation**
```cpp
// Future enhancement
if (slot.IsIdle(timeout)) {
  // Cancel the request
  backend->CancelRequest(slot.sequence_id);
  // Free the sequence
  backend->FreeSequence(slot.sequence_id);
  // Release slot
  ReleaseSlot(slot.slot_id);
}
```
- **Limitation**: llama.cpp may not support graceful cancellation

**Option C: Separate PagedKVCache** (Already exists!)
- InferFlux has `PagedKVCache` for model-level offloading
- Different layer than per-token KV cache
- Used for MODEL offloading, not sequence offloading
- **Limitation**: Doesn't help with llama.cpp's internal KV cache

---

## 4. Memory Scaling: Do 4 Concurrent Requests Use 4x Memory?

### Theory vs Reality

**Theoretical Calculation** (for Qwen 2.5 3B, Q4_K_M, 256 context):

```
Per slot KV cache:
  - Hidden dim: 128
  - Layers: 36
  - Context: 256 tokens
  - Data type: FP16 (2 bytes)

  Per token per layer: 128 × 2 = 256 bytes
  Per 256 tokens per layer: 256 × 256 = 65,536 bytes = 64 KB
  Per slot (all layers): 64 KB × 36 = 2,304 KB ≈ 2.25 MB

Wait, this is different from our previous calc!
Let's recalculate more carefully:

K cache per token: hidden_dim × 2 bytes = 128 × 2 = 256 bytes
V cache per token: hidden_dim × 2 bytes = 128 × 2 = 256 bytes
Total per token: 512 bytes

Per 256 tokens: 512 × 256 = 131,072 bytes = 128 KB
Per layer (256 tokens): 128 KB
All 36 layers: 128 KB × 36 = 4,608 KB ≈ 4.5 MB

So for 4 slots: 4.5 MB × 4 = 18 MB
```

### llama.cpp KV Cache Log

```
[INFO] llama_kv_cache: size = 72.00 MiB (256 cells, 36 layers, 128/128 seqs)
                             │       │      │         │      │
                             │       │      │         │      └─ Max sequences
                             │       │      │         └────────── Used/Max
                             │       │      └────────────────────── Layers
                             │       └─────────────────────────── Cells per layer
                             └────────────────────────────────── Total size
```

**Key Insight**: `128/128 seqs` means:
- Total KV cache allocated for 128 sequences
- **All 128 sequences share the SAME 72 MB pool**
- Memory is NOT `4.5 MB × 128 = 576 MB`
- It's **72 MB total**, divided among active sequences

**So, does memory scale 4x for 4 concurrent requests?**

**NO** - because llama.cpp uses a **fixed-size KV cache pool**!

```
Configuration:
  max_parallel_sequences: 128
  n_ctx: 256

llama.cpp allocates:
  Total KV cache = 72 MB (for all 128 sequences)

Usage:
  1 request: Uses 1/128 of pool (~0.56 MB per slot's worth)
  4 requests: Uses 4/128 of pool (~2.25 MB per slot)
  128 requests: Uses entire pool (72 MB)
```

### But Wait - Our Logs Show Different Behavior!

From the earlier stress test:
```
Baseline: 3799 MB
After 4 requests: 3799 MB (no increase!)
```

**Explanation**: The KV cache is pre-allocated when the model loads!

```cpp
// llama_backend.cpp
ctx_params.n_seq_max = static_cast<uint32_t>(config.max_parallel_sequences);
// This allocates KV cache for ALL 128 sequences upfront
```

**Memory Timeline**:
1. Server starts → Allocate model (1.5 GB) + KV cache (72 MB) = ~1.57 GB
2. Request 1 → Use slot 0 from pre-allocated pool
3. Request 4 → Use slots 0-3 from pre-allocated pool
4. Total GPU memory: Constant at ~1.57 GB (no per-request allocation)

---

## 5. Context Window, Slot Size, FlashAttention Relationships

### How They Work Together

```
┌─────────────────────────────────────────────────────────────────┐
│                    REQUEST FLOW                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. CONTEXT WINDOW (n_ctx)                                      │
│     - Maximum tokens per sequence                              │
│     - Configured per model: n_ctx: 256, 512, 2048, etc.       │
│     - Determines KV cache size PER SLOT                        │
│                                                                 │
│     KV_cache_per_slot = (n_ctx × hidden_dim × 2 × layers)     │
│                                                                 │
│     Examples:                                                   │
│       n_ctx=256:  4.5 MB per slot                              │
│       n_ctx=512:  9 MB per slot                                │
│       n_ctx=2048: 36 MB per slot                               │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. MAX PARALLEL SEQUENCES (max_parallel_sequences)            │
│     - Number of concurrent users/sequences                    │
│     - Total KV cache = per_slot × max_sequences                │
│                                                                 │
│     Total KV = 4.5 MB × 128 = 576 MB for n_ctx=256            │
│     Total KV = 36 MB × 128 = 4.6 GB for n_ctx=2048            │
│                                                                 │
│     This is PRE-ALLOCATED when model loads                     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. SLOT LIMIT (SequenceSlotManager)                           │
│     - Orchestrates which sequences are active                  │
│     - Slots = max_parallel_sequences (128)                     │
│     - NOT the same as llama.cpp KV cache (encapsulated)        │
│                                                                 │
│     SequenceSlotManager tracks:                                │
│     - Which slot is used by which request                      │
│     - How long each slot has been idle                         │
│     - When to evict (timeout-based)                            │
│                                                                 │
│     But does NOT directly manage llama.cpp KV memory            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. FlashAttention                                            │
│     - OPTIMIZES attention computation                          │
│     - Does NOT change memory allocation                        │
│     - Reduces FLOPs by avoiding O(n²) materialization           │
│                                                                 │
│     Without FA:  O(n²) memory for attention matrix              │
│     With FA:     O(n) memory, computes in blocks               │
│                                                                 │
│     KV cache size is UNCHANGED by FlashAttention               │
│     FlashAttention optimizes COMPUTE, not MEMORY              │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Calculation Examples

| n_ctx | Layers | Per Slot | 128 Slots | With FA2 |
|-------|--------|----------|-----------|----------|
| 256 | 36 | 4.5 MB | 576 MB | Same KV, faster compute |
| 512 | 36 | 9 MB | 1.1 GB | Same KV, faster compute |
| 1024 | 36 | 18 MB | 2.3 GB | Same KV, faster compute |
| 2048 | 36 | 36 MB | 4.6 GB | Same KV, faster compute |
| 8192 | 36 | 144 MB | 18.4 GB | Same KV, faster compute |

**FlashAttention Impact**:
- ✅ Faster inference (less memory bandwidth)
- ✅ Lower power consumption
- ✅ Enables larger batch sizes
- ❌ Does NOT reduce KV cache memory
- ❌ Does NOT increase slot capacity

---

## 6. Production Recommendations

### Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| SequenceSlotManager | ✅ Implemented | 128 slots, timeout eviction |
| llama.cpp KV cache | ✅ Working | Pre-allocated, opaque |
| Timeout eviction | ⚠️ Partial | Marks evicted, doesn't free llama.cpp |
| LRU eviction | ❌ Not implemented | Simple timeout only |
| CPU offload | ❌ Not integrated | PagedKVCache exists but separate |
| Tiered lifecycle | ❌ Not implemented | Single tier only |

### Recommended Improvements

**Priority 1: Fix Eviction (Critical)**
```cpp
// In EvictionWorkerLoop, actually free the backend
auto evicted = slot_manager_->EvictIdleSlots(timeout);
for (auto [slot_id, seq_id] : evicted) {
  // Need to find which backend owns this sequence_id
  // Then call:
  backend->FreeSequence(seq_id);  // Actually free llama.cpp memory
  FreeSeqSlot(slot_id);           // Free scheduler slot
}
```

**Priority 2: Add LRU Scoring**
```cpp
struct SequenceSlot {
  double GetEvictionScore() const {
    auto idle = std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::steady_clock::now() - last_access).count();
    // Prefer idle slots, but penalize frequently used ones
    return idle - (access_count * 0.5);
  }
};
```

**Priority 3: Add Prometheus Metrics**
```yaml
Metrics to expose:
  - inferflux_kv_slots_used_total
  - inferflux_kv_slots_free_total
  - inferflux_kv_slots_evicted_total
  - inferflux_kv_slot_idle_duration_seconds
  - inferflux_kv_slot_access_count
```

**Priority 4: Configuration**
```yaml
runtime:
  max_parallel_sequences: 128
  sequence_idle_timeout: 300000     # 5 minutes
  sequence_eviction_policy: lru    # or timeout, lfu, hybrid
```

---

## 7. Key Takeaways

1. **Memory is pre-allocated**: llama.cpp allocates KV cache for ALL 128 sequences upfront
2. **Per-request memory is negligible**: Using a slot doesn't allocate more GPU memory
3. **FlashAttention doesn't save memory**: It optimizes computation, not storage
4. **Current eviction is incomplete**: We mark slots as evicted but don't free llama.cpp memory
5. **LLM integration is limited**: Cannot directly manage llama.cpp's internal KV cache
6. **Context window is key**: Larger n_ctx → More memory per slot → Fewer slots possible

**For Production**: Keep current implementation (128 slots, timeout eviction) but:
- Monitor slot usage via metrics
- Increase `max_parallel_sequences` if needed
- Set timeout based on workload (5 min for interactive, longer for batch)
- Plan for manual server restarts if slots leak (workaround for incomplete eviction)
