# Large Context Windows: Memory, Slots, and Configuration

**Date**: 2026-03-04
**Question**: How do context size (n_ctx), slots, and memory relate for 32K/64K/128K context models?

---

## 1. The Math: Context Window × Slot Size

### Base Formula

```
Per-slot KV cache = (n_ctx × hidden_dim × 2 × layers × 2)

Where:
  n_ctx = context window size (tokens)
  hidden_dim = model hidden size (128 for Qwen 2.5 3B)
  2 = key and value (two vectors)
  layers = number of transformer layers (36 for Qwen 2.5 3B)
  2 = FP16 bytes per element

Simplified:
  Per-slot MB = (n_ctx × hidden_dim × layers) / (1024 × 1024)
```

### Memory Table: Qwen 2.5 3B (Q4_K_M)

| Context (n_ctx) | Per Slot | 4 Slots | 16 Slots | 32 Slots | 64 Slots | 128 Slots |
|-----------------|----------|---------|----------|----------|----------|-----------|
| 256 | 4.5 MB | 18 MB | 72 MB | 144 MB | 288 MB | 576 MB |
| 512 | 9 MB | 36 MB | 144 MB | 288 MB | 576 MB | 1.1 GB |
| 1,024 | 18 MB | 72 MB | 288 MB | 576 MB | 1.1 GB | 2.3 GB |
| 2,048 | 36 MB | 144 MB | 576 MB | 1.1 GB | 2.3 GB | 4.6 GB |
| 4,096 | 72 MB | 288 MB | 1.1 GB | 2.3 GB | 4.6 GB | 9.2 GB |
| 8,192 (8K) | 144 MB | 576 MB | 2.3 GB | 4.6 GB | 9.2 GB | 18.4 GB |
| 16,384 (16K) | 288 MB | 1.1 GB | 4.6 GB | 9.2 GB | 18.4 GB | 36.9 GB |
| 32,768 (32K) | 576 MB | 2.3 GB | 9.2 GB | 18.4 GB | 36.9 GB | 73.8 GB |
| 65,536 (64K) | 1.1 GB | 4.6 GB | 18.4 GB | 36.9 GB | 73.8 GB | 147.5 GB |
| 131,072 (128K) | 2.3 GB | 9.2 GB | 36.9 GB | 73.8 GB | 147.5 GB | 295 GB |

### Key Insight

**Memory scales LINEARLY with both context size AND slot count**

```
Total KV memory = per_slot × num_slots

For 32K context with 128 slots:
  576 MB × 128 = 73.8 GB ❌ Exceeds RTX 4090 (24 GB)

For 32K context with 16 slots:
  576 MB × 16 = 9.2 GB ✅ Fits in RTX 4090 (24 GB)
```

---

## 2. GPU Memory Budget Breakdown

For an RTX 4090 (24 GB VRAM):

```
Total available: 24 GB
─────────────────────────────
Model weights: ~1.5 GB (Q4_K_M)
KV cache: varies
Overhead: ~1 GB (CUDA, fragmentation)
─────────────────────────────
Available for KV: ~21 GB
```

### Slot Capacity by Context Window (RTX 4090)

| Context | Per Slot | Max Slots (21 GB) | Practical (16 GB) |
|---------|----------|-------------------|-------------------|
| 4K | 72 MB | 291 slots | 222 slots |
| 8K | 144 MB | 145 slots | 111 slots |
| 16K | 288 MB | 72 slots | 55 slots |
| 32K | 576 MB | 36 slots | 27 slots |
| 64K | 1.1 GB | 18 slots | 14 slots |
| 128K | 2.3 GB | 9 slots | 6 slots |

**RTX 6000 Ada (48 GB)**:
| Context | Max Slots | Practical (40 GB) |
|---------|-----------|-------------------|
| 32K | 69 slots | 55 slots |
| 64K | 36 slots | 29 slots |
| 128K | 17 slots | 14 slots |

---

## 3. Configuration in InferFlux

### Current Config (small context)

```yaml
# config/server.yaml

models:
  - id: "qwen-2.5-3b"
    path: "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    format: gguf
    backend: cuda_universal
    # n_ctx not specified → uses llama.cpp default (usually 2048 or 4096)

runtime:
  llama:
    # Context window (default varies by model)
    n_ctx: 2048  # or 512, 1024, etc.

    # Max concurrent sequences
    # Higher n_ctx → Lower max_parallel_sequences (memory trade-off)
    max_parallel_sequences: 128

    # GPU memory limit (MB) - auto-detected if not set
    # llama.cpp will auto-fit based on available VRAM
```

### Large Context Config (32K, 64K, 128K)

```yaml
models:
  - id: "qwen-2.5-3b-32k"
    backend: cuda_universal
    # For 32K context, reduce slots to fit in GPU memory
    runtime:
      llama:
        n_ctx: 32768  # 32K context
        max_parallel_sequences: 16  # Only 16 concurrent users
        # Memory: 576 MB × 16 = 9.2 GB (fits in 24 GB GPU)

  - id: "qwen-2.5-3b-128k"
    backend: cuda_universal
    runtime:
      llama:
        n_ctx: 131072  # 128K context
        max_parallel_sequences: 4  # Only 4 concurrent users
        # Memory: 2.3 GB × 4 = 9.2 GB (fits in 24 GB GPU)
```

### Auto-Configuration (Recommended)

llama.cpp can auto-detect optimal values:

```yaml
runtime:
  llama:
    # Let llama.cpp auto-detect based on GPU memory
    # Leave n_ctx and max_parallel_sequences unset

    # Set memory limit instead
    vram_size: 20000  # MB (leave 4 GB for model + overhead)
```

llama.cpp will then:
1. Load model (~1.5 GB)
2. Calculate available VRAM
3. Allocate largest possible KV cache
4. Print actual n_ctx and max_parallel_sequences to logs

---

## 4. Dynamic Configuration Strategy

### Strategy 1: Context Window vs Concurrency Trade-off

```yaml
# For HIGH concurrency (many users, short prompts)
runtime:
  llama:
    n_ctx: 4096  # 4K context
    max_parallel_sequences: 128  # Many concurrent users
    # Use case: Chat, Q&A, short responses
    # Memory: 72 MB × 128 = 9.2 GB

# For LONG context (few users, long documents)
runtime:
  llama:
    n_ctx: 32768  # 32K context
    max_parallel_sequences: 16  # Few concurrent users
    # Use case: Document analysis, code review
    # Memory: 576 MB × 16 = 9.2 GB
```

### Strategy 2: Multi-Model Deployment

Serve different context variants on different ports:

```yaml
# config/short_context.yaml
server:
  port: 8080
models:
  - id: "qwen-4k"
    backend: cuda_universal
runtime:
  llama:
    n_ctx: 4096
    max_parallel_sequences: 128

# config/long_context.yaml
server:
  port: 8081
models:
  - id: "qwen-32k"
    backend: cuda_universal
runtime:
  llama:
    n_ctx: 32768
    max_parallel_sequences: 16
```

### Strategy 3: Runtime Adaptation

```cpp
// Dynamic slot allocation based on context size
int CalculateMaxSlots(int requested_ctx) {
  int per_slot_mb = (requested_ctx * 128 * 36) / (1024 * 1024);
  int available_mb = 21000;  // 21 GB available for KV
  int max_slots = available_mb / per_slot_mb;

  // Clamp to reasonable range
  return std::min(128, std::max(4, max_slots));
}

// Examples:
// n_ctx=4096 → per_slot=72 MB → max_slots=291 → clamp to 128
// n_ctx=32768 → per_slot=576 MB → max_slots=36 → use 36
// n_ctx=131072 → per_slot=2304 MB → max_slots=9 → use 9
```

---

## 5. llama.cpp Configuration Details

### Where n_ctx is Defined

```cpp
// llama.cpp initialization
struct llama_context_params {
  uint32_t n_ctx;          // Context window (tokens)
  uint32_t n_seq_max;      // Max sequences (slots)
  uint32_t n_batch;        // Batch size
  // ...
};

// From InferFlux llama_backend.cpp
ctx_params.n_ctx = config.n_ctx;
ctx_params.n_seq_max = static_cast<uint32_t>(config.max_parallel_sequences);
```

### KV Cache Allocation

```cpp
// llama.cpp internal allocation
size_t kv_cache_size =
  n_ctx *              // tokens per sequence
  n_seq_max *          // max sequences
  hidden_dim *         // model hidden size
  n_layer *            // number of layers
  n_embd_head *        // attention heads
  sizeof(ggml_fp16_t); // bytes per element

// Actual allocation is optimized:
// - Paged allocation (llama.cpp 3.0+)
// - Only allocate for active layers
// - FP16 quantization
```

### Logging Output

```
[INFO] llama_kv_cache: size = 144.00 MiB (512 cells, 36 layers, 128/128 seqs)
                             │       │      │         │      │
                             │       │      │         │      └─ Max sequences (slots)
                             │       │      │         └────────── Used/Max
                             │       │      └────────────────────── Layers
                             │       └─────────────────────────── Context per layer
                             └────────────────────────────────── Total KV cache size
```

---

## 6. Configuration Examples for Different GPUs

### RTX 4090 (24 GB)

```yaml
# 4K context, high concurrency
runtime:
  llama:
    n_ctx: 4096
    max_parallel_sequences: 128
    # Memory: 72 MB × 128 = 9.2 GB

# 16K context, medium concurrency
runtime:
  llama:
    n_ctx: 16384
    max_parallel_sequences: 32
    # Memory: 288 MB × 32 = 9.2 GB

# 32K context, low concurrency
runtime:
  llama:
    n_ctx: 32768
    max_parallel_sequences: 16
    # Memory: 576 MB × 16 = 9.2 GB
```

### RTX 6000 Ada (48 GB)

```yaml
# 8K context, high concurrency
runtime:
  llama:
    n_ctx: 8192
    max_parallel_sequences: 128
    # Memory: 144 MB × 128 = 18.4 GB

# 32K context, medium concurrency
runtime:
  llama:
    n_ctx: 32768
    max_parallel_sequences: 64
    # Memory: 576 MB × 64 = 36.9 GB

# 64K context, low concurrency
runtime:
  llama:
    n_ctx: 65536
    max_parallel_sequences: 32
    # Memory: 1.1 GB × 32 = 36.9 GB
```

### H100 (80 GB)

```yaml
# 32K context, high concurrency
runtime:
  llama:
    n_ctx: 32768
    max_parallel_sequences: 128
    # Memory: 576 MB × 128 = 73.8 GB

# 128K context, medium concurrency
runtime:
  llama:
    n_ctx: 131072
    max_parallel_sequences: 32
    # Memory: 2.3 GB × 32 = 73.8 GB
```

---

## 7. Advanced: Mixed Context Workloads

### Problem: Different Requests Need Different Context

```cpp
// Request A: "Hi" → needs 100 tokens
// Request B: Long document → needs 32K tokens

// With fixed n_ctx=32768:
// - Request A wastes 32,668 tokens of capacity
// - Request B fits perfectly
```

### Solution: Variable Context Allocation (Future)

```yaml
runtime:
  llama:
    n_ctx_max: 32768  # Maximum context window
    n_ctx_min: 512    # Minimum context window
    max_parallel_sequences: 128  # For min context
    # Effective slots decrease as context increases
```

### Current Workaround

Use multiple model instances:

```bash
# Start short-context server
INFERFLUX_PORT_OVERRIDE=8080 ./build/inferfluxd --config config/4k_context.yaml

# Start long-context server
INFERFLUX_PORT_OVERRIDE=8081 ./build/inferfluxd --config config/32k_context.yaml

# Router directs requests based on prompt length
```

---

## 8. Production Configuration Calculator

### Quick Reference Formula

```
Step 1: Calculate per-slot memory
  per_slot_mb = (n_ctx × 128 × 36) / 1048576

Step 2: Calculate available memory
  available_mb = gpu_vram_mb - 1500 (model) - 1000 (overhead)

Step 3: Calculate max slots
  max_slots = available_mb / per_slot_mb

Step 4: Clamp to practical range
  max_slots = min(128, max(4, max_slots))
```

### Example Calculations

**RTX 4090 (24 GB), 32K context**:
```
per_slot_mb = (32768 × 128 × 36) / 1048576 = 576 MB
available_mb = 24000 - 1500 - 1000 = 21500 MB
max_slots = 21500 / 576 = 37 slots
final = min(128, 37) = 37 slots

Configuration:
  n_ctx: 32768
  max_parallel_sequences: 37
```

**RTX 4090 (24 GB), 128K context**:
```
per_slot_mb = (131072 × 128 × 36) / 1048576 = 2304 MB
available_mb = 21500 MB
max_slots = 21500 / 2304 = 9 slots
final = 9 slots

Configuration:
  n_ctx: 131072
  max_parallel_sequences: 9
```

---

## 9. Key Takeaways

1. **Context and slots trade off linearly**: 2× context = 2× memory per slot, so you need 2× fewer slots

2. **Large context requires fewer concurrent users**:
   - 4K context: 128 concurrent users
   - 32K context: 16-36 concurrent users
   - 128K context: 4-9 concurrent users

3. **Memory is pre-allocated**: KV cache for max_slots × n_ctx is allocated at model load time

4. **Configure based on use case**:
   - Chat/QA (short): 4K context, 128 slots
   - Documents (medium): 32K context, 16 slots
   - Books/code (long): 128K context, 4 slots

5. **Check llama.cpp logs** for actual allocation:
   ```
   [INFO] llama_kv_cache: size = X MiB (Y cells, Z layers, A/B seqs)
   ```

6. **For production**: Use multi-model deployment with different context/slot configurations for different endpoints
