# Fused Kernel Redesign Specification

## Status: DESIGN REVIEW — DO NOT IMPLEMENT

## 1. Problem Statement

InferFlux CUDA native decode throughput is **59 tok/s** vs llama.cpp's **101 tok/s** on Qwen2.5-3B Q4_K_M (RTX 4000 Ada, Windows WDDM). The gap decomposes as:

| Component | inferflux | llama.cpp (est.) | Delta |
|-----------|-----------|-----------------|-------|
| GPU forward compute | 11.4ms | ~8.5ms | +2.9ms |
| WDDM driver overhead | ~4.5ms | ~1.4ms | +3.1ms |
| Sampling + sync | ~0.5ms | ~0ms | +0.5ms |
| CPU batch loop | ~0.5ms | ~0ms | +0.5ms |

The driver overhead is proportional to kernel launch count (490 graph nodes vs ~200). The GPU compute gap comes from lower memory bandwidth utilization (75% vs 85%) and overhead kernels (QuantizeRows, BiasAdd, ResidualAdd).

## 2. Evidence Base (ncu profiling, 500 kernels, 15.3ms total GPU time)

### Current per-token kernel budget (B=1 decode, 36 layers)

| Phase | Time (µs) | % | Launches | Mem BW% | Problem |
|-------|-----------|---|----------|---------|---------|
| GEMV Q4_K paired group | 7,697 | 50.4% | 40 | 77.9% | Primary compute — good but not peak |
| GEMV Q4_K single | 3,020 | 19.8% | 45 | 74.7% | V/O/down projections |
| GEMV Q6_K | 2,070 | 13.6% | 24 | **62.1%** | Norm layers — **20% below optimal** |
| SiluMul+Quantize | 573 | 3.7% | 28 | 3.2% | Grid starvation (5 blocks / 48 SMs) |
| QuantizeRows | 423 | 2.8% | **86** | 1.4% | **Redundant** — eliminated by Q8_1 path |
| FlashAttention2 | 299 | 2.0% | 29 | 1.9% | Expected for B=1 GQA decode |
| RmsNorm | 238 | 1.6% | 57 | 3.1% | Overhead kernel — fuseable |
| BiasAdd | 178 | 1.2% | **87** | 2.0% | **87 launches for 3 per layer** |
| ResidualAdd | 120 | 0.8% | 57 | 6.3% | Fuseable into GEMV accumulate |
| RoPE | 62 | 0.4% | 29 | 3.9% | Already efficient |
| Embedding | 2 | 0.0% | 1 | 3.1% | Negligible |

**Current total: ~500 kernels/token, 15.3ms GPU time**

### Target: match llama.cpp kernel efficiency

llama.cpp achieves ~8.5ms/token by:
- Fewer kernel launches (~200 graph nodes)
- Higher DRAM bandwidth utilization (~85% on GEMV)
- No separate QuantizeRows (activations stay FP16, quantized on-the-fly in GEMV)
- Fused norm+GEMV in some paths

## 3. Redesign Principles

### P1: Minimize stream operations per token
Every kernel launch costs ~5-10µs on Linux (CUDA graph amortizes this) but ~20-50µs effective on Windows WDDM due to driver overhead. With 490 kernels × 20µs = 9.8ms of launch overhead — nearly equal to the GPU compute time.

**Target: ≤200 kernels per token** (11 per layer → 6 per layer).

### P2: Maximize memory bandwidth utilization
The RTX 4000 Ada has 360 GB/s DRAM bandwidth. Reading 1.69 GB of weights per token at 85% utilization = 5.5ms theoretical minimum. Current 75% utilization wastes 1.3ms per token.

**Target: ≥82% average DRAM bandwidth across all GEMV kernels.**

### P3: Eliminate overhead kernels entirely
QuantizeRows (86 launches, 423µs), BiasAdd (87 launches, 178µs), and ResidualAdd (57 launches, 120µs) are all small kernels with <10% bandwidth utilization. They exist because the current design treats each operation as an independent kernel.

**Target: zero standalone QuantizeRows, BiasAdd, and ResidualAdd kernels.**

### P4: Stream-through data flow
Each layer's output is the next layer's input. Currently, intermediate results bounce through global memory between kernels. The redesign should maximize L2 cache residency for inter-kernel data.

**Target: activation data stays in L2 between consecutive kernels within a layer.**

## 4. Redesigned Layer Pipeline

### 4.1 Current pipeline (11 kernels per layer)

```
[FusedRmsNormQuantQ8_1] → [MMVQ_QKV_triple] → [BiasAddTriple] →
[RoPE] → [KvAppend] → [FlashAttn2] →
[MMVQ_O_accum] →
[QuantQ8_1] → [FusedGateUpSiLU_MMVQ] → [MMVQ_Down_accum] →
[ResidualAddRmsNorm]
```

### 4.2 Proposed pipeline (6 kernels per layer)

```
[FusedNormQuantGemvBiasQKV] → [RoPEKvAppend] → [FlashAttn2] →
[FusedGemvAccumNormQuant_O] →
[FusedGateUpSiLU_MMVQ] → [FusedGemvAccumNormQuant_Down]
```

**Savings: 5 kernels per layer × 36 layers = 180 fewer launches per token.**
**New total: 6 × 36 + 5 (global) = 221 kernels per token.**

### 4.3 Kernel-by-kernel design

---

#### Kernel 1: `FusedNormQuantGemvBiasQKV`

**Fuses:** RmsNorm + QuantizeQ8_1 + MMVQ_Q + MMVQ_K + MMVQ_V + BiasAdd_Q + BiasAdd_K + BiasAdd_V

**Replaces:** 3-4 current kernels (FusedRmsNormQuantQ8_1 + MMVQ_QKV_triple + BiasAddTriple, or separate Q/K/V + separate BiasAdd)

**Design:**
- **Phase 1 (RmsNorm + Quantize):** Thread block processes one row of the residual stream. Compute RMS norm, multiply by weight, quantize to Q8_1 blocks — all in registers/shared memory. Output: Q8_1 activation blocks in shared memory (not global memory).
- **Phase 2 (MMVQ × 3 + Bias):** Each warp group reads Q4_K weight blocks for Q/K/V projections. Dot-product with Q8_1 activations from shared memory. After reduction, add bias in-register before writing to global memory.

**Key insight:** The Q8_1 activations never touch global memory. They flow from RmsNorm → shared memory → MMVQ dot products within the same thread block. This eliminates the QuantizeRows kernel AND the global memory round-trip.

**Thread/block layout:**
- Block: 256 threads (8 warps)
- Grid: `dim3(N_out, 1)` where N_out = total output columns across Q+K+V
- Warps 0-3: process weight blocks for Q/K/V (weight-read-first pattern)
- Warps 4-7: overlap with next weight block prefetch
- Shared memory: `hidden_size * sizeof(block_q8_1) / 32` ≈ 2048 * 36/32 = 2.3 KB for Q8_1 blocks + 256 * sizeof(float) for reduction

**Expected bandwidth:** 80-85% (single global memory pass over weights + single pass over activations, bias added in-register)

**Risk:** High shared memory pressure. If hidden_size > 4096, Q8_1 blocks may not fit in shared memory (4096 * 36/32 = 4.6 KB). Fallback to global memory Q8_1 path for large models.

**Fallback:** If shared memory is insufficient, split into:
- `FusedNormQuant` (writes Q8_1 to global, same as today)
- `FusedGemvBiasQKV` (reads Q8_1 from global, adds bias in epilogue)

---

#### Kernel 2: `FusedRoPEKvAppend`

**Fuses:** RoPE + KvAppend (currently 2 separate kernels)

**Replaces:** RoPEKernel + BatchedKvAppendStrided

**Design:**
- Each thread processes one (Q_re, Q_im) or (K_re, K_im) pair
- For K pairs: after rotation, scatter-write directly to KV cache slot
- For Q pairs: after rotation, write to Q output buffer (same as today)
- Grid: `dim3((num_heads * half_dim + num_kv_heads * half_dim + 255) / 256)`
- Block: 256 threads

**Key insight:** K values are currently rotated in-place, then a separate kernel copies them to the KV cache. By fusing, K values are rotated and written to their KV cache destination in a single pass — eliminating one global memory read+write cycle for K.

**Expected bandwidth:** Similar to current RoPE (compute-bound due to sin/cos), but eliminates the separate KvAppend kernel launch + K re-read.

**Savings:** 1 kernel launch per layer × 36 layers = 36 fewer launches. ~62µs GPU time savings.

---

#### Kernel 3: `FlashAttention2` (unchanged)

FlashAttention-2 is already highly optimized. No changes needed for B=1 decode. The 1.9% bandwidth utilization is inherent to single-token attention (seq_len=1 query against seq_len=N keys).

For batched decode (B>1), the existing `FlashDecodeMultiSeq` kernel handles multiple sequences efficiently.

---

#### Kernel 4: `FusedGemvAccumNormQuant_O`

**Fuses:** MMVQ_O + ResidualAdd + RmsNorm(post_attn) + QuantizeQ8_1

**Replaces:** MMVQ_O_accum + ResidualAddRmsNorm + QuantizeQ8_1 (the separate quantize before FFN)

**Design:**
- **Phase 1 (MMVQ O-projection):** Standard weight-read-first MMVQ for O projection. Accumulates partial sums across warps.
- **Phase 2 (Residual + Norm + Quantize epilogue):** After final warp reduction:
  1. Add O-projection output to residual: `residual[i] += o_proj[i]`
  2. Compute RMS over the updated residual (cross-thread reduction via shared memory)
  3. Apply post-attention norm weights: `normed[i] = residual[i] * rsqrt(rms) * weight[i]`
  4. Quantize normed output to Q8_1 blocks in shared memory or global memory

**Key insight:** The O-projection result, residual update, normalization, and activation quantization all happen on the same data. By keeping it in registers/shared memory through all four steps, we eliminate 3 global memory round-trips.

**Thread/block layout:**
- Same as current MMVQ_O_accum, but with epilogue logic after the reduction
- Block: 128 threads (4 warps), same weight-read-first pattern
- Grid: `dim3(hidden_size)` — one block per output element
- **Challenge:** RmsNorm requires a cross-element reduction (sum of squares across hidden_size). With one block per output element, this requires a global memory sync between phases.

**Resolution:** Two-phase approach:
1. Each block computes its partial sum-of-squares and writes to a small global buffer (hidden_size/block_size entries)
2. A single-block reduction kernel computes the final RMS value
3. Each block reads the RMS value and applies normalization + quantization

This adds 1 tiny reduction kernel but eliminates the standalone ResidualAdd + RmsNorm + QuantizeRows (3 kernels → 1+1 = 2 kernels, but the reduction kernel is ~1µs).

**Alternative:** If hidden_size ≤ 1024 (Qwen2.5-3B has 2048), a single thread block can process the entire row. Use `dim3(1)` grid with cooperative groups for the reduction. For hidden_size=2048, this requires 2048/128 = 16 warps = 512 threads — feasible on sm_89 (max 1024 threads/block).

**Preferred approach for hidden_size ≤ 4096:**
- Single block, 512-1024 threads
- Phase 1: Each thread handles hidden_size/threads elements of MMVQ
- Phase 2: Shared memory reduction for RMS
- Phase 3: Normalize + quantize in-register, write Q8_1 to global

**Expected bandwidth:** 80%+ (single pass over O-projection weights + single write of Q8_1 output)

---

#### Kernel 5: `FusedGateUpSiLU_MMVQ` (mostly unchanged)

The existing fused gate+up+SiLU MMVQ kernel is already well-designed:
- Reads gate and up weights in a single pass
- Computes SiLU(gate) * up in-register
- Writes fused result

**Proposed improvement:** Add Q8_1 quantization as an epilogue. Currently the output is FP16, and a separate QuantizeQ8_1 kernel runs before the down projection. By quantizing in the epilogue, we eliminate 1 kernel launch.

**New output:** Q8_1 blocks instead of FP16. The down projection MMVQ reads Q8_1 directly.

**Expected savings:** 1 kernel launch per layer (the QuantizeQ8_1 between gate+up+silu and down).

---

#### Kernel 6: `FusedGemvAccumNormQuant_Down`

**Fuses:** MMVQ_Down + ResidualAdd + RmsNorm(next_layer_input) + QuantizeQ8_1(next_layer)

**Identical structure to Kernel 4** but for the down projection:
- MMVQ down projection
- Accumulate to residual stream
- Compute next layer's input norm
- Quantize to Q8_1 for next layer's QKV projection

This is the inter-layer fusion kernel. For the last layer, skip the norm+quantize epilogue and write FP16 to the residual for the final LM head projection.

---

## 5. Kernel Count Summary

| Component | Current | Proposed | Savings |
|-----------|---------|----------|---------|
| Per-layer kernels | 11 | 6 | **5 per layer** |
| 36 layers total | 396 | 216 | **180 launches** |
| Global (embed+head) | 5 | 5 | 0 |
| Sampling | ~3 | ~3 | 0 |
| **Total per token** | **~500** | **~224** | **~276 fewer** |

On Windows WDDM, 276 fewer launches × ~20µs driver overhead = **~5.5ms saved per token**.

Combined with GPU compute improvements (higher bandwidth from fusion):
- GPU compute: 11.4ms → ~9.5ms (Q6_K bandwidth fix + epilogue fusion)
- WDDM overhead: 4.5ms → ~2.0ms (224 vs 500 launches)
- **Projected: ~11.5ms/token → ~76 tok/s** (vs current 59, vs llama.cpp 101)

## 6. Q6_K Bandwidth Optimization

The Q6_K GEMV kernel runs at only **62.1% bandwidth** vs Q4_K's 77.9%. This kernel handles norm weight projections (24 invocations, 2.07ms total).

### Root cause analysis

Q6_K has a more complex bit layout than Q4_K:
- Q4_K: 4-bit values packed 2 per byte, simple mask+shift extraction
- Q6_K: 6-bit values split across `ql` (low 4 bits) and `qh` (high 2 bits), requiring:
  1. Load ql (128 bytes per 256-element block)
  2. Load qh (32 bytes per 256-element block)
  3. Combine: `q = (ql & 0xF) | ((qh & 3) << 4)` — 3 bitwise ops per element
  4. Signed conversion: `q -= 32`

The extra load (qh) and bitwise recombination reduce the ratio of useful arithmetic to memory operations.

### Proposed fix

**Vectorized Q6_K extraction:** Load `ql` and `qh` as `uint4` (128-bit) vectors. Process 16 Q6_K values per vector operation instead of 1-2. Use `__byte_perm` for efficient bit manipulation.

**Pre-pack Q6_K weights at load time:** Convert Q6_K blocks to a layout where ql and qh are interleaved for sequential access. This eliminates the non-unit-stride `qh` read pattern.

**Expected improvement:** 62% → 78-82% bandwidth utilization, saving ~0.5ms/token.

## 7. Non-greedy Sampling Fix

The current `EnqueueSampleBatch` for non-greedy (temperature > 0) has a critical serialization bug at line 671-677 of `gpu_sampler.cu`:

```cpp
for (int i = 0; i < batch_size; ++i) {
  EnqueueSample(logits_i, temperatures[i], top_ks[i], top_ps[i], seed);
  h_result_batch_pinned_[i] = CollectSample();  // BLOCKING SYNC PER SEQUENCE
}
```

Each iteration blocks on `cudaEventSynchronize`. For B=8 with stochastic sampling, this adds 8 × ~0.3ms = 2.4ms per batch step.

### Fix

True async batched stochastic sampling:
1. Launch all B sampling kernels (softmax + top-p + multinomial) on the same stream
2. Use a single batched D2H memcpy for all B results
3. Single `cudaEventSynchronize` at the end

```cpp
// Proposed: async batch enqueue
for (int i = 0; i < batch_size; ++i) {
  EnqueueSampleAsync(logits_i, temperatures[i], top_ks[i], top_ps[i], seed);
}
// Single batched D2H + event
cudaMemcpyAsync(h_result_batch_pinned_, d_result_batch_,
                batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_);
cudaEventRecord(completion_event_, stream_);
completion_pending_ = true;
```

**Expected improvement:** For B=8 stochastic: 2.4ms → 0.3ms per batch step.

## 8. Implementation Priority

| Phase | Change | Expected Gain | Risk | Effort |
|-------|--------|---------------|------|--------|
| **P0** | Fix non-greedy sampling serialization | +15% concurrent | Low | 1 day |
| **P1** | `FusedRoPEKvAppend` | -36 launches, -62µs | Low | 2 days |
| **P2** | `FusedGemvAccumNormQuant` (O + Down) | -72 launches, -350µs + BW gain | Medium | 1 week |
| **P3** | `FusedNormQuantGemvBiasQKV` | -108 launches, -600µs | High | 2 weeks |
| **P4** | Q6_K vectorized extraction | +0.5ms bandwidth gain | Medium | 3 days |
| **P5** | Gate+Up+SiLU Q8_1 epilogue | -36 launches, -423µs | Low | 2 days |

### Projected cumulative throughput

| After Phase | Launches/token | GPU time | WDDM overhead | Total | tok/s |
|-------------|----------------|----------|---------------|-------|-------|
| Current | 500 | 11.4ms | 4.5ms | 16.9ms | 59 |
| P0 (sampling) | 500 | 11.4ms | 4.5ms | 16.9ms | 59 (single unchanged) |
| P1 (RoPE+KV) | 464 | 11.3ms | 4.2ms | 16.0ms | 63 |
| P2 (O+Down fuse) | 392 | 10.7ms | 3.5ms | 14.7ms | 68 |
| P3 (QKV fuse) | 284 | 10.1ms | 2.5ms | 13.1ms | 76 |
| P4 (Q6_K BW) | 284 | 9.6ms | 2.5ms | 12.6ms | 79 |
| P5 (SiLU Q8_1 epilogue) | 248 | 9.2ms | 2.2ms | 11.9ms | 84 |

**End state: 84 tok/s (0.83x llama.cpp)** — remaining gap is in llama.cpp's tighter ggml kernel integration and lower-level driver optimizations.

## 9. Validation Plan

Each phase must pass:
1. **Numerical parity:** `first_token_probe` logit comparison (top-5 match within 1e-3)
2. **Unit tests:** All 734 existing tests pass
3. **Throughput gate:** `benchmark.sh throughput-gate` — no regression
4. **ncu validation:** New kernel bandwidth ≥ target %
5. **Stress test:** 1000 sequential tokens, verify no memory leaks or drift

## 10. Architecture Invariants (DO NOT VIOLATE)

1. **Weight-read-first:** All GEMV kernels must iterate over weight blocks in the outer loop and batch rows in the inner loop. This amortizes weight cache misses across batch elements.
2. **Q8_1 activation format:** All fused kernels must produce/consume Q8_1 blocks (32-element, scale+sum) for activation data. This matches llama.cpp's runtime format and enables future cross-backend compatibility.
3. **CUDA graph compatibility:** No dynamic allocations, no host-visible side effects, no cuBLAS calls within graph-captured code paths. Guard cuBLAS fallbacks with `if (!capturing)`.
4. **Accumulate mode:** O-projection and down-projection must support direct accumulation to the residual stream (no intermediate buffer + separate ResidualAdd).
5. **Dispatch thresholds:** Fused kernels must only be selected when M ≤ threshold (batch size small enough that bandwidth savings from quantized weights exceed cuBLAS compute advantage).
