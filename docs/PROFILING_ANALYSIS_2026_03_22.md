# Profiling Analysis: Fused Kernel Redesign (2026-03-22)

## Setup
- GPU: NVIDIA RTX 4000 Ada (sm_89, 20GB, 360 GB/s)
- Model: Qwen2.5-3B Q4_K_M (2GB weights)
- Platform: Windows 11 WDDM
- Build: Release, CUDA 13.2, P1+P2 fused kernels enabled

## Decode-Only Throughput (corrected measurement)

| Backend | tok/s (decode-only) | Method |
|---------|-------------------|--------|
| inferflux_cuda | 65.4 | 64→128 token delta |
| llama_cpp_cuda | 128.4 | 64→128 token delta |
| **Ratio** | **0.51x** | |

## Per-Token GPU Time Breakdown (nsys, 16 tokens)

| Category | Time | % of Decode | Calls/token |
|----------|------|-------------|-------------|
| MMVQ GEMV (compute) | 1.95ms | 83.6% | ~41 |
| Q8_1 Quantize | 0.12ms | 5.3% | 18 |
| Sampling (argmax) | 0.09ms | 3.8% | 1 |
| Norm+Quantize (P2) | 0.09ms | 3.8% | ~27 |
| Attention (FlashDecode) | 0.05ms | 2.3% | 9 |
| Utility (RoPE+KV, Bias) | 0.03ms | 1.1% | ~19 |
| **Total GPU compute** | **2.33ms** | **100%** | |

## The Bottleneck: WDDM Scheduling Latency

### Token-by-token wall clock
- Tokens 1-3 (warmup, non-graph): 11-16ms, 455 kernels each
- Tokens 4-16 (graph replay): **10-14ms**, only 1 visible kernel (argmax)
- GPU busy per graph-replay token: **~2ms**
- GPU idle per token: **~10-12ms (82% of wall time)**

### Inter-token gap breakdown
| Component | Time |
|-----------|------|
| cudaGraphLaunch API | 280us |
| WDDM scheduling delay | **8,000-12,000us** |
| GPU forward pass (graph) | ~2,000us |
| Sampler (argmax + D2H) | ~100us |
| **Total per-token** | **~10,500us** |

**82% of decode time is GPU idle**, waiting for WDDM to schedule work.

## ncu: Hottest Kernel (mmvq_q4k_fused_gate_up_silu, 43% of GPU compute)

| Metric | Value | Implication |
|--------|-------|-------------|
| Duration | 110us | |
| Memory Throughput | 231 GB/s | 64% of 360 GB/s peak |
| DRAM Throughput | 67.7% peak | Weight reads |
| Compute Throughput | 65.1% peak | Dual-bottleneck |
| Registers/Thread | **52** | Limits occupancy to 75% |
| Achieved Occupancy | 70.8% | Register-limited |
| L1 Hit Rate | 81% | Good weight caching |
| L2 Hit Rate | 2.1% | Expected (weights >> L2) |
| No Eligible Warps | **46.2%** | Can't hide memory latency |
| Active Threads/Warp | 24.1/32 | 25% divergence |
| Branch Efficiency | 80% | Divergent break/if conditions |

## Optimization Priority

### P0: WDDM scheduling latency (82% of wall time)
- **Impact**: 2-5x decode throughput
- **Options**:
  1. Device-side decode loop (keep token on GPU, only sync for EOS/stop)
  2. Multi-token graph (generate N tokens per graph, 1 sync per N)
  3. `cudaLaunchHostFunc` callback instead of event polling
  4. On Linux TCC: problem disappears (~10us vs ~10ms latency)

### P1: Register pressure (52 → ≤40 regs)
- **Impact**: +15-20% GPU kernel throughput
- **Fix**: `__launch_bounds__`, reduce locals, merge accumulators
- **Moves occupancy**: 75% → 100%, warps/scheduler: 0.95 → 1.5+

### P2: Thread divergence (25% wasted compute)
- **Impact**: +10% effective throughput
- **Fix**: Predicated zero instead of `if (row >= M) break`

### P3: Eliminate 607KB logits D2H (when logprobs not requested)
- **Impact**: -94us/token, reduces WDDM command pressure
- **Fix**: Skip CopyLogitsToHost in non-logprob decode path
