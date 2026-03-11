# Native KV Device Addressing

## Goal

Remove the remaining host-built KV pointer slabs from native CUDA decode so
batched append and read paths derive addresses on device from compact metadata.

This is the next throughput step after:

- restoring the native Q4_K hot paths
- collapsing batch metadata H2D copies into one slab
- collapsing KV pointer H2D copies from 4 uploads to 2 uploads
- separating startup from steady-state `nsys` profiling

## Current State

Native decode still performs a host-side pointer build in
[transformer_forward.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/native/transformer_forward.cu):

- host loops over `layer x batch`
- fills `h_all_k_append_ptrs_`, `h_all_v_append_ptrs_`
- fills `h_all_k_read_ptrs_`, `h_all_v_read_ptrs_`
- uploads two pointer slabs per batch

The backing KV layout is already regular in
[kv_cache_gpu.cu](/home/vsingh/code/inferflux/runtime/backends/cuda/native/kv_cache_gpu.cu):

- `GetK(layer, seq_id) = buffer + seq_id * slot_stride + layer * layer_stride`
- `GetV(layer, seq_id) = buffer + seq_id * slot_stride + layer * layer_stride + kv_stride`

So the pointer slabs are not encoding sparse topology. They are encoding simple
address arithmetic that the device can compute directly.

## Why This Matters

Steady-state `nsys` profiling shows startup allocator churn is no longer the
main problem. The remaining native-vs-llama gap is now a real decode hot-path
gap, and `cudaMemcpyAsync` is still dominated by runtime metadata traffic.

That means the next safe win is not another host-side slab compression. It is
removing the host-built pointer slabs entirely.

## Safe First Slice

Do this in two phases, not one.

### Phase A: Device-side append addressing

Scope:

- keep KV storage layout unchanged
- keep FlashDecode read path unchanged
- remove only append pointer slab generation/upload

Implementation:

1. Extend native batched KV append to accept:
   - base KV buffer
   - `slot_stride`
   - `layer_stride`
   - `kv_stride`
   - `seq_ids`
   - `n_past`
   - `layer`
2. Compute `k_dst` and `v_dst` inside the append kernel:
   - `k_base = buffer + seq_id * slot_stride + layer * layer_stride`
   - `v_base = k_base + kv_stride`
   - `dst = base + n_past * kv_dim`
3. Delete the append host slab and its H2D upload.

Why first:

- append path is simpler than FlashDecode
- destination rows are written once and consumed later
- correctness is easy to verify against the existing path

### Phase B: Device-side read addressing for FlashDecode

Scope:

- keep the same KV storage layout
- stop uploading read pointer slabs

Implementation:

1. Modify `FlashDecodeMultiSeq` to accept layout/stride metadata plus:
   - `seq_ids`
   - `kv_lens`
   - `layer`
2. Compute `k_rd` and `v_rd` inside the kernel from the same address formula.
3. Remove the remaining host read slab and its H2D upload.

Why second:

- FlashDecode is the more sensitive path
- it touches read addressing on every decode step
- it should be changed only after append-side parity is proven

## Constraints

- no KV storage layout change in this slice
- no paged-KV ABI change in this slice
- no session/slot ownership change in this slice
- no per-request CUDA allocation
- preserve CUDA graph safety for fixed device addresses

## Required Contracts

Before rollout:

1. Unit parity:
   - append path writes identical KV rows vs current pointer-slab path
   - FlashDecode outputs match baseline for the same logits/token sequence
2. Integration:
   - short native vs llama.cpp benchmark keeps current accuracy envelope
3. Profiling:
   - steady-state native `cudaMemcpyAsync` count drops again
   - no new steady-state `cudaMalloc/cudaFree`

## Success Metric

The steady-state profiler should move from:

- host-built pointer slabs
- two KV pointer H2D uploads per batch

to:

- metadata-only batch upload
- zero KV pointer H2D uploads per batch

That is the cleanest next step toward exceeding llama.cpp on native decode
throughput without changing model math or destabilizing batching.
