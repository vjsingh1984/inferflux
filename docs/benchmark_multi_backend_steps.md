# Multi-backend Benchmark Checklist

This doc captures the repeatable steps and instrumentation that keep `run_gguf_comparison_benchmark.sh` trustworthy while the native row-pair kernels run alongside `llama.cpp`.

## 1. Flags and defaults
* `INFERFLUX_ENABLE_EXPERIMENTAL_Q8_1_GROUPED_ROWPAIR_W4` now defaults to `true` in `NativeExecutionPolicy`. The benchmark still accepts the environment variable to opt-out for controlled experiments, but day-to-day runs are now stacked on the row-pair path.
* Keep `INFERFLUX_ENABLE_BATCHED_DECODE=1` in the benchmark so multi-row decode batches naturally occur and exercise the row-pair operator per the metrics below.

## 2. Run order and reset hook
The benchmark runs `cuda_native` first, then `cuda_llama_cpp`. To avoid CUDA state leaking between the two:
1. The script already calls `stop_server cuda_native` once the native run is done.
2. We added `reset_cuda_device()` which issues `cudaDeviceReset()` (via `libcudart`) immediately after native shutdown. That ensures the GPU context is fully torn down before theama.cpp start.
3. Only then does the script launch `cuda_llama_cpp`; the 3-second sleep after the reset gives the GPU a final breathing room.

If you ever replicate the benchmark manually, follow the same order: stop native, reset the CUDA device (via `cudaDeviceReset()` or `./build-cuda/inferfluxd --reset-cuda` if available), then start the llama.cpp backend. This guarantees accurate throughput isolation for regression comparisons.

## 3. Metrics to validate row-pair activity
* Inspect `inferflux_native_rowpair_selection_total{phase="decode",operator="q8_1_group_row_pair_w4",bucket="2"}` and `...operator="q8_1_gemv_row_pair"` in the resulting `metrics_cuda_native.txt`. Successful runs record counts (>0) in bucket `2` or `3_4`, proving the specialized operators handled the multi-row batches.
* The benchmark also captures `inferflux_native_ffn_proj_operator_total` and `inferflux_native_down_proj_operator_total` summaries (written to `native_ffn_proj_summary_cuda_native.json` and `native_operator_summary_cuda_native.json`) so you can correlate which kernels were chosen.

## 4. Accuracy safeguards
* The existing similarity report (`similarity.json`) continues to compare paired requests for exact match / Jaccard / overlap. A stable run should remain ≥0.95 overlap and keep exact matches in the high single digits while the throughput benefit holds.
* Keep `INFERFLUX_DEBUG_OPERATOR_SELECTION=0`/`INFERFLUX_DEBUG_LOGITS=0` for normal benchmarks; enable them only for debugging because they add logging noise.

## 5. Release-note checklist
When promoting the row-pair flag for release:
* Update client-facing docs (this file) and point to the new metric so operators can verify row-pair usage.
* Mention that `cuda_llama_cpp` now runs against a clean GPU thanks to the reset hook—this avoids the sporadic `socket: Operation not permitted` issues that plagued earlier runs.
* Leave the instrumentation (metrics_capture hooks in the benchmark) so any regression gate re-running this benchmark automatically records operator breakdown, row-pair counters, and similarity data.
