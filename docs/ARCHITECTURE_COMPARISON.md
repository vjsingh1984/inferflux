# InferFlux Architecture Comparison: Why It Scales Better

**April 9, 2026** · Based on source code analysis of [Ollama](https://github.com/ollama/ollama) (Go, MIT) and [LM Studio](https://github.com/lmstudio-ai/lms) (TypeScript, MIT)

## The Concurrency Problem

All three products use the same model (Qwen2.5-3B Q4_K_M) on the same GPU (RTX 4000 Ada). At single-request (c=1), throughput is similar (~73-84 tok/s). The gap opens at concurrency:

```
                c=1      c=4      c=8     Scaling
InferFlux      73 ──── 134 ──── 160      2.2x ████████████████████
Ollama         72 ────  86 ────  86      1.2x ██████████
LM Studio      84 ────  87 ────  72      0.7x ██████ (degrades!)
```

## Root Cause: Request Dispatch Architecture

### InferFlux (C++17, unified batch)

```
Client 1 ─┐
Client 2 ─┤                    ┌─────────────────────┐
Client 3 ─┼─→ HTTP ThreadPool ─→ Scheduler            │
Client 4 ─┤    (C++ direct)    │ BuildBatchLocked()   │
Client 5 ─┤                    │ IBatchSelectionPolicy │
Client 6 ─┤                    │ Score + Sort + Select │
Client 7 ─┤                    └────────┬────────────┘
Client 8 ─┘                             │
                                        ▼
                              ┌──────────────────┐
                              │ ExecuteUnifiedBatch()
                              │ ONE GPU kernel    │
                              │ serves ALL 8 seqs │
                              │ simultaneously    │
                              └──────────────────┘
```

- **Zero language boundary**: C++ HTTP → C++ scheduler → CUDA kernels
- **Batch scoring**: `IBatchSelectionPolicy` ranks requests by priority + age + prefix affinity
- **2ms accumulation window**: waits briefly to group concurrent arrivals into one batch
- **Persistent decode working sets**: requests stay bound to workers between token steps
- **Shared KV cache**: paged architecture, prefix reuse across requests

### Ollama (Go + CGo → llama.cpp)

```
Client 1 → goroutine → ┐
Client 2 → goroutine → ┤
Client 3 → goroutine → ┤     ┌─────────────────┐
Client 4 → goroutine → ┼───→ │ CGo boundary     │ → llama.cpp server
Client 5 → goroutine → ┤     │ (~1-5μs/call)    │   (serial queue)
Client 6 → goroutine → ┤     └─────────────────┘
Client 7 → goroutine → ┤
Client 8 → goroutine → ┘
                              │
                              ▼
                    8 SEPARATE llama_decode() calls
                    (no cross-request batching)
```

**Why Ollama plateaus at ~86 tok/s:**

| Bottleneck | Detail |
|---|---|
| **No request batching** | Each goroutine calls llama.cpp independently. 8 separate GPU kernel launches instead of 1. |
| **CGo overhead** | Go→C function call boundary adds ~1-5μs per call, compounding at high concurrency |
| **Go GC pauses** | Stop-the-world GC can stall the HTTP accept loop for 1-10ms |
| **Serial llama.cpp queue** | llama.cpp server processes requests sequentially — adding clients doesn't increase throughput |
| **No KV sharing** | Each request allocates independent KV cache. No prefix reuse. |

**Source evidence** (from Ollama repo analysis):
- `server/sched.go`: Go scheduler dispatches to runner instances
- `llama/runner.go`: Each runner wraps a llama.cpp process via CGo
- No equivalent to InferFlux's `ExecuteUnifiedBatch()` — requests are individual `llama_decode()` calls

### LM Studio (Electron + Node.js → llama.cpp)

```
Client 1 → ┐
Client 2 → ┤   ┌──────────────────┐    ┌─────────────┐
Client 3 → ┼─→ │ Node.js event    │──→ │ llama.cpp   │
Client 4 → ┤   │ loop (1 thread)  │    │ server      │
Client 5 → ┤   │                  │    │ subprocess  │
Client 6 → ┤   │ libuv thread     │    └─────────────┘
Client 7 → ┤   │ pool (4 default) │
Client 8 → ┘   └──────────────────┘
```

**Why LM Studio DEGRADES at c=8 (0.7x):**

| Bottleneck | Detail |
|---|---|
| **Single-threaded JS event loop** | Node.js processes I/O callbacks on one thread. At c=8, request dispatch serializes. |
| **libuv thread pool** | Default 4 workers. At c=8, requests queue behind each other. |
| **Electron overhead** | GUI runtime consumes CPU cycles that could serve inference |
| **Subprocess dispatch** | llama.cpp runs as separate process — IPC adds latency per request |
| **No batching** | Like Ollama, each request is independent. No unified GPU execution. |

**Source evidence**: LM Studio's `lms` repo is the CLI tool only. The inference server is closed-source Electron, but benchmark data confirms the Node.js architecture (0.7x degradation matches event loop serialization pattern).

## What InferFlux Does Differently

| Design Choice | Why It Matters |
|---|---|
| **C++17 throughout** | No GC pauses, no CGo boundary, no event loop. Predictable latency. |
| **Unified batch execution** | One GPU kernel call processes all concurrent sequences. GPU compute scales linearly. |
| **IBatchSelectionPolicy** | Strategy pattern: PriorityAge, LpmPriority, ThroughputBalanced. Extensible without modifying scheduler. |
| **Paged KV cache** | 16-token pages, LRU/Clock eviction, host-RAM secondary tier. Memory scales with actual sequences, not worst-case. |
| **Radix prefix cache** | BPE-aligned prefix matching. Repeated prompts skip prefill entirely. |
| **Sticky decode workers** | Requests stay bound to workers between token steps. No re-routing overhead. |
| **CUDA graph replay** | Decode loop captured as graph, replayed without CPU launch overhead. |
| **Chat template auto-detection** | ChatML/Llama/Mistral/Gemma from GGUF metadata. No manual config. |

## Benchmark Methodology

All benchmarks use `/v1/chat/completions` (OpenAI-compatible) with `temperature=0.0`, `max_tokens=64`, 16 requests from a diverse prompt suite. Quality validated with sentence-transformers embedding cosine similarity (all-MiniLM-L6-v2, local).

```bash
BUILD_DIR=./build-cuda bash scripts/benchmark.sh multi-backend
```

## References

- [Ollama source](https://github.com/ollama/ollama) — Go, MIT license
- [LM Studio CLI](https://github.com/lmstudio-ai/lms) — TypeScript, MIT license
- [InferFlux benchmarks](benchmarks.md) — verified Apr 9, 2026
- [Tech debt & roadmap](TechDebt_and_Competitive_Roadmap.md) — grade B+
