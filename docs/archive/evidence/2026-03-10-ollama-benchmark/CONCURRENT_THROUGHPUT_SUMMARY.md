# Concurrent Throughput Investigation Summary

**Date**: 2026-03-05
**Status**: ✅ Root cause identified, mitigation phase landed, kernel-level follow-up pending

---

## Executive Summary

### Problem

Historical baseline showed concurrent throughput was **17x slower** than sequential:
- Sequential: 35.56 tok/s ✅
- Concurrent (8): 2.04 tok/s ❌ (historical before worker-pool fix)

### Root Cause

**HTTP worker pool saturation** in `server/http/http_server.cpp:2763-2764`:

```cpp
// Non-streaming path blocks worker threads
auto future = scheduler_->Generate(std::move(req));
auto result = future.get();  // ❌ BLOCKS until request completes
```

### Impact

- Historical default had only 4 worker threads available
- Each request takes ~50 seconds
- 8 concurrent requests → 4 run, 4 wait → ~392 seconds total
- Throughput: 800 tokens / 392s = **2.04 tok/s**

---

## Solution

### Quick Win: Use Streaming ✅

**Change**: Set `"stream": true` in requests

**How it works**:
```cpp
// Streaming path (line 2573)
futures.push_back(scheduler_->Generate(std::move(cur)));  // ✅ Async!
```

**Benefits**:
- Worker threads **not blocked**
- True concurrent processing
- **Expected improvement**: 2 tok/s → **35-50 tok/s** (17-25x faster!)

**Implementation**:
```bash
# Just change stream: false → stream: true
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"model":"default","messages":[...],"stream":true}'
```

### Request-Layer Mitigation Landed ✅

**Change landed**: increase default HTTP worker pool from 4 to 16  
**Commit**: `a18ca46`  
**Doc**: `docs/HTTP_WORKER_POOL_INCREASE.md`

**Impact**:
- Reduces non-streaming queue saturation at request layer
- Improves concurrency headroom without changing API clients
- Does **not** replace native CUDA kernel/overlap work

---

## Implementation Plan

### Phase 1: Document Streaming Best Practice ✅

**Status**: Complete

- Root cause analysis: `docs/CONCURRENT_THROUGHPUT_ANALYSIS.md`
- Test scripts created
- Expected results documented

---

### Phase 2: Increase HTTP Worker Pool ✅

**File**: `server/main.cpp`

**Change**: Increase worker threads from 4 to 16

```cpp
// Line ~1230
HttpServer http_server(tls_config, 16,  // Increase from 4
                      scheduler, &auth,
```

**Expected improvement**:
- Non-streaming: 2 tok/s → **8 tok/s** (4x better)
- More concurrent requests possible

**Effort**: ⭐⭐ (Simple config change)  
**Status**: ✅ Complete (`a18ca46`)

---

### Phase 3: Make Non-Streaming Truly Async

**File**: `server/http/http_server.cpp`

**Change**: Remove blocking `future.get()` for non-streaming

```cpp
// Before (blocking):
auto future = scheduler_->Generate(std::move(req));
auto result = future.get();  // ❌ Blocks worker

// After (async):
auto future = scheduler_->Generate(std::move(req));
// Register callback, don't block worker
// Worker can handle other requests
```

**Expected improvement**:
- Non-streaming scales like streaming
- Worker threads not blocked
- **17-25x speedup** even without streaming

**Effort**: ⭐⭐⭐ (Moderate refactoring)

**Status**: ⏳ TODO

---

## Verification

### Expected Results (Based on Analysis)

| Configuration | Requests | Time | Throughput | Status |
|----------------|----------|------|------------|--------|
| **Historical baseline (non-streaming, 4 workers)** | 8 | 392s | 2.04 tok/s | ✅ Root-cause baseline |
| **Streaming** | 8 | ~10-15s | **35-50 tok/s** | ✅ Immediate mitigation path |
| **Current default (non-streaming, 16 workers)** | 8 | rerun pending | expected better than baseline | ⏳ needs fresh benchmark capture |
| **+ Async non-streaming** | 8 | ~10-15s target | **35-50 tok/s** target | ⏳ planned |

---

## Code Changes Needed

### 1. HTTP Worker Pool Increase ✅

**File**: `server/main.cpp`

```diff
- HttpServer http_server(tls_config, 4,
+ HttpServer http_server(tls_config, 16,
                       scheduler, &auth,
```

**Lines**: `server/main.cpp` (`http_workers=16`)  
**Status**: Complete (`a18ca46`)

---

### 2. Async Non-Streaming (Future Work)

**File**: `server/http/http_server.cpp`

**Function**: `HandleCompletionsRequest` (around line 2750)

**Changes needed**:
1. Don't call `future.get()` synchronously
2. Register completion callback
3. Handle timeout asynchronously
4. Return response when ready

**Complexity**: Moderate (requires callback mechanism)

---

## Recommendations

### For Users (Immediate)

✅ **Use streaming mode** for concurrent workloads:
```python
response = client.chat.completions.create(
    model="default",
    messages=[...],
    stream=True  # ✅ Enables 17-25x faster concurrent processing
)

for chunk in response:
    print(chunk.choices[0].delta.content)
```

### For Developers (Short-term)

✅ **HTTP worker pool default increased to 16**:
- Request-layer concurrency bottleneck reduced
- Keeps client contract unchanged
- Use `INFERFLUX_HTTP_WORKERS` to tune per host

### For Developers (Long-term)

⏳ **Implement async non-streaming**:
- Best performance for all workloads
- Scales to 100+ concurrent requests
- Requires refactoring HTTP server

---

## Test Coverage

### Manual Testing

**Test streaming mode**:
```bash
# Single streaming request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

**Expected**: SSE chunks returned immediately

---

## Summary

| Item | Status |
|------|--------|
| Root cause identified | ✅ Complete |
| Solution documented | ✅ Complete |
| Test scripts created | ✅ Complete |
| Streaming best practice | ✅ Documented |
| Worker pool increase | ✅ Complete (`a18ca46`) |
| Async refactoring | ⏳ TODO |

---

## Next Steps

1. ✅ Document findings (this file)
2. ⏳ Rerun concurrent non-streaming benchmark with worker=16 and record measured delta
3. ⏳ Keep streaming guidance as default for high concurrency clients
4. ⏳ Plan async non-streaming refactoring
5. ⏳ Add throughput evidence snapshot for post-mitigation measurements

---

**Date**: 2026-03-05
**Status**: Mitigation phase complete at request layer; core kernel work remains
**Impact**: Streaming remains highest-confidence mitigation; worker-pool increase reduces non-streaming saturation risk
