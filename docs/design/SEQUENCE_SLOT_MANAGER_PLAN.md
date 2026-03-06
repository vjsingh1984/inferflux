# KV Cache Slot Manager Implementation Plan

**Goal**: Universal sequence slot management for both cuda_native and cuda_llama_cpp backends

## Phase 1: Create SequenceSlotManager (Week 1)

### Files to Create

**File**: `runtime/scheduler/sequence_slot_manager.h`

```cpp
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace inferflux {
namespace scheduler {

enum class SequenceState {
  kIdle,       // Available
  kPrefilling,  // Processing prompt
  kDecoding,   // Generating tokens
  kCompleted,  // Finished, waiting for cleanup
  kEvicted     // Timed out
};

struct SequenceSlot {
  int slot_id;                           // Slot number
  int64_t request_id;                     // Associated request
  int sequence_id;                       // Backend sequence ID
  SequenceState state;
  std::chrono::steady_clock::time_point last_access;
  std::chrono::steady_clock::time_point acquired_at;
  int token_count;                       // Tokens processed

  // For timeout-based eviction
  bool IsIdle(std::chrono::milliseconds timeout) const {
    if (state != kDecoding) return true;  // Not actively generating
    auto idle = std::chrono::steady_clock::now() - last_access;
    return idle > timeout;
  }
};

class SequenceSlotManager {
public:
  explicit SequenceSlotManager(size_t max_slots = 128);

  // Acquire a slot for a request
  std::optional<int> AcquireSlot(int64_t request_id);

  // Release a slot when sequence completes
  void ReleaseSlot(int slot_id);

  // Evict idle sequences based on timeout
  size_t EvictIdleSlots(std::chrono::milliseconds timeout);

  // Mark slot as actively processing
  void MarkProcessing(int slot_id);

  // Get statistics
  size_t GetUsedSlotCount() const;
  size_t GetFreeSlotCount() const;
  std::vector<SequenceSlot> GetSlotStatus() const;

  // Configuration
  void SetMaxSlots(size_t max_slots);
  void SetIdleTimeout(std::chrono::milliseconds timeout);

private:
  std::vector<SequenceSlot> slots_;
  mutable std::shared_mutex mutex_;
  size_t max_slots_;
  std::chrono::milliseconds idle_timeout_{300000};  // 5 minutes default

  // Find available slot
  std::optional<int> FindFreeSlot();

  // Find specific slot by request_id
  std::optional<int> FindSlotByRequestId(int64_t request_id);
};

} // namespace scheduler
} // namespace inferflux
```

**File**: `runtime/scheduler/sequence_slot_manager.cpp`

```cpp
#include "runtime/scheduler/sequence_slot_manager.h"
#include "server/logging/logger.h"
#include <algorithm>

namespace inferflux {
namespace scheduler {

SequenceSlotManager::SequenceSlotManager(size_t max_slots)
    : max_slots_(max_slots), slots_(max_slots) {

  // Initialize all slots as idle
  for (size_t i = 0; i < max_slots_; ++i) {
    slots_[i].slot_id = static_cast<int>(i);
    slots_[i].state = SequenceState::kIdle;
    slots_[i].request_id = -1;
    slots_[i].sequence_id = -1;
  }

  log::Info("slot_manager", "Initialized with " + std::to_string(max_slots_) + " slots");
}

std::optional<int> SequenceSlotManager::AcquireSlot(int64_t request_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  // Try to evict idle slots first
  EvictIdleSlots(idle_timeout_);

  // Find free slot
  auto slot = FindFreeSlot();
  if (!slot) {
    log::Warn("slot_manager", "No free slots available (request " +
             std::to_string(request_id) + ")");
    return std::nullopt;
  }

  // Mark as acquired
  slots_[*slot].state = SequenceState::kPrefilling;
  slots_[*slot].request_id = request_id;
  slots_[*slot].acquired_at = std::chrono::steady_clock::now();
  slots_[*slot].last_access = slots_[*slot].acquired_at;
  slots_[*slot].token_count = 0;

  log::Debug("slot_manager", "Acquired slot " + std::to_string(*slot) +
             " for request " + std::to_string(request_id) +
             " (free slots: " + std::to_string(GetFreeSlotCount()) + ")");

  return slot;
}

void SequenceSlotManager::ReleaseSlot(int slot_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (slot_id < 0 || slot_id >= static_cast<int>(max_slots_)) {
    log::Error("slot_manager", "Invalid slot_id: " + std::to_string(slot_id));
    return;
  }

  auto &slot = slots_[slot_id];

  log::Debug("slot_manager", "Releasing slot " + std::to_string(slot_id) +
             " (request " + std::to_string(slot.request_id) + ")");

  // Mark as idle
  slot.state = SequenceState::kIdle;
  slot.request_id = -1;
  slot.sequence_id = -1;
  slot.token_count = 0;
}

size_t SequenceSlotManager::EvictIdleSlots(std::chrono::milliseconds timeout) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  size_t evicted = 0;

  for (auto &slot : slots_) {
    if (slot.state == SequenceState::kDecoding && slot.IsIdle(timeout)) {
      log::Info("slot_manager", "Evicting idle slot " + std::to_string(slot.slot_id) +
                " (idle for " + std::to_string(timeout.count()) + "ms)");

      // Call backend to free the sequence
      // NOTE: This is where we'd call backend->FreeSequence(slot.sequence_id)
      slot.state = SequenceState::kEvicted;
      evicted++;
    }
  }

  if (evicted > 0) {
    log::Info("slot_manager", "Evicted " + std::to_string(evicted) + " idle slots");
  }

  return evicted;
}

size_t SequenceSlotManager::GetUsedSlotCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  return std::count_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state != SequenceState::kIdle &&
                                s.state != SequenceState::kEvicted;
                       });
}

std::optional<int> SequenceSlotManager::FindFreeSlot() {
  auto it = std::find_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state == SequenceState::kIdle ||
                                s.state == SequenceState::kEvicted;
                       });

  if (it != slots_.end()) {
    return it - slots_.begin();
  }

  return std::nullopt;
}

size_t SequenceSlotManager::GetFreeSlotCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  return std::count_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state == SequenceState::kIdle ||
                                s.state == SequenceState::kEvicted;
                       });
}

} // namespace scheduler
} // namespace inferflux
```

## Phase 2: Integrate with Scheduler (Week 2)

**File**: `scheduler/scheduler.cpp`

```cpp
// In Scheduler::ProcessRequest()

// Before processing request:
auto slot = slot_manager_->AcquireSlot(request->id);
if (!slot) {
  return ErrorResponse("All slots in use (max: 128)");
}

// Associate slot with request
request->slot_id = *slot;

// Call backend (same for both cuda_native and cuda_llama_cpp)
backend->Prefill(..., request->slot_id);

// After request completes:
slot_manager_->ReleaseSlot(request->slot_id);
backend->FreeSequence(request->slot_id);
```

## Phase 3: Add Timeout-Based Eviction (Week 3)

Add periodic eviction:

```cpp
class Scheduler {
  ...
  void StartEvictionThread() {
    eviction_thread_ = std::thread([this]() {
      while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        slot_manager_->EvictIdleSlots(std::chrono::minutes(5));
      }
    });
  }
};
```

## Phase 4: Metrics and Monitoring (Week 4)

```cpp
// Expose metrics:
// - inferflux_kv_slots_used_total
// - inferflux_kv_slots_free_total
// - inferflux_kv_slots_evicted_total
// - inferflux_kv_slot_utilization_percent
```

## Phase 5: Configuration (Week 4)

```yaml
# config/server.yaml
runtime:
  # Slot manager configuration
  max_parallel_sequences: 128        # Max slots (was: 16)

  # Timeout-based eviction
  sequence_idle_timeout: 300000     # 5 minutes (ms)

  # Paged KV cache (separate system, for model offloading)
  paged_kv:
    cpu_pages: 512
    gpu_pages: 32
```

## Summary

| Aspect | Approach |
|--------|----------|
| **llama.cpp KV cache** | Keep private (can't be shared) |
| **InferFlux PagedKVCache** | Use for model offloading (different layer) |
| **SequenceSlotManager** | NEW: Universal slot orchestration for both backends |
| **Configuration** | `max_parallel_sequences: 128` |
| **Eviction** | Timeout-based (5 min idle) |

**Key Principle**: Separation of concerns at different architectural layers.
