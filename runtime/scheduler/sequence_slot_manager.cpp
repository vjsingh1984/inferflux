#include "runtime/scheduler/sequence_slot_manager.h"
#include "server/logging/logger.h"
#include <algorithm>
#include <numeric>

namespace inferflux {
namespace scheduler {

SequenceSlotManager::SequenceSlotManager(size_t max_slots)
    : max_slots_(max_slots) {
  InitializeSlots();
  log::Info("slot_manager", "Initialized SequenceSlotManager with " +
            std::to_string(max_slots_) + " slots (timeout: " +
            std::to_string(idle_timeout_.count()) + "ms)");
}

void SequenceSlotManager::InitializeSlots() {
  slots_.resize(max_slots_);

  for (size_t i = 0; i < max_slots_; ++i) {
    slots_[i].slot_id = static_cast<int>(i);
    slots_[i].state = SequenceState::kIdle;
    slots_[i].request_id = -1;
    slots_[i].sequence_id = -1;
    slots_[i].token_count = 0;
    slots_[i].last_access = std::chrono::steady_clock::now();
    slots_[i].acquired_at = std::chrono::steady_clock::now();
  }
}

std::optional<int> SequenceSlotManager::AcquireSlot(int64_t request_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  // Try to evict idle slots first (internal version, assumes lock held)
  EvictIdleSlotsLocked(idle_timeout_);

  // Find free slot
  auto slot = FindFreeSlot();
  if (!slot) {
    log::Warn("slot_manager", "No free slots available (request " +
              std::to_string(request_id) + ", used: " +
              std::to_string(GetUsedSlotCountLocked()) + "/" +
              std::to_string(max_slots_) + ")");
    return std::nullopt;
  }

  // Mark as acquired (start in prefill state)
  auto now = std::chrono::steady_clock::now();
  slots_[*slot].state = SequenceState::kPrefilling;
  slots_[*slot].request_id = request_id;
  slots_[*slot].sequence_id = -1;  // Will be set by backend
  slots_[*slot].acquired_at = now;
  slots_[*slot].last_access = now;
  slots_[*slot].token_count = 0;

  log::Debug("slot_manager", "Acquired slot " + std::to_string(*slot) +
             " for request " + std::to_string(request_id) +
             " (free slots: " + std::to_string(GetFreeSlotCountLocked()) + ")");

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
             " (request " + std::to_string(slot.request_id) +
             ", tokens: " + std::to_string(slot.token_count) + ")");

  // Mark as idle
  slot.state = SequenceState::kIdle;
  slot.request_id = -1;
  slot.sequence_id = -1;
  slot.token_count = 0;
}

// Internal version - assumes lock is already held
std::vector<std::pair<int, int>> SequenceSlotManager::EvictIdleSlotsLocked(std::chrono::milliseconds timeout) {
  // Note: mutex_ should already be locked by caller
  std::vector<std::pair<int, int>> evicted_slots;

  for (auto &slot : slots_) {
    if (slot.state == SequenceState::kDecoding && slot.IsIdle(timeout)) {
      log::Info("slot_manager", "Evicting idle slot " + std::to_string(slot.slot_id) +
                " (request " + std::to_string(slot.request_id) +
                ", seq_id " + std::to_string(slot.sequence_id) +
                ", idle for >" + std::to_string(timeout.count()) + "ms, " +
                std::to_string(slot.token_count) + " tokens)");

      // Record evicted slot with its sequence ID for backend cleanup
      evicted_slots.push_back({slot.slot_id, slot.sequence_id});

      // Mark as evicted - will be reused after backend cleanup
      slot.state = SequenceState::kEvicted;
    }
  }

  if (!evicted_slots.empty()) {
    log::Info("slot_manager", "Evicted " + std::to_string(evicted_slots.size()) +
              " idle slots (timeout: " + std::to_string(timeout.count()) + "ms)");
  }

  return evicted_slots;
}

// Public version - acquires lock
std::vector<std::pair<int, int>> SequenceSlotManager::EvictIdleSlots(std::chrono::milliseconds timeout) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return EvictIdleSlotsLocked(timeout);
}

void SequenceSlotManager::MarkProcessing(int slot_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (slot_id < 0 || slot_id >= static_cast<int>(max_slots_)) {
    return;
  }

  slots_[slot_id].last_access = std::chrono::steady_clock::now();

  // Transition from prefill to decode if needed
  if (slots_[slot_id].state == SequenceState::kPrefilling) {
    slots_[slot_id].state = SequenceState::kDecoding;
  }
}

void SequenceSlotManager::UpdateTokenCount(int slot_id, int token_count) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (slot_id < 0 || slot_id >= static_cast<int>(max_slots_)) {
    return;
  }

  slots_[slot_id].token_count = token_count;
}

size_t SequenceSlotManager::GetUsedSlotCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  return std::count_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state == SequenceState::kPrefilling ||
                                s.state == SequenceState::kDecoding;
                       });
}

size_t SequenceSlotManager::GetFreeSlotCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  return std::count_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state == SequenceState::kIdle ||
                                s.state == SequenceState::kEvicted;
                       });
}

std::vector<SequenceSlot> SequenceSlotManager::GetSlotStatus() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  // Return copies of slot status
  return slots_;
}

std::optional<int> SequenceSlotManager::FindFreeSlot() {
  // Note: mutex_ should already be locked by caller
  auto it = std::find_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state == SequenceState::kIdle ||
                                s.state == SequenceState::kEvicted;
                       });

  if (it != slots_.end()) {
    return static_cast<int>(it - slots_.begin());
  }

  return std::nullopt;
}

std::optional<int> SequenceSlotManager::FindSlotByRequestId(int64_t request_id) {
  // Note: mutex_ should already be locked by caller
  auto it = std::find_if(slots_.begin(), slots_.end(),
                       [request_id](const SequenceSlot &s) {
                         return s.request_id == request_id &&
                                (s.state == SequenceState::kPrefilling ||
                                 s.state == SequenceState::kDecoding);
                       });

  if (it != slots_.end()) {
    return static_cast<int>(it - slots_.begin());
  }

  return std::nullopt;
}

void SequenceSlotManager::SetMaxSlots(size_t max_slots) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (max_slots == max_slots_) {
    return;  // No change
  }

  log::Info("slot_manager", "Resizing slots: " + std::to_string(max_slots_) +
            " -> " + std::to_string(max_slots));

  max_slots_ = max_slots;
  InitializeSlots();
}

void SequenceSlotManager::SetIdleTimeout(std::chrono::milliseconds timeout) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  idle_timeout_ = timeout;

  log::Info("slot_manager", "Idle timeout set to " +
            std::to_string(timeout.count()) + "ms");
}

size_t SequenceSlotManager::GetFreeSlotCountLocked() const {
  // Note: mutex_ should already be locked by caller
  return std::count_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state == SequenceState::kIdle ||
                                s.state == SequenceState::kEvicted;
                       });
}

size_t SequenceSlotManager::GetUsedSlotCountLocked() const {
  // Note: mutex_ should already be locked by caller
  return std::count_if(slots_.begin(), slots_.end(),
                       [](const SequenceSlot &s) {
                         return s.state == SequenceState::kPrefilling ||
                                s.state == SequenceState::kDecoding;
                       });
}

} // namespace scheduler
} // namespace inferflux
