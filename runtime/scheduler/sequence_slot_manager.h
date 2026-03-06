#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <vector>
#include <string>

namespace inferflux {
namespace scheduler {

/**
 * @brief State of a sequence slot in the KV cache
 */
enum class SequenceState {
  kIdle,       // Available for new requests
  kPrefilling,  // Processing prompt (prefill phase)
  kDecoding,   // Generating tokens (decode phase)
  kCompleted,  // Finished, waiting for cleanup
  kEvicted     // Timed out and evicted
};

/**
 * @brief Represents a single KV cache slot for one sequence
 */
struct SequenceSlot {
  int slot_id;                           // Slot number (0 to max_slots-1)
  int64_t request_id;                     // Associated request ID
  int sequence_id;                       // Backend sequence ID
  SequenceState state;
  std::chrono::steady_clock::time_point last_access;
  std::chrono::steady_clock::time_point acquired_at;
  int token_count;                       // Tokens processed in this sequence

  /**
   * @brief Check if slot has been idle longer than timeout
   * @param timeout Idle timeout threshold
   * @return true if slot is idle (not actively generating or past timeout)
   */
  bool IsIdle(std::chrono::milliseconds timeout) const {
    if (state != SequenceState::kDecoding) return true;  // Not actively generating
    auto idle = std::chrono::steady_clock::now() - last_access;
    return idle > timeout;
  }
};

/**
 * @brief Manages KV cache slots for both cuda_native and cuda_llama_cpp backends
 *
 * This provides universal sequence slot management at the orchestration layer,
 * above both backend implementations. Each backend maintains its own private
 * KV cache internally, but slot acquisition/release is coordinated here.
 *
 * Key benefits:
 * - Configurable slot count (increase from 16 to 128+ for production)
 * - Timeout-based eviction to prevent slot exhaustion
 * - Statistics and metrics for monitoring
 * - Thread-safe slot allocation
 */
class SequenceSlotManager {
public:
  /**
   * @brief Construct slot manager
   * @param max_slots Maximum number of concurrent sequences (default: 128)
   */
  explicit SequenceSlotManager(size_t max_slots = 128);

  /**
   * @brief Acquire a slot for a request
   * @param request_id Request ID for tracking
   * @return Slot ID if available, nullopt if all slots in use
   */
  std::optional<int> AcquireSlot(int64_t request_id);

  /**
   * @brief Release a slot when sequence completes
   * @param slot_id Slot ID to release
   */
  void ReleaseSlot(int slot_id);

  /**
   * @brief Evict idle sequences based on timeout
   * @param timeout Idle timeout threshold
   * @return Vector of evicted slot IDs with their sequence IDs
   */
  std::vector<std::pair<int, int>> EvictIdleSlots(std::chrono::milliseconds timeout);

  /**
   * @brief Mark slot as actively processing (update last_access time)
   * @param slot_id Slot ID to mark as processing
   */
  void MarkProcessing(int slot_id);

  /**
   * @brief Update token count for a slot
   * @param slot_id Slot ID
   * @param token_count Total tokens processed
   */
  void UpdateTokenCount(int slot_id, int token_count);

  /**
   * @brief Get number of currently used slots
   * @return Used slot count
   */
  size_t GetUsedSlotCount() const;

  /**
   * @brief Get number of available slots
   * @return Free slot count
   */
  size_t GetFreeSlotCount() const;

  /**
   * @brief Get status of all slots
   * @return Vector of slot status copies
   */
  std::vector<SequenceSlot> GetSlotStatus() const;

  /**
   * @brief Set maximum slot count (requires reinitialization)
   * @param max_slots New maximum slot count
   */
  void SetMaxSlots(size_t max_slots);

  /**
   * @brief Set idle timeout for eviction
   * @param timeout Timeout in milliseconds
   */
  void SetIdleTimeout(std::chrono::milliseconds timeout);

  /**
   * @brief Get current idle timeout
   * @return Idle timeout in milliseconds
   */
  std::chrono::milliseconds GetIdleTimeout() const { return idle_timeout_; }

  /**
   * @brief Get maximum slot count
   * @return Maximum slots
   */
  size_t GetMaxSlots() const { return max_slots_; }

  /**
   * @brief Check if we can accept a new request based on memory pressure
   * @return true if there's sufficient memory, false if we should reject/degrade
   */
  bool CanAcceptRequest() const;

  /**
   * @brief Get current memory pressure status
   * @return Memory pressure as percentage (0-100), or -1 if unavailable
   */
  int GetMemoryPressure() const;

  /**
   * @brief Perform graceful degradation if under memory pressure
   * @return true if degradation was performed, false otherwise
   */
  bool PerformGracefulDegradation();

private:
  mutable std::shared_mutex mutex_;
  std::vector<SequenceSlot> slots_;
  size_t max_slots_;
  std::chrono::milliseconds idle_timeout_{300000};  // 5 minutes default

  /**
   * @brief Find available slot
   * @return Slot ID if found, nullopt if all slots in use
   */
  std::optional<int> FindFreeSlot();

  /**
   * @brief Find slot by request ID
   * @param request_id Request ID to find
   * @return Slot ID if found, nullopt if not found
   */
  std::optional<int> FindSlotByRequestId(int64_t request_id);

  /**
   * @brief Initialize all slots to idle state
   */
  void InitializeSlots();

  /**
   * @brief Internal eviction (assumes lock already held)
   * @param timeout Idle timeout threshold
   * @return Vector of evicted slot IDs with their sequence IDs
   */
  std::vector<std::pair<int, int>> EvictIdleSlotsLocked(std::chrono::milliseconds timeout);

  /**
   * @brief Internal free slot count (assumes lock already held)
   * @return Free slot count
   */
  size_t GetFreeSlotCountLocked() const;

  /**
   * @brief Internal used slot count (assumes lock already held)
   * @return Used slot count
   */
  size_t GetUsedSlotCountLocked() const;
};

} // namespace scheduler
} // namespace inferflux
