#include "runtime/scheduler/sequence_slot_manager.h"
#include <catch2/catch_amalgamated.hpp>
#include <chrono>
#include <thread>

namespace inferflux {
namespace scheduler {

TEST_CASE("SequenceSlotManager basic operations", "[slot_manager]") {
  SequenceSlotManager manager(16);

  SECTION("Initial state") {
    REQUIRE(manager.GetMaxSlots() == 16);
    REQUIRE(manager.GetUsedSlotCount() == 0);
    REQUIRE(manager.GetFreeSlotCount() == 16);
  }

  SECTION("Acquire and release slots") {
    auto slot1 = manager.AcquireSlot(100);
    REQUIRE(slot1.has_value());
    REQUIRE(*slot1 >= 0);
    REQUIRE(*slot1 < 16);
    REQUIRE(manager.GetUsedSlotCount() == 1);
    REQUIRE(manager.GetFreeSlotCount() == 15);

    auto slot2 = manager.AcquireSlot(101);
    REQUIRE(slot2.has_value());
    REQUIRE(*slot2 != *slot1); // Different slots
    REQUIRE(manager.GetUsedSlotCount() == 2);

    manager.ReleaseSlot(*slot1);
    REQUIRE(manager.GetUsedSlotCount() == 1);
    REQUIRE(manager.GetFreeSlotCount() == 15);

    manager.ReleaseSlot(*slot2);
    REQUIRE(manager.GetUsedSlotCount() == 0);
    REQUIRE(manager.GetFreeSlotCount() == 16);
  }

  SECTION("Slot exhaustion") {
    std::vector<int> acquired_slots;

    // Acquire all 16 slots
    for (int i = 0; i < 16; i++) {
      auto slot = manager.AcquireSlot(200 + i);
      REQUIRE(slot.has_value());
      acquired_slots.push_back(*slot);
    }

    REQUIRE(manager.GetUsedSlotCount() == 16);
    REQUIRE(manager.GetFreeSlotCount() == 0);

    // Next acquire should fail
    auto slot = manager.AcquireSlot(999);
    REQUIRE(!slot.has_value());

    // Release one slot
    manager.ReleaseSlot(acquired_slots[0]);
    REQUIRE(manager.GetUsedSlotCount() == 15);
    REQUIRE(manager.GetFreeSlotCount() == 1);

    // Now acquire should succeed
    auto new_slot = manager.AcquireSlot(1000);
    REQUIRE(new_slot.has_value());
  }
}

TEST_CASE("SequenceSlotManager timeout-based eviction", "[slot_manager]") {
  SequenceSlotManager manager(16);

  // Set short timeout for testing (100ms)
  manager.SetIdleTimeout(std::chrono::milliseconds(100));

  // Acquire a slot
  auto slot = manager.AcquireSlot(100);
  REQUIRE(slot.has_value());

  // Mark as processing (transitions to Decoding state)
  manager.MarkProcessing(*slot);

  // Wait longer than timeout
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  // Run eviction - should evict the idle slot
  auto evicted = manager.EvictIdleSlots(std::chrono::milliseconds(100));
  REQUIRE(evicted.size() == 1);
  REQUIRE(evicted[0].first == *slot); // Evicted slot ID matches

  // Slot should now be available
  REQUIRE(manager.GetUsedSlotCount() == 0);
}

TEST_CASE("SequenceSlotManager slot status", "[slot_manager]") {
  SequenceSlotManager manager(4);

  auto slot = manager.AcquireSlot(100);
  REQUIRE(slot.has_value());

  auto status = manager.GetSlotStatus();
  REQUIRE(status.size() == 4);

  // Find the acquired slot
  bool found = false;
  for (const auto &s : status) {
    if (s.slot_id == *slot) {
      REQUIRE(s.request_id == 100);
      REQUIRE(s.state == SequenceState::kPrefilling);
      found = true;
      break;
    }
  }
  REQUIRE(found);
}

} // namespace scheduler
} // namespace inferflux
