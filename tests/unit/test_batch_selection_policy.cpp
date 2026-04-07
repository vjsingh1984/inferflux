#include <catch2/catch_amalgamated.hpp>

#include "scheduler/batch_selection_policy.h"
#include "scheduler/scheduler.h"

#include <algorithm>
#include <vector>

using namespace inferflux;

TEST_CASE("Factory creates correct policy types",
          "[batch_selection_policy]") {
  auto pa = CreateBatchSelectionPolicy(SchedulerBatchPolicy::kPriorityAge);
  REQUIRE(std::string(pa->Name()) == "priority_age");

  auto lpm = CreateBatchSelectionPolicy(SchedulerBatchPolicy::kLpmPriority);
  REQUIRE(std::string(lpm->Name()) == "lpm_priority");

  auto tb =
      CreateBatchSelectionPolicy(SchedulerBatchPolicy::kThroughputBalanced);
  REQUIRE(std::string(tb->Name()) == "throughput_balanced");
}

TEST_CASE("PriorityAge ignores prefix affinity",
          "[batch_selection_policy]") {
  auto policy =
      CreateBatchSelectionPolicy(SchedulerBatchPolicy::kPriorityAge);
  REQUIRE_FALSE(policy->UsesPrefixAffinity());

  ScoringItem without{/*age_ms=*/1000.0, /*priority=*/3, /*sequence=*/1,
                      /*prefix_affinity_tokens=*/0.0};
  ScoringItem with{/*age_ms=*/1000.0, /*priority=*/3, /*sequence=*/2,
                   /*prefix_affinity_tokens=*/128.0};
  REQUIRE(policy->Score(without) == policy->Score(with));
}

TEST_CASE("LpmPriority boosts by affinity/32",
          "[batch_selection_policy]") {
  auto policy =
      CreateBatchSelectionPolicy(SchedulerBatchPolicy::kLpmPriority);
  REQUIRE(policy->UsesPrefixAffinity());

  ScoringItem base{1000.0, 3, 1, 0.0};
  ScoringItem with_affinity{1000.0, 3, 2, 64.0};

  double delta = policy->Score(with_affinity) - policy->Score(base);
  REQUIRE(delta == Catch::Approx(64.0 / 32.0));
}

TEST_CASE("ThroughputBalanced boosts by affinity/64",
          "[batch_selection_policy]") {
  auto policy =
      CreateBatchSelectionPolicy(SchedulerBatchPolicy::kThroughputBalanced);
  REQUIRE(policy->UsesPrefixAffinity());

  ScoringItem base{1000.0, 3, 1, 0.0};
  ScoringItem with_affinity{1000.0, 3, 2, 128.0};

  double delta = policy->Score(with_affinity) - policy->Score(base);
  REQUIRE(delta == Catch::Approx(128.0 / 64.0));
}

TEST_CASE("Age contribution is consistent across policies",
          "[batch_selection_policy]") {
  ScoringItem item{4000.0, 0, 1, 0.0}; // 4s age, no priority, no affinity

  auto pa = CreateBatchSelectionPolicy(SchedulerBatchPolicy::kPriorityAge);
  auto lpm = CreateBatchSelectionPolicy(SchedulerBatchPolicy::kLpmPriority);
  auto tb =
      CreateBatchSelectionPolicy(SchedulerBatchPolicy::kThroughputBalanced);

  // 4000ms / 2000.0 = 2.0
  REQUIRE(pa->Score(item) == Catch::Approx(2.0));
  REQUIRE(lpm->Score(item) == Catch::Approx(2.0));
  REQUIRE(tb->Score(item) == Catch::Approx(2.0));
}

TEST_CASE("Priority adds directly to score", "[batch_selection_policy]") {
  auto policy =
      CreateBatchSelectionPolicy(SchedulerBatchPolicy::kPriorityAge);

  ScoringItem item{0.0, 5, 1, 0.0}; // no age, priority=5
  REQUIRE(policy->Score(item) == Catch::Approx(5.0));
}

TEST_CASE("Sort by Score produces correct ordering",
          "[batch_selection_policy]") {
  auto policy =
      CreateBatchSelectionPolicy(SchedulerBatchPolicy::kLpmPriority);

  // Three items: low-priority old, high-priority new, medium with affinity
  ScoringItem old_low{6000.0, 1, 10, 0.0};     // 1 + 3.0 = 4.0
  ScoringItem new_high{0.0, 5, 20, 0.0};        // 5 + 0.0 = 5.0
  ScoringItem mid_affinity{2000.0, 2, 30, 96.0}; // 2 + 1.0 + 3.0 = 6.0

  std::vector<ScoringItem> items = {old_low, new_high, mid_affinity};
  std::stable_sort(items.begin(), items.end(),
                   [&](const ScoringItem &a, const ScoringItem &b) {
                     double sa = policy->Score(a);
                     double sb = policy->Score(b);
                     if (sa != sb)
                       return sa > sb;
                     return a.sequence < b.sequence;
                   });

  // mid_affinity (6.0) > new_high (5.0) > old_low (4.0)
  REQUIRE(items[0].sequence == 30);
  REQUIRE(items[1].sequence == 20);
  REQUIRE(items[2].sequence == 10);
}
