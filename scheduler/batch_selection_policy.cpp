#include "scheduler/batch_selection_policy.h"

#include "scheduler/scheduler.h"

namespace inferflux {

namespace {

// Every 2 seconds of queue wait adds +1 to effective priority.
constexpr double kFairnessAgingDivisorMs = 2000.0;

class PriorityAgeBatchPolicy final : public IBatchSelectionPolicy {
public:
  double Score(const ScoringItem &item) const override {
    return static_cast<double>(item.priority) +
           item.age_ms / kFairnessAgingDivisorMs;
  }
  bool UsesPrefixAffinity() const override { return false; }
  const char *Name() const override { return "priority_age"; }
};

class LpmPriorityBatchPolicy final : public IBatchSelectionPolicy {
public:
  double Score(const ScoringItem &item) const override {
    return static_cast<double>(item.priority) +
           item.age_ms / kFairnessAgingDivisorMs +
           item.prefix_affinity_tokens / 32.0;
  }
  bool UsesPrefixAffinity() const override { return true; }
  const char *Name() const override { return "lpm_priority"; }
};

class ThroughputBalancedBatchPolicy final : public IBatchSelectionPolicy {
public:
  double Score(const ScoringItem &item) const override {
    return static_cast<double>(item.priority) +
           item.age_ms / kFairnessAgingDivisorMs +
           item.prefix_affinity_tokens / 64.0;
  }
  bool UsesPrefixAffinity() const override { return true; }
  const char *Name() const override { return "throughput_balanced"; }
};

} // namespace

std::unique_ptr<IBatchSelectionPolicy>
CreateBatchSelectionPolicy(SchedulerBatchPolicy policy) {
  switch (policy) {
  case SchedulerBatchPolicy::kLpmPriority:
    return std::make_unique<LpmPriorityBatchPolicy>();
  case SchedulerBatchPolicy::kThroughputBalanced:
    return std::make_unique<ThroughputBalancedBatchPolicy>();
  case SchedulerBatchPolicy::kPriorityAge:
  default:
    return std::make_unique<PriorityAgeBatchPolicy>();
  }
}

} // namespace inferflux
