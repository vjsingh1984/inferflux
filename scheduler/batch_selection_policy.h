#pragma once

#include <memory>

namespace inferflux {

enum class SchedulerBatchPolicy;

// Input data for scoring a single pending request.
struct ScoringItem {
  double age_ms{0.0};                 // time in queue (milliseconds)
  int priority{0};                    // request priority (higher = more urgent)
  uint64_t sequence{0};               // monotonic arrival order (tiebreaker)
  double prefix_affinity_tokens{0.0}; // prefix cache match length
};

// Strategy interface for batch selection scoring.
//
// The Scheduler builds a list of ScoringItems from its pending queues,
// calls Score() on each, sorts by descending score (sequence as tiebreaker),
// then selects up to max_batch_size/max_batch_tokens requests.
//
// Implementations control only the scoring formula. Queue management,
// prefix affinity probing, token budget enforcement, and swap-in remain
// in the Scheduler.
class IBatchSelectionPolicy {
public:
  virtual ~IBatchSelectionPolicy() = default;

  // Compute effective priority for a pending request.
  // Higher scores are selected first.
  virtual double Score(const ScoringItem &item) const = 0;

  // Whether this policy uses prefix affinity data in scoring.
  // When false, the scheduler skips prefix cache probing (saves work).
  virtual bool UsesPrefixAffinity() const = 0;

  // Human-readable name for metrics and logging.
  virtual const char *Name() const = 0;
};

// Factory: create the policy matching the given enum value.
std::unique_ptr<IBatchSelectionPolicy>
CreateBatchSelectionPolicy(SchedulerBatchPolicy policy);

} // namespace inferflux
