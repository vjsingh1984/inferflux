#include "scheduler/fairness_controller.h"
#include "scheduler/request_batch.h"
#include <algorithm>

namespace inferflux {

FairnessController::FairnessController() = default;

FairnessController::FairnessController(FairnessConfig config)
    : config_(config) {}

void FairnessController::UpdateConfig(const FairnessConfig &config) {
  config_ = config;
}

FairnessDecision
FairnessController::Evaluate(const std::vector<FairnessEntry> &batch,
                             const std::vector<FairnessEntry> &queue) const {
  FairnessDecision decision;
  if (!config_.enable_preemption || batch.empty() || queue.empty()) {
    return decision;
  }

  auto high_it =
      std::max_element(queue.begin(), queue.end(),
                       [](const FairnessEntry &a, const FairnessEntry &b) {
                         return a.priority_level < b.priority_level;
                       });
  if (high_it == queue.end() ||
      high_it->priority_level < config_.high_priority_threshold) {
    return decision;
  }

  auto low_it =
      std::min_element(batch.begin(), batch.end(),
                       [](const FairnessEntry &a, const FairnessEntry &b) {
                         return a.priority_level < b.priority_level;
                       });
  if (low_it == batch.end()) {
    return decision;
  }
  if (low_it->priority_level >= high_it->priority_level) {
    return decision;
  }

  decision.swap = true;
  decision.batch_index =
      static_cast<std::size_t>(std::distance(batch.begin(), low_it));
  decision.queue_index =
      static_cast<std::size_t>(std::distance(queue.begin(), high_it));
  return decision;
}

void FairnessController::ApplyTimeslice(
    std::vector<FairnessEntry> *batch) const {
  if (!batch || config_.max_timeslice_tokens <= 0) {
    return;
  }
  for (auto &entry : *batch) {
    if (!entry.request) {
      continue;
    }
    if (entry.priority_level >= config_.high_priority_threshold) {
      continue;
    }
    int limit = config_.max_timeslice_tokens;
    if (entry.request->remaining_decode_tokens > 0) {
      limit = std::min(limit, entry.request->remaining_decode_tokens);
    }
    entry.request->timeslice_tokens = limit;
  }
}

} // namespace inferflux
