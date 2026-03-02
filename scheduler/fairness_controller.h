#pragma once

#include <cstddef>
#include <vector>

namespace inferflux {

struct InferenceRequest;

struct FairnessConfig {
  int high_priority_threshold{5};
  int max_timeslice_tokens{0};
  bool enable_preemption{false};
};

struct FairnessEntry {
  InferenceRequest *request{nullptr};
  int priority_level{0};
  std::size_t queue_index{0};
};

struct FairnessDecision {
  bool swap{false};
  std::size_t batch_index{0};
  std::size_t queue_index{0};
};

class FairnessController {
public:
  FairnessController();
  explicit FairnessController(FairnessConfig config);

  void UpdateConfig(const FairnessConfig &config);
  FairnessDecision Evaluate(const std::vector<FairnessEntry> &batch,
                            const std::vector<FairnessEntry> &queue) const;
  void ApplyTimeslice(std::vector<FairnessEntry> *batch) const;

private:
  FairnessConfig config_;
};

} // namespace inferflux
