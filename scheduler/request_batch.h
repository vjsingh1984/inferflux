#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "runtime/multimodal/image_preprocessor.h"
#include "runtime/structured_output/structured_constraint.h"

namespace inferflux {

// Phase of an inference request in the continuous batching pipeline.
enum class RequestPhase {
  kPending,  // Queued, not yet scheduled.
  kPrefill,  // In prefill (prompt processing).
  kDecode,   // In decode (token generation).
  kFinished, // Generation complete.
  kAborted,  // Cancelled or timed out.
};

// InferenceRequest holds per-request state for the scheduler.
// It replaces the legacy GenerateRequest DTO with a richer structure that
// supports continuous batching, priority scheduling, and per-request
// accounting.
struct InferenceRequest {
  uint64_t id{0};
  std::string model;          // Requested model ID (empty = default).
  std::string resolved_model; // Assigned model ID after router resolution.
  std::string prompt;
  int max_tokens{256};
  int priority{0};         // Higher = more urgent. 0 = default.
  int priority_level{0};   // Cached priority level (for fairness controller).
  int service_tokens{0};   // Tokens consumed so far (fairness accounting).
  int timeslice_tokens{0}; // Timeslice cap when fairness preemption is enabled.
  int remaining_decode_tokens{-1}; // Remaining tokens after fairness slice.
  int reported_prompt_tokens{
      -1}; // Prompt tokens counted once for metrics/logs.
  int total_completion_tokens{
      0}; // Completion tokens accumulated across slices.
  int last_timeslice_tokens{
      0}; // Last applied fairness slice limit (observability).
  bool fairness_yielded{
      false}; // True when the last slice yielded for fairness.
  bool json_mode{false};
  // Phased prefill/decode state (§2.5 Option A).
  // Set by Scheduler::ProcessBatch after calling LlamaCPUBackend::Prefill().
  // BatchExecutor::ExecuteRequest() calls Decode() when n_past >= 0 instead of
  // Generate().
  int n_past{-1}; // KV position after prefill; -1 = use legacy Generate() path.
  int sequence_id{-1};    // KV cache sequence slot; -1 = unassigned.
  bool has_images{false}; // §2.2: true when request contains image_url parts.
  bool stream{false};
  bool cancelled{false};
  RequestPhase phase{RequestPhase::kPending};
  bool has_response_format{false};
  std::string response_format_type;    // "json_object", "grammar", etc.
  std::string response_format_schema;  // Raw JSON schema string (if provided).
  std::string response_format_grammar; // Resolved GBNF string.
  std::string response_format_root{"root"};
  bool response_format_ready{
      false}; // True once grammar string compiled/resolved.
  bool response_format_supported{true};
  std::string response_format_error;
  StructuredConstraint response_constraint;

  std::vector<DecodedImage>
      images; // §2.2: decoded images (parallel to <__media__> markers).

  // Token state (populated during scheduling).
  std::vector<int> prompt_tokens;
  std::vector<int> output_tokens;

  // Timing for SLO tracking.
  std::chrono::steady_clock::time_point enqueue_time;
  std::chrono::steady_clock::time_point first_token_time;

  // W3C trace-id propagated from the incoming HTTP traceparent header.
  // Empty string if no traceparent was present in the request.
  std::string trace_id;

  // Callback for streaming: invoked with each generated token string.
  // Null for non-streaming requests.
  std::function<void(const std::string &)> on_token;
  std::string
      accumulated_output; // Aggregated completion text across fairness slices.

  // Shared cancellation flag toggled when the HTTP connection closes.
  std::shared_ptr<std::atomic<bool>> cancellation_flag;
  int kv_page{-1};

  // Logprob collection (OpenAI logprobs API).
  // logprob_top_n == 0 means disabled; > 0 means collect selected-token logprob
  // and the top-N alternatives at each decode step.
  int logprob_top_n{0};
};

// RequestBatch groups requests that will execute together in a single
// forward pass. The scheduler fills batches from the pending queue,
// respecting memory limits and priority order.
//
// This abstraction is the foundation for:
//   - Continuous batching (mix prefill + decode in one batch)
//   - Preemption (evict low-priority requests to make room)
//   - Token accounting (track total tokens in flight)
//   - Disaggregated prefill/decode (tag batches by phase)
struct RequestBatch {
  uint64_t batch_id{0};
  std::vector<InferenceRequest *> requests; // Non-owning pointers.

  // Capacity tracking.
  int total_tokens() const {
    int sum = 0;
    for (const auto *req : requests) {
      sum += static_cast<int>(req->prompt_tokens.size());
      sum += static_cast<int>(req->output_tokens.size());
    }
    return sum;
  }

  bool empty() const { return requests.empty(); }
  std::size_t size() const { return requests.size(); }
};

} // namespace inferflux
