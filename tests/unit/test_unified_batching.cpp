#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/execution/batch_executor.h"
#include "scheduler/single_model_router.h"
#include <catch2/catch_amalgamated.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace inferflux;

// Mock backend to intercept ExecuteUnifiedBatch calls.
class MockUnifiedBackend : public LlamaCPUBackend {
public:
  struct CallInfo {
    std::vector<UnifiedBatchInput> inputs;
  };
  std::vector<CallInfo> calls;
  int token_capacity{512};

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    calls.push_back({inputs});
    std::vector<UnifiedBatchOutput> outputs;
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      if (in.request_logits) {
        out.token = 42; // Always return token 42
        out.piece = "x";
      }
      outputs.push_back(out);
    }
    return outputs;
  }

  bool IsReady() const override { return true; }
  int TokenCount(const std::string &) const override { return 5; }
  int UnifiedBatchTokenCapacity() const override { return token_capacity; }
};

class MockAsyncUnifiedBackend : public LlamaCPUBackend {
public:
  enum class EventType {
    kSubmitDecode,
    kSubmitPrefill,
    kSubmitAuto,
    kCollect,
  };

  struct SubmitCall {
    UnifiedBatchLane lane{UnifiedBatchLane::kAuto};
    std::vector<UnifiedBatchInput> inputs;
  };

  bool supports_async{true};
  int token_capacity{512};
  std::vector<SubmitCall> submit_calls;
  std::vector<EventType> events;
  UnifiedBatchHandle next_handle{1};
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      pending_outputs;

  bool SupportsAsyncUnifiedBatch() const override { return supports_async; }
  int UnifiedBatchTokenCapacity() const override { return token_capacity; }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    submit_calls.push_back({lane, inputs});
    if (lane == UnifiedBatchLane::kDecode) {
      events.push_back(EventType::kSubmitDecode);
    } else if (lane == UnifiedBatchLane::kPrefill) {
      events.push_back(EventType::kSubmitPrefill);
    } else {
      events.push_back(EventType::kSubmitAuto);
    }
    std::vector<UnifiedBatchOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      if (in.request_logits) {
        out.token = 42;
        out.piece = "x";
      }
      outputs.push_back(out);
    }
    const auto handle = next_handle++;
    pending_outputs.emplace(handle, std::move(outputs));
    return handle;
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    events.push_back(EventType::kCollect);
    if (!outputs) {
      return false;
    }
    auto it = pending_outputs.find(handle);
    if (it == pending_outputs.end()) {
      return false;
    }
    *outputs = std::move(it->second);
    pending_outputs.erase(it);
    return true;
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    std::vector<UnifiedBatchOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      if (in.request_logits) {
        out.token = 42;
        out.piece = "x";
      }
      outputs.push_back(out);
    }
    return outputs;
  }

  bool IsReady() const override { return true; }
  int TokenCount(const std::string &) const override { return 5; }
};

TEST_CASE("BatchExecutor: Unified Batching & Chunked Prefill",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache =
      std::make_shared<RadixPrefixCache>(cache, [](int) {}, 1024, 12);
  auto router = std::make_shared<SingleModelRouter>();
  auto executor = std::make_unique<BatchExecutor>(&tokenizer, device, cache,
                                                  router, nullptr);

  auto mock_backend = std::make_shared<MockUnifiedBackend>();

  SECTION("Chunked Prefill for a large prompt") {
    InferenceRequest req;
    req.model = "mock";
    req.phase = RequestPhase::kPrefill;
    req.n_past = 0;      // Required for eligibility
    req.sequence_id = 0; // Required for eligibility
    // Large prompt: 1200 tokens (will take 3 chunks of 512, 512, 176)
    req.bpe_prompt_tokens.assign(1200, 1);
    req.max_tokens = 5;
    req.block_table = {1, 2, 3};

    RequestBatch batch;
    batch.requests = {&req};

    auto results = executor->ExecuteBatch(batch, {mock_backend});

    // kMaxPrefillChunkSize is 512.
    // 1200 tokens -> 512 (chunk 0), 512 (chunk 1), 176 (chunk 2 + first token
    // sampled). Then 4 more tokens to reach max_tokens=5. Total steps in while
    // loop: 3 (prefill) + 4 (remaining decode) = 7 steps.
    REQUIRE(mock_backend->calls.size() == 7);

    // First call: 512 tokens, request_logits=false
    REQUIRE(mock_backend->calls[0].inputs[0].tokens.size() == 512);
    REQUIRE_FALSE(mock_backend->calls[0].inputs[0].request_logits);

    // Third call: 176 tokens, request_logits=true (final chunk)
    REQUIRE(mock_backend->calls[2].inputs[0].tokens.size() == 176);
    REQUIRE(mock_backend->calls[2].inputs[0].request_logits);

    REQUIRE(req.phase == RequestPhase::kFinished);
    REQUIRE(req.total_completion_tokens == 5);
  }

  SECTION("Interleaved Prefill and Decode") {
    InferenceRequest req_prefill;
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 0;
    req_prefill.sequence_id = 1;
    req_prefill.bpe_prompt_tokens = {1, 2, 3};
    req_prefill.max_tokens = 5;
    req_prefill.block_table = {4};

    InferenceRequest req_decode;
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 10;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 5;
    req_decode.sequence_id = 2;
    req_decode.block_table = {5};

    RequestBatch batch;
    batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteBatch(batch, {mock_backend, mock_backend});

    // Step 1: prefill (chunk 0), decode (step 0) -> 2 inputs
    // Step 2: prefill-now-decode (step 1), decode (step 1) -> 2 inputs
    REQUIRE(mock_backend->calls.size() >= 2);
    REQUIRE(mock_backend->calls[0].inputs.size() == 2);
    REQUIRE(mock_backend->calls[1].inputs.size() == 2);
  }

  SECTION("Adaptive prefill chunking respects backend token capacity") {
    auto bounded_backend = std::make_shared<MockUnifiedBackend>();
    bounded_backend->token_capacity = 8;

    InferenceRequest req_a;
    req_a.model = "mock";
    req_a.phase = RequestPhase::kPrefill;
    req_a.n_past = 0;
    req_a.sequence_id = 21;
    req_a.bpe_prompt_tokens.assign(10, 1);
    req_a.max_tokens = 1;
    req_a.block_table = {21};

    InferenceRequest req_b;
    req_b.model = "mock";
    req_b.phase = RequestPhase::kPrefill;
    req_b.n_past = 0;
    req_b.sequence_id = 22;
    req_b.bpe_prompt_tokens.assign(10, 2);
    req_b.max_tokens = 1;
    req_b.block_table = {22};

    RequestBatch batch;
    batch.requests = {&req_a, &req_b};

    executor->ExecuteBatch(batch, {bounded_backend, bounded_backend});

    REQUIRE_FALSE(bounded_backend->calls.empty());
    REQUIRE(bounded_backend->calls.front().inputs.size() == 2);
    REQUIRE(bounded_backend->calls.front().inputs[0].tokens.size() == 4);
    REQUIRE(bounded_backend->calls.front().inputs[1].tokens.size() == 4);
    for (const auto &call : bounded_backend->calls) {
      std::size_t total_tokens = 0;
      for (const auto &input : call.inputs) {
        total_tokens += input.tokens.size();
      }
      REQUIRE(total_tokens <=
              static_cast<std::size_t>(bounded_backend->token_capacity));
    }
    REQUIRE(req_a.phase == RequestPhase::kFinished);
    REQUIRE(req_b.phase == RequestPhase::kFinished);
  }

  SECTION("Deferred prefill resumes from non-zero prefill_offset") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 4;
    req_prefill.prefill_offset = 4;
    req_prefill.sequence_id = 6;
    req_prefill.bpe_prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    req_prefill.max_tokens = 2;
    req_prefill.block_table = {16};

    RequestBatch batch;
    batch.requests = {&req_prefill};

    auto results = executor->ExecuteBatch(batch, {async_backend});

    REQUIRE_FALSE(async_backend->submit_calls.empty());
    REQUIRE(async_backend->submit_calls.front().lane ==
            LlamaCPUBackend::UnifiedBatchLane::kPrefill);
    REQUIRE(async_backend->submit_calls.front().inputs.size() == 1);
    REQUIRE(async_backend->submit_calls.front().inputs[0].n_past == 4);
    REQUIRE(async_backend->submit_calls.front().inputs[0].tokens.size() == 4);
    REQUIRE(req_prefill.phase == RequestPhase::kFinished);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].completion != "[batch state error]");
  }

  SECTION("Groups phased unified execution by backend instance") {
    auto mock_backend_a = std::make_shared<MockUnifiedBackend>();
    auto mock_backend_b = std::make_shared<MockUnifiedBackend>();

    InferenceRequest req_a;
    req_a.model = "model-a";
    req_a.phase = RequestPhase::kPrefill;
    req_a.n_past = 0;
    req_a.sequence_id = 3;
    req_a.bpe_prompt_tokens = {1, 2, 3};
    req_a.max_tokens = 2;
    req_a.block_table = {10};

    InferenceRequest req_b;
    req_b.model = "model-b";
    req_b.phase = RequestPhase::kPrefill;
    req_b.n_past = 0;
    req_b.sequence_id = 4;
    req_b.bpe_prompt_tokens = {4, 5, 6};
    req_b.max_tokens = 2;
    req_b.block_table = {11};

    RequestBatch batch;
    batch.requests = {&req_a, &req_b};

    executor->ExecuteBatch(batch, {mock_backend_a, mock_backend_b});

    REQUIRE_FALSE(mock_backend_a->calls.empty());
    REQUIRE_FALSE(mock_backend_b->calls.empty());
    REQUIRE(mock_backend_a->calls.front().inputs.size() == 1);
    REQUIRE(mock_backend_b->calls.front().inputs.size() == 1);
    REQUIRE(req_a.phase == RequestPhase::kFinished);
    REQUIRE(req_b.phase == RequestPhase::kFinished);
  }

  SECTION("Async lane split prefers decode lane in mixed-phase steps") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 0;
    req_prefill.sequence_id = 7;
    req_prefill.bpe_prompt_tokens.assign(640, 1); // > one chunk and prefill
    req_prefill.max_tokens = 2;
    req_prefill.block_table = {12};

    InferenceRequest req_decode;
    req_decode.model = "mock";
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 10;
    req_decode.sequence_id = 8;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 2;
    req_decode.block_table = {13};

    RequestBatch batch;
    batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteBatch(batch, {async_backend, async_backend});

    bool saw_decode_lane = false;
    bool saw_prefill_lane = false;
    for (const auto &call : async_backend->submit_calls) {
      if (call.lane == LlamaCPUBackend::UnifiedBatchLane::kDecode) {
        saw_decode_lane = true;
      }
      if (call.lane == LlamaCPUBackend::UnifiedBatchLane::kPrefill) {
        saw_prefill_lane = true;
      }
    }
    REQUIRE(saw_decode_lane);
    REQUIRE(saw_prefill_lane);
    REQUIRE(req_prefill.phase == RequestPhase::kFinished);
    REQUIRE(req_decode.phase == RequestPhase::kFinished);
  }

  SECTION("Async mixed-phase step submits both lanes before first collection") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 0;
    req_prefill.sequence_id = 9;
    req_prefill.bpe_prompt_tokens.assign(640, 1);
    req_prefill.max_tokens = 2;
    req_prefill.block_table = {14};

    InferenceRequest req_decode;
    req_decode.model = "mock";
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 10;
    req_decode.sequence_id = 10;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 2;
    req_decode.block_table = {15};

    RequestBatch batch;
    batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteBatch(batch, {async_backend, async_backend});

    REQUIRE(async_backend->events.size() >= 3);
    REQUIRE(async_backend->events[0] ==
            MockAsyncUnifiedBackend::EventType::kSubmitDecode);
    REQUIRE(async_backend->events[1] ==
            MockAsyncUnifiedBackend::EventType::kSubmitPrefill);
    REQUIRE(async_backend->events[2] ==
            MockAsyncUnifiedBackend::EventType::kCollect);
  }

  SECTION("ExecuteUnifiedBatchStep splits async submissions by lane") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();

    InferenceRequest req_prefill;
    req_prefill.model = "mock";
    req_prefill.sequence_id = 11;
    req_prefill.n_past = 0;
    req_prefill.exec_active = true;
    req_prefill.exec_in_prefill = true;
    req_prefill.prefill_offset = 0;
    req_prefill.exec_decode_limit = 2;
    req_prefill.exec_tokens_generated = 0;
    req_prefill.bpe_prompt_tokens.assign(48, 1);

    InferenceRequest req_decode;
    req_decode.model = "mock";
    req_decode.sequence_id = 12;
    req_decode.n_past = 8;
    req_decode.exec_active = true;
    req_decode.exec_in_prefill = false;
    req_decode.exec_decode_limit = 2;
    req_decode.exec_tokens_generated = 0;
    req_decode.exec_current_token = 42;

    RequestBatch step_batch;
    step_batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteUnifiedBatchStep(step_batch, async_backend);

    bool saw_decode_lane = false;
    bool saw_prefill_lane = false;
    bool saw_auto_lane = false;
    for (const auto &call : async_backend->submit_calls) {
      if (call.lane == LlamaCPUBackend::UnifiedBatchLane::kDecode) {
        saw_decode_lane = true;
      } else if (call.lane == LlamaCPUBackend::UnifiedBatchLane::kPrefill) {
        saw_prefill_lane = true;
      } else {
        saw_auto_lane = true;
      }
    }
    REQUIRE(saw_decode_lane);
    REQUIRE(saw_prefill_lane);
    REQUIRE_FALSE(saw_auto_lane);
  }

  SECTION("ExecuteUnifiedBatchStep respects backend token capacity for prefill") {
    auto async_backend = std::make_shared<MockAsyncUnifiedBackend>();
    async_backend->token_capacity = 6;

    InferenceRequest req_prefill_a;
    req_prefill_a.model = "mock";
    req_prefill_a.sequence_id = 30;
    req_prefill_a.n_past = 0;
    req_prefill_a.exec_active = true;
    req_prefill_a.exec_in_prefill = true;
    req_prefill_a.prefill_offset = 0;
    req_prefill_a.exec_decode_limit = 1;
    req_prefill_a.exec_tokens_generated = 0;
    req_prefill_a.bpe_prompt_tokens.assign(12, 1);

    InferenceRequest req_prefill_b;
    req_prefill_b.model = "mock";
    req_prefill_b.sequence_id = 31;
    req_prefill_b.n_past = 0;
    req_prefill_b.exec_active = true;
    req_prefill_b.exec_in_prefill = true;
    req_prefill_b.prefill_offset = 0;
    req_prefill_b.exec_decode_limit = 1;
    req_prefill_b.exec_tokens_generated = 0;
    req_prefill_b.bpe_prompt_tokens.assign(12, 2);

    RequestBatch step_batch;
    step_batch.requests = {&req_prefill_a, &req_prefill_b};

    executor->ExecuteUnifiedBatchStep(step_batch, async_backend);

    REQUIRE_FALSE(async_backend->submit_calls.empty());
    REQUIRE(async_backend->submit_calls.front().lane ==
            LlamaCPUBackend::UnifiedBatchLane::kPrefill);
    REQUIRE(async_backend->submit_calls.front().inputs.size() == 2);
    REQUIRE(async_backend->submit_calls.front().inputs[0].tokens.size() == 3);
    REQUIRE(async_backend->submit_calls.front().inputs[1].tokens.size() == 3);
  }
}
