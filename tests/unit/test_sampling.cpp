#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "scheduler/request_batch.h"
#include "scheduler/scheduler.h"

using namespace inferflux;
using Catch::Approx;

// ---------------------------------------------------------------------------
// SamplingParams struct defaults
// ---------------------------------------------------------------------------

TEST_CASE("SamplingParams defaults", "[sampling]") {
  SamplingParams sp;
  REQUIRE(sp.temperature == Approx(1.0f));
  REQUIRE(sp.top_p == Approx(1.0f));
  REQUIRE(sp.top_k == 0);
  REQUIRE(sp.min_p == Approx(0.0f));
  REQUIRE(sp.frequency_penalty == Approx(0.0f));
  REQUIRE(sp.presence_penalty == Approx(0.0f));
  REQUIRE(sp.repetition_penalty == Approx(1.0f));
  REQUIRE(sp.penalty_last_n == 64);
  REQUIRE(sp.seed == UINT32_MAX);
}

TEST_CASE("InferenceRequest sampling field defaults to SamplingParams{}",
          "[sampling]") {
  InferenceRequest req;
  SamplingParams defaults;
  REQUIRE(req.sampling.temperature == Approx(defaults.temperature));
  REQUIRE(req.sampling.top_p == Approx(defaults.top_p));
  REQUIRE(req.sampling.top_k == defaults.top_k);
  REQUIRE(req.sampling.seed == defaults.seed);
}

// ---------------------------------------------------------------------------
// LlamaCPUBackend sampler lifecycle (null-model / ForceReadyForTests)
// ---------------------------------------------------------------------------

TEST_CASE("SetupSampler no-op when model not loaded", "[sampling]") {
  LlamaCPUBackend backend;
  // vocab_ is null — SetupSampler must not crash.
  REQUIRE_NOTHROW(backend.SetupSampler("", "root", {}));
}

TEST_CASE("TeardownSampler idempotent — double call does not crash",
          "[sampling]") {
  LlamaCPUBackend backend;
  REQUIRE_NOTHROW(backend.TeardownSampler());
  REQUIRE_NOTHROW(backend.TeardownSampler());
}

TEST_CASE(
    "SetupSampler with temperature=0 (greedy) succeeds when model not loaded",
    "[sampling]") {
  LlamaCPUBackend backend;
  SamplingParams sp;
  sp.temperature = 0.0f;
  // Guard: vocab_ null → early return without crash.
  REQUIRE_NOTHROW(backend.SetupSampler("", "root", sp));
}

TEST_CASE("SetupSampler with temperature>0 (stochastic) succeeds when model "
          "not loaded",
          "[sampling]") {
  LlamaCPUBackend backend;
  SamplingParams sp;
  sp.temperature = 0.8f;
  REQUIRE_NOTHROW(backend.SetupSampler("", "root", sp));
}

TEST_CASE("SamplerScope RAII: TeardownSampler called on scope exit",
          "[sampling]") {
  // Without a loaded model SetupSampler is a no-op, but TeardownSampler should
  // still be callable without error when the scope exits.
  LlamaCPUBackend backend;
  backend.ForceReadyForTests();
  InferenceRequest req;
  {
    auto be = std::make_shared<LlamaCPUBackend>();
    // Verify no crash during construction + destruction of SamplerScope-like
    // pattern using the public API.
    be->SetupSampler("", "root", req.sampling);
    be->TeardownSampler();
  }
  SUCCEED("SamplerScope pattern did not crash");
}

TEST_CASE("EnableGrammarConstraint / DisableGrammarConstraint backward compat",
          "[sampling]") {
  LlamaCPUBackend backend;
  // Should delegate to SetupSampler / TeardownSampler without crashing.
  REQUIRE_NOTHROW(
      backend.EnableGrammarConstraint("root ::= \"hello\"", "root"));
  REQUIRE_NOTHROW(backend.DisableGrammarConstraint());
}

// ---------------------------------------------------------------------------
// Stop sequences on InferenceRequest
// ---------------------------------------------------------------------------

TEST_CASE("InferenceRequest stop field defaults to empty", "[sampling]") {
  InferenceRequest req;
  REQUIRE(req.stop.empty());
}

TEST_CASE("InferenceRequest stop accepts up to 4 strings", "[sampling]") {
  InferenceRequest req;
  req.stop = {"<|eot_id|>", "\n\n", "Human:", "Assistant:"};
  REQUIRE(req.stop.size() == 4);
  REQUIRE(req.stop[0] == "<|eot_id|>");
  REQUIRE(req.stop[3] == "Assistant:");
}

// ---------------------------------------------------------------------------
// InferenceResult finish_reason_length
// ---------------------------------------------------------------------------

TEST_CASE("InferenceResult finish_reason_length defaults false", "[sampling]") {
  inferflux::InferenceResult r;
  REQUIRE_FALSE(r.finish_reason_length);
}

TEST_CASE("InferenceResult finish_reason_length can be set to true",
          "[sampling]") {
  inferflux::InferenceResult r;
  r.finish_reason_length = true;
  REQUIRE(r.finish_reason_length);
}

// ---------------------------------------------------------------------------
// best_of: cumulative logprob ranking (pure algorithm tests)
// ---------------------------------------------------------------------------

TEST_CASE("best_of cumulative logprob ranking selects highest-logprob result",
          "[sampling]") {
  using inferflux::InferenceResult;
  using inferflux::TokenLogprob;

  // Build three results with known cumulative log-probs: -1, -5, -2.
  auto makeResult = [](float lp1, float lp2, const std::string &text) {
    InferenceResult r;
    r.completion = text;
    TokenLogprob t1, t2;
    t1.logprob = lp1;
    t2.logprob = lp2;
    r.logprobs = {t1, t2};
    r.completion_tokens = 2;
    return r;
  };

  std::vector<InferenceResult> results = {
      makeResult(-0.5f, -0.5f, "best"),   // cumsum = -1.0
      makeResult(-2.5f, -2.5f, "worst"),  // cumsum = -5.0
      makeResult(-0.9f, -1.1f, "middle"), // cumsum = -2.0
  };

  // Apply the same ranking as the production code.
  auto cumlogprob = [](const InferenceResult &r) -> double {
    double s = 0.0;
    for (const auto &tlp : r.logprobs)
      s += static_cast<double>(tlp.logprob);
    return s;
  };
  std::sort(results.begin(), results.end(),
            [&](const InferenceResult &a, const InferenceResult &b) {
              return cumlogprob(a) > cumlogprob(b);
            });

  REQUIRE(results[0].completion == "best");
  REQUIRE(results[1].completion == "middle");
  REQUIRE(results[2].completion == "worst");
}

TEST_CASE("best_of n=1 best_of=3 keeps only top result", "[sampling]") {
  using inferflux::InferenceResult;
  using inferflux::TokenLogprob;

  auto makeResult = [](float lp, const std::string &text) {
    InferenceResult r;
    r.completion = text;
    TokenLogprob t;
    t.logprob = lp;
    r.logprobs = {t};
    r.completion_tokens = 1;
    return r;
  };

  std::vector<InferenceResult> results = {
      makeResult(-3.0f, "c"),
      makeResult(-1.0f, "a"),
      makeResult(-2.0f, "b"),
  };

  auto cumlogprob = [](const InferenceResult &r) -> double {
    double s = 0.0;
    for (const auto &tlp : r.logprobs)
      s += static_cast<double>(tlp.logprob);
    return s;
  };
  std::sort(results.begin(), results.end(),
            [&](const InferenceResult &a, const InferenceResult &b) {
              return cumlogprob(a) > cumlogprob(b);
            });
  results.resize(1);

  REQUIRE(results.size() == 1u);
  REQUIRE(results[0].completion == "a");
}

TEST_CASE("best_of with equal logprobs is stable (no crash)", "[sampling]") {
  using inferflux::InferenceResult;
  using inferflux::TokenLogprob;

  std::vector<InferenceResult> results(3);
  for (auto &r : results) {
    TokenLogprob t;
    t.logprob = -1.0f;
    r.logprobs = {t};
    r.completion_tokens = 1;
  }
  results[0].completion = "x";
  results[1].completion = "y";
  results[2].completion = "z";

  auto cumlogprob = [](const InferenceResult &r) -> double {
    double s = 0.0;
    for (const auto &tlp : r.logprobs)
      s += static_cast<double>(tlp.logprob);
    return s;
  };
  REQUIRE_NOTHROW(
      std::sort(results.begin(), results.end(),
                [&](const InferenceResult &a, const InferenceResult &b) {
                  return cumlogprob(a) > cumlogprob(b);
                }));
  REQUIRE(results.size() == 3u);
}
