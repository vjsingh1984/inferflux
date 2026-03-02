#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/mlx/mlx_execution.h"
#include "runtime/backends/mlx/mlx_loader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

using namespace inferflux;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Write a minimal config.json for a tiny LLaMA model:
//   hidden=64, layers=2, heads=4, kv_heads=2, vocab=32, ffn=128
static void WriteSmallConfig(const fs::path &dir) {
  nlohmann::json cfg;
  cfg["model_type"] = "llama";
  cfg["hidden_size"] = 64;
  cfg["num_hidden_layers"] = 2;
  cfg["num_attention_heads"] = 4;
  cfg["num_key_value_heads"] = 2;
  cfg["vocab_size"] = 32;
  cfg["intermediate_size"] = 128;
  cfg["rms_norm_eps"] = 1e-5;
  cfg["rope_theta"] = 10000.0;
  cfg["max_position_embeddings"] = 128;
  std::ofstream f(dir / "config.json");
  f << cfg.dump(2);
}

#ifdef INFERFLUX_HAS_MLX
// ---------------------------------------------------------------------------
// MLX-only helpers: create a tiny all-zeros weight store matching the config.
// ---------------------------------------------------------------------------
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"

// Helper: create a zero float16 tensor [rows, cols] and insert into a map.
static void AddZeroTensor(mlx_map_string_to_array &m, const char *name,
                          int rows, int cols, mlx_stream s) {
  mlx_array arr{};
  int shape[2] = {rows, cols};
  mlx_zeros(&arr, shape, 2, MLX_BFLOAT16, s);
  mlx_synchronize(s);
  mlx_map_string_to_array_insert(m, name, arr);
  mlx_array_free(arr);
}

static void AddZeroVec(mlx_map_string_to_array &m, const char *name, int size,
                       mlx_stream s) {
  mlx_array arr{};
  // RMS-norm weights must be 1.0 (not zero) or the computation collapses.
  // Use mlx_ones instead.
  int shape[1] = {size};
  mlx_ones(&arr, shape, 1, MLX_BFLOAT16, s);
  mlx_synchronize(s);
  mlx_map_string_to_array_insert(m, name, arr);
  mlx_array_free(arr);
}

// Build a minimal MlxWeightStore for the tiny 2-layer model.
static MlxWeightStore BuildTinyStore(const MlxModelConfig &cfg) {
  const int H = cfg.hidden_size;          // 64
  const int V = cfg.vocab_size;           // 32
  const int KV = cfg.num_key_value_heads; // 2
  const int NH = cfg.num_attention_heads; // 4
  const int HD = H / NH;                  // 16
  const int FF = cfg.intermediate_size;   // 128

  MlxWeightStore store;
  mlx_stream s = mlx_default_gpu_stream_new();
  auto &m = store.weights;

  // Embedding + final norm + lm_head.
  AddZeroTensor(m, "model.embed_tokens.weight", V, H, s);
  AddZeroVec(m, "model.norm.weight", H, s);
  AddZeroTensor(m, "lm_head.weight", V, H, s);

  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    std::string pfx = "model.layers." + std::to_string(i) + ".";
    // Norms.
    AddZeroVec(m, (pfx + "input_layernorm.weight").c_str(), H, s);
    AddZeroVec(m, (pfx + "post_attention_layernorm.weight").c_str(), H, s);
    // Attention.
    AddZeroTensor(m, (pfx + "self_attn.q_proj.weight").c_str(), NH * HD, H, s);
    AddZeroTensor(m, (pfx + "self_attn.k_proj.weight").c_str(), KV * HD, H, s);
    AddZeroTensor(m, (pfx + "self_attn.v_proj.weight").c_str(), KV * HD, H, s);
    AddZeroTensor(m, (pfx + "self_attn.o_proj.weight").c_str(), H, H, s);
    // MLP.
    AddZeroTensor(m, (pfx + "mlp.gate_proj.weight").c_str(), FF, H, s);
    AddZeroTensor(m, (pfx + "mlp.up_proj.weight").c_str(), FF, H, s);
    AddZeroTensor(m, (pfx + "mlp.down_proj.weight").c_str(), H, FF, s);
    store.count += 9;
  }
  store.count += 3; // embed + norm + lm_head

  mlx_synchronize(s);
  mlx_stream_free(s);
  store.ok = true;
  return store;
}
#endif // INFERFLUX_HAS_MLX

// ---------------------------------------------------------------------------
// Tests — engine lifecycle (no MLX required)
// ---------------------------------------------------------------------------

TEST_CASE("MlxExecutionEngine not loaded by default", "[mlx_execution]") {
  MlxExecutionEngine eng;
  REQUIRE_FALSE(eng.WeightsLoaded());
  REQUIRE(eng.NPast() == 0);
}

TEST_CASE("MlxExecutionEngine Initialize returns false without MLX",
          "[mlx_execution]") {
#ifdef INFERFLUX_HAS_MLX
  // With MLX: Initialize should succeed.
  MlxExecutionEngine eng;
  REQUIRE(eng.Initialize());
  eng.Shutdown();
#else
  MlxExecutionEngine eng;
  REQUIRE_FALSE(eng.Initialize());
#endif
}

TEST_CASE("MlxExecutionEngine Step returns -1 when not loaded",
          "[mlx_execution]") {
  MlxExecutionEngine eng;
  REQUIRE(eng.Step({1, 2, 3}) == -1);
}

TEST_CASE("MlxExecutionEngine LoadWeights rejects invalid store",
          "[mlx_execution]") {
  MlxWeightStore empty_store;
  MlxModelConfig cfg;
  MlxExecutionEngine eng;
  eng.Initialize();
  REQUIRE_FALSE(eng.LoadWeights(empty_store, cfg));
  eng.Shutdown();
}

TEST_CASE("MlxExecutionEngine Reset clears n_past", "[mlx_execution]") {
  MlxExecutionEngine eng;
  // Reset on a fresh engine is a no-op (shouldn't crash).
  eng.Reset();
  REQUIRE(eng.NPast() == 0);
}

// ---------------------------------------------------------------------------
// Tests — full forward pass (MLX build only)
// ---------------------------------------------------------------------------

#ifdef INFERFLUX_HAS_MLX

TEST_CASE("MlxExecutionEngine single-token prefill runs without crash",
          "[mlx_execution]") {
  // Build tiny config + weight store.
  MlxModelConfig cfg;
  cfg.model_type = "llama";
  cfg.hidden_size = 64;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.vocab_size = 32;
  cfg.intermediate_size = 128;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 10000.0f;
  cfg.max_position_embeddings = 128;
  cfg.valid = true;

  MlxWeightStore store = BuildTinyStore(cfg);
  REQUIRE(store.ok);

  MlxExecutionEngine eng;
  REQUIRE(eng.Initialize());
  REQUIRE(eng.LoadWeights(store, cfg));
  REQUIRE(eng.WeightsLoaded());

  // Single-token prefill.
  int32_t tok = eng.Step({1});
  REQUIRE(tok >= 0);
  REQUIRE(tok < cfg.vocab_size);
  REQUIRE(eng.NPast() == 1);
}

TEST_CASE("MlxExecutionEngine multi-token prefill returns valid token",
          "[mlx_execution]") {
  MlxModelConfig cfg;
  cfg.hidden_size = 64;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.vocab_size = 32;
  cfg.intermediate_size = 128;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 10000.0f;
  cfg.max_position_embeddings = 128;
  cfg.valid = true;

  MlxWeightStore store = BuildTinyStore(cfg);
  REQUIRE(store.ok);

  MlxExecutionEngine eng;
  REQUIRE(eng.Initialize());
  REQUIRE(eng.LoadWeights(store, cfg));

  // 4-token prefill.
  int32_t tok = eng.Step({1, 5, 7, 3});
  REQUIRE(tok >= 0);
  REQUIRE(tok < cfg.vocab_size);
  REQUIRE(eng.NPast() == 4);
}

TEST_CASE("MlxExecutionEngine decode loop grows n_past correctly",
          "[mlx_execution]") {
  MlxModelConfig cfg;
  cfg.hidden_size = 64;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.vocab_size = 32;
  cfg.intermediate_size = 128;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 10000.0f;
  cfg.max_position_embeddings = 128;
  cfg.valid = true;

  MlxWeightStore store = BuildTinyStore(cfg);
  MlxExecutionEngine eng;
  REQUIRE(eng.Initialize());
  REQUIRE(eng.LoadWeights(store, cfg));

  // Prefill 2 tokens.
  int32_t tok = eng.Step({1, 2});
  REQUIRE(tok >= 0);
  REQUIRE(eng.NPast() == 2);

  // 3 decode steps.
  for (int i = 0; i < 3; ++i) {
    tok = eng.Step({tok});
    REQUIRE(tok >= 0);
    REQUIRE(tok < cfg.vocab_size);
  }
  REQUIRE(eng.NPast() == 5);
}

TEST_CASE("MlxExecutionEngine Reset clears KV cache and restarts sequence",
          "[mlx_execution]") {
  MlxModelConfig cfg;
  cfg.hidden_size = 64;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.vocab_size = 32;
  cfg.intermediate_size = 128;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 10000.0f;
  cfg.max_position_embeddings = 128;
  cfg.valid = true;

  MlxWeightStore store = BuildTinyStore(cfg);
  MlxExecutionEngine eng;
  REQUIRE(eng.Initialize());
  REQUIRE(eng.LoadWeights(store, cfg));

  // Run a few steps then reset.
  eng.Step({1, 2, 3});
  REQUIRE(eng.NPast() == 3);
  eng.Reset();
  REQUIRE(eng.NPast() == 0);

  // Should be able to run again from position 0.
  int32_t tok = eng.Step({1});
  REQUIRE(tok >= 0);
  REQUIRE(eng.NPast() == 1);
}

#endif // INFERFLUX_HAS_MLX

// ---------------------------------------------------------------------------
// Sampling tests — [mlx_execution_sampling] (MLX build only)
// ---------------------------------------------------------------------------

#ifdef INFERFLUX_HAS_MLX

// Helper: keep both the weight store and the engine alive together.
// MlxExecutionEngine holds a raw pointer into MlxWeightStore, so the store
// must outlive the engine.
struct TinySetup {
  MlxWeightStore store;
  MlxExecutionEngine eng;
};

static TinySetup BuildTinySetup() {
  MlxModelConfig cfg;
  cfg.model_type = "llama";
  cfg.hidden_size = 64;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2;
  cfg.vocab_size = 32;
  cfg.intermediate_size = 128;
  cfg.rms_norm_eps = 1e-5f;
  cfg.rope_theta = 10000.0f;
  cfg.max_position_embeddings = 128;
  cfg.valid = true;

  TinySetup s;
  s.store = BuildTinyStore(cfg);
  s.eng.Initialize();
  s.eng.LoadWeights(s.store, cfg);
  return s;
}

TEST_CASE("temperature=0 is deterministic", "[mlx_execution_sampling]") {
  inferflux::SamplingParams sp;
  sp.temperature = 0.0f; // greedy

  auto s1 = BuildTinySetup();
  int32_t tok1 = s1.eng.Step({1, 2, 3}, sp);
  auto s2 = BuildTinySetup();
  int32_t tok2 = s2.eng.Step({1, 2, 3}, sp);

  REQUIRE(tok1 >= 0);
  REQUIRE(tok1 == tok2);
}

TEST_CASE("top_k=1 is deterministic", "[mlx_execution_sampling]") {
  inferflux::SamplingParams sp;
  sp.temperature = 1.0f;
  sp.top_k = 1;

  auto s1 = BuildTinySetup();
  int32_t tok1 = s1.eng.Step({1}, sp);
  auto s2 = BuildTinySetup();
  int32_t tok2 = s2.eng.Step({1}, sp);

  REQUIRE(tok1 == tok2);
}

TEST_CASE("seed=42 reproduces 5-token sequence", "[mlx_execution_sampling]") {
  inferflux::SamplingParams sp;
  sp.temperature = 1.0f;
  sp.seed = 42;

  auto run = [&]() {
    auto s = BuildTinySetup();
    std::vector<int32_t> seq;
    int32_t tok = s.eng.Step({1}, sp);
    seq.push_back(tok);
    for (int i = 0; i < 4; ++i) {
      tok = s.eng.Step({tok}, sp);
      seq.push_back(tok);
    }
    return seq;
  };

  auto seq1 = run();
  auto seq2 = run();
  REQUIRE(seq1.size() == 5u);
  REQUIRE(seq1 == seq2);
}

TEST_CASE("top_p=0.0001 always returns highest-prob token",
          "[mlx_execution_sampling]") {
  // With top_p very small, only the single highest-probability token survives
  // the nucleus filter, so the result must equal the greedy token.
  inferflux::SamplingParams sp_greedy;
  sp_greedy.temperature = 0.0f;

  inferflux::SamplingParams sp_top_p;
  sp_top_p.temperature = 1.0f;
  sp_top_p.top_p = 0.0001f;
  sp_top_p.seed = 1;

  // Both setups use zero weights so logits are identical.
  auto sg = BuildTinySetup();
  int32_t greedy_tok = sg.eng.Step({1}, sp_greedy);

  auto st = BuildTinySetup();
  int32_t top_p_tok = st.eng.Step({1}, sp_top_p);

  REQUIRE(greedy_tok >= 0);
  REQUIRE(top_p_tok == greedy_tok);
}

TEST_CASE("SamplingParams defaults are valid", "[mlx_execution_sampling]") {
  inferflux::SamplingParams sp;
  REQUIRE(sp.temperature == 1.0f);
  REQUIRE(sp.top_p == 1.0f);
  REQUIRE(sp.top_k == 0);
  REQUIRE(sp.seed == UINT32_MAX);

  auto s = BuildTinySetup();
  // Default params should not crash.
  int32_t tok = s.eng.Step({1}, sp);
  REQUIRE(tok >= 0);
}

#endif // INFERFLUX_HAS_MLX (sampling tests)

// ---------------------------------------------------------------------------
// [mlx_execution_phased] — slot API tests (INF-8)
// Only compiled + run when INFERFLUX_HAS_MLX is defined.
// ---------------------------------------------------------------------------

#ifdef INFERFLUX_HAS_MLX

TEST_CASE("AllocSlot / SeqNPast / FreeSlot lifecycle",
          "[mlx_execution_phased]") {
  auto s = BuildTinySetup();
  // Before allocation: slot does not exist → SeqNPast returns -1.
  REQUIRE(s.eng.SeqNPast(0) == -1);

  s.eng.AllocSlot(0);
  // After allocation: n_past is 0.
  REQUIRE(s.eng.SeqNPast(0) == 0);

  // AllocSlot is idempotent.
  s.eng.AllocSlot(0);
  REQUIRE(s.eng.SeqNPast(0) == 0);

  s.eng.FreeSlot(0);
  // After free: slot gone.
  REQUIRE(s.eng.SeqNPast(0) == -1);
}

TEST_CASE("StepSeq updates slot n_past", "[mlx_execution_phased]") {
  auto s = BuildTinySetup();
  // Two tokens in prefill step → n_past should be 2.
  int32_t tok = s.eng.StepSeq(3, {1, 2});
  REQUIRE(tok >= 0);
  REQUIRE(s.eng.SeqNPast(3) == 2);

  // One decode step → n_past should be 3.
  int32_t tok2 = s.eng.StepSeq(3, {tok});
  REQUIRE(tok2 >= 0);
  REQUIRE(s.eng.SeqNPast(3) == 3);
}

TEST_CASE("Multiple slots are independent", "[mlx_execution_phased]") {
  auto s = BuildTinySetup();

  // Prefill different lengths into different slots.
  s.eng.StepSeq(0, {1, 2, 3}); // n_past = 3
  s.eng.StepSeq(1, {4, 5});    // n_past = 2

  REQUIRE(s.eng.SeqNPast(0) == 3);
  REQUIRE(s.eng.SeqNPast(1) == 2);

  // Step slot 0 once more → n_past = 4; slot 1 stays at 2.
  s.eng.StepSeq(0, {10});
  REQUIRE(s.eng.SeqNPast(0) == 4);
  REQUIRE(s.eng.SeqNPast(1) == 2);
}

TEST_CASE("CopySlotPrefix creates dst with n_tokens of KV",
          "[mlx_execution_phased]") {
  auto s = BuildTinySetup();

  // Fill slot 5 with a 4-token prefill.
  s.eng.StepSeq(5, {1, 2, 3, 4});
  REQUIRE(s.eng.SeqNPast(5) == 4);

  // Copy first 2 tokens of KV into slot 7.
  s.eng.CopySlotPrefix(5, 7, 2);
  REQUIRE(s.eng.SeqNPast(7) == 2);
  // Source slot is unaffected.
  REQUIRE(s.eng.SeqNPast(5) == 4);

  // We can continue decoding slot 7 from position 2.
  int32_t tok = s.eng.StepSeq(7, {1});
  REQUIRE(tok >= 0);
  REQUIRE(s.eng.SeqNPast(7) == 3);
}

TEST_CASE("FreeSlot of active slot does not crash", "[mlx_execution_phased]") {
  auto s = BuildTinySetup();
  s.eng.StepSeq(2, {1, 2});
  // seq 2 is now the active slot; freeing it should be safe.
  REQUIRE_NOTHROW(s.eng.FreeSlot(2));
  REQUIRE(s.eng.SeqNPast(2) == -1);
}

#endif // INFERFLUX_HAS_MLX (phased tests)
