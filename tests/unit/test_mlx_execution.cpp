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
