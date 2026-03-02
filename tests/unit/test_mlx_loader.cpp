#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/mlx/mlx_loader.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "nlohmann/json.hpp"

using namespace inferflux;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Write a minimal valid safetensors binary into a file.
// The header JSON describes the tensors in `entries`, each as:
//   {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}
// The tensor data region is zero-filled with the appropriate size.
static void WriteSafetensors(
    const fs::path &path,
    const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>,
                                 uint64_t>> &entries) {
  // Build header JSON.
  nlohmann::json hdr;
  uint64_t cursor = 0;
  for (const auto &[name, dtype, shape, nbytes] : entries) {
    nlohmann::json td;
    td["dtype"] = dtype;
    td["shape"] = shape;
    td["data_offsets"] = {cursor, cursor + nbytes};
    hdr[name] = td;
    cursor += nbytes;
  }
  const std::string hdr_str = hdr.dump();
  const uint64_t hdr_len = hdr_str.size();

  std::ofstream f(path, std::ios::binary);
  // 8-byte LE header length.
  for (int b = 0; b < 8; ++b)
    f.put(static_cast<char>((hdr_len >> (b * 8)) & 0xFF));
  f.write(hdr_str.data(), static_cast<std::streamsize>(hdr_len));
  // Zero-fill the data region.
  std::vector<char> zeros(static_cast<size_t>(cursor), '\0');
  f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

// Write a minimal config.json into a file.
static void WriteConfig(const fs::path &path, const std::string &model_type,
                        int hidden_size = 512, int num_layers = 4,
                        int num_heads = 8, int vocab_size = 1024) {
  nlohmann::json j;
  j["model_type"] = model_type;
  j["hidden_size"] = hidden_size;
  j["num_hidden_layers"] = num_layers;
  j["num_attention_heads"] = num_heads;
  j["vocab_size"] = vocab_size;
  j["intermediate_size"] = hidden_size * 4;
  j["max_position_embeddings"] = 2048;
  std::ofstream f(path);
  f << j.dump(2);
}

// ---------------------------------------------------------------------------
// MlxDtype defaults
// ---------------------------------------------------------------------------

TEST_CASE("MlxTensorDescriptor defaults", "[mlx_loader]") {
  MlxTensorDescriptor td;
  REQUIRE(td.name.empty());
  REQUIRE(td.dtype == MlxDtype::Unknown);
  REQUIRE(td.shape.empty());
  REQUIRE(td.data_offset == 0);
  REQUIRE(td.data_length == 0);
  REQUIRE(td.shard_file.empty());
}

// ---------------------------------------------------------------------------
// MlxModelConfig defaults
// ---------------------------------------------------------------------------

TEST_CASE("MlxModelConfig defaults", "[mlx_loader]") {
  MlxModelConfig cfg;
  REQUIRE_FALSE(cfg.valid);
  REQUIRE(cfg.model_type.empty());
  REQUIRE(cfg.hidden_size == 0);
  REQUIRE(cfg.num_hidden_layers == 0);
}

// ---------------------------------------------------------------------------
// ParseModelConfig
// ---------------------------------------------------------------------------

TEST_CASE("ParseModelConfig missing file returns invalid", "[mlx_loader]") {
  auto cfg = MlxWeightLoader::ParseModelConfig(
      "/tmp/inferflux_nonexistent_config.json");
  REQUIRE_FALSE(cfg.valid);
}

TEST_CASE("ParseModelConfig malformed JSON returns invalid", "[mlx_loader]") {
  const auto tmp = fs::temp_directory_path() / "ifx_bad_config.json";
  {
    std::ofstream f(tmp);
    f << "{ not valid json !!!";
  }
  auto cfg = MlxWeightLoader::ParseModelConfig(tmp);
  REQUIRE_FALSE(cfg.valid);
  fs::remove(tmp);
}

TEST_CASE("ParseModelConfig valid LLaMA-style config", "[mlx_loader]") {
  const auto tmp = fs::temp_directory_path() / "ifx_llama_config.json";
  WriteConfig(tmp, "llama", /*hidden_size=*/4096, /*layers=*/32,
              /*heads=*/32, /*vocab=*/32000);
  auto cfg = MlxWeightLoader::ParseModelConfig(tmp);
  REQUIRE(cfg.valid);
  REQUIRE(cfg.model_type == "llama");
  REQUIRE(cfg.hidden_size == 4096);
  REQUIRE(cfg.num_hidden_layers == 32);
  REQUIRE(cfg.num_attention_heads == 32);
  // num_key_value_heads not in WriteConfig → falls back to num_attention_heads.
  REQUIRE(cfg.num_key_value_heads == 32);
  REQUIRE(cfg.vocab_size == 32000);
  fs::remove(tmp);
}

TEST_CASE("ParseModelConfig GQA num_key_value_heads present", "[mlx_loader]") {
  const auto tmp = fs::temp_directory_path() / "ifx_gqa_config.json";
  {
    nlohmann::json j;
    j["model_type"] = "mistral";
    j["hidden_size"] = 4096;
    j["num_hidden_layers"] = 32;
    j["num_attention_heads"] = 32;
    j["num_key_value_heads"] = 8; // GQA
    j["vocab_size"] = 32000;
    std::ofstream f(tmp);
    f << j.dump();
  }
  auto cfg = MlxWeightLoader::ParseModelConfig(tmp);
  REQUIRE(cfg.valid);
  REQUIRE(cfg.num_attention_heads == 32);
  REQUIRE(cfg.num_key_value_heads == 8);
  fs::remove(tmp);
}

// ---------------------------------------------------------------------------
// ReadSafetensorsHeader
// ---------------------------------------------------------------------------

TEST_CASE("ReadSafetensorsHeader missing file returns false", "[mlx_loader]") {
  std::vector<MlxTensorDescriptor> out;
  bool ok = MlxWeightLoader::ReadSafetensorsHeader(
      "/tmp/inferflux_no_such_shard.safetensors", out);
  REQUIRE_FALSE(ok);
  REQUIRE(out.empty());
}

TEST_CASE("ReadSafetensorsHeader valid single-tensor file", "[mlx_loader]") {
  const auto tmp = fs::temp_directory_path() / "ifx_one_tensor.safetensors";
  // One F16 tensor named "embed.weight", shape [1000, 512], 1000*512*2 =
  // 1024000 bytes.
  WriteSafetensors(tmp, {{"embed.weight", "F16", {1000, 512}, 1024000}});

  std::vector<MlxTensorDescriptor> out;
  REQUIRE(MlxWeightLoader::ReadSafetensorsHeader(tmp, out));
  REQUIRE(out.size() == 1);
  REQUIRE(out[0].name == "embed.weight");
  REQUIRE(out[0].dtype == MlxDtype::Float16);
  REQUIRE(out[0].shape == std::vector<int64_t>{1000, 512});
  REQUIRE(out[0].data_offset == 0);
  REQUIRE(out[0].data_length == 1024000);
  REQUIRE(out[0].shard_file == tmp.filename().string());
  fs::remove(tmp);
}

TEST_CASE("ReadSafetensorsHeader multi-tensor with __metadata__ skipped",
          "[mlx_loader]") {
  const auto tmp = fs::temp_directory_path() / "ifx_multi_tensor.safetensors";
  WriteSafetensors(
      tmp, {
               {"model.weight", "BF16", {4096, 4096}, 4096ULL * 4096 * 2},
               {"model.bias", "F32", {4096}, 4096 * 4},
           });

  // Inject __metadata__ by writing the file manually (WriteSafetensors doesn't
  // add it). Re-write with nlohmann directly so we can include __metadata__.
  {
    nlohmann::json hdr;
    hdr["__metadata__"] = {{"format", "pt"}};
    hdr["model.weight"] = {{"dtype", "BF16"},
                           {"shape", {4096, 4096}},
                           {"data_offsets", {0, 4096ULL * 4096 * 2}}};
    hdr["model.bias"] = {
        {"dtype", "F32"},
        {"shape", {4096}},
        {"data_offsets", {4096ULL * 4096 * 2, 4096ULL * 4096 * 2 + 4096 * 4}}};
    const std::string hdr_str = hdr.dump();
    const uint64_t hdr_len = hdr_str.size();
    std::ofstream f(tmp, std::ios::binary);
    for (int b = 0; b < 8; ++b)
      f.put(static_cast<char>((hdr_len >> (b * 8)) & 0xFF));
    f.write(hdr_str.data(), static_cast<std::streamsize>(hdr_len));
  }

  std::vector<MlxTensorDescriptor> out;
  REQUIRE(MlxWeightLoader::ReadSafetensorsHeader(tmp, out));
  // __metadata__ must be skipped; 2 real tensors expected.
  REQUIRE(out.size() == 2);
  const bool has_weight =
      std::any_of(out.begin(), out.end(),
                  [](const auto &td) { return td.name == "model.weight"; });
  const bool has_bias = std::any_of(out.begin(), out.end(), [](const auto &td) {
    return td.name == "model.bias";
  });
  REQUIRE(has_weight);
  REQUIRE(has_bias);
  fs::remove(tmp);
}

// ---------------------------------------------------------------------------
// LoadDirectory
// ---------------------------------------------------------------------------

TEST_CASE("LoadDirectory nonexistent path returns invalid", "[mlx_loader]") {
  MlxWeightLoader loader;
  auto desc = loader.LoadDirectory("/tmp/inferflux_no_such_dir_xyz");
  REQUIRE_FALSE(desc.valid);
}

TEST_CASE("LoadDirectory without config.json returns invalid", "[mlx_loader]") {
  const auto dir = fs::temp_directory_path() / "ifx_no_config_dir";
  fs::create_directories(dir);
  // Add a shard but NO config.json.
  WriteSafetensors(dir / "model.safetensors",
                   {{"w", "F32", {16, 16}, 16 * 16 * 4}});

  MlxWeightLoader loader;
  auto desc = loader.LoadDirectory(dir);
  REQUIRE_FALSE(desc.valid);
  fs::remove_all(dir);
}

TEST_CASE("LoadDirectory valid directory with single shard", "[mlx_loader]") {
  const auto dir = fs::temp_directory_path() / "ifx_valid_model_dir";
  fs::create_directories(dir);
  WriteConfig(dir / "config.json", "llama", 512, 4, 8, 1024);
  WriteSafetensors(
      dir / "model.safetensors",
      {{"embed_tokens.weight", "BF16", {1024, 512}, 1024ULL * 512 * 2}});

  MlxWeightLoader loader;
  auto desc = loader.LoadDirectory(dir);
  REQUIRE(desc.valid);
  REQUIRE(desc.config.model_type == "llama");
  REQUIRE(desc.config.num_hidden_layers == 4);
  REQUIRE(desc.shard_files.size() == 1);
  REQUIRE(desc.tensors.size() == 1);
  REQUIRE(desc.tensors[0].name == "embed_tokens.weight");
  REQUIRE(desc.tensors[0].dtype == MlxDtype::BFloat16);
  fs::remove_all(dir);
}

TEST_CASE("LoadDirectory valid directory with multiple shards",
          "[mlx_loader]") {
  const auto dir = fs::temp_directory_path() / "ifx_sharded_model_dir";
  fs::create_directories(dir);
  WriteConfig(dir / "config.json", "mistral", 1024, 8, 16, 2048);
  WriteSafetensors(
      dir / "model-00001-of-00002.safetensors",
      {{"layer.0.weight", "F16", {1024, 1024}, 1024ULL * 1024 * 2}});
  WriteSafetensors(
      dir / "model-00002-of-00002.safetensors",
      {{"layer.1.weight", "F16", {1024, 1024}, 1024ULL * 1024 * 2},
       {"lm_head.weight", "F16", {2048, 1024}, 2048ULL * 1024 * 2}});

  MlxWeightLoader loader;
  auto desc = loader.LoadDirectory(dir);
  REQUIRE(desc.valid);
  REQUIRE(desc.config.model_type == "mistral");
  REQUIRE(desc.shard_files.size() == 2);
  REQUIRE(desc.tensors.size() == 3);
  // Shards should be in name-sorted order.
  REQUIRE(fs::path(desc.shard_files[0]).filename() ==
          "model-00001-of-00002.safetensors");
  REQUIRE(fs::path(desc.shard_files[1]).filename() ==
          "model-00002-of-00002.safetensors");
  // Each tensor knows which shard it lives in.
  const bool shard1_tensor =
      std::any_of(desc.tensors.begin(), desc.tensors.end(), [](const auto &td) {
        return td.shard_file == "model-00001-of-00002.safetensors";
      });
  REQUIRE(shard1_tensor);
  fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// MlxWeightStore — Stage 2
// ---------------------------------------------------------------------------

TEST_CASE("MlxWeightStore defaults — not loaded", "[mlx_loader]") {
  MlxWeightStore store;
  REQUIRE_FALSE(store.ok);
  REQUIRE(store.count == 0);
  REQUIRE_FALSE(store.HasTensor("any.tensor"));
}

TEST_CASE("LoadWeights with invalid descriptor returns empty store",
          "[mlx_loader]") {
  MlxWeightLoader loader;
  MlxModelDescriptor bad; // valid=false
  auto store = loader.LoadWeights(bad);
  REQUIRE_FALSE(store.ok);
  REQUIRE(store.count == 0);
}

TEST_CASE("LoadWeights with no shard files returns empty store",
          "[mlx_loader]") {
  MlxWeightLoader loader;
  MlxModelDescriptor desc;
  desc.valid = true;
  // shard_files is empty — LoadWeights must return gracefully.
  auto store = loader.LoadWeights(desc);
  REQUIRE_FALSE(store.ok);
  REQUIRE(store.count == 0);
}

#if INFERFLUX_HAS_MLX
TEST_CASE("LoadWeights loads real safetensors on MLX build", "[mlx_loader]") {
  // Create a minimal model directory with one BF16 tensor.
  const auto dir = fs::temp_directory_path() / "ifx_stage2_mlx_dir";
  fs::create_directories(dir);
  WriteConfig(dir / "config.json", "llama", 64, 2, 4, 128);
  // One small tensor: shape [128, 64] BF16 — 128*64*2 = 16384 bytes.
  WriteSafetensors(dir / "model.safetensors",
                   {{"model.embed_tokens.weight", "BF16", {128, 64}, 16384}});

  MlxWeightLoader loader;
  auto desc = loader.LoadDirectory(dir);
  REQUIRE(desc.valid);

  auto store = loader.LoadWeights(desc);
  REQUIRE(store.ok);
  REQUIRE(store.count == 1);
  REQUIRE(store.HasTensor("model.embed_tokens.weight"));
  REQUIRE_FALSE(store.HasTensor("nonexistent.weight"));

  fs::remove_all(dir);
}

TEST_CASE("LoadWeights with two shards merges all tensors", "[mlx_loader]") {
  const auto dir = fs::temp_directory_path() / "ifx_stage2_twoshard_dir";
  fs::create_directories(dir);
  WriteConfig(dir / "config.json", "mistral", 64, 2, 4, 128);
  // Shard 1: one tensor.
  WriteSafetensors(dir / "model-00001-of-00002.safetensors",
                   {{"layer.0.weight", "F16", {64, 64}, 64 * 64 * 2}});
  // Shard 2: two tensors.
  WriteSafetensors(dir / "model-00002-of-00002.safetensors",
                   {{"layer.1.weight", "F16", {64, 64}, 64 * 64 * 2},
                    {"lm_head.weight", "F16", {128, 64}, 128 * 64 * 2}});

  MlxWeightLoader loader;
  auto desc = loader.LoadDirectory(dir);
  REQUIRE(desc.valid);

  auto store = loader.LoadWeights(desc);
  REQUIRE(store.ok);
  REQUIRE(store.count == 3);
  REQUIRE(store.HasTensor("layer.0.weight"));
  REQUIRE(store.HasTensor("layer.1.weight"));
  REQUIRE(store.HasTensor("lm_head.weight"));

  fs::remove_all(dir);
}
#endif // INFERFLUX_HAS_MLX
