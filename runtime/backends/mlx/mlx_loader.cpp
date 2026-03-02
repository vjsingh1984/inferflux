#include "runtime/backends/mlx/mlx_loader.h"
#include "server/logging/logger.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

#ifdef INFERFLUX_HAS_MLX
#include "mlx/c/array.h"
#include "mlx/c/io.h"
#include "mlx/c/stream.h"
#endif

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace inferflux {

namespace {

// Map safetensors dtype tag strings to MlxDtype.
MlxDtype ParseDtype(const std::string &s) {
  if (s == "F16")
    return MlxDtype::Float16;
  if (s == "F32")
    return MlxDtype::Float32;
  if (s == "BF16")
    return MlxDtype::BFloat16;
  if (s == "I8")
    return MlxDtype::Int8;
  if (s == "I16")
    return MlxDtype::Int16;
  if (s == "I32")
    return MlxDtype::Int32;
  if (s == "I64")
    return MlxDtype::Int64;
  if (s == "U8")
    return MlxDtype::UInt8;
  if (s == "U16")
    return MlxDtype::UInt16;
  if (s == "U32")
    return MlxDtype::UInt32;
  if (s == "U64")
    return MlxDtype::UInt64;
  if (s == "BOOL")
    return MlxDtype::Bool;
  return MlxDtype::Unknown;
}

// Collect all *.safetensors files in dir, sorted by name so that
// shards appear in canonical order (model-00001-of-00005 <
// model-00002-of-00005).
std::vector<std::filesystem::path>
CollectShards(const std::filesystem::path &dir) {
  std::vector<std::filesystem::path> shards;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
      shards.push_back(entry.path());
    }
  }
  std::sort(shards.begin(), shards.end());
  return shards;
}

} // namespace

// ---------------------------------------------------------------------------
// ParseModelConfig
// ---------------------------------------------------------------------------

MlxModelConfig
MlxWeightLoader::ParseModelConfig(const std::filesystem::path &config_path) {
  MlxModelConfig cfg;

  std::ifstream f(config_path);
  if (!f.is_open()) {
    log::Error("mlx_loader", "Cannot open " + config_path.string());
    return cfg;
  }

  json j;
  try {
    f >> j;
  } catch (const std::exception &e) {
    log::Error("mlx_loader",
               std::string("config.json parse error: ") + e.what());
    return cfg;
  }

  auto get_str = [&](const char *key, std::string &out) {
    if (j.contains(key) && j[key].is_string())
      out = j[key].get<std::string>();
  };
  auto get_int = [&](const char *key, int &out) {
    if (j.contains(key) && j[key].is_number_integer())
      out = j[key].get<int>();
  };
  auto get_float = [&](const char *key, float &out) {
    if (j.contains(key) && j[key].is_number())
      out = j[key].get<float>();
  };

  get_str("model_type", cfg.model_type);
  get_int("hidden_size", cfg.hidden_size);
  get_int("num_hidden_layers", cfg.num_hidden_layers);
  get_int("num_attention_heads", cfg.num_attention_heads);
  get_int("num_key_value_heads", cfg.num_key_value_heads);
  get_int("vocab_size", cfg.vocab_size);
  get_int("intermediate_size", cfg.intermediate_size);
  get_int("max_position_embeddings", cfg.max_position_embeddings);
  get_float("rms_norm_eps", cfg.rms_norm_eps);
  get_float("rope_theta", cfg.rope_theta);

  // RoPE scaling: read from rope_scaling.factor if present.
  if (j.contains("rope_scaling") && j["rope_scaling"].is_object()) {
    const auto &rs = j["rope_scaling"];
    if (rs.contains("factor") && rs["factor"].is_number_float()) {
      float f = rs["factor"].get<float>();
      if (f > 0.0f)
        cfg.rope_freq_scale = 1.0f / f;
    }
  }
  // Allow direct override via rope_freq_scale key.
  if (j.contains("rope_freq_scale") && j["rope_freq_scale"].is_number_float())
    cfg.rope_freq_scale = j["rope_freq_scale"].get<float>();

  // GQA fallback: if num_key_value_heads is absent, MHA applies (KV heads == Q
  // heads).
  if (cfg.num_key_value_heads == 0 && cfg.num_attention_heads > 0)
    cfg.num_key_value_heads = cfg.num_attention_heads;

  cfg.valid = !cfg.model_type.empty() && cfg.num_hidden_layers > 0;
  return cfg;
}

// ---------------------------------------------------------------------------
// ReadSafetensorsHeader
// ---------------------------------------------------------------------------

bool MlxWeightLoader::ReadSafetensorsHeader(
    const std::filesystem::path &shard_path,
    std::vector<MlxTensorDescriptor> &out_tensors) {
  std::ifstream f(shard_path, std::ios::binary);
  if (!f.is_open()) {
    log::Error("mlx_loader", "Cannot open shard " + shard_path.string());
    return false;
  }

  // First 8 bytes: little-endian uint64 = header JSON length in bytes.
  uint8_t len_buf[8];
  if (!f.read(reinterpret_cast<char *>(len_buf), 8)) {
    log::Error("mlx_loader",
               "Shard " + shard_path.string() + " too small for header");
    return false;
  }
  const uint64_t header_len = static_cast<uint64_t>(len_buf[0]) |
                              (static_cast<uint64_t>(len_buf[1]) << 8) |
                              (static_cast<uint64_t>(len_buf[2]) << 16) |
                              (static_cast<uint64_t>(len_buf[3]) << 24) |
                              (static_cast<uint64_t>(len_buf[4]) << 32) |
                              (static_cast<uint64_t>(len_buf[5]) << 40) |
                              (static_cast<uint64_t>(len_buf[6]) << 48) |
                              (static_cast<uint64_t>(len_buf[7]) << 56);

  // Sanity-check: real safetensors headers are never larger than 100 MB.
  constexpr uint64_t kMaxHeaderBytes = 100ULL * 1024 * 1024;
  if (header_len == 0 || header_len > kMaxHeaderBytes) {
    log::Error("mlx_loader", "Shard " + shard_path.string() +
                                 " has implausible header length " +
                                 std::to_string(header_len));
    return false;
  }

  std::string header_str(header_len, '\0');
  if (!f.read(header_str.data(), static_cast<std::streamsize>(header_len))) {
    log::Error("mlx_loader",
               "Shard " + shard_path.string() + " truncated before header end");
    return false;
  }

  json hdr;
  try {
    hdr = json::parse(header_str);
  } catch (const std::exception &e) {
    log::Error("mlx_loader", "Shard " + shard_path.string() +
                                 " header JSON parse error: " + e.what());
    return false;
  }

  const std::string shard_name = shard_path.filename().string();
  for (const auto &[name, meta] : hdr.items()) {
    if (name == "__metadata__")
      continue;
    if (!meta.is_object())
      continue;

    MlxTensorDescriptor td;
    td.name = name;
    td.shard_file = shard_name;

    if (meta.contains("dtype") && meta["dtype"].is_string())
      td.dtype = ParseDtype(meta["dtype"].get<std::string>());

    if (meta.contains("shape") && meta["shape"].is_array()) {
      for (const auto &dim : meta["shape"])
        td.shape.push_back(dim.get<int64_t>());
    }

    if (meta.contains("data_offsets") && meta["data_offsets"].is_array() &&
        meta["data_offsets"].size() == 2) {
      const uint64_t start = meta["data_offsets"][0].get<uint64_t>();
      const uint64_t end = meta["data_offsets"][1].get<uint64_t>();
      td.data_offset = start;
      td.data_length = end - start;
    }

    out_tensors.push_back(std::move(td));
  }
  return true;
}

// ---------------------------------------------------------------------------
// LoadDirectory
// ---------------------------------------------------------------------------

MlxModelDescriptor
MlxWeightLoader::LoadDirectory(const std::filesystem::path &model_dir) {
  MlxModelDescriptor desc;
  desc.source_dir = model_dir;

  if (!std::filesystem::is_directory(model_dir)) {
    log::Error("mlx_loader", "Not a directory: " + model_dir.string());
    return desc;
  }

  // Parse architecture config.
  desc.config = ParseModelConfig(model_dir / "config.json");
  if (!desc.config.valid) {
    log::Error("mlx_loader",
               "config.json missing or invalid in " + model_dir.string());
    return desc;
  }

  // Discover and catalogue all safetensors shards.
  const auto shards = CollectShards(model_dir);
  if (shards.empty()) {
    log::Error("mlx_loader",
               "No *.safetensors files found in " + model_dir.string());
    return desc;
  }

  for (const auto &shard : shards) {
    desc.shard_files.push_back(shard.string());
    if (!ReadSafetensorsHeader(shard, desc.tensors)) {
      log::Error("mlx_loader",
                 "Failed to read shard header: " + shard.string());
      return desc; // partial load — leave valid=false
    }
  }

  desc.valid = true;
  log::Info("mlx_loader",
            "Descriptor loaded: model_type=" + desc.config.model_type +
                " layers=" + std::to_string(desc.config.num_hidden_layers) +
                " tensors=" + std::to_string(desc.tensors.size()) +
                " shards=" + std::to_string(desc.shard_files.size()));
  return desc;
}

// ---------------------------------------------------------------------------
// LoadWeights
// ---------------------------------------------------------------------------

MlxWeightStore
MlxWeightLoader::LoadWeights(const MlxModelDescriptor &descriptor) {
  MlxWeightStore store;

  if (!descriptor.valid) {
    log::Error("mlx_loader",
               "LoadWeights: descriptor is invalid — skipping weight load");
    return store;
  }
  if (descriptor.shard_files.empty()) {
    log::Error("mlx_loader", "LoadWeights: no shard files in descriptor");
    return store;
  }

#ifndef INFERFLUX_HAS_MLX
  (void)descriptor;
  log::Error(
      "mlx_loader",
      "LoadWeights: built without INFERFLUX_HAS_MLX — no weights loaded");
  return store;
#else
  // Use the default Metal GPU stream for loading.
  mlx_stream stream = mlx_default_gpu_stream_new();

  size_t total = 0;
  for (const auto &shard_path : descriptor.shard_files) {
    mlx_map_string_to_array shard_map = mlx_map_string_to_array_new();
    mlx_map_string_to_string metadata = mlx_map_string_to_string_new();

    const int rc =
        mlx_load_safetensors(&shard_map, &metadata, shard_path.c_str(), stream);
    mlx_map_string_to_string_free(metadata);

    if (rc != 0) {
      log::Error("mlx_loader", "mlx_load_safetensors failed for " + shard_path);
      mlx_map_string_to_array_free(shard_map);
      mlx_stream_free(stream);
      return store; // leave ok=false
    }

    // Synchronise before iterating so all Metal I/O for this shard is complete.
    mlx_synchronize(stream);

    // Merge shard arrays into the combined store map.
    auto it = mlx_map_string_to_array_iterator_new(shard_map);
    const char *key = nullptr;
    mlx_array arr{};
    while (mlx_map_string_to_array_iterator_next(&key, &arr, it) == 0) {
      mlx_map_string_to_array_insert(store.weights, key, arr);
      mlx_array_free(arr);
      arr = {}; // reset to avoid dangling pointer on next iteration
      ++total;
    }
    mlx_map_string_to_array_iterator_free(it);
    mlx_map_string_to_array_free(shard_map);
  }

  mlx_stream_free(stream);

  store.count = total;
  store.ok = true;
  log::Info("mlx_loader",
            "Weights loaded: " + std::to_string(total) + " tensors across " +
                std::to_string(descriptor.shard_files.size()) + " shard(s)");
  return store;
#endif
}

} // namespace inferflux
