#pragma once

#include <filesystem>
#include <string>
#include <vector>

#ifdef INFERFLUX_HAS_MLX
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#endif

namespace inferflux {

// Dtype enum mirroring the safetensors / mlx_dtype vocabulary.
// Usable without including any mlx-c headers.
enum class MlxDtype {
  Float16,
  Float32,
  BFloat16,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Bool,
  Unknown,
};

// Per-tensor metadata extracted from a safetensors file header.
// No tensor data is loaded into memory at this stage.
struct MlxTensorDescriptor {
  std::string name;
  MlxDtype dtype{MlxDtype::Unknown};
  std::vector<int64_t> shape;
  uint64_t data_offset{0}; // byte offset within shard's data region
  uint64_t data_length{0}; // byte length of tensor data
  std::string shard_file;  // basename of the .safetensors file
};

// Model architecture parameters parsed from config.json.
struct MlxModelConfig {
  std::string model_type; // e.g. "llama", "mistral", "phi"
  int hidden_size{0};
  int num_hidden_layers{0};
  int num_attention_heads{0};
  int num_key_value_heads{0}; // GQA; defaults to num_attention_heads if absent
  int vocab_size{0};
  float rms_norm_eps{1e-5f};
  float rope_theta{10000.0f};
  float rope_freq_scale{
      1.0f}; // 1.0 = no scaling; derived from rope_scaling.factor
  int intermediate_size{0};
  int max_position_embeddings{4096};
  bool valid{false};
};

// Descriptor for a fully-catalogued MLX-native model checkpoint.
// Holds only metadata — no weights are resident in RAM.
struct MlxModelDescriptor {
  std::filesystem::path source_dir;
  MlxModelConfig config;
  std::vector<std::string> shard_files; // absolute paths, shard-name order
  std::vector<MlxTensorDescriptor>
      tensors; // full tensor catalogue across all shards
  bool valid{false};
};

// ---------------------------------------------------------------------------
// MlxWeightStore — RAII container for materialized MLX weight tensors.
//
// When built with INFERFLUX_HAS_MLX=1: owns a merged mlx_map_string_to_array
// covering all shards. The arrays live on the Metal device.
// When built without MLX: a lightweight stub (ok=false, count=0).
// ---------------------------------------------------------------------------
struct MlxWeightStore {
  bool ok{false};
  size_t count{0}; // number of tensors materialised

#ifdef INFERFLUX_HAS_MLX
  mlx_map_string_to_array weights{}; // merged tensor name → array map

  MlxWeightStore() : weights(mlx_map_string_to_array_new()) {}

  ~MlxWeightStore() { mlx_map_string_to_array_free(weights); }

  // Non-copyable: MLX arrays own device memory.
  MlxWeightStore(const MlxWeightStore &) = delete;
  MlxWeightStore &operator=(const MlxWeightStore &) = delete;

  // Movable.
  MlxWeightStore(MlxWeightStore &&o) noexcept
      : ok(o.ok), count(o.count), weights(o.weights) {
    o.ok = false;
    o.count = 0;
    o.weights = mlx_map_string_to_array_new(); // leave o in a freeable state
  }

  MlxWeightStore &operator=(MlxWeightStore &&o) noexcept {
    if (this != &o) {
      mlx_map_string_to_array_free(weights);
      ok = o.ok;
      count = o.count;
      weights = o.weights;
      o.ok = false;
      o.count = 0;
      o.weights = mlx_map_string_to_array_new();
    }
    return *this;
  }

  // Returns true if a tensor with the given name was loaded.
  bool HasTensor(const std::string &name) const {
    if (!ok)
      return false;
    mlx_array arr{};
    return mlx_map_string_to_array_get(&arr, weights, name.c_str()) == 0;
  }
#else
  bool HasTensor(const std::string &) const { return false; }
#endif
};

// ---------------------------------------------------------------------------
// MlxWeightLoader
// ---------------------------------------------------------------------------

class MlxWeightLoader {
public:
  MlxWeightLoader() = default;

  // Main entry point: loads an MLX-native model directory containing
  // config.json and one or more *.safetensors files.
  // Returns an invalid descriptor on any error.
  MlxModelDescriptor LoadDirectory(const std::filesystem::path &model_dir);

  // Materialise all shard weights into MLX arrays on the default GPU stream.
  // When INFERFLUX_HAS_MLX=0, returns an empty store (ok=false).
  MlxWeightStore LoadWeights(const MlxModelDescriptor &descriptor);

  // Parse config.json at config_path and return a populated MlxModelConfig.
  // Returns config.valid=false on any error. Exposed for unit-testing.
  static MlxModelConfig
  ParseModelConfig(const std::filesystem::path &config_path);

  // Parse the binary header of one safetensors shard (no tensor data loaded).
  // Appends discovered MlxTensorDescriptors to out_tensors.
  // Returns false if the file cannot be read or has an invalid header.
  static bool
  ReadSafetensorsHeader(const std::filesystem::path &shard_path,
                        std::vector<MlxTensorDescriptor> &out_tensors);
};

} // namespace inferflux
