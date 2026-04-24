#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/cuda/inferflux_cuda_backend.h"
#include "runtime/backends/llama/llama_cpp_backend.h"

#include "runtime/backends/common/backend_interface.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace inferflux {
namespace {

using json = nlohmann::json;

struct Options {
  std::string backend;
  std::optional<std::string> compare_backend;
  std::filesystem::path model_path;
  std::optional<std::string> prompt;
  int max_tokens{2};
  int top_n{8};
  int ctx_size{4096};
  int batch_size{512};
  int gpu_layers{99};
  bool use_flash_attention{true};
  int max_parallel_sequences{16};
  std::optional<std::filesystem::path> output_file;
};

void PrintUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --backend <inferflux_cuda|llama_cpp_cuda>"
               " --model <path> [--prompt <text>] [--max-tokens <int>]"
               " [--compare-backend <backend>] [--output <path>]\n"
               "\n"
               "Options:\n"
               "  --backend <name>         Primary backend to test\n"
               "  --compare-backend <name>  Secondary backend for comparison (optional)\n"
               "  --model <path>            Path to GGUF model file\n"
               "  --prompt <text>           Prompt text (default: \"Hello world\")\n"
               "  --max-tokens <int>        Number of tokens to generate (default: 2)\n"
               "  --output <path>           Output JSON file (default: stdout)\n"
               "\n"
               "Examples:\n"
               "  " << argv0 << " --backend inferflux_cuda --model model.gguf\n"
               "  " << argv0 << " --backend inferflux_cuda --compare-backend llama_cpp_cuda "
                  "--model model.gguf --output comparison.json\n";
}

bool ParseArgs(int argc, char **argv, Options *out) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--backend") {
      const char *value = require_value("--backend");
      if (!value)
        return false;
      out->backend = value;
    } else if (arg == "--compare-backend") {
      const char *value = require_value("--compare-backend");
      if (!value)
        return false;
      out->compare_backend = std::string(value);
    } else if (arg == "--model") {
      const char *value = require_value("--model");
      if (!value)
        return false;
      out->model_path = value;
    } else if (arg == "--prompt") {
      const char *value = require_value("--prompt");
      if (!value)
        return false;
      out->prompt = std::string(value);
    } else if (arg == "--max-tokens") {
      const char *value = require_value("--max-tokens");
      if (!value)
        return false;
      out->max_tokens = std::stoi(value);
    } else if (arg == "--output") {
      const char *value = require_value("--output");
      if (!value)
        return false;
      out->output_file = std::filesystem::path(value);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (out->backend.empty() || out->model_path.empty()) {
    return false;
  }
  return true;
}

std::unique_ptr<LlamaCppBackend> CreateBackend(const std::string &backend) {
  if (backend == "inferflux_cuda") {
    return std::make_unique<InferfluxCudaBackend>();
  }
  if (backend == "llama_cpp_cuda") {
    return std::make_unique<CudaBackend>();
  }
  return nullptr;
}

// Similarity metrics
struct SimilarityMetrics {
  float cosine_similarity{0.0f};
  float max_abs_diff{0.0f};
  float rmse{0.0f};
  float pearson_correlation{0.0f};
};

SimilarityMetrics ComputeSimilarity(const std::vector<float> &a,
                                     const std::vector<float> &b) {
  SimilarityMetrics metrics;

  if (a.size() != b.size() || a.empty()) {
    return metrics;
  }

  const size_t n = a.size();

  // Compute dot product, norms, and differences
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  float max_diff = 0.0f;
  float sum_diff_sq = 0.0f;
  float sum_a = 0.0f;
  float sum_b = 0.0f;
  float sum_a_sq = 0.0f;
  float sum_b_sq = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    const float ai = a[i];
    const float bi = b[i];
    const float diff = ai - bi;

    dot += ai * bi;
    norm_a += ai * ai;
    norm_b += bi * bi;
    max_diff = std::max(max_diff, std::abs(diff));
    sum_diff_sq += diff * diff;

    sum_a += ai;
    sum_b += bi;
    sum_a_sq += ai * ai;
    sum_b_sq += bi * bi;
  }

  // Cosine similarity
  const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  metrics.cosine_similarity = (denom > 0.0f) ? dot / denom : 0.0f;

  // Max absolute difference
  metrics.max_abs_diff = max_diff;

  // RMSE
  metrics.rmse = std::sqrt(sum_diff_sq / static_cast<float>(n));

  // Pearson correlation
  const float mean_a = sum_a / static_cast<float>(n);
  const float mean_b = sum_b / static_cast<float>(n);

  float covariance = 0.0f;
  float var_a = 0.0f;
  float var_b = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    const float ai = a[i];
    const float bi = b[i];
    const float diff_a = ai - mean_a;
    const float diff_b = bi - mean_b;

    covariance += diff_a * diff_b;
    var_a += diff_a * diff_a;
    var_b += diff_b * diff_b;
  }

  const float pearson_denom = std::sqrt(var_a) * std::sqrt(var_b);
  metrics.pearson_correlation = (pearson_denom > 0.0f) ? covariance / pearson_denom : 0.0f;

  return metrics;
}

json SerializeMetrics(const SimilarityMetrics &metrics) {
  return {
      {"cosine_similarity", metrics.cosine_similarity},
      {"max_abs_diff", metrics.max_abs_diff},
      {"rmse", metrics.rmse},
      {"pearson_correlation", metrics.pearson_correlation},
  };
}

int Run(const Options &options) {
  const std::string prompt = options.prompt.value_or("Hello world");

  // Create primary backend
  auto backend = CreateBackend(options.backend);
  if (!backend) {
    std::cerr << "Error: unsupported backend: " << options.backend << "\n";
    return 2;
  }

  // Configure backend
  if (options.backend == "inferflux_cuda") {
#ifdef _WIN32
    _putenv_s("INFERFLUX_CUDA_STRICT", "1");
    _putenv_s("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE", "1");
#else
    setenv("INFERFLUX_CUDA_STRICT", "1", 1);
    setenv("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE", "1", 1);
#endif
  }

  // Enable attention tensor capture
#ifdef _WIN32
  _putenv_s("INFERFLUX_DEBUG_ATTENTION_TENSORS", "1");
#else
  setenv("INFERFLUX_DEBUG_ATTENTION_TENSORS", "1", 1);
#endif

  LlamaBackendConfig config;
  config.ctx_size = options.ctx_size;
  config.batch_size = options.batch_size;
  config.gpu_layers = options.gpu_layers;
  config.use_flash_attention = options.use_flash_attention;
  config.max_parallel_sequences = options.max_parallel_sequences;

  SamplingParams sampling;
  sampling.temperature = 0.0f;
  sampling.top_k = 1;
  sampling.seed = 0;

  if (!backend->LoadModel(options.model_path, config)) {
    std::cerr << "Error: LoadModel failed\n";
    return 3;
  }
  if (!backend->IsReady()) {
    std::cerr << "Error: backend not ready\n";
    return 3;
  }

  backend->SetupSampler("", "", sampling);

  // Generate tokens
  std::vector<TokenLogprob> logprobs;
  const std::string output =
      backend->Generate(prompt, options.max_tokens, {}, {}, options.top_n, &logprobs, {});

  // Capture attention tensors
  AttentionTensorData primary_tensors = backend->CaptureAttentionTensors();

  backend->TeardownSampler();

  // Build result JSON
  json result = {
      {"ok", true},
      {"backend", options.backend},
      {"model_path", options.model_path.string()},
      {"prompt", prompt},
      {"max_tokens", options.max_tokens},
      {"output", output},
      {"token_count", static_cast<int>(logprobs.size())},
  };

  // Add attention tensor data
  if (primary_tensors.ok) {
    json layers = json::array();
    for (const auto &snapshot : primary_tensors.snapshots) {
      json layer_data = {
          {"layer_idx", snapshot.layer_idx},
          {"operation", snapshot.operation},
          {"shape", snapshot.shape},
          {"data", snapshot.data},
          {"data_size", snapshot.data.size()},
      };
      layers.push_back(layer_data);
    }
    result["attention_tensors"] = layers;
  } else {
    result["attention_tensors"] = nullptr;
    result["attention_tensors_error"] = primary_tensors.error;
  }

  // Compare with secondary backend if specified
  if (options.compare_backend) {
    auto compare_backend = CreateBackend(*options.compare_backend);
    if (!compare_backend) {
      std::cerr << "Warning: unsupported compare backend: " << *options.compare_backend
                << "\n";
    } else {
      // Configure compare backend
      if (*options.compare_backend == "inferflux_cuda") {
#ifdef _WIN32
        _putenv_s("INFERFLUX_CUDA_STRICT", "1");
        _putenv_s("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE", "1");
#else
        setenv("INFERFLUX_CUDA_STRICT", "1", 1);
        setenv("INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE", "1", 1);
#endif
      }

      if (!compare_backend->LoadModel(options.model_path, config)) {
        std::cerr << "Warning: compare backend LoadModel failed\n";
      } else {
        compare_backend->SetupSampler("", "", sampling);

        // Generate tokens (same prompt)
        std::vector<TokenLogprob> compare_logprobs;
        compare_backend->Generate(prompt, options.max_tokens, {}, {}, options.top_n,
                                  &compare_logprobs, {});

        // Capture attention tensors
        AttentionTensorData compare_tensors = compare_backend->CaptureAttentionTensors();

        compare_backend->TeardownSampler();

        // Add comparison data to result
        result["compare_backend"] = *options.compare_backend;
        result["compare_output"] = output;

        if (compare_tensors.ok && primary_tensors.ok) {
          // Compare tensors layer by layer
          json comparison = json::array();

          // Assume both backends return snapshots in the same order
          const size_t n_snapshots =
              std::min(primary_tensors.snapshots.size(), compare_tensors.snapshots.size());

          for (size_t i = 0; i < n_snapshots; ++i) {
            const auto &primary_snap = primary_tensors.snapshots[i];
            const auto &compare_snap = compare_tensors.snapshots[i];

            if (primary_snap.layer_idx == compare_snap.layer_idx &&
                primary_snap.operation == compare_snap.operation &&
                primary_snap.data.size() == compare_snap.data.size()) {

              SimilarityMetrics metrics =
                  ComputeSimilarity(primary_snap.data, compare_snap.data);

              comparison.push_back({
                  {"layer_idx", primary_snap.layer_idx},
                  {"operation", primary_snap.operation},
                  {"metrics", SerializeMetrics(metrics)},
              });
            }
          }

          result["tensor_comparison"] = comparison;
        } else {
          result["tensor_comparison"] = nullptr;
          if (!compare_tensors.ok) {
            result["compare_error"] = compare_tensors.error;
          }
        }
      }
    }
  }

  // Output result
  std::string json_str = result.dump(2);
  if (options.output_file) {
    std::ofstream out(*options.output_file);
    if (!out) {
      std::cerr << "Error: cannot open output file: " << *options.output_file << "\n";
      return 4;
    }
    out << json_str << std::endl;
  } else {
    std::cout << json_str << std::endl;
  }

  return 0;
}

} // namespace
} // namespace inferflux

int main(int argc, char **argv) {
  inferflux::Options options;
  if (!inferflux::ParseArgs(argc, argv, &options)) {
    inferflux::PrintUsage(argv[0]);
    return 2;
  }
  try {
    return inferflux::Run(options);
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Error: unknown exception\n";
    return 1;
  }
}
