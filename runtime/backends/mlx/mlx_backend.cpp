#include "runtime/backends/mlx/mlx_backend.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace inferflux {

MlxBackend::MlxBackend() = default;

bool MlxBackend::LoadModel(const std::filesystem::path &model_path,
                           const LlamaBackendConfig &config) {
#if !INFERFLUX_HAS_MLX
  std::cerr << "[mlx] backend requested but binary was built without "
               "ENABLE_MLX support. Rebuild with -DENABLE_MLX=ON.\n";
  return false;
#else
  // MLX-native path: a model directory containing config.json + *.safetensors.
  if (std::filesystem::is_directory(model_path)) {
    // Stage 1: parse config.json and catalogue all tensors from shard headers.
    descriptor_ = loader_.LoadDirectory(model_path);
    if (!descriptor_.valid) {
      std::cerr << "[mlx] Directory load failed: " << model_path << "\n";
      return false;
    }

    // Stage 2: materialise all weight tensors into MLX arrays on Metal.
    weight_store_ = loader_.LoadWeights(descriptor_);
    if (!weight_store_.ok) {
      std::cerr << "[mlx] Weight materialisation failed for " << model_path
                << "\n";
      return false;
    }

    // Initialise the execution engine and hand it the loaded weight store.
    if (!engine_.Initialize()) {
      std::cerr << "[mlx] Engine initialisation failed\n";
      return false;
    }
    if (!engine_.LoadWeights(weight_store_, descriptor_.config)) {
      std::cerr << "[mlx] Engine weight loading failed\n";
      return false;
    }
    // Stage 3: load the HuggingFace tokenizer from the same directory.
    if (!tokenizer_.Load(model_path)) {
      std::cerr << "[mlx] Tokenizer load failed for " << model_path
                << " (continuing without tokenizer)\n";
      // Not fatal — tokenizer is optional until Stage 4 inference.
    }

    engine_ready_ = true;

    std::cout << "[mlx] Model ready: " << descriptor_.config.model_type << " ("
              << descriptor_.config.num_hidden_layers << " layers, "
              << weight_store_.count << " tensors). "
              << "Stage 3 (kernel execution) is the next slice.\n";
    return true;
  }

  // GGUF file path: delegate to the llama.cpp backend.
  return LlamaCPUBackend::LoadModel(model_path, config);
#endif
}

std::string MlxBackend::InferText(const std::string &prompt, int max_new_tokens,
                                  bool add_bos) const {
  if (!engine_ready_ || !engine_.WeightsLoaded()) {
    std::cerr << "[mlx] InferText: engine not ready\n";
    return "";
  }
  if (!tokenizer_.Loaded()) {
    std::cerr << "[mlx] InferText: tokenizer not loaded\n";
    return "";
  }

  // Encode prompt (const_cast needed because engine_.Reset()/Step() are not
  // const, but we use a mutable helper engine_ member).
  auto enc = tokenizer_.Encode(prompt, add_bos);
  if (!enc.ok || enc.ids.empty())
    return "";

  // Const method — borrow a mutable reference via const_cast.
  // engine_ is logically part of mutable inference state.
  MlxExecutionEngine &eng = const_cast<MlxExecutionEngine &>(engine_);
  eng.Reset();

  // Prefill: run all prompt tokens in one forward pass.
  int32_t next_tok = eng.Step(enc.ids);
  if (next_tok < 0)
    return "";

  // Decode loop: generate up to max_new_tokens.
  int32_t eos_id = tokenizer_.EosId();
  std::vector<int32_t> generated;
  generated.push_back(next_tok);

  for (int t = 1; t < max_new_tokens; ++t) {
    if (next_tok == eos_id)
      break;
    next_tok = eng.Step({next_tok});
    if (next_tok < 0)
      break;
    generated.push_back(next_tok);
  }

  return tokenizer_.Decode(generated, /*skip_special=*/true);
}

// ---------------------------------------------------------------------------
// LlamaCPUBackend virtual overrides
// ---------------------------------------------------------------------------

bool MlxBackend::IsReady() const {
  return engine_ready_ || LlamaCPUBackend::IsReady();
}

int MlxBackend::TokenCount(const std::string &text) const {
  if (tokenizer_.Loaded()) {
    auto enc = tokenizer_.Encode(text, /*add_bos=*/false);
    return enc.ok ? static_cast<int>(enc.ids.size()) : 0;
  }
  return LlamaCPUBackend::TokenCount(text);
}

std::string
MlxBackend::Generate(const std::string &prompt, int max_tokens,
                     const std::function<bool(const std::string &)> &on_chunk,
                     const std::function<bool()> &should_stop,
                     int /*logprob_top_n*/,
                     std::vector<TokenLogprob> * /*out_logprobs*/,
                     const std::vector<std::string> &stop_seqs) {
  // Not engine-ready: delegate to the llama.cpp base (GGUF path).
  if (!engine_ready_) {
    return LlamaCPUBackend::Generate(prompt, max_tokens, on_chunk, should_stop,
                                     0, nullptr, stop_seqs);
  }
  if (!tokenizer_.Loaded()) {
    std::cerr << "[mlx] Generate: tokenizer not loaded\n";
    return "";
  }

  auto enc = tokenizer_.Encode(prompt, /*add_bos=*/true);
  if (!enc.ok || enc.ids.empty())
    return "";

  engine_.Reset();
  const int32_t eos_id = tokenizer_.EosId();

  // Prefill: evaluate all prompt tokens in one forward pass.
  int32_t next_tok = engine_.Step(enc.ids);
  if (next_tok < 0)
    return "";

  std::string output;

  for (int t = 0; t < max_tokens; ++t) {
    if (next_tok == eos_id)
      break;
    if (should_stop && should_stop())
      break;

    std::string piece = tokenizer_.Decode({next_tok}, /*skip_special=*/true);
    output += piece;

    // Stop sequence check: strip trailing stop seq and halt.
    bool stop_hit = false;
    for (const auto &seq : stop_seqs) {
      if (seq.empty())
        continue;
      if (output.size() >= seq.size() &&
          output.compare(output.size() - seq.size(), seq.size(), seq) == 0) {
        output.resize(output.size() - seq.size());
        stop_hit = true;
        break;
      }
    }
    if (stop_hit)
      break;

    // Emit streaming chunk (after stop-seq check so we never stream the stop).
    if (on_chunk && !piece.empty()) {
      if (!on_chunk(piece))
        break;
    }

    // Decode step: generate next token from the just-sampled one.
    next_tok = engine_.Step({next_tok});
    if (next_tok < 0)
      break;
  }

  return output;
}

MlxBackend::~MlxBackend() {
  if (engine_ready_) {
    engine_.Shutdown();
  }
}

} // namespace inferflux
