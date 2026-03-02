#include "runtime/backends/mlx/mlx_execution.h"
#include "runtime/backends/backend_utils.h"
#include "server/logging/logger.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>

#ifdef INFERFLUX_HAS_MLX
#include "mlx/c/array.h"
#include "mlx/c/fast.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#endif

namespace inferflux {

// ---------------------------------------------------------------------------
// RAII array wrapper + primitive op helpers (MLX build only)
// ---------------------------------------------------------------------------

#ifdef INFERFLUX_HAS_MLX
namespace {

// Non-copyable RAII owner for mlx_array.
struct Arr {
  mlx_array a{};

  Arr() = default;
  explicit Arr(mlx_array x) : a(x) {}
  ~Arr() { mlx_array_free(a); }
  Arr(const Arr &) = delete;
  Arr &operator=(const Arr &) = delete;
  Arr(Arr &&o) noexcept : a(o.a) { o.a = {}; }
  Arr &operator=(Arr &&o) noexcept {
    if (this != &o) {
      mlx_array_free(a);
      a = o.a;
      o.a = {};
    }
    return *this;
  }

  mlx_array get() const { return a; }
  bool valid() const { return a.ctx != nullptr; }
  mlx_array release() {
    mlx_array x = a;
    a = {};
    return x;
  }
};

// Fetch weight by name; logs if missing.
Arr GetW(const mlx_map_string_to_array &m, const char *name) {
  mlx_array a{};
  if (mlx_map_string_to_array_get(&a, m, name) != 0 || !a.ctx) {
    inferflux::log::Error("mlx_execution",
                          std::string("missing weight: ") + name);
    return Arr{};
  }
  return Arr(a);
}

// y = x @ W.T  (W: [out, in], x: [..., in])
Arr Linear(const Arr &x, const Arr &W, mlx_stream s) {
  mlx_array Wt{}, res{};
  mlx_transpose(&Wt, W.get(), s);
  mlx_matmul(&res, x.get(), Wt, s);
  mlx_array_free(Wt);
  return Arr(res);
}

// RMS norm via the MLX fast kernel.
Arr RMSNorm(const Arr &x, const Arr &w, float eps, mlx_stream s) {
  mlx_array res{};
  mlx_fast_rms_norm(&res, x.get(), w.get(), eps, s);
  return Arr(res);
}

// SiLU(x) = x / (1 + exp(-x))
Arr Silu(const Arr &x, mlx_stream s) {
  mlx_array neg{}, en{}, one_p{}, res{};
  mlx_negative(&neg, x.get(), s);
  mlx_exp(&en, neg, s);
  mlx_array_free(neg);
  mlx_array one_sc = mlx_array_new_float(1.0f);
  mlx_add(&one_p, one_sc, en, s);
  mlx_array_free(one_sc);
  mlx_array_free(en);
  mlx_divide(&res, x.get(), one_p, s);
  mlx_array_free(one_p);
  return Arr(res);
}

Arr Add(const Arr &a, const Arr &b, mlx_stream s) {
  mlx_array res{};
  mlx_add(&res, a.get(), b.get(), s);
  return Arr(res);
}

Arr Mul(const Arr &a, const Arr &b, mlx_stream s) {
  mlx_array res{};
  mlx_multiply(&res, a.get(), b.get(), s);
  return Arr(res);
}

Arr Reshape(const Arr &x, std::initializer_list<int> shape, mlx_stream s) {
  std::vector<int> sh(shape);
  mlx_array res{};
  mlx_reshape(&res, x.get(), sh.data(), sh.size(), s);
  return Arr(res);
}

Arr Transpose(const Arr &x, std::initializer_list<int> axes, mlx_stream s) {
  std::vector<int> ax(axes);
  mlx_array res{};
  mlx_transpose_axes(&res, x.get(), ax.data(), ax.size(), s);
  return Arr(res);
}

// Concatenate two raw arrays (without transferring ownership) along axis.
Arr Cat2(mlx_array a, mlx_array b, int axis, mlx_stream s) {
  mlx_vector_array va = mlx_vector_array_new();
  mlx_vector_array_append_value(va, a);
  mlx_vector_array_append_value(va, b);
  mlx_array res{};
  mlx_concatenate_axis(&res, va, axis, s);
  mlx_vector_array_free(va);
  return Arr(res);
}

// ---------------------------------------------------------------------------
// SampleToken — pure C++ sampling from a float logit array.
// Applies repetition/frequency/presence penalties, then optionally samples
// stochastically (top-k → min-p → top-p → temperature → categorical).
// Returns greedy token when sp.temperature <= 0.
// ---------------------------------------------------------------------------
int32_t SampleToken(const float *logits, int vocab_size,
                    const std::vector<int32_t> &token_history,
                    const SamplingParams &sp, std::mt19937 &rng) {
  // Make a mutable copy for penalty application.
  std::vector<float> l(logits, logits + vocab_size);

  // Apply repetition / frequency / presence penalties.
  const bool has_penalties =
      (sp.frequency_penalty != 0.0f || sp.presence_penalty != 0.0f ||
       sp.repetition_penalty != 1.0f);
  if (has_penalties && !token_history.empty()) {
    const int last_n =
        std::min(sp.penalty_last_n, static_cast<int>(token_history.size()));
    // Count token occurrences in the lookback window.
    std::vector<int> counts(vocab_size, 0);
    for (int i = static_cast<int>(token_history.size()) - last_n;
         i < static_cast<int>(token_history.size()); ++i) {
      int32_t t = token_history[i];
      if (t >= 0 && t < vocab_size)
        counts[t]++;
    }
    for (int i = 0; i < vocab_size; ++i) {
      if (counts[i] == 0)
        continue;
      // Repetition penalty (multiplicative).
      if (sp.repetition_penalty != 1.0f) {
        if (l[i] < 0.0f)
          l[i] *= sp.repetition_penalty;
        else
          l[i] /= sp.repetition_penalty;
      }
      // Frequency penalty (additive, per-occurrence).
      if (sp.frequency_penalty != 0.0f)
        l[i] -= sp.frequency_penalty * counts[i];
      // Presence penalty (additive, flat per unique token).
      if (sp.presence_penalty != 0.0f)
        l[i] -= sp.presence_penalty;
    }
  }

  // Greedy shortcut.
  if (sp.temperature <= 0.0f) {
    return static_cast<int32_t>(
        std::distance(l.begin(), std::max_element(l.begin(), l.end())));
  }

  // Temperature scaling.
  for (auto &v : l)
    v /= sp.temperature;

  // Softmax.
  float max_l = *std::max_element(l.begin(), l.end());
  double sum_exp = 0.0;
  for (auto &v : l) {
    v = std::exp(v - max_l);
    sum_exp += v;
  }
  for (auto &v : l)
    v = static_cast<float>(v / sum_exp);

  // Build index array for filtering.
  std::vector<int32_t> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), 0);

  // Top-K filter.
  if (sp.top_k > 0 && sp.top_k < vocab_size) {
    std::partial_sort(indices.begin(), indices.begin() + sp.top_k,
                      indices.end(),
                      [&](int32_t a, int32_t b) { return l[a] > l[b]; });
    indices.resize(sp.top_k);
  } else {
    std::sort(indices.begin(), indices.end(),
              [&](int32_t a, int32_t b) { return l[a] > l[b]; });
  }

  // Min-P filter: remove tokens with prob < min_p * max_prob.
  if (sp.min_p > 0.0f) {
    float max_p = l[indices[0]];
    float threshold = sp.min_p * max_p;
    auto it = std::find_if(indices.begin(), indices.end(),
                           [&](int32_t i) { return l[i] < threshold; });
    if (it != indices.begin()) // keep at least one
      indices.erase(it, indices.end());
  }

  // Top-P (nucleus) filter.
  if (sp.top_p < 1.0f) {
    float cum = 0.0f;
    size_t keep = indices.size();
    for (size_t i = 0; i < indices.size(); ++i) {
      cum += l[indices[i]];
      if (cum >= sp.top_p) {
        keep = i + 1;
        break;
      }
    }
    indices.resize(keep);
  }

  // Re-normalise the surviving logits and sample.
  std::vector<float> probs;
  probs.reserve(indices.size());
  float tot = 0.0f;
  for (int32_t idx : indices)
    tot += l[idx];
  for (int32_t idx : indices)
    probs.push_back(l[idx] / tot);

  std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
  int32_t sampled = dist(rng);
  return indices[static_cast<size_t>(sampled)];
}

} // anonymous namespace
#endif // INFERFLUX_HAS_MLX

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

MlxExecutionEngine::~MlxExecutionEngine() { Shutdown(); }

bool MlxExecutionEngine::Initialize() {
#ifndef INFERFLUX_HAS_MLX
  return false;
#else
  if (initialized_)
    return true;
  stream_ = mlx_default_gpu_stream_new();
  initialized_ = true;
  log::Info("mlx_execution", "Engine initialised on Metal GPU stream.");
  return true;
#endif
}

void MlxExecutionEngine::Shutdown() {
#ifdef INFERFLUX_HAS_MLX
  if (initialized_) {
    // Free all phased sequence slots.
    for (auto &[id, slot] : slots_) {
      for (auto &k : slot.key_cache)
        mlx_array_free(k);
      for (auto &v : slot.val_cache)
        mlx_array_free(v);
    }
    slots_.clear();
    active_seq_ = -1;
    Reset();
    mlx_stream_free(stream_);
    stream_ = {};
    weights_ = nullptr;
    initialized_ = false;
  }
#endif
}

bool MlxExecutionEngine::LoadWeights(const MlxWeightStore &store,
                                     const MlxModelConfig &cfg) {
#ifndef INFERFLUX_HAS_MLX
  (void)store;
  (void)cfg;
  return false;
#else
  if (!store.ok) {
    log::Error("mlx_execution", "LoadWeights: store not ready");
    return false;
  }
  weights_ = &store;
  config_ = cfg;
  Reset();
  key_cache_.resize(cfg.num_hidden_layers);
  val_cache_.resize(cfg.num_hidden_layers);
  log::Info("mlx_execution",
            "Weights wired: " + std::to_string(store.count) + " tensors, " +
                std::to_string(cfg.num_hidden_layers) + " layers.");
  return true;
#endif
}

void MlxExecutionEngine::Reset() {
#ifdef INFERFLUX_HAS_MLX
  for (auto &k : key_cache_) {
    mlx_array_free(k);
    k = {};
  }
  for (auto &v : val_cache_) {
    mlx_array_free(v);
    v = {};
  }
  n_past_ = 0;
  token_history_.clear();
  rng_seeded_ = false;
  // Reset does NOT touch phased slots — those are managed by FreeSlot().
  active_seq_ = -1;
#endif
}

// ---------------------------------------------------------------------------
// Phased slot management (INF-8)
// ---------------------------------------------------------------------------

void MlxExecutionEngine::AllocSlot(int seq_id) {
#ifdef INFERFLUX_HAS_MLX
  if (slots_.count(seq_id))
    return; // already allocated
  SlotState &s = slots_[seq_id];
  s.key_cache.assign(config_.num_hidden_layers, mlx_array{});
  s.val_cache.assign(config_.num_hidden_layers, mlx_array{});
  s.n_past = 0;
  s.rng_seeded = false;
#else
  (void)seq_id;
#endif
}

void MlxExecutionEngine::FreeSlot(int seq_id) {
#ifdef INFERFLUX_HAS_MLX
  if (active_seq_ == seq_id) {
    // The slot's arrays are currently in the flat members — free them.
    for (auto &k : key_cache_) {
      mlx_array_free(k);
      k = {};
    }
    for (auto &v : val_cache_) {
      mlx_array_free(v);
      v = {};
    }
    n_past_ = 0;
    token_history_.clear();
    rng_seeded_ = false;
    active_seq_ = -1;
  }
  auto it = slots_.find(seq_id);
  if (it == slots_.end())
    return;
  for (auto &k : it->second.key_cache)
    mlx_array_free(k);
  for (auto &v : it->second.val_cache)
    mlx_array_free(v);
  slots_.erase(it);
#else
  (void)seq_id;
#endif
}

void MlxExecutionEngine::SaveActiveSlot() {
#ifdef INFERFLUX_HAS_MLX
  if (active_seq_ < 0)
    return;
  SlotState &s = slots_[active_seq_];
  // Transfer flat arrays into the slot.
  s.key_cache = std::move(key_cache_);
  s.val_cache = std::move(val_cache_);
  s.n_past = n_past_;
  s.rng = rng_;
  s.rng_seeded = rng_seeded_;
  s.token_history = token_history_;
  // Re-allocate the flat arrays so Forward() can still use them.
  key_cache_.assign(config_.num_hidden_layers, mlx_array{});
  val_cache_.assign(config_.num_hidden_layers, mlx_array{});
  n_past_ = 0;
  token_history_.clear();
  rng_seeded_ = false;
  active_seq_ = -1;
#endif
}

void MlxExecutionEngine::LoadSlot(int seq_id) {
#ifdef INFERFLUX_HAS_MLX
  if (active_seq_ == seq_id)
    return;
  SaveActiveSlot();
  // Create slot if it doesn't exist yet.
  if (!slots_.count(seq_id)) {
    AllocSlot(seq_id);
  }
  SlotState &s = slots_[seq_id];
  key_cache_ = std::move(s.key_cache);
  val_cache_ = std::move(s.val_cache);
  n_past_ = s.n_past;
  rng_ = s.rng;
  rng_seeded_ = s.rng_seeded;
  token_history_ = s.token_history;
  if (key_cache_.size() < static_cast<size_t>(config_.num_hidden_layers))
    key_cache_.resize(config_.num_hidden_layers);
  if (val_cache_.size() < static_cast<size_t>(config_.num_hidden_layers))
    val_cache_.resize(config_.num_hidden_layers);
  active_seq_ = seq_id;
#else
  (void)seq_id;
#endif
}

void MlxExecutionEngine::CopySlotPrefix(int src_seq, int dst_seq,
                                        int n_tokens) {
#ifdef INFERFLUX_HAS_MLX
  if (n_tokens <= 0 || !initialized_)
    return;

  // Get source arrays (may be in flat members if src is active).
  const std::vector<mlx_array> *src_k = nullptr;
  const std::vector<mlx_array> *src_v = nullptr;
  int src_n_past = 0;
  if (active_seq_ == src_seq) {
    src_k = &key_cache_;
    src_v = &val_cache_;
    src_n_past = n_past_;
  } else {
    auto it = slots_.find(src_seq);
    if (it == slots_.end())
      return;
    src_k = &it->second.key_cache;
    src_v = &it->second.val_cache;
    src_n_past = it->second.n_past;
  }
  if (n_tokens > src_n_past)
    n_tokens = src_n_past;

  // Free any existing dst slot, then create a fresh one.
  // Note: dst must not be the currently active slot (would corrupt flat state).
  if (active_seq_ == dst_seq) {
    for (auto &k : key_cache_) {
      mlx_array_free(k);
      k = {};
    }
    for (auto &v : val_cache_) {
      mlx_array_free(v);
      v = {};
    }
    n_past_ = 0;
    token_history_.clear();
    rng_seeded_ = false;
    active_seq_ = -1;
  }
  {
    auto it = slots_.find(dst_seq);
    if (it != slots_.end()) {
      for (auto &k : it->second.key_cache)
        mlx_array_free(k);
      for (auto &v : it->second.val_cache)
        mlx_array_free(v);
      slots_.erase(it);
    }
  }
  SlotState &dst = slots_[dst_seq];
  dst.key_cache.assign(config_.num_hidden_layers, mlx_array{});
  dst.val_cache.assign(config_.num_hidden_layers, mlx_array{});
  dst.n_past = n_tokens;
  dst.rng_seeded = false;

  // Slice first n_tokens from each layer: KV shape [1, n_kv, total, head_dim].
  for (int i = 0; i < config_.num_hidden_layers; ++i) {
    if (!(*src_k)[i].ctx)
      continue;
    const int *sh = mlx_array_shape((*src_k)[i]);
    if (!sh)
      continue;
    // start = {0, 0, 0, 0}, stop = {sh[0], sh[1], n_tokens, sh[3]}, stride
    // all 1.
    int start[4] = {0, 0, 0, 0};
    int stop[4] = {sh[0], sh[1], n_tokens, sh[3]};
    int stride[4] = {1, 1, 1, 1};
    mlx_slice(&dst.key_cache[i], (*src_k)[i], start, 4, stop, 4, stride, 4,
              stream_);
    mlx_slice(&dst.val_cache[i], (*src_v)[i], start, 4, stop, 4, stride, 4,
              stream_);
    mlx_array_eval(dst.key_cache[i]);
    mlx_array_eval(dst.val_cache[i]);
  }
#else
  (void)src_seq;
  (void)dst_seq;
  (void)n_tokens;
#endif
}

int32_t MlxExecutionEngine::StepSeq(int seq_id,
                                    const std::vector<int32_t> &token_ids,
                                    const SamplingParams &sp, int logprob_top_n,
                                    std::vector<TokenLogprob> *out_logprobs) {
#ifndef INFERFLUX_HAS_MLX
  (void)seq_id;
  (void)token_ids;
  (void)sp;
  (void)logprob_top_n;
  (void)out_logprobs;
  return -1;
#else
  if (!weights_)
    return -1;
  LoadSlot(seq_id);
  return Forward(token_ids, sp, logprob_top_n, out_logprobs);
#endif
}

int MlxExecutionEngine::SeqNPast(int seq_id) const {
#ifdef INFERFLUX_HAS_MLX
  if (active_seq_ == seq_id)
    return n_past_;
  auto it = slots_.find(seq_id);
  if (it == slots_.end())
    return -1;
  return it->second.n_past;
#else
  (void)seq_id;
  return -1;
#endif
}

int32_t MlxExecutionEngine::Step(const std::vector<int32_t> &token_ids,
                                 const SamplingParams &sp, int logprob_top_n,
                                 std::vector<TokenLogprob> *out_logprobs) {
#ifndef INFERFLUX_HAS_MLX
  (void)token_ids;
  (void)sp;
  (void)logprob_top_n;
  (void)out_logprobs;
  return -1;
#else
  if (!weights_) {
    log::Error("mlx_execution", "Step called before LoadWeights");
    return -1;
  }
  return Forward(token_ids, sp, logprob_top_n, out_logprobs);
#endif
}

// ---------------------------------------------------------------------------
// Transformer forward pass — LLaMA/Mistral architecture
// ---------------------------------------------------------------------------

#ifdef INFERFLUX_HAS_MLX

int32_t MlxExecutionEngine::Forward(const std::vector<int32_t> &token_ids,
                                    const SamplingParams &sp, int logprob_top_n,
                                    std::vector<TokenLogprob> *out_logprobs) {
  const int seq_len = static_cast<int>(token_ids.size());
  const int n_heads = config_.num_attention_heads;
  const int n_kv = config_.num_key_value_heads;
  const int hidden = config_.hidden_size;
  const int head_dim = hidden / n_heads;
  const float eps = config_.rms_norm_eps;
  const float attn_sc = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const auto &wm = weights_->weights;
  mlx_stream s = stream_;

  // Causal mask for multi-token prefill; empty = no mask for single decode
  // step.
  const char *mask = (seq_len > 1) ? "causal" : "";

  // RoPE base θ (from config, default 10000).
  const mlx_optional_float rope_base = {config_.rope_theta, true};

  // ── 1. Token embedding ──────────────────────────────────────────────────
  Arr emb_w = GetW(wm, "model.embed_tokens.weight");
  if (!emb_w.valid())
    return -1;

  Arr x;
  {
    int shape1[1] = {seq_len};
    Arr idx(mlx_array_new_data(token_ids.data(), shape1, 1, MLX_INT32));
    mlx_array x2d{};
    mlx_take_axis(&x2d, emb_w.get(), idx.get(), 0, s); // [seq, hidden]
    mlx_array x3d{};
    int s3[3] = {1, seq_len, hidden};
    mlx_reshape(&x3d, x2d, s3, 3, s); // [1, seq, hidden]
    mlx_array_free(x2d);
    x = Arr(x3d);
  }

  // ── 2. Transformer layers ─────────────────────────────────────────────
  for (int i = 0; i < config_.num_hidden_layers; ++i) {
    const std::string pfx = "model.layers." + std::to_string(i) + ".";

    // 2a. Attention input norm.
    Arr anorm_w = GetW(wm, (pfx + "input_layernorm.weight").c_str());
    if (!anorm_w.valid())
      return -1;
    Arr xn = RMSNorm(x, anorm_w, eps, s);

    // 2b. Q / K / V linear projections.
    Arr wq = GetW(wm, (pfx + "self_attn.q_proj.weight").c_str());
    Arr wk = GetW(wm, (pfx + "self_attn.k_proj.weight").c_str());
    Arr wv = GetW(wm, (pfx + "self_attn.v_proj.weight").c_str());
    if (!wq.valid() || !wk.valid() || !wv.valid())
      return -1;

    Arr q = Linear(xn, wq, s); // [1, seq, n_heads*head_dim]
    Arr k = Linear(xn, wk, s); // [1, seq, n_kv*head_dim]
    Arr v = Linear(xn, wv, s); // [1, seq, n_kv*head_dim]

    // 2c. Reshape: [1, seq, n_heads, head_dim].
    q = Reshape(q, {1, seq_len, n_heads, head_dim}, s);
    k = Reshape(k, {1, seq_len, n_kv, head_dim}, s);
    v = Reshape(v, {1, seq_len, n_kv, head_dim}, s);

    // 2d. RoPE (applied before transposing to attention layout).
    {
      mlx_array null_freqs{};
      mlx_array qr{}, kr{};
      mlx_fast_rope(&qr, q.get(), head_dim, false, rope_base,
                    config_.rope_freq_scale, n_past_, null_freqs, s);
      mlx_fast_rope(&kr, k.get(), head_dim, false, rope_base,
                    config_.rope_freq_scale, n_past_, null_freqs, s);
      q = Arr(qr);
      k = Arr(kr);
    }

    // 2e. Transpose to [1, n_heads/n_kv, seq, head_dim] for SDPA.
    q = Transpose(q, {0, 2, 1, 3}, s);
    k = Transpose(k, {0, 2, 1, 3}, s);
    v = Transpose(v, {0, 2, 1, 3}, s);

    // 2f. KV-cache update.
    //   Concatenate old cache with new K/V along sequence axis (axis=2).
    if (key_cache_[i].ctx != nullptr) {
      Arr new_k = Cat2(key_cache_[i], k.get(), 2, s);
      Arr new_v = Cat2(val_cache_[i], v.get(), 2, s);
      mlx_array_free(key_cache_[i]);
      mlx_array_free(val_cache_[i]);
      key_cache_[i] = new_k.release();
      val_cache_[i] = new_v.release();
    } else {
      key_cache_[i] = k.release();
      val_cache_[i] = v.release();
    }

    // 2g. Scaled dot-product attention.
    mlx_array attn_out{};
    mlx_array null_arr{};
    mlx_fast_scaled_dot_product_attention(&attn_out, q.get(), key_cache_[i],
                                          val_cache_[i], attn_sc, mask,
                                          null_arr, null_arr, s);
    // attn_out: [1, n_heads, seq, head_dim]

    // Merge heads: [1, n_heads, seq, head_dim] → [1, seq, hidden]
    Arr ao(attn_out);
    ao = Transpose(ao, {0, 2, 1, 3}, s);       // [1, seq, n_heads, head_dim]
    ao = Reshape(ao, {1, seq_len, hidden}, s); // [1, seq, hidden]

    // 2h. Output projection + residual.
    Arr wo = GetW(wm, (pfx + "self_attn.o_proj.weight").c_str());
    if (!wo.valid())
      return -1;
    x = Add(x, Linear(ao, wo, s), s);

    // 2i. Post-attention norm + SwiGLU MLP.
    Arr pnorm_w = GetW(wm, (pfx + "post_attention_layernorm.weight").c_str());
    if (!pnorm_w.valid())
      return -1;
    Arr xn2 = RMSNorm(x, pnorm_w, eps, s);

    Arr wg = GetW(wm, (pfx + "mlp.gate_proj.weight").c_str());
    Arr wu = GetW(wm, (pfx + "mlp.up_proj.weight").c_str());
    Arr wd = GetW(wm, (pfx + "mlp.down_proj.weight").c_str());
    if (!wg.valid() || !wu.valid() || !wd.valid())
      return -1;

    // SwiGLU: silu(gate) * up
    Arr mlp_out =
        Linear(Mul(Silu(Linear(xn2, wg, s), s), Linear(xn2, wu, s), s), wd, s);
    x = Add(x, mlp_out, s);
  }

  // ── 3. Final norm ─────────────────────────────────────────────────────
  Arr final_w = GetW(wm, "model.norm.weight");
  if (!final_w.valid())
    return -1;
  x = RMSNorm(x, final_w, eps, s);

  // ── 4. LM head: logits [1, seq, vocab] ────────────────────────────────
  Arr lm_w = GetW(wm, "lm_head.weight");
  if (!lm_w.valid())
    return -1;
  Arr logits = Linear(x, lm_w, s);

  // ── 5. Extract last-position logits [1, vocab] ────────────────────────
  Arr last_idx(mlx_array_new_int(seq_len - 1));
  mlx_array last_logits_raw{};
  mlx_take_axis(&last_logits_raw, logits.get(), last_idx.get(), 1, s);
  Arr ll(last_logits_raw);

  // Cast to float32 and force GPU→CPU evaluation so we can read logits.
  {
    mlx_array lf32{};
    mlx_astype(&lf32, ll.get(), MLX_FLOAT32, s);
    ll = Arr(lf32);
  }
  mlx_array_eval(ll.get());

  const int vocab_size = config_.vocab_size;
  const float *lp = mlx_array_data_float32(ll.get());

  // Seed the PRNG lazily.
  if (!rng_seeded_) {
    if (sp.seed != UINT32_MAX) {
      rng_.seed(sp.seed);
    } else {
      std::random_device rd;
      rng_.seed(rd());
    }
    rng_seeded_ = true;
  } else if (sp.seed != UINT32_MAX) {
    // Re-seed for deterministic sequences when caller provides a fixed seed.
    rng_.seed(sp.seed + static_cast<uint32_t>(n_past_));
  }

  // Collect logprobs before sampling (logits still valid here).
  int32_t token_id = -1;
  if (lp && vocab_size > 0) {
    // Sample token using SamplingParams (greedy when temperature <= 0).
    token_id = SampleToken(lp, vocab_size, token_history_, sp, rng_);

    if (out_logprobs && logprob_top_n > 0) {
      // id_to_token function: return empty string for out-of-range ids.
      auto id_to_token_fn = [](int32_t /*id*/) -> std::string { return ""; };
      out_logprobs->push_back(ComputeLogprob(lp, vocab_size, token_id, "",
                                             logprob_top_n, id_to_token_fn));
    }
  } else {
    // Fallback: read scalar via argmax (shouldn't happen with valid config).
    int ndim = static_cast<int>(mlx_array_ndim(ll.get()));
    mlx_array tok_arr{};
    mlx_argmax_axis(&tok_arr, ll.get(), ndim - 1, false, s);
    Arr tok(tok_arr);
    mlx_array_eval(tok.get());
    mlx_array_item_int32(&token_id, tok.get());
  }

  // Append sampled token to lookback history.
  if (token_id >= 0)
    token_history_.push_back(token_id);

  // Materialise KV cache for the next step (avoids lazy-graph accumulation).
  for (int i = 0; i < config_.num_hidden_layers; ++i) {
    if (key_cache_[i].ctx)
      mlx_array_eval(key_cache_[i]);
    if (val_cache_[i].ctx)
      mlx_array_eval(val_cache_[i]);
  }

  n_past_ += seq_len;
  return token_id;
}

#endif // INFERFLUX_HAS_MLX

} // namespace inferflux
