#include "runtime/backends/mlx/mlx_execution.h"

#include <cmath>
#include <iostream>
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
    std::cerr << "[mlx_exec] missing weight: " << name << "\n";
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
  std::cout << "[mlx_exec] Engine initialised on Metal GPU stream.\n";
  return true;
#endif
}

void MlxExecutionEngine::Shutdown() {
#ifdef INFERFLUX_HAS_MLX
  if (initialized_) {
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
    std::cerr << "[mlx_exec] LoadWeights: store not ready\n";
    return false;
  }
  weights_ = &store;
  config_ = cfg;
  Reset();
  key_cache_.resize(cfg.num_hidden_layers);
  val_cache_.resize(cfg.num_hidden_layers);
  std::cout << "[mlx_exec] Weights wired: " << store.count << " tensors, "
            << cfg.num_hidden_layers << " layers.\n";
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
#endif
}

int32_t MlxExecutionEngine::Step(const std::vector<int32_t> &token_ids) {
#ifndef INFERFLUX_HAS_MLX
  (void)token_ids;
  return -1;
#else
  if (!weights_) {
    std::cerr << "[mlx_exec] Step called before LoadWeights\n";
    return -1;
  }
  return Forward(token_ids);
#endif
}

// ---------------------------------------------------------------------------
// Transformer forward pass — LLaMA/Mistral architecture
// ---------------------------------------------------------------------------

#ifdef INFERFLUX_HAS_MLX

int32_t MlxExecutionEngine::Forward(const std::vector<int32_t> &token_ids) {
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
      mlx_fast_rope(&qr, q.get(), head_dim, false, rope_base, 1.0f, n_past_,
                    null_freqs, s);
      mlx_fast_rope(&kr, k.get(), head_dim, false, rope_base, 1.0f, n_past_,
                    null_freqs, s);
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

  // ── 5. Greedy sample from last-position logits ────────────────────────
  // Extract position seq_len-1 along axis 1 → [1, vocab] (scalar idx removes
  // dim).
  Arr last_idx(mlx_array_new_int(seq_len - 1));
  mlx_array last_logits{};
  mlx_take_axis(&last_logits, logits.get(), last_idx.get(), 1, s);
  Arr ll(last_logits);

  // Argmax over vocab (last) axis.
  int ndim = static_cast<int>(mlx_array_ndim(ll.get()));
  mlx_array tok_arr{};
  mlx_argmax_axis(&tok_arr, ll.get(), ndim - 1, false, s);
  Arr tok(tok_arr);

  // Force evaluation and read scalar token ID.
  mlx_array_eval(tok.get());
  int32_t token_id = -1;
  mlx_array_item_int32(&token_id, tok.get());

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
