#include "model/tokenizer/simple_tokenizer.h"
#include "policy/policy_store.h"
#include "runtime/backends/backend_manager.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/prefix_cache/prefix_cache.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/model_router.h"
#include "scheduler/scheduler.h"
#include "scheduler/single_model_router.h"
#include "server/auth/api_key_auth.h"
#include "server/auth/oidc_validator.h"
#include "server/auth/rate_limiter.h"
#include "server/http/http_server.h"
#include "server/logging/audit_logger.h"
#include "server/metrics/metrics.h"
#include "server/policy/guardrail.h"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct ModelConfig {
  std::string id;
  std::string path;
  std::string backend;
  bool make_default{false};
};

namespace {
std::string Trim(const std::string& input) {
  auto start = input.find_first_not_of(" \t");
  auto end = input.find_last_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  return input.substr(start, end - start + 1);
}

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::size_t ParsePositiveSize(const std::string& text) {
  try {
    long long value = std::stoll(text);
    if (value > 0) {
      return static_cast<std::size_t>(value);
    }
  } catch (...) {
  }
  return 0;
}

bool ParseBool(const std::string& value) {
  auto lowered = ToLower(value);
  return lowered == "true" || lowered == "1" || lowered == "yes";
}

std::vector<ModelConfig> ParseModelsEnv(const std::string& raw) {
  std::vector<ModelConfig> entries;
  std::stringstream ss(raw);
  std::string segment;
  while (std::getline(ss, segment, ';')) {
    ModelConfig cfg;
    std::stringstream kv_stream(segment);
    std::string pair;
    while (std::getline(kv_stream, pair, ',')) {
      auto eq = pair.find('=');
      if (eq == std::string::npos) {
        continue;
      }
      auto key = Trim(pair.substr(0, eq));
      auto value = Trim(pair.substr(eq + 1));
      if (key == "id") {
        cfg.id = value;
      } else if (key == "path") {
        cfg.path = value;
      } else if (key == "backend") {
        cfg.backend = ToLower(value);
      } else if (key == "default") {
        cfg.make_default = ParseBool(value);
      }
    }
    if (!cfg.path.empty()) {
      entries.push_back(cfg);
    }
  }
  return entries;
}
}

static std::atomic<bool> g_running{true};

void SignalHandler(int) { g_running = false; }

int main(int argc, char** argv) {
  std::string config_path = "config/server.yaml";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    }
  }

  std::string host = "0.0.0.0";
  int port = 8080;
  auto auth = std::make_shared<inferflux::ApiKeyAuth>();
  std::string model_path;
  int mps_layers = 0;
  int rate_limit_per_minute = 0;
  std::string audit_log_path;
  std::vector<std::string> guard_blocklist;
  std::string oidc_issuer;
  std::string oidc_audience;
  bool tls_enabled = false;
  std::string tls_cert_path;
  std::string tls_key_path;
  bool cuda_enabled = false;
  bool cuda_flash_attention_enabled = false;
  int cuda_flash_attention_tile = 128;
  bool speculative_enabled = false;
  std::string speculative_draft_model;
  int speculative_max_prefill_tokens = 256;
  int speculative_chunk_size = 4;
  std::string nvme_offload_path;
  std::string opa_endpoint;
  std::size_t paged_kv_pages = 32;
  std::string paged_kv_policy = "lru";
  std::size_t nvme_writer_workers = 1;
  std::size_t nvme_writer_queue_depth = 256;
  std::size_t prefix_cache_capacity = 256;
  std::vector<ModelConfig> configured_models;
  std::string default_model_override;

  struct ApiKeyEntry {
    std::string key;
    std::vector<std::string> scopes;
  };
  std::vector<ApiKeyEntry> api_key_entries;

  if (std::filesystem::exists(config_path)) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);

        // Server config
        if (config["server"]) {
            if (config["server"]["host"]) host = config["server"]["host"].as<std::string>();
            if (config["server"]["http_port"]) port = config["server"]["http_port"].as<int>();
        }

        // Legacy model config
        if (config["model"] && config["model"]["path"]) {
            model_path = config["model"]["path"].as<std::string>();
        }
        
        // Models config
        if (config["models"] && config["models"].IsSequence()) {
            for (const auto& model_node : config["models"]) {
                ModelConfig mc;
                if (model_node["id"]) mc.id = model_node["id"].as<std::string>();
                if (model_node["path"]) mc.path = model_node["path"].as<std::string>();
                if (model_node["backend"]) mc.backend = ToLower(model_node["backend"].as<std::string>());
                if (model_node["default"]) mc.make_default = model_node["default"].as<bool>();
                if (!mc.path.empty()) {
                    configured_models.push_back(mc);
                }
            }
        }

        // Runtime config
        if (config["runtime"]) {
            if (config["runtime"]["mps_layers"]) mps_layers = config["runtime"]["mps_layers"].as<int>();
            if (config["runtime"]["prefix_cache_capacity"]) prefix_cache_capacity = config["runtime"]["prefix_cache_capacity"].as<std::size_t>();
            if (config["runtime"]["cuda"] && config["runtime"]["cuda"]["enabled"]) {
                cuda_enabled = config["runtime"]["cuda"]["enabled"].as<bool>();
            }
            if (config["runtime"]["cuda"] && config["runtime"]["cuda"]["flash_attention"] && config["runtime"]["cuda"]["flash_attention"]["enabled"]) {
                cuda_flash_attention_enabled = config["runtime"]["cuda"]["flash_attention"]["enabled"].as<bool>();
            }
             if (config["runtime"]["cuda"] && config["runtime"]["cuda"]["flash_attention"] && config["runtime"]["cuda"]["flash_attention"]["tile_size"]) {
                cuda_flash_attention_tile = config["runtime"]["cuda"]["flash_attention"]["tile_size"].as<int>();
            }
            if (config["runtime"]["speculative_decoding"]) {
                if(config["runtime"]["speculative_decoding"]["enabled"]) speculative_enabled = config["runtime"]["speculative_decoding"]["enabled"].as<bool>();
                if(config["runtime"]["speculative_decoding"]["draft_model"]) speculative_draft_model = config["runtime"]["speculative_decoding"]["draft_model"].as<std::string>();
                if(config["runtime"]["speculative_decoding"]["max_prefill_tokens"]) speculative_max_prefill_tokens = config["runtime"]["speculative_decoding"]["max_prefill_tokens"].as<int>();
                if(config["runtime"]["speculative_decoding"]["chunk_size"]) speculative_chunk_size = config["runtime"]["speculative_decoding"]["chunk_size"].as<int>();
            }
            if (config["runtime"]["nvme_offload"]) {
                if(config["runtime"]["nvme_offload"]["path"]) nvme_offload_path = config["runtime"]["nvme_offload"]["path"].as<std::string>();
                if(config["runtime"]["nvme_offload"]["workers"]) nvme_writer_workers = config["runtime"]["nvme_offload"]["workers"].as<std::size_t>();
                if(config["runtime"]["nvme_offload"]["queue_depth"]) nvme_writer_queue_depth = config["runtime"]["nvme_offload"]["queue_depth"].as<std::size_t>();
            }
            if (config["runtime"]["paged_kv"]) {
                if(config["runtime"]["paged_kv"]["cpu_pages"]) paged_kv_pages = config["runtime"]["paged_kv"]["cpu_pages"].as<std::size_t>();
                if(config["runtime"]["paged_kv"]["eviction"]) paged_kv_policy = config["runtime"]["paged_kv"]["eviction"].as<std::string>();
            }
        }

        // Auth config
        if (config["auth"]) {
            if (config["auth"]["rate_limit_per_minute"]) rate_limit_per_minute = config["auth"]["rate_limit_per_minute"].as<int>();
            if (config["auth"]["oidc_issuer"]) oidc_issuer = config["auth"]["oidc_issuer"].as<std::string>();
            if (config["auth"]["oidc_audience"]) oidc_audience = config["auth"]["oidc_audience"].as<std::string>();
            if (config["auth"]["api_keys"] && config["auth"]["api_keys"].IsSequence()) {
                for (const auto& key_node : config["auth"]["api_keys"]) {
                    ApiKeyEntry entry;
                    if (key_node.IsScalar()) { // Simple key string
                        entry.key = key_node.as<std::string>();
                    } else if (key_node.IsMap() && key_node["key"]) {
                        entry.key = key_node["key"].as<std::string>();
                        if (key_node["scopes"] && key_node["scopes"].IsSequence()) {
                            for (const auto& scope_node : key_node["scopes"]) {
                                entry.scopes.push_back(scope_node.as<std::string>());
                            }
                        }
                    }
                    if (!entry.key.empty()) {
                        api_key_entries.push_back(entry);
                    }
                }
            }
        }

        // Guardrails config
        if (config["guardrails"]) {
            if (config["guardrails"]["blocklist"] && config["guardrails"]["blocklist"].IsSequence()) {
                for (const auto& item : config["guardrails"]["blocklist"]) {
                    guard_blocklist.push_back(item.as<std::string>());
                }
            }
            if (config["guardrails"]["opa_endpoint"]) opa_endpoint = config["guardrails"]["opa_endpoint"].as<std::string>();
        }

        // Logging config
        if (config["logging"] && config["logging"]["audit_log"]) {
            audit_log_path = config["logging"]["audit_log"].as<std::string>();
        }

        // TLS config
        if (config["tls"]) {
            if (config["tls"]["enabled"]) tls_enabled = config["tls"]["enabled"].as<bool>();
            if (config["tls"]["cert_path"]) tls_cert_path = config["tls"]["cert_path"].as<std::string>();
            if (config["tls"]["key_path"]) tls_key_path = config["tls"]["key_path"].as<std::string>();
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing config file " << config_path << ": " << e.what() << std::endl;
    }
  }

  for (const auto& entry : api_key_entries) {
    auth->AddKey(entry.key, entry.scopes);
  }

  if (const char* env_keys = std::getenv("INFERFLUX_API_KEYS")) {
    std::stringstream ss(env_keys);
    std::string key;
    while (std::getline(ss, key, ',')) {
      auto trimmed = Trim(key);
      if (!trimmed.empty()) {
        auth->AddKey(trimmed, {});
      }
    }
  }
  // INFERCTL_API_KEY: convenience single-key override (used in dev/test environments).
  if (const char* env_key = std::getenv("INFERCTL_API_KEY")) {
    std::string key = Trim(env_key);
    if (!key.empty()) {
      auth->AddKey(key, {"generate", "read", "admin"});
    }
  }

  if (const char* env_model = std::getenv("INFERFLUX_MODEL_PATH")) {
    model_path = env_model;
  }
  if (const char* env_mps = std::getenv("INFERFLUX_MPS_LAYERS")) {
    mps_layers = std::stoi(env_mps);
  }
  if (const char* env_rate = std::getenv("INFERFLUX_RATE_LIMIT_PER_MINUTE")) {
    rate_limit_per_minute = std::stoi(env_rate);
  }
  if (const char* env_audit = std::getenv("INFERFLUX_AUDIT_LOG")) {
    audit_log_path = env_audit;
  }
  if (const char* env_blk = std::getenv("INFERFLUX_GUARDRAIL_BLOCKLIST")) {
    std::stringstream ss(env_blk);
    std::string item;
    while (std::getline(ss, item, ',')) {
      auto word = Trim(item);
      if (!word.empty()) {
        guard_blocklist.push_back(word);
      }
    }
  }
  if (const char* env_oidc_iss = std::getenv("INFERFLUX_OIDC_ISSUER")) {
    oidc_issuer = env_oidc_iss;
  }
  if (const char* env_oidc_aud = std::getenv("INFERFLUX_OIDC_AUDIENCE")) {
    oidc_audience = env_oidc_aud;
  }
  if (const char* env_spec = std::getenv("INFERFLUX_SPECULATIVE_ENABLED")) {
    speculative_enabled = std::string(env_spec) == "true";
  }
  if (const char* env_spec_draft = std::getenv("INFERFLUX_SPEC_DRAFT_MODEL")) {
    speculative_draft_model = env_spec_draft;
  }
  if (const char* env_spec_prefill = std::getenv("INFERFLUX_SPEC_MAX_PREFILL")) {
    speculative_max_prefill_tokens = std::stoi(env_spec_prefill);
  }
  if (const char* env_spec_chunk = std::getenv("INFERFLUX_SPEC_CHUNK_SIZE")) {
    auto chunk = ParsePositiveSize(env_spec_chunk);
    if (chunk > 0) {
      speculative_chunk_size = static_cast<int>(chunk);
    }
  }
  if (const char* env_nvme = std::getenv("INFERFLUX_NVME_OFFLOAD_PATH")) {
    nvme_offload_path = env_nvme;
  }
  if (const char* env_kv_pages = std::getenv("INFERFLUX_KV_CPU_PAGES")) {
    auto pages_value = ParsePositiveSize(env_kv_pages);
    if (pages_value > 0) {
      paged_kv_pages = pages_value;
    }
  }
  if (const char* env_kv_policy = std::getenv("INFERFLUX_KV_EVICTION")) {
    paged_kv_policy = env_kv_policy;
  }
  if (const char* env_nvme_workers = std::getenv("INFERFLUX_NVME_WORKERS")) {
    auto workers = ParsePositiveSize(env_nvme_workers);
    if (workers > 0) {
      nvme_writer_workers = workers;
    }
  }
  if (const char* env_nvme_depth = std::getenv("INFERFLUX_NVME_QUEUE_DEPTH")) {
    auto depth = ParsePositiveSize(env_nvme_depth);
    if (depth > 0) {
      nvme_writer_queue_depth = depth;
    }
  }
  if (const char* env_opa = std::getenv("INFERFLUX_OPA_ENDPOINT")) {
    opa_endpoint = env_opa;
  }
  if (const char* env_port = std::getenv("INFERFLUX_PORT_OVERRIDE")) {
    port = std::stoi(env_port);
  }
  if (const char* env_host = std::getenv("INFERFLUX_HOST_OVERRIDE")) {
    host = env_host;
  }
  if (const char* env_tls = std::getenv("INFERFLUX_TLS_ENABLED")) {
    std::string value = env_tls;
    tls_enabled = (value == "true" || value == "1");
  }
  if (const char* env_tls_cert = std::getenv("INFERFLUX_TLS_CERT_PATH")) {
    tls_cert_path = env_tls_cert;
  }
  if (const char* env_tls_key = std::getenv("INFERFLUX_TLS_KEY_PATH")) {
    tls_key_path = env_tls_key;
  }
  if (const char* env_prefix_cap = std::getenv("INFERFLUX_PREFIX_CACHE_CAPACITY")) {
    auto cap = ParsePositiveSize(env_prefix_cap);
    if (cap > 0) {
      prefix_cache_capacity = cap;
    }
  }
  if (const char* env_cuda = std::getenv("INFERFLUX_CUDA_ENABLED")) {
    std::string value = env_cuda;
    cuda_enabled = (value == "true" || value == "1");
  }
  if (const char* env_cuda_fa = std::getenv("INFERFLUX_CUDA_FLASH_ATTENTION")) {
    std::string value = env_cuda_fa;
    cuda_flash_attention_enabled = (value == "true" || value == "1");
  }
  if (const char* env_cuda_tile = std::getenv("INFERFLUX_CUDA_FLASH_TILE")) {
    auto tile = ParsePositiveSize(env_cuda_tile);
    if (tile > 0) {
      cuda_flash_attention_tile = static_cast<int>(tile);
    }
  }
  inferflux::FairnessConfig fairness_config;
  fairness_config.enable_preemption = false;
  fairness_config.high_priority_threshold = 5;
  fairness_config.max_timeslice_tokens = 0;
  if (const char* env_preempt = std::getenv("INFERFLUX_FAIRNESS_ENABLE_PREEMPTION")) {
    fairness_config.enable_preemption = std::string(env_preempt) == "true" ||
                                        std::string(env_preempt) == "1";
  }
  if (const char* env_threshold = std::getenv("INFERFLUX_FAIRNESS_PRIORITY_THRESHOLD")) {
    fairness_config.high_priority_threshold = std::stoi(env_threshold);
  }
  if (const char* env_timeslice = std::getenv("INFERFLUX_FAIRNESS_MAX_TIMESLICE")) {
    fairness_config.max_timeslice_tokens = std::stoi(env_timeslice);
  }
  if (const char* env_models = std::getenv("INFERFLUX_MODELS")) {
    auto parsed = ParseModelsEnv(env_models);
    if (!parsed.empty()) {
      configured_models = parsed;
    }
  }
  if (const char* env_default_model = std::getenv("INFERFLUX_DEFAULT_MODEL_ID")) {
    default_model_override = env_default_model;
  }

  if (configured_models.empty() && !model_path.empty()) {
    ModelConfig cfg;
    cfg.path = model_path;
    cfg.make_default = true;
    configured_models.push_back(cfg);
  }

  inferflux::HttpServer::TlsConfig tls_config;
  tls_config.enabled = tls_enabled;
  tls_config.cert_path = tls_cert_path;
  tls_config.key_path = tls_key_path;
  if (tls_config.enabled && (tls_config.cert_path.empty() || tls_config.key_path.empty())) {
    std::cerr << "[server] TLS enabled without cert/key paths; disabling in-process TLS" << std::endl;
    tls_config.enabled = false;
  }

  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto normalized_policy = ToLower(paged_kv_policy);
  inferflux::PagedKVCache::EvictionPolicy policy = inferflux::PagedKVCache::EvictionPolicy::kLRU;
  if (normalized_policy == "clock") {
    policy = inferflux::PagedKVCache::EvictionPolicy::kClock;
  }
  std::size_t cache_pages = paged_kv_pages == 0 ? 32 : paged_kv_pages;
  auto cache = std::make_shared<inferflux::PagedKVCache>(cache_pages, /*page_size_bytes=*/16384, policy);
  cache->ConfigureAsyncWriter(nvme_writer_workers, nvme_writer_queue_depth);
  if (!nvme_offload_path.empty()) {
    cache->SetOffloadPath(nvme_offload_path);
  }
  std::cout << "Paged KV cache pages: " << cache_pages << " eviction=" << normalized_policy << std::endl;
  inferflux::BackendManager backend_manager;
  std::string backend_label = "stub";
  std::string primary_model_id;
  inferflux::LlamaBackendConfig primary_cfg;
  primary_cfg.gpu_layers = mps_layers;
  primary_cfg.use_flash_attention = cuda_enabled && cuda_flash_attention_enabled;
  primary_cfg.flash_attention_tile = cuda_flash_attention_tile;

#ifndef INFERFLUX_HAS_CUDA
  if (cuda_enabled) {
    std::cerr << "[server] CUDA requested but binary was built without CUDA support. "
              << "Set ENABLE_CUDA=ON with a CUDA toolkit installed to enable GPU mode.\n";
    cuda_enabled = false;
  }
  if (cuda_flash_attention_enabled) {
    std::cerr << "[server] FlashAttention requested but CUDA support is unavailable. Disabling flash_attention.\n";
    cuda_flash_attention_enabled = false;
  }
#else
  if (cuda_enabled) {
    std::cout << "[server] CUDA support enabled; FlashAttention executor will be wired in upcoming releases.\n";
  }
#endif
  if (cuda_flash_attention_enabled && !cuda_enabled) {
    std::cerr << "[server] FlashAttention requires CUDA. Disable or build with CUDA support.\n";
    cuda_flash_attention_enabled = false;
  }
#ifdef INFERFLUX_HAS_CUDA
  if (cuda_flash_attention_enabled) {
    std::cout << "[server] FlashAttention toggled on (tile_size=" << cuda_flash_attention_tile << ").\n";
  }
#endif

  auto router = std::make_shared<inferflux::SingleModelRouter>();
  std::string resolved_default_model_id;
  std::string resolved_default_path = model_path;
  if (!configured_models.empty()) {
    for (auto& cfg : configured_models) {
      auto assigned_id = router->LoadModel(cfg.path, cfg.backend, cfg.id);
      if (assigned_id.empty()) {
        std::cerr << "[server] Failed to load model from " << cfg.path << std::endl;
        continue;
      }
      cfg.id = assigned_id;
      std::cout << "InferFlux loaded model: " << cfg.path
                << " (backend=" << (cfg.backend.empty() ? "cpu" : cfg.backend)
                << ", id=" << assigned_id << ")" << std::endl;
      if (resolved_default_model_id.empty()) {
        resolved_default_model_id = assigned_id;
        resolved_default_path = cfg.path;
      }
      if (cfg.make_default) {
        resolved_default_model_id = assigned_id;
        resolved_default_path = cfg.path;
      }
    }
  }
  if (!default_model_override.empty()) {
    if (router->SetDefaultModel(default_model_override)) {
      resolved_default_model_id = default_model_override;
    } else {
      std::cerr << "[server] INFERFLUX_DEFAULT_MODEL_ID=" << default_model_override
                << " did not match a loaded model\n";
    }
  } else if (!resolved_default_model_id.empty()) {
    router->SetDefaultModel(resolved_default_model_id);
  }
  auto llama_backend = resolved_default_model_id.empty()
                           ? nullptr
                           : router->GetBackend(resolved_default_model_id);
  primary_model_id = resolved_default_model_id;
  if (!resolved_default_path.empty()) {
    model_path = resolved_default_path;
  }
  if (!primary_model_id.empty()) {
    auto models = router->ListModels();
    for (const auto& info : models) {
      if (info.id == primary_model_id) {
        backend_label = info.backend.empty() ? backend_label : info.backend;
        break;
      }
    }
  }
  if (llama_backend && backend_label == "stub") {
    backend_label = cuda_enabled ? "cuda" : (mps_layers > 0 ? "mps" : "cpu");
  }
  std::string policy_store_path = "config/policy_store.conf";
  if (const char* env_policy = std::getenv("INFERFLUX_POLICY_STORE")) {
    policy_store_path = env_policy;
  }
  std::string policy_passphrase;
  if (const char* env_pass = std::getenv("INFERFLUX_POLICY_PASSPHRASE")) {
    policy_passphrase = env_pass;
  }
  inferflux::PolicyStore policy_store(policy_store_path, policy_passphrase);
  policy_store.Load();

  // Keys from PolicyStore are already SHA-256 hashed on disk — load them
  // directly into ApiKeyAuth without re-hashing.
  auto stored_keys = policy_store.ApiKeys();
  for (const auto& entry : stored_keys) {
    auth->AddKeyHashed(entry.key, entry.scopes);
  }

  // Fresh keys from config/env are plaintext — hash them into both auth and
  // the policy store (PolicyStore::SetApiKey hashes before storing).
  if (stored_keys.empty()) {
    for (const auto& entry : api_key_entries) {
      auth->AddKey(entry.key, entry.scopes);
      policy_store.SetApiKey(entry.key, entry.scopes);
    }
  }
  int store_limit = policy_store.RateLimitPerMinute();
  if (store_limit > 0) {
    rate_limit_per_minute = store_limit;
  } else if (rate_limit_per_minute > 0) {
    policy_store.SetRateLimitPerMinute(rate_limit_per_minute);
  }
  auto store_guardrail = policy_store.GuardrailBlocklist();
  if (!store_guardrail.empty()) {
    guard_blocklist = store_guardrail;
  } else if (!guard_blocklist.empty()) {
    policy_store.SetGuardrailBlocklist(guard_blocklist);
  }
  policy_store.Save();

  if (speculative_chunk_size <= 0) {
    speculative_chunk_size = 1;
  }
  inferflux::SpeculativeConfig spec_config;
  spec_config.enabled = speculative_enabled;
  spec_config.max_prefill_tokens = speculative_max_prefill_tokens;
  spec_config.draft_model = speculative_draft_model;
  spec_config.chunk_size = speculative_chunk_size;
  std::shared_ptr<inferflux::LlamaCPUBackend> draft_backend = llama_backend;
  if (speculative_enabled && !speculative_draft_model.empty() &&
      speculative_draft_model != model_path) {
    auto draft_cfg = primary_cfg;
    auto backend = backend_manager.LoadBackend("draft", speculative_draft_model, draft_cfg, cuda_enabled);
    if (backend) {
      draft_backend = backend;
    }
  }
  std::shared_ptr<inferflux::SpeculativeDecoder> speculative_decoder;
  if (speculative_enabled) {
    speculative_decoder = std::make_shared<inferflux::SpeculativeDecoder>(spec_config, device, &tokenizer,
                                                                          draft_backend);
  }

  // Build ModelRouter (SingleModelRouter is the default; future multi-model
  // routers can be substituted here without touching Scheduler or HttpServer).
  auto prefix_cache = std::make_shared<inferflux::PrefixCache>(prefix_cache_capacity);
  inferflux::Scheduler scheduler(tokenizer,
                                 device,
                                 cache,
                                 router,
                                 speculative_decoder,
                                 prefix_cache,
                                 fairness_config);
  auto& metrics = inferflux::GlobalMetrics();
  metrics.SetBackend(backend_label);
  inferflux::OIDCValidator oidc_validator(oidc_issuer, oidc_audience);
  inferflux::RateLimiter rate_limiter(rate_limit_per_minute);
  inferflux::Guardrail guardrail;
  guardrail.SetBlocklist(guard_blocklist);
  guardrail.SetOPAEndpoint(opa_endpoint);
  bool audit_debug_mode = false;
  if (const char* env_audit_debug = std::getenv("INFERFLUX_AUDIT_DEBUG")) {
    audit_debug_mode = std::string(env_audit_debug) == "true";
  }
  inferflux::AuditLogger audit_logger(audit_log_path, audit_debug_mode);
  std::cout << "Speculative decoding: " << (speculative_enabled ? "enabled" : "disabled")
            << " draft_model=" << (speculative_draft_model.empty() ? "<none>" : speculative_draft_model)
            << " max_prefill_tokens=" << speculative_max_prefill_tokens
            << " chunk_size=" << speculative_chunk_size << std::endl;
  if (!nvme_offload_path.empty()) {
    std::cout << "NVMe KV offload path: " << nvme_offload_path << " (workers=" << nvme_writer_workers
              << ", queue_depth=" << nvme_writer_queue_depth << ")" << std::endl;
  }
  if (!opa_endpoint.empty()) {
    std::cout << "OPA guardrail endpoint: " << opa_endpoint << std::endl;
  }
  int http_workers = 4;
  if (const char* env_workers = std::getenv("INFERFLUX_HTTP_WORKERS")) {
    try { http_workers = std::stoi(env_workers); } catch (...) {}
  }
  inferflux::HttpServer server(host, port, &scheduler, auth, &metrics, &oidc_validator,
                               rate_limit_per_minute > 0 ? &rate_limiter : nullptr,
                               guardrail.Enabled() ? &guardrail : nullptr,
                               audit_logger.Enabled() ? &audit_logger : nullptr,
                               &policy_store,
                               speculative_decoder,
                               tls_config,
                               http_workers);

  if (!router->DefaultModelId().empty()) {
    server.SetModelReady(true);
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  server.Start();
  std::cout << "InferFlux listening on " << host << ":" << port
            << (tls_config.enabled ? " (TLS enabled)" : "")
            << " prefix_cache_capacity=" << prefix_cache_capacity << std::endl;

  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  server.Stop();
  std::cout << "InferFlux shutting down" << std::endl;
  return 0;
}
