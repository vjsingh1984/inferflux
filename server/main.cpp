#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/http/http_server.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace {
std::string Trim(const std::string& input) {
  auto start = input.find_first_not_of(" \t");
  auto end = input.find_last_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  return input.substr(start, end - start + 1);
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

  if (std::filesystem::exists(config_path)) {
    std::ifstream input(config_path);
    std::string line;
    bool in_auth = false;
    bool in_model = false;
    bool in_runtime = false;
    bool in_api_keys = false;
    while (std::getline(input, line)) {
      auto trimmed = Trim(line);
      if (trimmed.empty() || trimmed[0] == '#') {
        continue;
      }
      if (trimmed == "server:") {
        in_auth = false;
        in_runtime = false;
        in_api_keys = false;
        continue;
      }
      if (trimmed == "model:") {
        in_model = true;
        in_runtime = false;
        continue;
      }
      if (trimmed == "auth:") {
        in_model = false;
        in_runtime = false;
        in_auth = true;
        continue;
      }
      if (trimmed == "runtime:") {
        in_model = false;
        in_auth = false;
        in_api_keys = false;
        in_runtime = true;
        continue;
      }
      if (trimmed == "adapters:" || trimmed == "logging:") {
        in_model = false;
        in_auth = false;
        in_runtime = false;
        in_api_keys = false;
        continue;
      }
      if (in_auth && trimmed.rfind("api_keys:", 0) == 0) {
        in_api_keys = true;
        continue;
      }
      if (in_api_keys && trimmed.rfind("-", 0) == 0) {
        auto key = Trim(trimmed.substr(1));
        if (!key.empty()) {
          auth->AddKey(key);
        }
        continue;
      }
      if (in_model && trimmed.rfind("path:", 0) == 0) {
        model_path = Trim(trimmed.substr(trimmed.find(':') + 1));
        continue;
      }
      if (in_runtime && trimmed.rfind("mps_layers:", 0) == 0) {
        mps_layers = std::stoi(trimmed.substr(trimmed.find(':') + 1));
        continue;
      }
      if (trimmed.rfind("http_port:", 0) == 0) {
        port = std::stoi(trimmed.substr(trimmed.find(':') + 1));
      }
      if (trimmed.rfind("host:", 0) == 0) {
        host = Trim(trimmed.substr(trimmed.find(':') + 1));
      }
    }
  }

  if (const char* env_keys = std::getenv("INFERFLUX_API_KEYS")) {
    std::stringstream ss(env_keys);
    std::string key;
    while (std::getline(ss, key, ',')) {
      auto trimmed = Trim(key);
      if (!trimmed.empty()) {
        auth->AddKey(trimmed);
      }
    }
  }

  if (const char* env_model = std::getenv("INFERFLUX_MODEL_PATH")) {
    model_path = env_model;
  }
  if (const char* env_mps = std::getenv("INFERFLUX_MPS_LAYERS")) {
    mps_layers = std::stoi(env_mps);
  }

  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(/*pages=*/32, /*page_size_bytes=*/16384);
  std::shared_ptr<inferflux::LlamaCPUBackend> llama_backend;
  std::string backend_label = "stub";
  if (!model_path.empty()) {
    llama_backend = std::make_shared<inferflux::LlamaCPUBackend>();
    inferflux::LlamaBackendConfig backend_config;
    backend_config.gpu_layers = std::max(0, mps_layers);
    if (!llama_backend->LoadModel(model_path, backend_config)) {
      std::cerr << "Failed to load llama.cpp model at " << model_path
                << ". Falling back to stub responses." << std::endl;
      llama_backend.reset();
    } else {
      backend_label = backend_config.gpu_layers > 0 ? "mps" : "cpu";
      std::cout << "InferFlux loaded model: " << model_path << " (backend=" << backend_label << ")" << std::endl;
    }
  }
  inferflux::Scheduler scheduler(tokenizer, device, cache, llama_backend);
  auto& metrics = inferflux::GlobalMetrics();
  metrics.SetBackend(backend_label);
  inferflux::HttpServer server(host, port, &scheduler, auth, &metrics);

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  server.Start();
  std::cout << "InferFlux listening on " << host << ":" << port << std::endl;

  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  server.Stop();
  std::cout << "InferFlux shutting down" << std::endl;
  return 0;
}
