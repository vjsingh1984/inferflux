#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "scheduler/model_selection.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace inferflux;

namespace {

class StubRouter : public ModelRouter {
public:
  void AddModel(const ModelInfo &info,
                std::shared_ptr<LlamaCPUBackend> backend) {
    models_[info.id] = info;
    backends_[info.id] = std::move(backend);
    if (default_model_id_.empty()) {
      default_model_id_ = info.id;
    }
  }

  std::vector<ModelInfo> ListModels() const override {
    std::vector<ModelInfo> out;
    out.reserve(models_.size());
    for (const auto &entry : models_) {
      out.push_back(entry.second);
    }
    return out;
  }

  std::string LoadModel(const std::string &, const std::string &,
                        const std::string &) override {
    return "";
  }

  bool UnloadModel(const std::string &id) override {
    return models_.erase(id) > 0;
  }

  ModelInfo *Resolve(const std::string &requested_model) override {
    if (models_.empty()) {
      return nullptr;
    }
    if (!requested_model.empty()) {
      auto it = models_.find(requested_model);
      if (it != models_.end()) {
        return &it->second;
      }
      auto def = models_.find(default_model_id_);
      return def == models_.end() ? nullptr : &def->second;
    }
    auto def = models_.find(default_model_id_);
    return def == models_.end() ? nullptr : &def->second;
  }

  std::shared_ptr<LlamaCPUBackend>
  GetBackend(const std::string &model_id) override {
    auto it = backends_.find(model_id);
    if (it == backends_.end()) {
      return nullptr;
    }
    return it->second;
  }

  bool SetDefaultModel(const std::string &model_id) override {
    if (models_.find(model_id) == models_.end()) {
      return false;
    }
    default_model_id_ = model_id;
    return true;
  }

  std::string DefaultModelId() const override { return default_model_id_; }

  std::string Name() const override { return "stub_router"; }

private:
  std::unordered_map<std::string, ModelInfo> models_;
  std::unordered_map<std::string, std::shared_ptr<LlamaCPUBackend>> backends_;
  std::string default_model_id_;
};

std::shared_ptr<LlamaCPUBackend> ReadyBackend() {
  auto backend = std::make_shared<LlamaCPUBackend>();
  backend->ForceReadyForTests();
  return backend;
}

} // namespace

TEST_CASE("SelectModelForRequest falls back for default routing when primary "
          "lacks required capability",
          "[model_selection]") {
  StubRouter router;

  ModelInfo primary;
  primary.id = "tinyllama-cuda";
  primary.path = "/models/tinyllama.gguf";
  primary.backend = "cuda";
  primary.capabilities.supports_logprobs = false;
  router.AddModel(primary, ReadyBackend());

  ModelInfo fallback;
  fallback.id = "tinyllama-cpu";
  fallback.path = "/models/tinyllama.gguf";
  fallback.backend = "cpu";
  fallback.capabilities.supports_logprobs = true;
  router.AddModel(fallback, ReadyBackend());

  REQUIRE(router.SetDefaultModel(primary.id));

  BackendFeatureRequirements req;
  req.needs_logprobs = true;
  auto selection = SelectModelForRequest(
      &router, "", req,
      ModelSelectionOptions{/*allow_capability_fallback_for_default=*/true,
                            /*require_ready_backend=*/true});

  REQUIRE(selection.status == ModelSelectionStatus::kSelected);
  REQUIRE(selection.used_fallback);
  REQUIRE(selection.info.id == fallback.id);
  REQUIRE(selection.fallback_from_backend == primary.backend);
  REQUIRE(selection.fallback_feature == "logprobs");
  REQUIRE(selection.backend != nullptr);
}

TEST_CASE("SelectModelForRequest keeps explicit model pinned",
          "[model_selection]") {
  StubRouter router;

  ModelInfo primary;
  primary.id = "tinyllama-cuda";
  primary.path = "/models/tinyllama.gguf";
  primary.backend = "cuda";
  primary.capabilities.supports_logprobs = false;
  router.AddModel(primary, ReadyBackend());

  ModelInfo fallback;
  fallback.id = "tinyllama-cpu";
  fallback.path = "/models/tinyllama.gguf";
  fallback.backend = "cpu";
  fallback.capabilities.supports_logprobs = true;
  router.AddModel(fallback, ReadyBackend());

  BackendFeatureRequirements req;
  req.needs_logprobs = true;
  auto selection = SelectModelForRequest(
      &router, primary.id, req,
      ModelSelectionOptions{/*allow_capability_fallback_for_default=*/true,
                            /*require_ready_backend=*/true});

  REQUIRE(selection.status == ModelSelectionStatus::kUnsupported);
  REQUIRE_FALSE(selection.used_fallback);
  REQUIRE(selection.missing_feature == "logprobs");
}

TEST_CASE(
    "SelectModelForRequest falls back when primary backend is unavailable",
    "[model_selection]") {
  StubRouter router;

  ModelInfo primary;
  primary.id = "tinyllama-cuda";
  primary.path = "/models/tinyllama.gguf";
  primary.backend = "cuda";
  primary.capabilities.supports_logprobs = true;
  router.AddModel(primary, std::make_shared<LlamaCPUBackend>());

  ModelInfo fallback;
  fallback.id = "tinyllama-cpu";
  fallback.path = "/models/tinyllama.gguf";
  fallback.backend = "cpu";
  fallback.capabilities.supports_logprobs = true;
  router.AddModel(fallback, ReadyBackend());

  REQUIRE(router.SetDefaultModel(primary.id));

  BackendFeatureRequirements req;
  auto selection = SelectModelForRequest(
      &router, "", req,
      ModelSelectionOptions{/*allow_capability_fallback_for_default=*/true,
                            /*require_ready_backend=*/true});

  REQUIRE(selection.status == ModelSelectionStatus::kSelected);
  REQUIRE(selection.used_fallback);
  REQUIRE(selection.info.id == fallback.id);
  REQUIRE(selection.fallback_feature == "backend_unavailable");
}
