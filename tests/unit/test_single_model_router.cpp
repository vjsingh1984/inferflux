#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "scheduler/single_model_router.h"

namespace inferflux {

TEST_CASE("SingleModelRouter applies native capability contract",
          "[single_model_router]") {
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<LlamaCPUBackend>();
  backend->ForceReadyForTests();

  ModelInfo info;
  info.id = "native-contract";
  info.path = "models/native.gguf";
  info.backend = "cuda";
  info.backend_provider = "native";

  REQUIRE(router->RegisterModel(info, backend));
  auto models = router->ListModels();
  REQUIRE(models.size() == 1);

  const auto &caps = models.front().capabilities;
  REQUIRE_FALSE(caps.supports_logprobs);
  REQUIRE_FALSE(caps.supports_structured_output);
  REQUIRE_FALSE(caps.supports_embeddings);
  REQUIRE_FALSE(caps.supports_speculative_decoding);
  REQUIRE(caps.supports_kv_prefix_transfer);
}

TEST_CASE("SingleModelRouter keeps llama.cpp capability defaults",
          "[single_model_router]") {
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<LlamaCPUBackend>();
  backend->ForceReadyForTests();

  ModelInfo info;
  info.id = "llama-contract";
  info.path = "models/llama.gguf";
  info.backend = "cuda";
  info.backend_provider = "llama_cpp";

  REQUIRE(router->RegisterModel(info, backend));
  auto models = router->ListModels();
  REQUIRE(models.size() == 1);

  const auto &caps = models.front().capabilities;
  REQUIRE(caps.supports_logprobs);
  REQUIRE(caps.supports_structured_output);
  REQUIRE(caps.supports_embeddings);
}

} // namespace inferflux
