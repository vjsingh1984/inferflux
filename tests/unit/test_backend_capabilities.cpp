#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/backend_capabilities.h"

using namespace inferflux;

TEST_CASE("CheckBackendCapabilities succeeds when requirements are empty",
          "[backend_capabilities]") {
  BackendCapabilities capabilities;
  BackendFeatureRequirements requirements;

  auto result = CheckBackendCapabilities(capabilities, requirements);
  REQUIRE(result.supported);
  REQUIRE(result.missing_feature.empty());
  REQUIRE(result.reason.empty());
}

TEST_CASE("CheckBackendCapabilities rejects unsupported logprobs",
          "[backend_capabilities]") {
  BackendCapabilities capabilities;
  capabilities.supports_logprobs = false;

  BackendFeatureRequirements requirements;
  requirements.needs_logprobs = true;

  auto result = CheckBackendCapabilities(capabilities, requirements);
  REQUIRE_FALSE(result.supported);
  REQUIRE(result.missing_feature == "logprobs");
  REQUIRE(result.reason.find("logprobs") != std::string::npos);
}

TEST_CASE("CheckBackendCapabilities rejects unsupported vision",
          "[backend_capabilities]") {
  BackendCapabilities capabilities;
  capabilities.supports_vision = false;

  BackendFeatureRequirements requirements;
  requirements.needs_vision = true;

  auto result = CheckBackendCapabilities(capabilities, requirements);
  REQUIRE_FALSE(result.supported);
  REQUIRE(result.missing_feature == "vision");
  REQUIRE(result.reason.find("image inputs") != std::string::npos);
}
