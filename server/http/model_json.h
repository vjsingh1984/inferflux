#pragma once

/// @file model_json.h
/// @brief Model identity/metadata JSON builders extracted from http_server.cpp.

#include "runtime/backends/backend_capabilities.h"
#include "scheduler/model_router.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace inferflux {

/// Build capabilities JSON from BackendCapabilities.
nlohmann::json BuildCapabilitiesJson(const BackendCapabilities &capabilities);

/// Return the user-facing source path for a model.
std::string ModelSourcePath(const ModelInfo &info);

/// Return the effective load path (resolved from source/path).
std::string ModelEffectiveLoadPath(const ModelInfo &info);

/// Build backend exposure details (requested, exposed, provider, fallback).
nlohmann::json BuildBackendExposureJson(const ModelInfo &info);

/// Build model identity JSON with capabilities.
nlohmann::json BuildModelIdentityJson(const ModelInfo &info);

/// Build OpenAI-compatible model list entry.
nlohmann::json BuildOpenAIModelJson(const ModelInfo &info, int64_t created_ts);

/// Build admin-facing model JSON with provider and default flag.
nlohmann::json BuildAdminModelJson(const ModelInfo &info,
                                   const std::string &default_id);

} // namespace inferflux
