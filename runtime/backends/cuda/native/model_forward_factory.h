#pragma once

#include "runtime/backends/cuda/native/model_forward.h"
#include <memory>
#include <string>

namespace inferflux {

/**
 * Create a ModelForward implementation based on model_type string.
 *
 * Supported types: "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2"
 * These all use the standard Llama-style decoder architecture.
 *
 * Returns nullptr for unsupported model types.
 */
std::unique_ptr<ModelForward> CreateModelForward(const std::string &model_type);

/**
 * Typed factory: creates ModelForward with specified dtype.
 * T = half (FP16) or __nv_bfloat16 (BF16).
 */
template <typename T>
std::unique_ptr<ModelForward>
CreateModelForwardTyped(const std::string &model_type);

} // namespace inferflux
