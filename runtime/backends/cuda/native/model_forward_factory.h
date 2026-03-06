#pragma once

#include "runtime/backends/cuda/native/model_forward.h"
#include <memory>
#include <string>

namespace inferflux {

// Forward declaration for QuantizedForward (defined in quantized_forward.h)
// Note: This must be a full include where QuantizedForward is used as a
// complete type
class QuantizedForward;

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

/**
 * Create a QuantizedForward for GGUF quantized models.
 *
 * Returns a QuantizedForward instance that can handle GGUF quantization.
 * Returns nullptr for unsupported model types.
 *
 * Note: This function returns std::unique_ptr<ModelForward> (base class)
 * to maintain compatibility with the existing factory interface.
 * The actual object is a QuantizedForward instance.
 */
std::unique_ptr<ModelForward>
CreateQuantizedForwardAsModelForward(const std::string &model_type);

} // namespace inferflux
