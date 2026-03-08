#pragma once

#include "model/tokenizer.h"

#include <memory>
#include <string>

namespace inferflux {

/**
 * Create a tokenizer appropriate for the given model path and format.
 *
 * Resolution order:
 *   - gguf / auto-detected .gguf → LlamaTokenizer
 *   - safetensors / hf → HFTokenizer, fallback to LlamaTokenizer via GGUF
 *     sidecar
 *
 * Returns nullptr only if all loading strategies fail.
 */
std::unique_ptr<ITokenizer> CreateTokenizer(const std::string &model_path,
                                            const std::string &format = "auto");

} // namespace inferflux
