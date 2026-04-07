#pragma once

#include "runtime/core/gguf/igguf_parser.h"

namespace inferflux {
namespace runtime {
namespace core {
namespace gguf {

/**
 * @brief CPU-only GGUF parser implementation
 *
 * Provides CPU-side GGUF file parsing without any CUDA dependencies.
 * Can be used by both CPU-only and CUDA builds.
 */
class CpuGgufParser : public IGGUFParser {
public:
  bool ParseHeader(const std::filesystem::path &path,
                   GgufHeader *header) override;

  bool ReadTensorInfo(std::FILE *file, GgufTensorInfo *info) override;

  bool ReadKeyValue(std::FILE *file, std::string *key,
                    GgufValueType *type) override;

  bool SkipValue(std::FILE *file, GgufValueType type) override;

  bool ValidateHeader(const GgufHeader &header) const override;

  bool IsAvailable() const override;

private:
  size_t CalcQuantizedSize(GgufTensorType type,
                           const std::vector<size_t> &shape) const;
};

} // namespace gguf
} // namespace core
} // namespace runtime
} // namespace inferflux
