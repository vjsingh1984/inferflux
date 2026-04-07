#pragma once

#include "runtime/core/gguf/igguf_parser.h"
#include "server/quantization_detection.h"

namespace inferflux {
namespace server {

/**
 * @brief CPU-only quantization detector
 *
 * Uses CpuGgufParser to detect quantization types from GGUF files
 * without any CUDA dependencies.
 */
class CpuQuantizationDetector : public IQuantizationDetector {
public:
  QuantizationType
  DetectFromGgufMetadata(const std::filesystem::path &path) const override;

  bool IsAvailable() const override;

  const char *GetName() const override;

private:
  inferflux::QuantizationType GgufTensorTypeToQuantizationType(
      runtime::core::gguf::GgufTensorType type) const;
};

} // namespace server
} // namespace inferflux
