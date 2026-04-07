#pragma once

#include <filesystem>
#include <memory>

namespace inferflux {

// Forward declaration (QuantizationType defined in startup_advisor.h)
enum class QuantizationType;

namespace server {

/**
 * @brief Interface for quantization detection from GGUF files
 *
 * Separates quantization detection logic from CUDA dependencies.
 * Allows CPU-only and CUDA implementations.
 */
class IQuantizationDetector {
public:
  virtual ~IQuantizationDetector() = default;

  /**
   * @brief Detect quantization type from GGUF file metadata
   * @param path Path to GGUF model file
   * @return Detected quantization type, or kUnknown if detection fails
   */
  virtual QuantizationType
  DetectFromGgufMetadata(const std::filesystem::path &path) const = 0;

  /**
   * @brief Check if detector is available
   * @return true if detector can be used
   */
  virtual bool IsAvailable() const = 0;

  /**
   * @brief Get detector name (for logging)
   * @return Human-readable detector name
   */
  virtual const char *GetName() const = 0;
};

/**
 * @brief Factory function to create quantization detector
 * @return Best available detector (CPU or CUDA-based)
 */
std::unique_ptr<IQuantizationDetector> CreateQuantizationDetector();

} // namespace server
} // namespace inferflux
