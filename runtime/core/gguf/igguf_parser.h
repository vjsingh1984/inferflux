#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Forward declarations to avoid including CUDA headers
namespace inferflux::runtime::core::gguf {

// GGUF value types (from GGUF spec)
enum class GgufValueType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
  COUNT = 13,
};

// GGUF tensor types (quantization types)
enum class GgufTensorType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
};

// GGUF header structure
struct GgufHeader {
  uint32_t magic;
  uint32_t version;
  int64_t tensor_count;
  int64_t kv_count;
};

// GGUF tensor information
struct GgufTensorInfo {
  std::string name;
  std::vector<size_t> shape;
  GgufTensorType type;
  size_t offset;
  size_t byte_size;

  // Helper methods
  bool is_quantized() const {
    return type != GgufTensorType::F32 && type != GgufTensorType::F16;
  }

  size_t block_size() const;
  std::string type_string() const;
};

/**
 * @brief Interface for GGUF file parsing
 *
 * Provides CPU-only GGUF parsing functionality that can be used
 * by both CPU and CUDA builds. Separates parsing logic from GPU execution.
 */
class IGGUFParser {
public:
  virtual ~IGGUFParser() = default;

  /**
   * @brief Parse GGUF header from file
   * @param path Path to GGUF file
   * @param header Output header structure
   * @return true on success
   */
  virtual bool ParseHeader(const std::filesystem::path &path,
                           GgufHeader *header) = 0;

  /**
   * @brief Read tensor info from file
   * @param file Open FILE pointer
   * @param info Output tensor info
   * @return true on success
   */
  virtual bool ReadTensorInfo(std::FILE *file, GgufTensorInfo *info) = 0;

  /**
   * @brief Read key-value pair from file
   * @param file Open FILE pointer
   * @param key Output key
   * @param type Output value type
   * @return true on success
   */
  virtual bool ReadKeyValue(std::FILE *file, std::string *key,
                            GgufValueType *type) = 0;

  /**
   * @brief Skip a value in file based on type
   * @param file Open FILE pointer
   * @param type Value type to skip
   * @return true on success
   */
  virtual bool SkipValue(std::FILE *file, GgufValueType type) = 0;

  /**
   * @brief Validate GGUF header
   * @param header Header to validate
   * @return true if header is valid
   */
  virtual bool ValidateHeader(const GgufHeader &header) const = 0;

  /**
   * @brief Check if parser is available
   * @return true if parser can be used
   */
  virtual bool IsAvailable() const = 0;
};

/**
 * @brief Factory function to create parser
 * @return CPU GGUF parser instance
 */
std::unique_ptr<IGGUFParser> CreateCpuGgufParser();

} // namespace inferflux::runtime::core::gguf
