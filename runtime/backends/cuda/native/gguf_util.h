#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

/**
 * @brief GGUF file format constants
 *
 * GGUF (GPT-Generated Unified Format) is a binary file format
 * for storing tensors for GGML and associated data.
 *
 * Format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */
namespace GGUF {
  // Magic bytes: "GGUF"
  constexpr uint32_t MAGIC = 0x46554747;

  // Version numbers
  constexpr uint32_t VERSION = 3; // Latest version (as of 2024)

  // GGUF value types
  enum class ValueType : uint32_t {
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
    COUNT = 13, // Number of types
  };

  // GGML tensor types (quantization types)
  enum class TensorType : uint32_t {
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

  // Header structure
  struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t tensor_count;    // Number of tensors
    uint32_t kv_count;        // Number of key-value pairs
  };

  // Tensor information
  struct TensorInfo {
    std::string name;
    std::vector<uint32_t> shape;
    TensorType type;
    size_t offset;            // Offset in tensor data section
    size_t byte_size;         // Size in bytes

    // For quantized types: additional metadata
    bool is_quantized() const {
      return type != TensorType::F32 && type != TensorType::F16;
    }

    // Get block size for quantized types
    size_t block_size() const;

    // Get string representation of type
    std::string type_string() const;
  };
} // namespace GGUF

/**
 * @brief Utility class for reading GGUF files
 *
 * Provides low-level parsing functions for GGUF format.
 * Designed for testing and validation without CUDA dependencies.
 */
class GGUFReader {
public:
  /**
   * @brief Read variable-length integer from file
   * GGUF uses variable-length encoding for some values
   */
  static bool ReadVarint(FILE *file, uint64_t *value);

  /**
   * @brief Read string from file
   * Format: [len:uint64_t][data:len]
   */
  static bool ReadString(FILE *file, std::string *str);

  /**
   * @brief Read value from file based on type
   */
  static bool ReadValue(FILE *file, GGUF::ValueType type,
                        std::vector<uint8_t> *output);

  /**
   * @brief Parse tensor name to extract layer and component
   *
   * Examples:
   *   "blk.0.attn_q.weight" → layer=0, component=attn_q, type=weight
   *   "tok_emb.weight" → layer=-1, component=tok_emb, type=weight
   *   "output.weight" → layer=-2, component=output, type=weight
   */
  struct TensorNameParts {
    int layer{-1};      // -1 for embeddings, -2 for output
    std::string component; // attn_q, attn_k, ffn_gate, etc.
    std::string type;      // weight, bias
  };

  static TensorNameParts ParseTensorName(const std::string &name);

  /**
   * @brief Convert GGUF tensor name to internal convention
   *
   * Maps GGUF naming to internal naming used by the codebase:
   *   "blk.0.attn_q.weight" → "model.layers.0.self_attn.q_proj.weight"
   */
  static std::string MapTensorName(const std::string &gguf_name,
                                   const std::string &model_type = "qwen2");

private:
  // Tensor name mapping tables
  static const std::unordered_map<std::string, std::string>&
  GetQwen2TensorMap();

  static const std::unordered_map<std::string, std::string>&
  GetLlamaTensorMap();
};

/**
 * @brief Validate GGUF file header
 *
 * Checks magic number and version compatibility.
 */
bool ValidateGGUFHeader(const GGUF::Header &header);

/**
 * @brief Get string representation of value type
 */
std::string ValueTypeToString(GGUF::ValueType type);

/**
 * @brief Get string representation of tensor type
 */
std::string TensorTypeToString(GGUF::TensorType type);

/**
 * @brief Convert tensor type string to enum
 */
GGUF::TensorType StringToTensorType(const std::string &str);

/**
 * @brief Check if tensor type is quantized
 */
bool IsQuantizedType(GGUF::TensorType type);

/**
 * @brief Get quantization type string (e.g., "q4_k_m" from Q4_K)
 */
std::string GetQuantizationType(GGUF::TensorType type);

/**
 * @brief Calculate byte size for tensor type and dimensions
 */
size_t CalcTensorSize(GGUF::TensorType type,
                      const std::vector<uint32_t> &shape);

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
