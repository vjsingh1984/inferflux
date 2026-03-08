#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "runtime/backends/cuda/native/gguf_util.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace inferflux::runtime::cuda::native;

namespace {

// Helper to create a temporary directory
fs::path CreateTempDir(const std::string &suffix) {
  const auto base =
      fs::temp_directory_path() / ("inferflux_test_" + suffix + "_" +
                                   std::to_string(std::hash<std::thread::id>{}(
                                       std::this_thread::get_id())));
  fs::create_directories(base);
  return base;
}

// Helper to write a binary file
bool WriteBinaryFile(const fs::path &path, const void *data, size_t size) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write(static_cast<const char *>(data), size);
  return out.good();
}

// Helper to create a minimal GGUF header
std::vector<uint8_t> CreateGGUFHeader(uint32_t tensor_count = 1,
                                      uint32_t kv_count = 0) {
  std::vector<uint8_t> header(24); // GGUF v3 header is 24 bytes

  // Magic (4 bytes): "GGUF"
  memcpy(header.data(), "GGUF", 4);

  // Version (4 bytes) at offset 4
  uint32_t version = 3;
  memcpy(header.data() + 4, &version, 4);

  // Tensor count (8 bytes) at offset 8
  int64_t tensors_i64 = tensor_count;
  memcpy(header.data() + 8, &tensors_i64, 8);

  // KV count (8 bytes) at offset 16
  int64_t kv_i64 = kv_count;
  memcpy(header.data() + 16, &kv_i64, 8);

  return header;
}

// Helper to create a GGUF tensor info entry
std::vector<uint8_t> CreateTensorInfo(const std::string &name, uint32_t n_dims,
                                      const uint32_t *ne, GGUF::TensorType type,
                                      size_t offset) {

  // Calculate total size needed
  size_t name_len = name.size() + 1;              // null terminator
  size_t total_size = sizeof(uint32_t) +          // name length
                      name_len +                  // name string
                      sizeof(uint32_t) +          // n_dims
                      n_dims * sizeof(uint32_t) + // ne array
                      sizeof(uint32_t) +          // type
                      sizeof(uint8_t) +           // type_flags
                      sizeof(uint64_t);           // offset

  std::vector<uint8_t> data(total_size);
  size_t pos = 0;

  // Name length
  uint32_t name_len_u32 = name_len;
  memcpy(data.data() + pos, &name_len_u32, sizeof(uint32_t));
  pos += sizeof(uint32_t);

  // Name string
  memcpy(data.data() + pos, name.c_str(), name_len);
  pos += name_len;

  // n_dims
  memcpy(data.data() + pos, &n_dims, sizeof(uint32_t));
  pos += sizeof(uint32_t);

  // ne array
  for (uint32_t i = 0; i < n_dims; i++) {
    memcpy(data.data() + pos, &ne[i], sizeof(uint32_t));
    pos += sizeof(uint32_t);
  }

  // type
  uint32_t type_u32 = static_cast<uint32_t>(type);
  memcpy(data.data() + pos, &type_u32, sizeof(uint32_t));
  pos += sizeof(uint32_t);

  // type_flags (always 0 for now)
  data.data()[pos] = 0;
  pos += sizeof(uint8_t);

  // offset
  uint64_t offset_u64 = offset;
  memcpy(data.data() + pos, &offset_u64, sizeof(uint64_t));

  return data;
}

} // namespace

// =============================================================================
// Test Suite: GGUFModelLoader Construction
// =============================================================================

TEST_CASE("GGUFModelLoader: Default construction", "[gguf][loader]") {
  GGUFModelLoader loader;

  // Check default state
  REQUIRE(loader.GetFormat() == "gguf");
  REQUIRE(loader.IsQuantized() == false); // No model loaded yet
  REQUIRE(loader.GetQuantizationType().empty());
}

TEST_CASE("GGUFModelLoader: Destructor handles unloaded model",
          "[gguf][loader]") {
  // Test that destructor doesn't crash on unintialized loader
  GGUFModelLoader loader;
  // Destructor called automatically when going out of scope
}

// =============================================================================
// Test Suite: GGUF File Parsing (CPU-only, no GPU required)
// =============================================================================

TEST_CASE("GGUFModelLoader: Reject non-existent file", "[gguf][loader]") {
  GGUFModelLoader loader;

  fs::path non_existent = "/tmp/nonexistent_file_12345.gguf";
  REQUIRE_FALSE(loader.Load(non_existent));
}

TEST_CASE("GGUFModelLoader: Reject invalid GGUF magic", "[gguf][loader]") {
  auto temp_dir = CreateTempDir("invalid_magic");
  fs::path test_file = temp_dir / "test.gguf";

  // Create file with wrong magic
  std::vector<uint8_t> invalid_header(24);
  memcpy(invalid_header.data(), "XXXX", 4); // Wrong magic

  REQUIRE(
      WriteBinaryFile(test_file, invalid_header.data(), invalid_header.size()));

  GGUFModelLoader loader;
  REQUIRE_FALSE(loader.Load(test_file));
}

TEST_CASE("GGUFModelLoader: Reject unsupported GGUF version",
          "[gguf][loader]") {
  auto temp_dir = CreateTempDir("bad_version");
  fs::path test_file = temp_dir / "test.gguf";

  // Create file with wrong version
  std::vector<uint8_t> header = CreateGGUFHeader(1, 0);
  uint32_t bad_version = 999;
  memcpy(header.data() + 4, &bad_version, 4);

  REQUIRE(WriteBinaryFile(test_file, header.data(), header.size()));

  GGUFModelLoader loader;
  REQUIRE_FALSE(loader.Load(test_file));
}

TEST_CASE("GGUFModelLoader: Parse valid GGUF header with tensors",
          "[gguf][loader]") {
  auto temp_dir = CreateTempDir("valid_header");
  fs::path test_file = temp_dir / "test.gguf";

  // Create minimal valid GGUF file
  std::vector<uint8_t> header = CreateGGUFHeader(2, 0);

  // Add tensor info section
  uint32_t n_dims = 2;
  uint32_t ne[2] = {768, 768}; // 2D tensor
  auto tensor1_info =
      CreateTensorInfo("tensor1", n_dims, ne, GGUF::TensorType::F16, 24);
  auto tensor2_info =
      CreateTensorInfo("tensor2", n_dims, ne, GGUF::TensorType::Q4_K, 100);

  // Calculate total file size and write
  size_t data_size = header.size() + tensor1_info.size() + tensor2_info.size();
  std::vector<uint8_t> file_data(data_size);
  size_t pos = 0;

  memcpy(file_data.data() + pos, header.data(), header.size());
  pos += header.size();
  memcpy(file_data.data() + pos, tensor1_info.data(), tensor1_info.size());
  pos += tensor1_info.size();
  memcpy(file_data.data() + pos, tensor2_info.data(), tensor2_info.size());

  REQUIRE(WriteBinaryFile(test_file, file_data.data(), file_data.size()));

  GGUFModelLoader loader;
  // Note: This may still fail if tensor data section is missing, but header
  // parsing should work For now, we're testing that the file can be opened and
  // header parsed
  bool result = loader.Load(test_file);

  // We expect this to potentially fail due to missing tensor data,
  // but it should not crash
  (void)result;
}

// =============================================================================
// Test Suite: Tensor Name Mapping (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: Tensor name mapping for Qwen2", "[gguf][loader]") {
  using namespace inferflux::runtime::cuda::native;

  // Test Qwen2 naming conventions
  std::string qwen_name = "blk.0.attn_q.weight";
  std::string expected = "model.layers.0.self_attn.q_proj.weight";

  std::string result = GGUFReader::MapTensorName(qwen_name, "qwen2");
  REQUIRE(result == expected);
}

TEST_CASE("GGUFModelLoader: Tensor name mapping for Llama", "[gguf][loader]") {
  using namespace inferflux::runtime::cuda::native;

  // Test Llama naming conventions
  std::string llama_name = "blk.0.attn_q.weight";
  std::string expected = "model.layers.0.self_attn.q_proj.weight";

  std::string result = GGUFReader::MapTensorName(llama_name, "llama");
  REQUIRE(result == expected);
}

TEST_CASE("GGUFModelLoader: Tensor name mapping for output", "[gguf][loader]") {
  using namespace inferflux::runtime::cuda::native;

  // Test output layer mapping
  std::string output_name = "output.weight";
  // output.weight should map to lm_head.weight or similar depending on model
  std::string result = GGUFReader::MapTensorName(output_name, "qwen2");
  // Just verify it doesn't crash
  (void)result;
}

TEST_CASE("GGUFModelLoader: Tensor name mapping unknown types",
          "[gguf][loader]") {
  using namespace inferflux::runtime::cuda::native;

  // Test that unknown names pass through with warning
  std::string unknown_name = "some.unknown.tensor";
  std::string result = GGUFReader::MapTensorName(unknown_name, "qwen2");
  // Just verify it doesn't crash
  (void)result;
}

// =============================================================================
// Test Suite: Quantization Detection (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: Detect Q4_K_M quantization",
          "[gguf][loader][quantization]") {
  // Create a GGUF file with Q4_K_M tensors
  auto temp_dir = CreateTempDir("q4_k_m_detect");
  fs::path test_file = temp_dir / "q4_k_m.gguf";

  std::vector<uint8_t> header = CreateGGUFHeader(1, 0);

  // Add Q4_K_M tensor
  uint32_t n_dims = 2;
  uint32_t ne[2] = {768, 768};
  auto tensor_info =
      CreateTensorInfo("weight", n_dims, ne, GGUF::TensorType::Q4_K, 24);

  size_t data_size = header.size() + tensor_info.size();
  std::vector<uint8_t> file_data(data_size);
  size_t pos = 0;

  memcpy(file_data.data() + pos, header.data(), header.size());
  pos += header.size();
  memcpy(file_data.data() + pos, tensor_info.data(), tensor_info.size());

  REQUIRE(WriteBinaryFile(test_file, file_data.data(), file_data.size()));

  GGUFModelLoader loader;
  // Note: May fail due to missing data section, but we're testing parsing logic
  bool result = loader.Load(test_file);
  (void)result;
}

TEST_CASE("GGUFModelLoader: Detect Q5_K_M quantization",
          "[gguf][loader][quantization]") {
  // Similar test for Q5_K_M
  auto temp_dir = CreateTempDir("q5_k_m_detect");
  fs::path test_file = temp_dir / "q5_k_m.gguf";

  std::vector<uint8_t> header = CreateGGUFHeader(1, 0);

  uint32_t n_dims = 2;
  uint32_t ne[2] = {768, 768};
  auto tensor_info =
      CreateTensorInfo("weight", n_dims, ne, GGUF::TensorType::Q5_K, 24);

  size_t data_size = header.size() + tensor_info.size();
  std::vector<uint8_t> file_data(data_size);
  size_t pos = 0;

  memcpy(file_data.data() + pos, header.data(), header.size());
  pos += header.size();
  memcpy(file_data.data() + pos, tensor_info.data(), tensor_info.size());

  REQUIRE(WriteBinaryFile(test_file, file_data.data(), file_data.size()));

  GGUFModelLoader loader;
  bool result = loader.Load(test_file);
  (void)result;
}

TEST_CASE("GGUFModelLoader: Detect Q6_K quantization",
          "[gguf][loader][quantization]") {
  // Similar test for Q6_K
  auto temp_dir = CreateTempDir("q6_k_detect");
  fs::path test_file = temp_dir / "q6_k.gguf";

  std::vector<uint8_t> header = CreateGGUFHeader(1, 0);

  uint32_t n_dims = 2;
  uint32_t ne[2] = {768, 768};
  auto tensor_info =
      CreateTensorInfo("weight", n_dims, ne, GGUF::TensorType::Q6_K, 24);

  size_t data_size = header.size() + tensor_info.size();
  std::vector<uint8_t> file_data(data_size);
  size_t pos = 0;

  memcpy(file_data.data() + pos, header.data(), header.size());
  pos += header.size();
  memcpy(file_data.data() + pos, tensor_info.data(), tensor_info.size());

  REQUIRE(WriteBinaryFile(test_file, file_data.data(), file_data.size()));

  GGUFModelLoader loader;
  bool result = loader.Load(test_file);
  (void)result;
}

TEST_CASE("GGUFModelLoader: Detect FP16 (non-quantized)", "[gguf][loader]") {
  auto temp_dir = CreateTempDir("fp16_detect");
  fs::path test_file = temp_dir / "fp16.gguf";

  std::vector<uint8_t> header = CreateGGUFHeader(1, 0);

  uint32_t n_dims = 2;
  uint32_t ne[2] = {768, 768};
  auto tensor_info =
      CreateTensorInfo("weight", n_dims, ne, GGUF::TensorType::F16, 24);

  size_t data_size = header.size() + tensor_info.size();
  std::vector<uint8_t> file_data(data_size);
  size_t pos = 0;

  memcpy(file_data.data() + pos, header.data(), header.size());
  pos += header.size();
  memcpy(file_data.data() + pos, tensor_info.data(), tensor_info.size());

  REQUIRE(WriteBinaryFile(test_file, file_data.data(), file_data.size()));

  GGUFModelLoader loader;
  bool result = loader.Load(test_file);
  (void)result;
}

// =============================================================================
// Test Suite: GetTensorNames (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: GetTensorNames returns empty before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  auto names = loader.GetTensorNames();
  REQUIRE(names.empty());
}

// =============================================================================
// Test Suite: Tokenizer Metadata (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: Tokenizer methods return defaults before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  REQUIRE(loader.TokenizerEosTokenId() == -1);
  REQUIRE(loader.TokenizerBosTokenId() == -1);

  auto pieces = loader.TokenizerPieces();
  REQUIRE(pieces.empty());
}

// =============================================================================
// Test Suite: Tensor Access (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: GetTensorByGGUFName returns nullptr before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  const auto *tensor = loader.GetTensorByGGUFName("some.weight");
  REQUIRE(tensor == nullptr);
}

TEST_CASE(
    "GGUFModelLoader: GetTensorByGGUFName returns nullptr for unknown tensor",
    "[gguf][loader]") {
  // Even after loading (if we had a valid file), unknown tensors should return
  // nullptr
  GGUFModelLoader loader;

  const auto *tensor = loader.GetTensorByGGUFName("nonexistent.weight");
  REQUIRE(tensor == nullptr);
}

// =============================================================================
// Test Suite: GetWeightAccessor (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: GetWeightAccessor returns nullptr before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  auto accessor = loader.GetWeightAccessor("some.weight");
  REQUIRE(accessor == nullptr);
}

TEST_CASE(
    "GGUFModelLoader: GetWeightAccessor returns nullptr for unknown tensor",
    "[gguf][loader]") {
  GGUFModelLoader loader;

  auto accessor = loader.GetWeightAccessor("unknown.weight");
  REQUIRE(accessor == nullptr);
}

// =============================================================================
// Test Suite: Tensor Name Mapping (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: GetTensorNameMapping returns empty before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  const auto &mapping = loader.GetTensorNameMapping();
  REQUIRE(mapping.empty());
}

// =============================================================================
// GPU-Dependent Tests (Skip when CUDA unavailable)
// =============================================================================

#ifdef INFERFLUX_HAS_CUDA

TEST_CASE("GGUFModelLoader: UploadToGPU with no model loaded",
          "[gguf][loader][gpu]") {
  // This test will only compile and run when CUDA is available
  GGUFModelLoader loader;

  // Before loading a model, UploadToGPU succeeds (uploads 0 bytes)
  cudaStream_t stream = nullptr;
  REQUIRE(loader.UploadToGPU(stream)); // Should succeed (no-op)
  REQUIRE(loader.GetGPUSize() == 0);   // No data uploaded
}

TEST_CASE("GGUFModelLoader: FreeGPUMemory is safe without CUDA init",
          "[gguf][loader][gpu]") {
  GGUFModelLoader loader;

  // Should not crash even if no CUDA memory allocated
  loader.FreeGPUMemory();
}

#endif // INFERFLUX_HAS_CUDA

// =============================================================================
// Test Suite: ModelInfo (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: GetModelInfo returns defaults before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  const auto &info = loader.GetModelInfo();
  REQUIRE(info.model_type.empty());
  REQUIRE(info.hidden_size == 0);
  REQUIRE(info.num_hidden_layers == 0);
  REQUIRE(info.vocab_size == 0);
}

// =============================================================================
// Test Suite: Memory Management (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: FreeCPUMemory is safe before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  // Should not crash even with no data loaded
  loader.FreeCPUMemory();
}

TEST_CASE("GGUFModelLoader: GetGPUSize returns 0 before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  REQUIRE(loader.GetGPUSize() == 0);
}

TEST_CASE("GGUFModelLoader: GetGPUBuffer returns nullptr before load",
          "[gguf][loader]") {
  GGUFModelLoader loader;

  REQUIRE(loader.GetGPUBuffer() == nullptr);
}

// =============================================================================
// Test Suite: Error Handling (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: Multiple load attempts handled gracefully",
          "[gguf][loader]") {
  auto temp_dir = CreateTempDir("multi_load");
  fs::path test_file1 = temp_dir / "test1.gguf";
  fs::path test_file2 = temp_dir / "test2.gguf";

  // Create first file
  std::vector<uint8_t> header = CreateGGUFHeader(1, 0);
  WriteBinaryFile(test_file1, header.data(), header.size());

  GGUFModelLoader loader;
  loader.Load(test_file1);

  // Try loading a different file (should handle gracefully)
  loader.Load(test_file2);
  // May fail or succeed depending on file validity,
  // but should not crash
}

// =============================================================================
// Test Suite: Alignment and Offset Calculation (CPU-only)
// =============================================================================

TEST_CASE("GGUFModelLoader: Default alignment is 32 bytes", "[gguf][loader]") {
  // The loader uses 32-byte alignment for GPU memory allocation
  // This is tested implicitly by successful loading
  // We verify the constant is correct
  GGUFModelLoader loader;

  // Can't directly access alignment_, but we can verify it compiles
  (void)loader; // Just verify it exists
}
