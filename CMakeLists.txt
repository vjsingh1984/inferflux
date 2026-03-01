cmake_minimum_required(VERSION 3.22)
project(InferFlux VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Load dependencies from cmake/Dependencies.cmake
include(cmake/Dependencies.cmake)

option(ENABLE_CUDA "Enable CUDA runtime" ON)
option(ENABLE_ROCM "Enable ROCm runtime" ON)
option(ENABLE_MPS "Enable Metal/MPS runtime" ON)
option(ENABLE_BLAS "Enable BLAS acceleration" ON)
option(ENABLE_CPU "Enable CPU runtime" ON)
option(ENABLE_MTMD "Enable multimodal (vision) support via libmtmd" OFF)

add_library(inferflux_core
  runtime/device_context.h
  runtime/kv_cache/paged_kv_cache.cpp
  runtime/kv_cache/paged_kv_cache.h
  runtime/tensors/tensor.h
  runtime/backends/cpu/cpu_backend.cpp
  runtime/backends/cpu/cpu_backend.h
  runtime/backends/cpu/llama_backend.cpp
  runtime/backends/cpu/llama_backend.h
  runtime/backends/cuda/cuda_device_context.cpp
  runtime/backends/cuda/cuda_device_context.h
  runtime/backends/cuda/cuda_backend.cpp
  runtime/backends/cuda/cuda_backend.h
  runtime/backends/backend_manager.cpp
  runtime/backends/backend_manager.h
  runtime/speculative/speculative_decoder.cpp
  runtime/speculative/speculative_decoder.h
  runtime/execution/batch_executor.cpp
  runtime/execution/batch_executor.h
  runtime/prefix_cache/prefix_cache.cpp
  runtime/prefix_cache/prefix_cache.h
  runtime/prefix_cache/radix_prefix_cache.cpp
  runtime/prefix_cache/radix_prefix_cache.h
  model/gguf/gguf_loader.cpp
  model/gguf/gguf_loader.h
  model/tokenizer/simple_tokenizer.cpp
  model/tokenizer/simple_tokenizer.h
  scheduler/scheduler.cpp
  scheduler/fairness_controller.cpp
  scheduler/scheduler.h
  scheduler/single_model_router.cpp
  scheduler/single_model_router.h
  server/http/http_server.cpp
  server/http/http_server.h
  server/auth/api_key_auth.cpp
  server/auth/api_key_auth.h
  server/metrics/metrics.cpp
  server/metrics/metrics.h
  server/auth/rate_limiter.cpp
  server/auth/rate_limiter.h
  server/auth/oidc_validator.cpp
  server/auth/oidc_validator.h
  server/logging/audit_logger.cpp
  server/logging/audit_logger.h
  server/policy/guardrail.cpp
  server/policy/guardrail.h
  policy/policy_store.cpp
  policy/policy_store.h
  policy/opa_client.cpp
  policy/opa_client.h
  runtime/structured_output/structured_constraint.h
  runtime/structured_output/structured_output_adapter.cpp
  runtime/structured_output/structured_output_adapter.h
  runtime/structured_output/json_schema_shim.cpp
  runtime/disaggregated/kv_channel.cpp
  runtime/disaggregated/kv_channel.h
  runtime/disaggregated/shm_kv_transport.cpp
  runtime/disaggregated/shm_kv_transport.h
  runtime/multimodal/image_preprocessor.cpp
  runtime/multimodal/image_preprocessor.h
  external/llama.cpp/common/json-schema-to-grammar.cpp
  io/async_file_writer.cpp
  io/async_file_writer.h
  net/http_client.cpp
  net/http_client.h
)

target_include_directories(inferflux_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/external
)

find_package(OpenSSL REQUIRED)
find_package(CUDAToolkit QUIET)

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  message(STATUS "CUDA toolkit found: enabling CUDA build path")
  set(LLAMA_CUDA ON CACHE BOOL "" FORCE)
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_CUDA=1)
  target_link_libraries(inferflux_core PUBLIC CUDA::cudart)
else()
  if(ENABLE_CUDA)
    message(WARNING "ENABLE_CUDA=ON but CUDA toolkit not found; building without GPU support")
  endif()
  set(LLAMA_CUDA OFF CACHE BOOL "" FORCE)
endif()

if(ENABLE_MPS AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(STATUS "Metal/MPS enabled")
  set(LLAMA_METAL ON CACHE BOOL "" FORCE)
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_METAL=1)
else()
  set(LLAMA_METAL OFF CACHE BOOL "" FORCE)
endif()

if(ENABLE_BLAS)
  find_package(BLAS)
  if(BLAS_FOUND)
    message(STATUS "BLAS backend found: ${BLAS_LIBRARIES}")
    set(LLAMA_BLAS ON CACHE BOOL "" FORCE)
    target_link_libraries(inferflux_core PUBLIC BLAS::BLAS)
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_BLAS=1)
  else()
    message(WARNING "ENABLE_BLAS=ON but no BLAS implementation found; building without BLAS acceleration")
    set(LLAMA_BLAS OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(LLAMA_BLAS OFF CACHE BOOL "" FORCE)
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CUDA OFF CACHE BOOL "" FORCE)
add_subdirectory(external/llama.cpp EXCLUDE_FROM_ALL)
target_link_libraries(inferflux_core PUBLIC llama)
target_include_directories(inferflux_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/common
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/vendor
)
target_link_libraries(inferflux_core PUBLIC OpenSSL::Crypto OpenSSL::SSL yaml-cpp)
# POSIX SHM (shm_open/mmap) requires -lrt on Linux; macOS links it automatically.
if(UNIX AND NOT APPLE)
  target_link_libraries(inferflux_core PUBLIC rt)
endif()

# §2.2 Multimodal support via libmtmd (ENABLE_MTMD=OFF by default)
if(ENABLE_MTMD)
  set(MTMD_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/tools/mtmd)
  target_sources(inferflux_core PRIVATE
    ${MTMD_SRC_DIR}/mtmd.cpp
    ${MTMD_SRC_DIR}/mtmd-audio.cpp
    ${MTMD_SRC_DIR}/mtmd-helper.cpp
    ${MTMD_SRC_DIR}/clip.cpp
    ${MTMD_SRC_DIR}/models/cogvlm.cpp
    ${MTMD_SRC_DIR}/models/conformer.cpp
    ${MTMD_SRC_DIR}/models/glm4v.cpp
    ${MTMD_SRC_DIR}/models/internvl.cpp
    ${MTMD_SRC_DIR}/models/kimivl.cpp
    ${MTMD_SRC_DIR}/models/kimik25.cpp
    ${MTMD_SRC_DIR}/models/nemotron-v2-vl.cpp
    ${MTMD_SRC_DIR}/models/llama4.cpp
    ${MTMD_SRC_DIR}/models/llava.cpp
    ${MTMD_SRC_DIR}/models/minicpmv.cpp
    ${MTMD_SRC_DIR}/models/paddleocr.cpp
    ${MTMD_SRC_DIR}/models/pixtral.cpp
    ${MTMD_SRC_DIR}/models/qwen2vl.cpp
    ${MTMD_SRC_DIR}/models/qwen3vl.cpp
    ${MTMD_SRC_DIR}/models/siglip.cpp
    ${MTMD_SRC_DIR}/models/whisper-enc.cpp
    ${MTMD_SRC_DIR}/models/mobilenetv5.cpp
    ${MTMD_SRC_DIR}/models/youtuvl.cpp
  )
  target_include_directories(inferflux_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/tools
  )
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MTMD=1)
  if(NOT MSVC)
    target_compile_options(inferflux_core PRIVATE -Wno-cast-qual)
  endif()
  message(STATUS "ENABLE_MTMD=ON — libmtmd vision support enabled")
else()
  message(STATUS "ENABLE_MTMD=OFF — multimodal support disabled (pass -DENABLE_MTMD=ON to enable)")
endif()

add_executable(inferfluxd server/main.cpp)
target_link_libraries(inferfluxd PRIVATE inferflux_core)

add_executable(inferctl cli/main.cpp)
target_link_libraries(inferctl PRIVATE inferflux_core)

enable_testing()
add_executable(inferflux_tests
  tests/unit/test_tokenizer.cpp
  tests/unit/test_api_key_auth.cpp
  tests/unit/test_rate_limiter.cpp
  tests/unit/test_guardrail.cpp
  tests/unit/test_audit_logger.cpp
  tests/unit/test_metrics.cpp
  tests/unit/test_scheduler.cpp
  tests/unit/test_policy_store.cpp
  tests/unit/test_oidc_validator.cpp
  tests/unit/test_prefix_cache.cpp
  tests/unit/test_radix_prefix_cache.cpp
  tests/unit/test_span.cpp
  tests/unit/test_kv_channel.cpp
  tests/unit/test_phased_execution.cpp
  tests/unit/test_multimodal.cpp
  tests/unit/test_moe_routing.cpp
  tests/unit/test_flash_attn.cpp
  tests/unit/test_shm_transport.cpp
  external/catch2/catch_amalgamated.cpp
)
target_include_directories(inferflux_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(inferflux_tests PRIVATE inferflux_core)
add_test(NAME UnitTests COMMAND inferflux_tests)
add_test(NAME FairnessTests COMMAND inferflux_tests "[fairness]")
set_tests_properties(FairnessTests PROPERTIES LABELS fairness)
add_test(NAME MoETests COMMAND inferflux_tests "[moe]")
set_tests_properties(MoETests PROPERTIES LABELS moe)
add_test(NAME FlashAttnTests COMMAND inferflux_tests "[flash_attn]")
set_tests_properties(FlashAttnTests PROPERTIES LABELS flash_attn)
add_test(NAME ShmTransportTests COMMAND inferflux_tests "[shm_transport]")
set_tests_properties(ShmTransportTests PROPERTIES LABELS shm_transport)

# ---------------------------------------------------------------------------
# SBOM — generate CycloneDX / SPDX bill-of-materials after every build.
#
# The SBOM script (scripts/generate_sbom.py) walks the source tree for
# vendored dependencies, reads their declared versions, and writes two
# output files to ${CMAKE_BINARY_DIR}/sbom/:
#   inferflux-sbom.cdx.json   — CycloneDX 1.5 JSON (for supply-chain tooling)
#   inferflux-sbom.spdx        — SPDX 2.3 tag-value (for license compliance)
#
# The target is always defined but only runs when Python3 is available.
# Run manually: cmake --build build --target sbom
# ---------------------------------------------------------------------------
option(ENABLE_SBOM "Generate SBOM after build" ON)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  # Stub-mode integration tests — run without a model; always enabled.
  add_test(NAME StubIntegration
           COMMAND ${Python3_EXECUTABLE} tests/integration/stub_integration_test.py)
  set_tests_properties(StubIntegration PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60)

  add_test(NAME SSECancel
           COMMAND ${Python3_EXECUTABLE} tests/integration/sse_cancel_test.py)
  set_tests_properties(SSECancel PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60)

  if(NOT "$ENV{INFERFLUX_MODEL_PATH}" STREQUAL "")
    add_test(NAME IntegrationSSE COMMAND ${Python3_EXECUTABLE} tests/integration/sse_metrics_test.py)
    set_tests_properties(IntegrationSSE PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(STATUS "Skipping IntegrationSSE test (INFERFLUX_MODEL_PATH not set)")
  endif()

  if(ENABLE_SBOM)
    set(SBOM_OUT_DIR "${CMAKE_BINARY_DIR}/sbom")
    add_custom_target(sbom
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SBOM_OUT_DIR}
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_sbom.py
              --source-dir  ${CMAKE_CURRENT_SOURCE_DIR}
              --version     ${PROJECT_VERSION}
              --out-dir     ${SBOM_OUT_DIR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating SBOM (CycloneDX + SPDX) → ${SBOM_OUT_DIR}"
      VERBATIM
    )
    message(STATUS "SBOM target registered — run: cmake --build build --target sbom")
  endif()
else()
  message(STATUS "Python3 interpreter not found; skipping integration tests and SBOM")
endif()
