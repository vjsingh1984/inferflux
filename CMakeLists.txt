cmake_minimum_required(VERSION 3.22)
project(InferFlux VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_CUDA "Enable CUDA runtime" ON)
option(ENABLE_ROCM "Enable ROCm runtime" ON)
option(ENABLE_MPS "Enable Metal/MPS runtime" ON)
option(ENABLE_CPU "Enable CPU runtime" ON)

add_library(inferflux_core
  runtime/device_context.h
  runtime/kv_cache/paged_kv_cache.cpp
  runtime/kv_cache/paged_kv_cache.h
  runtime/tensors/tensor.h
  runtime/backends/cpu/cpu_backend.cpp
  runtime/backends/cpu/cpu_backend.h
  runtime/backends/cpu/llama_backend.cpp
  runtime/backends/cpu/llama_backend.h
  runtime/backends/backend_manager.cpp
  runtime/backends/backend_manager.h
  runtime/speculative/speculative_decoder.cpp
  runtime/speculative/speculative_decoder.h
  model/gguf/gguf_loader.cpp
  model/gguf/gguf_loader.h
  model/tokenizer/simple_tokenizer.cpp
  model/tokenizer/simple_tokenizer.h
  scheduler/scheduler.cpp
  scheduler/scheduler.h
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
  io/async_file_writer.cpp
  io/async_file_writer.h
  net/http_client.cpp
  net/http_client.h
)

target_include_directories(inferflux_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenSSL REQUIRED)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CUDA OFF CACHE BOOL "" FORCE)
add_subdirectory(external/llama.cpp EXCLUDE_FROM_ALL)
target_link_libraries(inferflux_core PUBLIC llama)
target_include_directories(inferflux_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/common
)
target_link_libraries(inferflux_core PUBLIC OpenSSL::Crypto)

add_executable(inferfluxd server/main.cpp)
target_link_libraries(inferfluxd PRIVATE inferflux_core)

add_executable(inferctl cli/main.cpp)
target_link_libraries(inferctl PRIVATE inferflux_core)

enable_testing()
add_executable(inferflux_tests tests/unit/test_tokenizer.cpp)
target_link_libraries(inferflux_tests PRIVATE inferflux_core)
add_test(NAME TokenizerTests COMMAND inferflux_tests)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  if(NOT "$ENV{INFERFLUX_MODEL_PATH}" STREQUAL "")
    add_test(NAME IntegrationSSE COMMAND ${Python3_EXECUTABLE} tests/integration/sse_metrics_test.py)
    set_tests_properties(IntegrationSSE PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(STATUS "Skipping IntegrationSSE test (INFERFLUX_MODEL_PATH not set)")
  endif()
else()
  message(STATUS "Python3 interpreter not found; skipping IntegrationSSE test")
endif()
