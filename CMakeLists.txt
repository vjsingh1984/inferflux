cmake_minimum_required(VERSION 3.22)
project(InferFlux VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Load dependencies from cmake/Dependencies.cmake
include(cmake/Dependencies.cmake)

option(ENABLE_CUDA "Enable CUDA runtime" ON)
option(ENABLE_ROCM "Enable ROCm runtime" ON)
option(ENABLE_MPS "Enable Metal/MPS runtime" ON)
option(ENABLE_BLAS "Enable BLAS acceleration" ON)
option(ENABLE_VULKAN "Enable Vulkan GPU backend via llama.cpp" ON)
option(ENABLE_WEBUI "Enable embedded WebUI (litehtml-based) support" OFF)
option(ENABLE_MLX "Enable experimental MLX backend" OFF)
# MLX_C_TAG: mlx-c release to fetch when MLX_C_ROOT is not provided.
# Bump this together with the `pip install mlx==X.Y.Z` version in the README.
set(MLX_C_TAG "v0.5.0" CACHE STRING "mlx-c git tag to use when building from source")

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
  runtime/backends/backend_capabilities.cpp
  runtime/backends/backend_capabilities.h
  runtime/backends/llama/llama_backend_traits.cpp
  runtime/backends/llama/llama_backend_traits.h
  runtime/backends/backend_factory.cpp
  runtime/backends/backend_factory.h
  runtime/backends/backend_manager.cpp
  runtime/backends/backend_manager.h
  runtime/speculative/speculative_decoder.cpp
  runtime/speculative/speculative_decoder.h
  runtime/execution/batch_executor.cpp
  runtime/execution/batch_executor.h
  runtime/execution/parallel_context.cpp
  runtime/execution/parallel_context.h
  runtime/backends/ep_dispatch.cpp
  runtime/backends/ep_dispatch.h
  runtime/prefix_cache/prefix_cache.cpp
  runtime/prefix_cache/prefix_cache.h
  runtime/prefix_cache/radix_prefix_cache.cpp
  runtime/prefix_cache/radix_prefix_cache.h
  model/tokenizer/simple_tokenizer.cpp
  model/tokenizer/simple_tokenizer.h
  scheduler/scheduler.cpp
  scheduler/fairness_controller.cpp
  scheduler/scheduler.h
  scheduler/model_selection.cpp
  scheduler/model_selection.h
  scheduler/single_model_router.cpp
  scheduler/single_model_router.h
  scheduler/model_registry.cpp
  scheduler/model_registry.h
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
  server/logging/logger.cpp
  server/logging/logger.h
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

if(ENABLE_VULKAN)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    message(STATUS "Vulkan SDK found: enabling Vulkan GPU backend")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_VULKAN=1)
    target_link_libraries(inferflux_core PUBLIC Vulkan::Vulkan)
  else()
    message(STATUS "ENABLE_VULKAN=ON but Vulkan SDK not found; building without Vulkan")
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()

if(ENABLE_WEBUI)
  message(STATUS "ENABLE_WEBUI=ON — embedding litehtml-based WebUI")
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_ENABLE_WEBUI=1)
  target_sources(inferflux_core PRIVATE
    webui/ui_renderer.cpp
    webui/ui_bundle.cpp
  )
  target_include_directories(inferflux_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/litehtml_stub
    ${CMAKE_CURRENT_SOURCE_DIR}/webui
  )
else()
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_ENABLE_WEBUI=0)
endif()

if(ENABLE_MLX)
  target_sources(inferflux_core PRIVATE
    runtime/backends/mlx/mlx_backend.cpp
    runtime/backends/mlx/mlx_loader.cpp
    runtime/backends/mlx/mlx_execution.cpp
    runtime/backends/mlx/mlx_tokenizer.cpp
  )
  message(STATUS "ENABLE_MLX=ON — configuring mlx-c dependency")
  set(MLX_C_HINTS $ENV{MLX_C_ROOT} ${MLX_C_ROOT})
  if(MLX_C_HINTS)
    # Fast path: user supplied a pre-built mlx-c installation.
    find_path(MLX_C_INCLUDE_DIR mlx/c/mlx.h HINTS ${MLX_C_HINTS} PATH_SUFFIXES include)
    find_library(MLX_C_LIBRARY NAMES mlxc mlx HINTS ${MLX_C_HINTS} PATH_SUFFIXES lib)
  else()
    # Slow path: fetch mlx-c from GitHub at the pinned tag and build it.
    # Requires `pip install mlx==${MLX_C_TAG compatible version}` first.
    message(STATUS "MLX_C_ROOT not set — fetching mlx-c ${MLX_C_TAG} via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      mlxc
      GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
      GIT_TAG        ${MLX_C_TAG}
      GIT_SHALLOW    TRUE
    )
    set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_C_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(mlxc)
    # FetchContent builds the 'mlxc' target; expose its include + lib paths.
    set(MLX_C_INCLUDE_DIR ${mlxc_SOURCE_DIR})
    set(MLX_C_LIBRARY mlxc)
  endif()
  if(MLX_C_INCLUDE_DIR AND MLX_C_LIBRARY)
    message(STATUS "mlx-c include dir: ${MLX_C_INCLUDE_DIR}")
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MLX=1)
    target_include_directories(inferflux_core PUBLIC ${MLX_C_INCLUDE_DIR})
    target_link_libraries(inferflux_core PUBLIC ${MLX_C_LIBRARY})
    # Record the directory of mlxc so ctest can set DYLD_LIBRARY_PATH / LD_LIBRARY_PATH.
    get_filename_component(MLX_C_LIB_DIR "${MLX_C_LIBRARY}" DIRECTORY)
    set(MLX_C_LIB_DIR "${MLX_C_LIB_DIR}" CACHE INTERNAL "")
    # Also find the MLX C++ library for rpath embedding.
    find_library(MLX_LIBRARY NAMES mlx
                 HINTS ${MLX_C_HINTS} $ENV{MLX_ROOT}
                 PATHS /opt/anaconda3/lib/python3.13/site-packages/mlx
                       /opt/homebrew/lib
                 PATH_SUFFIXES lib ../lib)
    if(MLX_LIBRARY)
      get_filename_component(MLX_LIB_DIR "${MLX_LIBRARY}" DIRECTORY)
      set_target_properties(inferflux_core PROPERTIES
        BUILD_RPATH   "${MLX_C_LIB_DIR};${MLX_LIB_DIR}"
        INSTALL_RPATH "${MLX_C_LIB_DIR};${MLX_LIB_DIR}")
    endif()
  else()
    message(WARNING "ENABLE_MLX=ON but mlx-c headers/libraries not found. Disabling MLX backend.")
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MLX=0)
  endif()
else()
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MLX=0)
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
# NOTE: LLAMA_CUDA, LLAMA_METAL, and LLAMA_BLAS are set by the hardware-detection
# blocks above.  Do NOT override them here — the former stale reset was a bug
# that silently disabled GPU support even when CUDAToolkit was found.
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

install(TARGETS inferfluxd inferctl
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES README.md docs/Quickstart.md docs/Troubleshooting.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR})

install(FILES config/server.yaml
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/inferflux
        RENAME inferflux.yaml)

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
  tests/unit/test_chat_template.cpp
  tests/unit/test_sampling.cpp
  tests/unit/test_logger.cpp
  tests/unit/test_structured_output.cpp
  tests/unit/test_paged_kv_cache.cpp
  tests/unit/test_unified_batching.cpp
  tests/unit/test_model_registry.cpp
  tests/unit/test_model_selection.cpp
  tests/unit/test_stop_sequences.cpp
  tests/unit/test_parallel_context.cpp
  tests/unit/test_ep_dispatch.cpp
  tests/unit/test_backend_factory.cpp
  tests/unit/test_backend_capabilities.cpp
  external/catch2/catch_amalgamated.cpp
)
target_include_directories(inferflux_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external)
target_link_libraries(inferflux_tests PRIVATE inferflux_core)
if(ENABLE_MLX)
  target_sources(inferflux_tests PRIVATE
    tests/unit/test_mlx_loader.cpp
    tests/unit/test_mlx_tokenizer.cpp
    tests/unit/test_mlx_execution.cpp
    tests/unit/test_mlx_backend.cpp
  )
endif()
add_test(NAME UnitTests COMMAND inferflux_tests)
add_test(NAME FairnessTests COMMAND inferflux_tests "[fairness]")
set_tests_properties(FairnessTests PROPERTIES LABELS fairness)
add_test(NAME PagedKVTests COMMAND inferflux_tests "[paged_kv]")
set_tests_properties(PagedKVTests PROPERTIES LABELS paged_kv)
add_test(NAME UnifiedBatchTests COMMAND inferflux_tests "[unified_batch]")
set_tests_properties(UnifiedBatchTests PROPERTIES LABELS unified_batch)
add_test(NAME ParallelTests COMMAND inferflux_tests "[parallel]")
set_tests_properties(ParallelTests PROPERTIES LABELS parallel)
add_test(NAME EPTests COMMAND inferflux_tests "[ep]")
set_tests_properties(EPTests PROPERTIES LABELS ep)
add_test(NAME BackendFactoryTests COMMAND inferflux_tests "[backend_factory]")
set_tests_properties(BackendFactoryTests PROPERTIES LABELS backend_factory)
add_test(NAME BackendCapabilitiesTests COMMAND inferflux_tests "[backend_capabilities]")
set_tests_properties(BackendCapabilitiesTests PROPERTIES LABELS backend_capabilities)
add_test(NAME MoETests COMMAND inferflux_tests "[moe]")
set_tests_properties(MoETests PROPERTIES LABELS moe)
add_test(NAME FlashAttnTests COMMAND inferflux_tests "[flash_attn]")
set_tests_properties(FlashAttnTests PROPERTIES LABELS flash_attn)
add_test(NAME ShmTransportTests COMMAND inferflux_tests "[shm_transport]")
set_tests_properties(ShmTransportTests PROPERTIES LABELS shm_transport)
add_test(NAME ChatTemplateTests COMMAND inferflux_tests "[chat_template]")
set_tests_properties(ChatTemplateTests PROPERTIES LABELS chat_template)
add_test(NAME SamplingTests COMMAND inferflux_tests "[sampling]")
set_tests_properties(SamplingTests PROPERTIES LABELS sampling)
add_test(NAME LoggerTests COMMAND inferflux_tests "[logger]")
set_tests_properties(LoggerTests PROPERTIES LABELS logger)
add_test(NAME StructuredTests COMMAND inferflux_tests "[structured]")
set_tests_properties(StructuredTests PROPERTIES LABELS structured)
add_test(NAME ModelRegistryTests COMMAND inferflux_tests "[model_registry]")
set_tests_properties(ModelRegistryTests PROPERTIES LABELS model_registry)
add_test(NAME StopSequenceTests COMMAND inferflux_tests "[stop_sequences]")
set_tests_properties(StopSequenceTests PROPERTIES LABELS stop_sequences)
if(ENABLE_MLX)
  add_test(NAME MlxLoaderTests COMMAND inferflux_tests "[mlx_loader]")
  set_tests_properties(MlxLoaderTests PROPERTIES LABELS mlx_loader)
  if(MLX_C_LIB_DIR)
    # Provide the mlx-c and MLX dylib locations to the test runner on macOS.
    set_tests_properties(MlxLoaderTests PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${MLX_C_LIB_DIR}:$ENV{DYLD_LIBRARY_PATH}")
  endif()
  add_test(NAME MlxTokenizerTests COMMAND inferflux_tests "[mlx_tokenizer]")
  set_tests_properties(MlxTokenizerTests PROPERTIES LABELS mlx_tokenizer)
  add_test(NAME MlxExecutionTests COMMAND inferflux_tests "[mlx_execution]")
  set_tests_properties(MlxExecutionTests PROPERTIES LABELS mlx_execution)
  if(MLX_C_LIB_DIR)
    set_tests_properties(MlxExecutionTests PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${MLX_C_LIB_DIR}:$ENV{DYLD_LIBRARY_PATH}")
  endif()
endif()

set(CPACK_PACKAGE_NAME "inferflux")
set(CPACK_PACKAGE_VENDOR "Inferencial Labs")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "InferFlux inference server and CLI")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT "support@inferencial.ai")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libssl3, libyaml-cpp0.7")
set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
set(CPACK_RPM_PACKAGE_REQUIRES "openssl, yaml-cpp")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "${CMAKE_INSTALL_SYSCONFDIR}/inferflux")
set(CPACK_GENERATOR "TGZ")
include(CPack)

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
# ---------------------------------------------------------------------------
# Coverage — gcov/lcov instrumentation.
# Enable with -DENABLE_COVERAGE=ON (automatically uses Debug build).
# Requires GCC or Clang + lcov installed.  Run:
#   cmake --build build-cov --target coverage
# to produce HTML report at build-cov/coverage/html/index.html and an
# lcov.info trace file for upload to Codecov.
# ---------------------------------------------------------------------------
option(ENABLE_COVERAGE "Instrument build for gcov/lcov coverage" OFF)

if(ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "ENABLE_COVERAGE requires GCC or Clang")
  endif()
  # Force Debug so optimiser doesn't collapse away instrumented lines.
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "ENABLE_COVERAGE: overriding build type to Debug")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
  endif()
  set(COVERAGE_FLAGS --coverage -O0 -fno-inline)
  target_compile_options(inferflux_core PUBLIC ${COVERAGE_FLAGS})
  target_compile_options(inferflux_tests PRIVATE ${COVERAGE_FLAGS})
  target_link_libraries(inferflux_core PUBLIC --coverage)
  target_link_libraries(inferflux_tests PRIVATE --coverage)

  find_program(LCOV_BIN lcov REQUIRED)
  find_program(GENHTML_BIN genhtml REQUIRED)

  set(COVERAGE_OUT_DIR "${CMAKE_BINARY_DIR}/coverage")
  set(LCOV_INFO "${COVERAGE_OUT_DIR}/lcov.info")

  add_custom_target(coverage
    DEPENDS inferflux_tests inferfluxd
    COMMENT "Collecting gcov data and generating coverage report"
    # 0. Ensure output directory exists
    COMMAND ${CMAKE_COMMAND} -E make_directory "${COVERAGE_OUT_DIR}"
    # 1. Zero any stale counters
    COMMAND ${LCOV_BIN} --zerocounters --directory "${CMAKE_BINARY_DIR}" --quiet
    # 2. Capture initial baseline (zero coverage)
    COMMAND ${LCOV_BIN} --capture --initial --directory "${CMAKE_BINARY_DIR}"
            --output-file "${COVERAGE_OUT_DIR}/baseline.info" --quiet
    # 3. Run the test suite
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_BINARY_DIR}"
            --output-on-failure --timeout 90
    # 4. Capture post-test counters
    COMMAND ${LCOV_BIN} --capture --directory "${CMAKE_BINARY_DIR}"
            --output-file "${COVERAGE_OUT_DIR}/tests.info" --quiet
    # 5. Merge baseline + test data
    COMMAND ${LCOV_BIN} --add-tracefile "${COVERAGE_OUT_DIR}/baseline.info"
            --add-tracefile "${COVERAGE_OUT_DIR}/tests.info"
            --output-file "${LCOV_INFO}" --quiet
    # 6. Strip external/vendored and system headers
    COMMAND ${LCOV_BIN} --remove "${LCOV_INFO}"
            "${CMAKE_CURRENT_SOURCE_DIR}/external/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*"
            "/usr/*"
            --output-file "${LCOV_INFO}" --quiet
    # 7. HTML report
    COMMAND ${CMAKE_COMMAND} -E make_directory "${COVERAGE_OUT_DIR}/html"
    COMMAND ${GENHTML_BIN} "${LCOV_INFO}"
            --output-directory "${COVERAGE_OUT_DIR}/html"
            --title "InferFlux coverage"
            --legend --demangle-cpp --quiet
    COMMAND ${CMAKE_COMMAND} -E echo
            "Coverage report: ${COVERAGE_OUT_DIR}/html/index.html"
    COMMAND ${CMAKE_COMMAND} -E echo
            "lcov.info for Codecov: ${LCOV_INFO}"
    VERBATIM
    USES_TERMINAL
  )
  message(STATUS "ENABLE_COVERAGE=ON — run: cmake --build <build-dir> --target coverage")
endif()

option(ENABLE_SBOM "Generate SBOM after build" ON)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  # INFERFLUX_SERVER_BIN tells integration tests which binary to launch.
  # CMake always knows the correct path (CMAKE_BINARY_DIR/inferfluxd), so we
  # inject it as a ctest environment variable — this makes the tests work
  # regardless of whether the build directory is "build", "build-cov", etc.
  set(_INFERFLUX_SERVER_BIN "${CMAKE_BINARY_DIR}/inferfluxd")

  # Stub-mode integration tests — run without a model; always enabled.
  add_test(NAME StubIntegration
           COMMAND ${Python3_EXECUTABLE} tests/integration/stub_integration_test.py)
  set_tests_properties(StubIntegration PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")

  add_test(NAME SSECancel
           COMMAND ${Python3_EXECUTABLE} tests/integration/sse_cancel_test.py)
  set_tests_properties(SSECancel PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")

  add_test(NAME ShmSmoke
           COMMAND ${Python3_EXECUTABLE} tests/integration/shm_smoke_test.py)
  set_tests_properties(ShmSmoke PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")

  if(NOT "$ENV{INFERFLUX_MODEL_PATH}" STREQUAL "")
    add_test(NAME IntegrationSSE COMMAND ${Python3_EXECUTABLE} tests/integration/sse_metrics_test.py)
    set_tests_properties(IntegrationSSE PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")
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
