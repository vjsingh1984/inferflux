include(FetchContent)

message(STATUS "Loading third-party dependencies from cmake/Dependencies.cmake")

# yaml-cpp: A robust YAML parser and emitter for C++
# Fetched at configure time to avoid needing to add it as a submodule.
# We are using a specific commit from the main branch because the latest
# stable release (0.8.0) has a CMakeLists.txt that is too old for our
# project's CMake version (3.22), causing a policy error.
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 05c050c6c14d5c3a82cbc368b50d985896922196 # HEAD of main as of 2026-02-28
)

# This modern CMake function handles the full process of checking if the
# dependency is populated and adding its subdirectory if needed. This also
# resolves a deprecation warning from the previous implementation.
FetchContent_MakeAvailable(yaml-cpp)