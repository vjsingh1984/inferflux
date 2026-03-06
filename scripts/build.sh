#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build}

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

CMAKE_ARGS=(
  -S .
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DINFERFLUX_CUDA_DEVICE_LTO=${INFERFLUX_CUDA_DEVICE_LTO:-OFF}
  -DINFERFLUX_CUDA_USE_FAST_MATH=${INFERFLUX_CUDA_USE_FAST_MATH:-OFF}
)

# Prefer GCC/G++ as the CUDA host compiler on Linux for maximum compatibility.
if [[ -z "${CMAKE_CUDA_HOST_COMPILER:-}" ]]; then
  if [[ -x /usr/bin/g++ ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++)
  fi
else
  CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER}")
fi

# Optional per-node CUDA architecture override (example: 89 for Ada RTX 4000).
if [[ -n "${INFERFLUX_CUDA_ARCHS:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${INFERFLUX_CUDA_ARCHS}")
fi

cmake "${GENERATOR_ARGS[@]}" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
