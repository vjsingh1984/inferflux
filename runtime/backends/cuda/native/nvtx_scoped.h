#pragma once

#ifdef INFERFLUX_HAS_CUDA
#include <nvtx3/nvToolsExt.h>

namespace inferflux {

class NvtxRange {
public:
  explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange &) = delete;
  NvtxRange &operator=(const NvtxRange &) = delete;
};

} // namespace inferflux

#define INFERFLUX_NVTX_CONCAT_IMPL(x, y) x##y
#define INFERFLUX_NVTX_CONCAT(x, y) INFERFLUX_NVTX_CONCAT_IMPL(x, y)
#define NVTX_SCOPE(name)                                                       \
  ::inferflux::NvtxRange INFERFLUX_NVTX_CONCAT(nvtx_scope_, __COUNTER__)(name)
#else
#define NVTX_SCOPE(name) ((void)0)
#endif
