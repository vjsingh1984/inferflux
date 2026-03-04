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

#define NVTX_SCOPE(name) ::inferflux::NvtxRange nvtx_scope_##__LINE__(name)
#else
#define NVTX_SCOPE(name) ((void)0)
#endif
