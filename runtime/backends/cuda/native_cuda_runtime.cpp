#include "runtime/backends/cuda/native_cuda_runtime.h"

#include "runtime/backends/cuda/native_kernel_executor.h"

namespace inferflux {

std::unique_ptr<NativeCudaRuntime> CreateNativeCudaRuntime() {
  return std::make_unique<NativeKernelExecutor>();
}

} // namespace inferflux
