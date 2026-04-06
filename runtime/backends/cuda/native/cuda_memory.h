#pragma once

#include "runtime/result.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <utility>

namespace inferflux {

/// RAII wrapper for cudaMalloc / cudaFree (device memory).
/// Move-only.  Releases the allocation on destruction.
template <typename T>
class CudaDevicePtr {
public:
  CudaDevicePtr() = default;
  ~CudaDevicePtr() { reset(); }

  CudaDevicePtr(const CudaDevicePtr &) = delete;
  CudaDevicePtr &operator=(const CudaDevicePtr &) = delete;

  CudaDevicePtr(CudaDevicePtr &&other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  CudaDevicePtr &operator=(CudaDevicePtr &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  /// Allocate @p count elements of T on device.
  static Result<CudaDevicePtr> Alloc(std::size_t count) {
#ifdef INFERFLUX_HAS_CUDA
    T *raw = nullptr;
    auto err = cudaMalloc(reinterpret_cast<void **>(&raw), count * sizeof(T));
    if (err != cudaSuccess) {
      return Err(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    CudaDevicePtr p;
    p.ptr_ = raw;
    p.count_ = count;
    return p;
#else
    (void)count;
    return Err("CUDA not available");
#endif
  }

  T *get() const { return ptr_; }
  T *operator->() const { return ptr_; }
  T &operator*() const { return *ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }
  std::size_t count() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

  /// Release ownership and return raw pointer.
  T *release() {
    T *tmp = ptr_;
    ptr_ = nullptr;
    count_ = 0;
    return tmp;
  }

  void reset() {
#ifdef INFERFLUX_HAS_CUDA
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
#endif
  }

private:
  T *ptr_{nullptr};
  std::size_t count_{0};
};

/// RAII wrapper for cudaMallocHost / cudaFreeHost (pinned host memory).
template <typename T>
class CudaPinnedPtr {
public:
  CudaPinnedPtr() = default;
  ~CudaPinnedPtr() { reset(); }

  CudaPinnedPtr(const CudaPinnedPtr &) = delete;
  CudaPinnedPtr &operator=(const CudaPinnedPtr &) = delete;

  CudaPinnedPtr(CudaPinnedPtr &&other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  CudaPinnedPtr &operator=(CudaPinnedPtr &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  static Result<CudaPinnedPtr> Alloc(std::size_t count) {
#ifdef INFERFLUX_HAS_CUDA
    T *raw = nullptr;
    auto err = cudaMallocHost(reinterpret_cast<void **>(&raw), count * sizeof(T));
    if (err != cudaSuccess) {
      return Err(std::string("cudaMallocHost failed: ") +
                 cudaGetErrorString(err));
    }
    CudaPinnedPtr p;
    p.ptr_ = raw;
    p.count_ = count;
    return p;
#else
    (void)count;
    return Err("CUDA not available");
#endif
  }

  T *get() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }
  std::size_t count() const { return count_; }

  T *release() {
    T *tmp = ptr_;
    ptr_ = nullptr;
    count_ = 0;
    return tmp;
  }

  void reset() {
#ifdef INFERFLUX_HAS_CUDA
    if (ptr_) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
#endif
  }

private:
  T *ptr_{nullptr};
  std::size_t count_{0};
};

/// RAII wrapper for cudaEvent_t.
class CudaEvent {
public:
  CudaEvent() = default;
  ~CudaEvent() { destroy(); }

  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;

  CudaEvent(CudaEvent &&other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  CudaEvent &operator=(CudaEvent &&other) noexcept {
    if (this != &other) {
      destroy();
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  static Result<CudaEvent> Create(unsigned flags = 0) {
#ifdef INFERFLUX_HAS_CUDA
    cudaEvent_t ev = nullptr;
    auto err = cudaEventCreateWithFlags(&ev, flags);
    if (err != cudaSuccess) {
      return Err(std::string("cudaEventCreate failed: ") +
                 cudaGetErrorString(err));
    }
    CudaEvent e;
    e.event_ = ev;
    return e;
#else
    (void)flags;
    return Err("CUDA not available");
#endif
  }

#ifdef INFERFLUX_HAS_CUDA
  cudaEvent_t get() const { return event_; }
  operator cudaEvent_t() const { return event_; } // NOLINT
#endif
  explicit operator bool() const { return event_ != nullptr; }

private:
  void destroy() {
#ifdef INFERFLUX_HAS_CUDA
    if (event_) {
      cudaEventDestroy(event_);
      event_ = nullptr;
    }
#endif
  }

#ifdef INFERFLUX_HAS_CUDA
  cudaEvent_t event_{nullptr};
#else
  void *event_{nullptr};
#endif
};

/// RAII wrapper for cudaStream_t.
class CudaStream {
public:
  CudaStream() = default;
  ~CudaStream() { destroy(); }

  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  CudaStream(CudaStream &&other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  CudaStream &operator=(CudaStream &&other) noexcept {
    if (this != &other) {
      destroy();
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  static Result<CudaStream> Create(unsigned flags = 0) {
#ifdef INFERFLUX_HAS_CUDA
    cudaStream_t s = nullptr;
    auto err = cudaStreamCreateWithFlags(&s, flags);
    if (err != cudaSuccess) {
      return Err(std::string("cudaStreamCreate failed: ") +
                 cudaGetErrorString(err));
    }
    CudaStream cs;
    cs.stream_ = s;
    return cs;
#else
    (void)flags;
    return Err("CUDA not available");
#endif
  }

#ifdef INFERFLUX_HAS_CUDA
  cudaStream_t get() const { return stream_; }
  operator cudaStream_t() const { return stream_; } // NOLINT
#endif
  explicit operator bool() const { return stream_ != nullptr; }

private:
  void destroy() {
#ifdef INFERFLUX_HAS_CUDA
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
#endif
  }

#ifdef INFERFLUX_HAS_CUDA
  cudaStream_t stream_{nullptr};
#else
  void *stream_{nullptr};
#endif
};

} // namespace inferflux
