#pragma once

#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace inferflux {

struct AsyncWriteTask {
  std::filesystem::path path;
  std::vector<char> buffer;
};

class AsyncFileWriter {
 public:
  explicit AsyncFileWriter(std::size_t max_queue_depth = 256);
  ~AsyncFileWriter();

  void Configure(std::size_t workers, std::size_t max_queue_depth);
  void Start(std::size_t workers = 0);
  void Stop();
  void Enqueue(AsyncWriteTask task);

 private:
  void Worker();

  std::vector<std::thread> workers_;
  std::queue<AsyncWriteTask> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable producer_cv_;
  bool running_{false};
  bool stop_{false};
  std::size_t preferred_workers_{1};
  std::size_t max_queue_depth_{256};
};

}  // namespace inferflux
