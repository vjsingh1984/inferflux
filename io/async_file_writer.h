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
  AsyncFileWriter();
  ~AsyncFileWriter();

  void Start(std::size_t workers = 1);
  void Stop();
  void Enqueue(AsyncWriteTask task);

 private:
  void Worker();

  std::vector<std::thread> workers_;
  std::queue<AsyncWriteTask> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool running_{false};
  bool stop_{false};
};

}  // namespace inferflux
