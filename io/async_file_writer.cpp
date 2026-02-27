#include "io/async_file_writer.h"

#include <fstream>

namespace inferflux {

AsyncFileWriter::AsyncFileWriter() = default;

AsyncFileWriter::~AsyncFileWriter() { Stop(); }

void AsyncFileWriter::Start(std::size_t workers) {
  if (running_) {
    return;
  }
  stop_ = false;
  running_ = true;
  if (workers == 0) {
    workers = 1;
  }
  for (std::size_t i = 0; i < workers; ++i) {
    workers_.emplace_back(&AsyncFileWriter::Worker, this);
  }
}

void AsyncFileWriter::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
  running_ = false;
}

void AsyncFileWriter::Enqueue(AsyncWriteTask task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
  }
  if (!running_) {
    Start();
  }
  cv_.notify_one();
}

void AsyncFileWriter::Worker() {
  while (true) {
    AsyncWriteTask task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    std::ofstream out(task.path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      continue;
    }
    out.write(task.buffer.data(), static_cast<std::streamsize>(task.buffer.size()));
  }
}

}  // namespace inferflux
