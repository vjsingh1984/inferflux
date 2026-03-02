#include "runtime/execution/parallel_context.h"
#include <stdexcept>
#include <thread>

namespace inferflux {

// Simple stub backend for single-node / non-distributed runs.
class StubCommBackend : public CommBackend {
public:
  StubCommBackend(int rank, int world_size)
      : rank_(rank), world_size_(world_size) {}
  int Rank() const override { return rank_; }
  int WorldSize() const override { return world_size_; }
  void Barrier() override {}
  void AllReduce(std::vector<float> &, CollectiveOp) override {}
  void Broadcast(std::vector<float> &, int) override {}
  void AllGather(const std::vector<float> &send,
                 std::vector<float> &recv) override {
    recv = send;
  }
  void Send(const std::vector<float> &, int) override {}
  void Recv(std::vector<float> &, int) override {}

private:
  int rank_;
  int world_size_;
};

ParallelContext &ParallelContext::Get() {
  static ParallelContext instance;
  return instance;
}

void ParallelContext::Initialize(int rank, int world_size,
                                 const std::string &backend_type) {
  if (initialized_)
    return;

  rank_ = rank;
  world_size_ = world_size;

  if (backend_type == "stub" || world_size == 1) {
    comm_ = std::make_unique<StubCommBackend>(rank, world_size);
  } else {
    throw std::runtime_error("Unsupported distributed backend: " +
                             backend_type);
  }

  initialized_ = true;
}

void ParallelContext::BroadcastBatch(const std::vector<int> &request_ids,
                                     const std::vector<int> &phases) {
  if (!comm_ || world_size_ == 1)
    return;

  std::vector<float> data;
  data.reserve(request_ids.size() + phases.size() + 1);
  data.push_back(static_cast<float>(request_ids.size()));
  for (int id : request_ids)
    data.push_back(static_cast<float>(id));
  for (int p : phases)
    data.push_back(static_cast<float>(p));

  comm_->Broadcast(data, 0);
}

bool ParallelContext::ReceiveBatch(std::vector<int> &out_request_ids,
                                   std::vector<int> &out_phases) {
  if (!comm_ || world_size_ == 1)
    return false;

  std::vector<float> data;
  comm_->Broadcast(data, 0);

  if (data.empty())
    return false;

  std::size_t n = static_cast<std::size_t>(data[0]);
  out_request_ids.clear();
  out_phases.clear();
  for (std::size_t i = 0; i < n; ++i)
    out_request_ids.push_back(static_cast<int>(data[i + 1]));
  for (std::size_t i = 0; i < n; ++i)
    out_phases.push_back(static_cast<int>(data[n + i + 1]));

  return true;
}

void ParallelContext::SendActivations(const std::vector<float> &data,
                                      int dest_rank) {
  if (comm_)
    comm_->Send(data, dest_rank);
}

void ParallelContext::RecvActivations(std::vector<float> &data, int src_rank) {
  if (comm_)
    comm_->Recv(data, src_rank);
}

} // namespace inferflux
