#pragma once

#include <memory>
#include <string>
#include <vector>

namespace inferflux {

// Types of collective communication operations.
enum class CollectiveOp { kSum, kMax, kMin, kProd };

// Abstract interface for distributed communication (§P1e).
// Implementations will wrap NCCL (NVIDIA), RCCL (AMD), or MPI (CPU).
class CommBackend {
public:
  virtual ~CommBackend() = default;

  virtual int Rank() const = 0;
  virtual int WorldSize() const = 0;

  // Synchronize all ranks in the communication group.
  virtual void Barrier() = 0;

  // Collective operations on vectors of floats (common for LLM activations).
  virtual void AllReduce(std::vector<float> &data, CollectiveOp op) = 0;
  virtual void Broadcast(std::vector<float> &data, int root_rank) = 0;
  virtual void AllGather(const std::vector<float> &send_data,
                         std::vector<float> &recv_data) = 0;

  // Point-to-Point communication for Pipeline Parallelism (§P1h).
  virtual void Send(const std::vector<float> &data, int dest_rank) = 0;
  virtual void Recv(std::vector<float> &data, int src_rank) = 0;
};

// ParallelContext manages the distributed topology and communication layer.
class ParallelContext {
public:
  static ParallelContext &Get();

  // Initialize the distributed environment.
  void Initialize(int rank, int world_size,
                  const std::string &backend_type = "stub");

  bool IsInitialized() const { return initialized_; }
  int Rank() const { return rank_; }
  int WorldSize() const { return world_size_; }
  bool IsMaster() const { return rank_ == 0; }

  // Synchronize batch decisions across all ranks (§P1g).
  // Master rank calls BroadcastBatch; Worker ranks call ReceiveBatch.
  void BroadcastBatch(const std::vector<int> &request_ids,
                      const std::vector<int> &phases);
  bool ReceiveBatch(std::vector<int> &out_request_ids,
                    std::vector<int> &out_phases);

  // P2P stage handover for Pipeline Parallelism (§P1h).
  void SendActivations(const std::vector<float> &data, int dest_rank);
  void RecvActivations(std::vector<float> &data, int src_src);

  CommBackend *Comm() const { return comm_.get(); }

private:
  ParallelContext() = default;

  bool initialized_{false};
  int rank_{0};
  int world_size_{1};
  std::unique_ptr<CommBackend> comm_;
};

} // namespace inferflux
