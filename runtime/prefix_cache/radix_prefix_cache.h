#pragma once

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

class PagedKVCache;
class LlamaCPUBackend;

struct RadixNode {
  std::vector<int> edge;
  std::unordered_map<int, std::unique_ptr<RadixNode>> children;
  RadixNode *parent{nullptr};

  // KV block IDs assigned to this prefix segment.
  std::vector<int> block_table;
  // The sequence_id where these blocks were originally computed (§P1b).
  int sequence_id{-1};
  // The backend that owns the sequence_id.
  std::weak_ptr<LlamaCPUBackend> backend;

  // Total BPE tokens covered by the prefix at this node (including edges up to
  // here).
  int n_tokens{0};

  uint64_t last_used{0};

  int NumTokens() const { return static_cast<int>(edge.size()); }
};

class RadixPrefixCache {
public:
  using EvictCallback = std::function<void(int)>;

  explicit RadixPrefixCache(std::shared_ptr<PagedKVCache> kv_cache,
                            EvictCallback on_evict_seq,
                            std::size_t capacity = 1024,
                            std::size_t max_sequences = 12);

  // Returns true if a full node match was found (allowing CopySequencePrefix).
  // matched_tokens is always filled with the longest common prefix length
  // including partial edge matches for metrics (§ Item 3).
  bool Lookup(const std::vector<int> &tokens, LlamaCPUBackend *backend,
              std::vector<int> *out_block_table, int *out_sequence_id,
              int *matched_tokens);

  // Insert a sequence of tokens and its corresponding block_table.
  void Insert(const std::vector<int> &tokens,
              const std::vector<int> &block_table, int sequence_id,
              std::shared_ptr<LlamaCPUBackend> backend);

  std::size_t Capacity() const { return capacity_; }
  std::size_t Size() const; // total nodes in tree
  std::size_t LiveSequences() const;

private:
  // Walk tree following tokens; return (deepest node reached, tokens matched).
  std::pair<RadixNode *, std::size_t>
  FindLongestPrefix(const std::vector<int> &tokens) const;

  // Split the child edge of parent keyed by first_token at split_at tokens in.
  void SplitEdge(RadixNode *parent, int first_token, std::size_t split_at);

  // Evict the leaf node with the smallest last_used timestamp.
  void EvictOne();
  // Evict the sequence-holding node with the smallest last_used timestamp (§
  // Item 1).
  void EvictOneSequence();

  // DFS: collect nodes matching a criteria.
  void CollectNodes(RadixNode *node,
                    std::function<bool(const RadixNode *)> criteria,
                    std::vector<std::pair<uint64_t, RadixNode *>> &out) const;

  std::shared_ptr<PagedKVCache> kv_cache_;
  EvictCallback on_evict_seq_;
  std::size_t capacity_;
  std::size_t max_sequences_;
  std::size_t size_{0};
  std::size_t live_sequences_{0};
  uint64_t clock_{0};
  std::unique_ptr<RadixNode> root_;
  mutable std::shared_mutex mutex_;
};

} // namespace inferflux
