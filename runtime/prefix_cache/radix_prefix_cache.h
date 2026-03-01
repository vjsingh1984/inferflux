#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

struct RadixNode {
  std::vector<int> edge;
  std::unordered_map<int, std::unique_ptr<RadixNode>> children;
  bool has_completion{false};
  std::string completion;
  int completion_tokens{0};
  uint64_t last_used{0};
};

class RadixPrefixCache {
 public:
  explicit RadixPrefixCache(std::size_t capacity = 256);

  // Returns true on an exact match; fills completion/completion_tokens.
  // matched_tokens is always filled (even on miss) — useful for partial-hit metrics.
  bool Lookup(const std::vector<int>& tokens,
              std::string* completion,
              int* completion_tokens,
              int* matched_tokens = nullptr);

  void Insert(const std::vector<int>& tokens,
              const std::string& completion,
              int completion_tokens);

  std::size_t Capacity() const { return capacity_; }
  std::size_t Size() const;  // nodes with has_completion == true

 private:
  // Walk tree following tokens; return (deepest node reached, tokens matched).
  std::pair<RadixNode*, std::size_t> FindLongestPrefix(
      const std::vector<int>& tokens) const;

  // Split the child edge of parent keyed by first_token at split_at tokens in.
  void SplitEdge(RadixNode* parent, int first_token, std::size_t split_at);

  // Evict the completion-holding leaf with the smallest last_used timestamp.
  void EvictOne();

  // DFS: collect all nodes where has_completion==true.
  void CollectLeaves(const RadixNode* node,
                     std::vector<std::pair<uint64_t, RadixNode*>>& out) const;

  std::size_t capacity_;
  std::size_t size_{0};
  uint64_t clock_{0};
  std::unique_ptr<RadixNode> root_;
  mutable std::shared_mutex mutex_;
};

}  // namespace inferflux
