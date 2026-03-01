#include "runtime/prefix_cache/radix_prefix_cache.h"

#include <algorithm>
#include <cassert>

namespace inferflux {

namespace {

// Return the length of the longest common prefix between two token spans.
std::size_t CommonPrefixLength(const std::vector<int>& edge,
                               const std::vector<int>& tokens,
                               std::size_t tokens_offset) {
  std::size_t len = 0;
  std::size_t edge_len = edge.size();
  std::size_t tokens_remaining = tokens.size() - tokens_offset;
  std::size_t limit = std::min(edge_len, tokens_remaining);
  while (len < limit && edge[len] == tokens[tokens_offset + len]) {
    ++len;
  }
  return len;
}

}  // namespace

RadixPrefixCache::RadixPrefixCache(std::size_t capacity)
    : capacity_(capacity), root_(std::make_unique<RadixNode>()) {}

std::size_t RadixPrefixCache::Size() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return size_;
}

std::pair<RadixNode*, std::size_t> RadixPrefixCache::FindLongestPrefix(
    const std::vector<int>& tokens) const {
  RadixNode* node = root_.get();
  std::size_t offset = 0;

  while (offset < tokens.size()) {
    int first = tokens[offset];
    auto it = node->children.find(first);
    if (it == node->children.end()) {
      break;
    }
    RadixNode* child = it->second.get();
    std::size_t common = CommonPrefixLength(child->edge, tokens, offset);
    if (common < child->edge.size()) {
      // Partial edge match — advance the offset by the matched portion and stop.
      offset += common;
      break;
    }
    offset += common;
    node = child;
  }

  return {node, offset};
}

bool RadixPrefixCache::Lookup(const std::vector<int>& tokens,
                              std::string* completion,
                              int* completion_tokens,
                              int* matched_tokens) {
  if (capacity_ == 0) {
    if (matched_tokens) *matched_tokens = 0;
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto [node, matched] = FindLongestPrefix(tokens);

  // Also count tokens matched into the first untraversed child edge (partial match).
  std::size_t total_matched = matched;
  if (matched_tokens && matched < tokens.size()) {
    int first_rem = tokens[matched];
    auto child_it = node->children.find(first_rem);
    if (child_it != node->children.end()) {
      total_matched += CommonPrefixLength(child_it->second->edge, tokens, matched);
    }
  }

  if (matched_tokens) {
    *matched_tokens = static_cast<int>(total_matched);
  }

  if (matched == tokens.size() && node != root_.get() && node->has_completion) {
    // Exact match.
    node->last_used = ++clock_;
    if (completion) *completion = node->completion;
    if (completion_tokens) *completion_tokens = node->completion_tokens;
    return true;
  }

  return false;
}

void RadixPrefixCache::SplitEdge(RadixNode* parent, int first_token, std::size_t split_at) {
  // Take ownership of the original child.
  auto original = std::move(parent->children[first_token]);

  // Create intermediate node: edge = original->edge[0..split_at)
  auto intermediate = std::make_unique<RadixNode>();
  intermediate->edge.assign(original->edge.begin(),
                            original->edge.begin() + static_cast<std::ptrdiff_t>(split_at));

  // Truncate original's edge to the suffix.
  int original_first = original->edge[split_at];
  original->edge.erase(original->edge.begin(),
                       original->edge.begin() + static_cast<std::ptrdiff_t>(split_at));

  // Wire: intermediate → original
  intermediate->children[original_first] = std::move(original);

  // Wire: parent → intermediate
  parent->children[first_token] = std::move(intermediate);
}

void RadixPrefixCache::Insert(const std::vector<int>& tokens,
                              const std::string& completion,
                              int completion_tokens) {
  if (capacity_ == 0) {
    return;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);

  RadixNode* node = root_.get();
  std::size_t offset = 0;

  while (offset < tokens.size()) {
    int first = tokens[offset];
    auto it = node->children.find(first);

    if (it == node->children.end()) {
      // Create a new leaf with edge = remaining tokens.
      auto leaf = std::make_unique<RadixNode>();
      leaf->edge.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset), tokens.end());
      leaf->has_completion = true;
      leaf->completion = completion;
      leaf->completion_tokens = completion_tokens;
      leaf->last_used = ++clock_;
      node->children[first] = std::move(leaf);
      ++size_;
      while (size_ > capacity_) {
        EvictOne();
      }
      return;
    }

    RadixNode* child = it->second.get();
    std::size_t common = CommonPrefixLength(child->edge, tokens, offset);

    if (common < child->edge.size()) {
      // Need to split the existing edge.
      SplitEdge(node, first, common);
      // After split, node->children[first] is now the intermediate node.
      child = node->children[first].get();
    }

    offset += common;
    node = child;
  }

  // node is the insertion point.
  if (!node->has_completion) {
    ++size_;
  }
  node->has_completion = true;
  node->completion = completion;
  node->completion_tokens = completion_tokens;
  node->last_used = ++clock_;

  while (size_ > capacity_) {
    EvictOne();
  }
}

void RadixPrefixCache::CollectLeaves(const RadixNode* node,
                                     std::vector<std::pair<uint64_t, RadixNode*>>& out) const {
  if (node->has_completion) {
    out.emplace_back(node->last_used, const_cast<RadixNode*>(node));
  }
  for (const auto& [key, child] : node->children) {
    CollectLeaves(child.get(), out);
  }
}

void RadixPrefixCache::EvictOne() {
  std::vector<std::pair<uint64_t, RadixNode*>> leaves;
  leaves.reserve(size_);
  CollectLeaves(root_.get(), leaves);

  if (leaves.empty()) {
    return;
  }

  // Pick the node with the smallest last_used.
  auto victim_it = std::min_element(
      leaves.begin(), leaves.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; });

  RadixNode* victim = victim_it->second;
  victim->has_completion = false;
  victim->completion.clear();
  victim->completion_tokens = 0;
  --size_;
}

}  // namespace inferflux
