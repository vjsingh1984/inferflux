#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/kv_cache/paged_kv_cache.h"

#include <algorithm>
#include <cassert>

namespace inferflux {

namespace {

// Return the length of the longest common prefix between two token spans.
std::size_t CommonPrefixLength(const std::vector<int> &edge,
                               const std::vector<int> &tokens,
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

} // namespace

RadixPrefixCache::RadixPrefixCache(std::shared_ptr<PagedKVCache> kv_cache,
                                   EvictCallback on_evict_seq,
                                   std::size_t capacity,
                                   std::size_t max_sequences)
    : kv_cache_(std::move(kv_cache)), on_evict_seq_(std::move(on_evict_seq)),
      capacity_(capacity), max_sequences_(max_sequences),
      root_(std::make_unique<RadixNode>()) {}

std::size_t RadixPrefixCache::Size() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return size_;
}

std::size_t RadixPrefixCache::LiveSequences() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return live_sequences_;
}

bool RadixPrefixCache::Lookup(const std::vector<int> &tokens,
                              LlamaCPUBackend *backend,
                              std::vector<int> *out_block_table,
                              int *out_sequence_id, int *matched_tokens) {
  if (capacity_ == 0 || tokens.empty()) {
    if (matched_tokens)
      *matched_tokens = 0;
    return false;
  }

  std::shared_lock<std::shared_mutex> lock(mutex_);
  RadixNode *node = root_.get();
  std::size_t offset = 0;
  std::vector<int> blocks;
  int last_seq_id = -1;
  bool node_hit = false;

  while (offset < tokens.size()) {
    int first = tokens[offset];
    auto it = node->children.find(first);
    if (it == node->children.end()) {
      break;
    }
    RadixNode *child = it->second.get();
    std::size_t common = CommonPrefixLength(child->edge, tokens, offset);

    if (common < child->edge.size()) {
      // Mid-edge divergence (§ Item 3): record partial match length but hit is
      // false.
      offset += common;
      node_hit = false;
      break;
    }

    // Backend validation logic for full node reuse.
    if (child->sequence_id >= 0) {
      auto locked_be = child->backend.lock();
      if (locked_be.get() != backend) {
        // Mismatch — we still count these tokens but can't reuse the node.
        offset += common;
        node_hit = false;
        break;
      }
    }

    offset += common;
    node = child;
    node_hit = true;
    if (!node->block_table.empty()) {
      blocks.insert(blocks.end(), node->block_table.begin(),
                    node->block_table.end());
    }
    if (node->sequence_id >= 0) {
      last_seq_id = node->sequence_id;
    }
    node->last_used = ++clock_;
  }

  if (matched_tokens) {
    *matched_tokens = static_cast<int>(offset);
  }

  if (node_hit && node != root_.get()) {
    if (out_block_table)
      *out_block_table = std::move(blocks);
    if (out_sequence_id)
      *out_sequence_id = last_seq_id;
    return true;
  }

  return false;
}

void RadixPrefixCache::SplitEdge(RadixNode *parent, int first_token,
                                 std::size_t split_at) {
  auto original = std::move(parent->children[first_token]);

  auto intermediate = std::make_unique<RadixNode>();
  intermediate->parent = parent;
  intermediate->edge.assign(original->edge.begin(),
                            original->edge.begin() +
                                static_cast<std::ptrdiff_t>(split_at));

  intermediate->block_table = {};
  intermediate->sequence_id = -1;

  int original_first = original->edge[split_at];
  original->parent = intermediate.get();
  original->edge.erase(original->edge.begin(),
                       original->edge.begin() +
                           static_cast<std::ptrdiff_t>(split_at));

  intermediate->children[original_first] = std::move(original);
  parent->children[first_token] = std::move(intermediate);
  size_++;
}

void RadixPrefixCache::Insert(const std::vector<int> &tokens,
                              const std::vector<int> &block_table,
                              int sequence_id,
                              std::shared_ptr<LlamaCPUBackend> backend) {
  if (capacity_ == 0 || tokens.empty() || block_table.empty()) {
    return;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);

  RadixNode *node = root_.get();
  std::size_t offset = 0;
  const int kTokensPerBlock = 16;

  while (offset < tokens.size()) {
    int first = tokens[offset];
    auto it = node->children.find(first);

    if (it == node->children.end()) {
      auto leaf = std::make_unique<RadixNode>();
      leaf->parent = node;
      leaf->edge.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                        tokens.end());

      std::size_t block_offset = offset / kTokensPerBlock;
      if (block_offset < block_table.size()) {
        leaf->block_table.assign(block_table.begin() +
                                     static_cast<std::ptrdiff_t>(block_offset),
                                 block_table.end());
      }

      leaf->sequence_id = sequence_id;
      leaf->backend = backend;
      if (sequence_id >= 0)
        live_sequences_++;

      leaf->last_used = ++clock_;
      node->children[first] = std::move(leaf);
      size_++;

      while (live_sequences_ > max_sequences_) {
        EvictOneSequence();
      }
      while (size_ > capacity_) {
        EvictOne();
      }
      return;
    }

    RadixNode *child = it->second.get();
    std::size_t common = CommonPrefixLength(child->edge, tokens, offset);

    if (common < child->edge.size()) {
      SplitEdge(node, first, common);
      child = node->children[first].get();
    }

    offset += common;
    node = child;
  }

  if (node->sequence_id < 0 && sequence_id >= 0)
    live_sequences_++;
  node->sequence_id = sequence_id;
  node->backend = backend;
  node->last_used = ++clock_;
}

void RadixPrefixCache::CollectNodes(
    RadixNode *node, std::function<bool(const RadixNode *)> criteria,
    std::vector<std::pair<uint64_t, RadixNode *>> &out) const {
  if (criteria(node)) {
    out.emplace_back(node->last_used, node);
  }
  for (const auto &[key, child] : node->children) {
    CollectNodes(child.get(), criteria, out);
  }
}

void RadixPrefixCache::EvictOne() {
  std::vector<std::pair<uint64_t, RadixNode *>> leaves;
  leaves.reserve(size_);
  CollectNodes(
      root_.get(),
      [](const RadixNode *n) {
        return n->children.empty() && n->parent != nullptr;
      },
      leaves);

  if (leaves.empty())
    return;

  auto victim_it = std::min_element(
      leaves.begin(), leaves.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });

  RadixNode *victim = victim_it->second;
  if (victim->sequence_id >= 0) {
    if (auto locked_be = victim->backend.lock()) {
      locked_be->FreeSequence(victim->sequence_id);
    }
    if (on_evict_seq_) {
      on_evict_seq_(victim->sequence_id);
    }
    live_sequences_--;
  }

  if (kv_cache_ && !victim->block_table.empty()) {
    kv_cache_->ReleaseBlocksRef(victim->block_table);
  }

  // Prune from trie.
  int first_token = victim->edge[0];
  victim->parent->children.erase(first_token);
  size_--;
}

void RadixPrefixCache::EvictOneSequence() {
  std::vector<std::pair<uint64_t, RadixNode *>> seq_nodes;
  CollectNodes(
      root_.get(), [](const RadixNode *n) { return n->sequence_id >= 0; },
      seq_nodes);

  if (seq_nodes.empty())
    return;

  auto victim_it = std::min_element(
      seq_nodes.begin(), seq_nodes.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });

  RadixNode *victim = victim_it->second;
  if (auto locked_be = victim->backend.lock()) {
    locked_be->FreeSequence(victim->sequence_id);
  }
  if (on_evict_seq_) {
    on_evict_seq_(victim->sequence_id);
  }

  // Correctness Fix (§ Item 2): release block references back to the cache!
  if (kv_cache_ && !victim->block_table.empty()) {
    kv_cache_->ReleaseBlocksRef(victim->block_table);
  }
  victim->block_table.clear();

  victim->sequence_id = -1;
  live_sequences_--;
}

} // namespace inferflux
