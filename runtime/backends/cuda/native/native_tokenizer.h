#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

/**
 * NativeTokenizer: detokenization from token IDs to text.
 *
 * Parses tokenizer.json (HuggingFace BPE format) to build a reverse
 * mapping from token ID to text piece. Handles byte-level BPE encoding.
 */
class NativeTokenizer {
public:
  NativeTokenizer() = default;

  /**
   * Load tokenizer from model directory.
   * Parses tokenizer.json for vocab and special tokens.
   */
  bool Load(const std::string &model_path);

  /**
   * Convert token ID to text piece.
   * Handles byte-level BPE decoding (e.g., \xC4\xA0 -> space).
   */
  std::string IdToString(int token_id) const;

  /**
   * Tokenize text to token IDs (basic implementation).
   * For production use, the full BPE merge algorithm is needed.
   */
  std::vector<int> Encode(const std::string &text) const;

  int EosTokenId() const { return eos_token_id_; }
  int BosTokenId() const { return bos_token_id_; }
  int VocabSize() const { return static_cast<int>(id_to_piece_.size()); }

private:
  std::string DecodeByteLevelBPE(const std::string &piece) const;

  std::unordered_map<int, std::string> id_to_piece_;
  std::unordered_map<std::string, int> piece_to_id_;
  int eos_token_id_{-1};
  int bos_token_id_{-1};
};

} // namespace inferflux
