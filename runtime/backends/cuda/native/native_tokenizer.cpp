#include "runtime/backends/cuda/native/native_tokenizer.h"
#include "server/logging/logger.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace inferflux {

using json = nlohmann::json;

namespace {

// Byte-level BPE uses a mapping from printable unicode chars back to raw bytes.
// The HuggingFace byte-level BPE maps bytes 0-255 to specific unicode chars.
// This table reverses that mapping.
std::unordered_map<char32_t, uint8_t> BuildByteDecoder() {
  std::unordered_map<char32_t, uint8_t> decoder;
  // Printable ASCII range mapped directly
  for (int b = '!'; b <= '~'; ++b) {
    decoder[static_cast<char32_t>(b)] = static_cast<uint8_t>(b);
  }
  for (int b = 0xA1; b <= 0xAC; ++b) {
    decoder[static_cast<char32_t>(b)] = static_cast<uint8_t>(b);
  }
  for (int b = 0xAE; b <= 0xFF; ++b) {
    decoder[static_cast<char32_t>(b)] = static_cast<uint8_t>(b);
  }
  // Non-printable bytes mapped to unicode offset starting at U+0100
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (decoder.find(static_cast<char32_t>(b)) == decoder.end()) {
      decoder[static_cast<char32_t>(256 + n)] = static_cast<uint8_t>(b);
      ++n;
    }
  }
  return decoder;
}

} // namespace

std::string
NativeTokenizer::DecodeByteLevelBPE(const std::string &piece) const {
  static auto byte_decoder = BuildByteDecoder();

  std::string result;
  result.reserve(piece.size());

  size_t i = 0;
  while (i < piece.size()) {
    // Decode UTF-8 codepoint
    char32_t cp = 0;
    uint8_t c = static_cast<uint8_t>(piece[i]);
    int len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c & 0xE0) == 0xC0) {
      cp = c & 0x1F;
      len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      cp = c & 0x0F;
      len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      cp = c & 0x07;
      len = 4;
    }
    for (int j = 1; j < len && i + j < piece.size(); ++j) {
      cp = (cp << 6) | (static_cast<uint8_t>(piece[i + j]) & 0x3F);
    }
    i += len;

    auto it = byte_decoder.find(cp);
    if (it != byte_decoder.end()) {
      result.push_back(static_cast<char>(it->second));
    } else {
      // Pass through as UTF-8
      for (int j = 0; j < len && (i - len + j) < piece.size(); ++j) {
        result.push_back(piece[i - len + j]);
      }
    }
  }
  return result;
}

bool NativeTokenizer::Load(const std::string &model_path) {
  id_to_piece_.clear();
  piece_to_id_.clear();
  eos_token_id_ = -1;
  bos_token_id_ = -1;

  std::string tokenizer_path = model_path + "/tokenizer.json";
  std::ifstream f(tokenizer_path);
  if (!f.is_open()) {
    log::Error("native_tokenizer",
               "Cannot open tokenizer.json: " + tokenizer_path);
    return false;
  }

  try {
    json tok = json::parse(f);

    // Parse vocabulary from model.vocab
    if (tok.contains("model") && tok["model"].contains("vocab")) {
      auto &vocab = tok["model"]["vocab"];
      for (auto &[piece, id] : vocab.items()) {
        int token_id = id.get<int>();
        id_to_piece_[token_id] = piece;
        piece_to_id_[piece] = token_id;
      }
    }

    // Parse added_tokens for special tokens
    if (tok.contains("added_tokens")) {
      for (auto &added : tok["added_tokens"]) {
        int id = added["id"].get<int>();
        std::string content = added["content"].get<std::string>();
        id_to_piece_[id] = content;
        piece_to_id_[content] = id;

        // Detect EOS/BOS
        if (added.contains("special") && added["special"].get<bool>()) {
          if (content == "</s>" || content == "<|endoftext|>" ||
              content == "<|end_of_text|>" || content == "<|im_end|>" ||
              content == "<eos>") {
            eos_token_id_ = id;
          }
          if (content == "<s>" || content == "<|startoftext|>" ||
              content == "<|begin_of_text|>" || content == "<|im_start|>" ||
              content == "<bos>") {
            bos_token_id_ = id;
          }
        }
      }
    }

    log::Info("native_tokenizer",
              "Loaded " + std::to_string(id_to_piece_.size()) +
                  " tokens, eos=" + std::to_string(eos_token_id_) +
                  ", bos=" + std::to_string(bos_token_id_));
    return true;

  } catch (const std::exception &e) {
    log::Error("native_tokenizer",
               "Failed to parse tokenizer.json: " + std::string(e.what()));
    return false;
  }
}

bool NativeTokenizer::LoadFromPieces(const std::vector<std::string> &pieces,
                                     int eos_token_id, int bos_token_id) {
  id_to_piece_.clear();
  piece_to_id_.clear();

  for (size_t i = 0; i < pieces.size(); ++i) {
    const int token_id = static_cast<int>(i);
    id_to_piece_[token_id] = pieces[i];
    // Keep first ID on duplicates.
    if (piece_to_id_.find(pieces[i]) == piece_to_id_.end()) {
      piece_to_id_[pieces[i]] = token_id;
    }
  }
  eos_token_id_ = eos_token_id;
  bos_token_id_ = bos_token_id;

  log::Info("native_tokenizer",
            "Loaded tokenizer pieces from GGUF metadata: " +
                std::to_string(id_to_piece_.size()) +
                " tokens, eos=" + std::to_string(eos_token_id_) +
                ", bos=" + std::to_string(bos_token_id_));
  return !id_to_piece_.empty();
}

std::string NativeTokenizer::IdToString(int token_id) const {
  auto it = id_to_piece_.find(token_id);
  if (it == id_to_piece_.end()) {
    return "";
  }
  return DecodeByteLevelBPE(it->second);
}

std::vector<int> NativeTokenizer::Encode(const std::string &text) const {
  // Simple greedy longest-match encoding.
  // For production use, the full BPE merge algorithm is needed.
  std::vector<int> tokens;
  size_t i = 0;
  while (i < text.size()) {
    int best_len = 0;
    int best_id = -1;
    // Try decreasing lengths
    for (size_t len = std::min(text.size() - i, size_t(64)); len > 0; --len) {
      auto it = piece_to_id_.find(text.substr(i, len));
      if (it != piece_to_id_.end()) {
        best_len = static_cast<int>(len);
        best_id = it->second;
        break;
      }
    }
    if (best_id >= 0) {
      tokens.push_back(best_id);
      i += best_len;
    } else {
      // Skip unknown byte
      ++i;
    }
  }
  return tokens;
}

} // namespace inferflux
