#include "policy/policy_store.h"
#include "server/auth/api_key_auth.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <cctype>

namespace inferflux {

namespace {
std::string Trim(const std::string& input) {
  auto start = input.find_first_not_of(" \t");
  auto end = input.find_last_not_of(" \t\r\n");
  if (start == std::string::npos || end == std::string::npos) {
    return "";
  }
  return input.substr(start, end - start + 1);
}
std::string HexEncode(const unsigned char* data, std::size_t len) {
  static const char* hex = "0123456789abcdef";
  std::string out;
  out.resize(len * 2);
  for (std::size_t i = 0; i < len; ++i) {
    out[2 * i] = hex[(data[i] >> 4) & 0xF];
    out[2 * i + 1] = hex[data[i] & 0xF];
  }
  return out;
}

std::vector<unsigned char> HexDecode(const std::string& hex) {
  std::vector<unsigned char> out;
  if (hex.size() % 2 != 0) {
    return out;
  }
  out.reserve(hex.size() / 2);
  for (std::size_t i = 0; i < hex.size(); i += 2) {
    unsigned char hi = static_cast<unsigned char>(std::tolower(hex[i]));
    unsigned char lo = static_cast<unsigned char>(std::tolower(hex[i + 1]));
    auto decode = [](unsigned char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      return -1;
    };
    int high = decode(hi);
    int low = decode(lo);
    if (high < 0 || low < 0) {
      out.clear();
      return out;
    }
    out.push_back(static_cast<unsigned char>((high << 4) | low));
  }
  return out;
}
}  // namespace

PolicyStore::PolicyStore(std::string path, std::string passphrase)
    : path_(std::move(path)), encryption_enabled_(!passphrase.empty()) {
  if (encryption_enabled_) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(passphrase.data()), passphrase.size(), hash);
    std::copy(hash, hash + key_.size(), key_.begin());
  }
}

void PolicyStore::EnsureParentDir() const {
  auto parent = std::filesystem::path(path_).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }
}

std::vector<std::string> PolicyStore::SplitCSV(const std::string& line) {
  std::vector<std::string> values;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto trimmed = Trim(item);
    if (!trimmed.empty()) {
      values.push_back(trimmed);
    }
  }
  return values;
}

std::string PolicyStore::JoinCSV(const std::vector<std::string>& values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  return out.str();
}

bool PolicyStore::Load() {
  std::lock_guard<std::mutex> lock(mutex_);
  api_keys_.clear();
  guardrail_blocklist_.clear();
  rate_limit_per_minute_ = 0;
  std::ifstream input(path_, std::ios::binary);
  if (!input.good()) {
    return false;
  }
  std::string raw((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  std::string plaintext = raw;
  if (raw.rfind("ENC", 0) == 0) {
    if (!encryption_enabled_) {
      return false;
    }
    if (!Decrypt(raw, &plaintext)) {
      return false;
    }
  }
  std::istringstream buffer(plaintext);
  std::string line;
  std::string section;
  while (std::getline(buffer, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      section = line.substr(1, line.size() - 2);
      continue;
    }
    auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    auto key = Trim(line.substr(0, eq));
    auto value = Trim(line.substr(eq + 1));
    if (section == "api_keys") {
      api_keys_[key] = SplitCSV(value);
    } else if (section == "guardrail" && key == "words") {
      guardrail_blocklist_ = SplitCSV(value);
    } else if (section == "rate_limit" && key == "tokens") {
      try {
        rate_limit_per_minute_ = std::stoi(value);
      } catch (...) {
        rate_limit_per_minute_ = 0;
      }
    }
  }
  return true;
}

bool PolicyStore::Save() const {
  std::lock_guard<std::mutex> lock(mutex_);
  EnsureParentDir();
  std::ostringstream plaintext;
  plaintext << "[api_keys]\n";
  for (const auto& [key, scopes] : api_keys_) {
    plaintext << key << "=" << JoinCSV(scopes) << "\n";
  }
  plaintext << "\n[guardrail]\n";
  plaintext << "words=" << JoinCSV(guardrail_blocklist_) << "\n";
  plaintext << "\n[rate_limit]\n";
  plaintext << "tokens=" << rate_limit_per_minute_ << "\n";
  std::string serialized = plaintext.str();

  std::ofstream output(path_, std::ios::trunc | std::ios::binary);
  if (!output.good()) {
    return false;
  }
  if (encryption_enabled_) {
    std::string encrypted;
    if (!Encrypt(serialized, &encrypted)) {
      return false;
    }
    output << encrypted;
  } else {
    output << serialized;
  }
  return true;
}

std::vector<PolicyKeyEntry> PolicyStore::ApiKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PolicyKeyEntry> keys;
  keys.reserve(api_keys_.size());
  for (const auto& [key, scopes] : api_keys_) {
    keys.push_back({key, scopes});
  }
  return keys;
}

void PolicyStore::SetApiKey(const std::string& key, const std::vector<std::string>& scopes) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Hash the plaintext key before storing — never persist raw keys to disk.
  api_keys_[ApiKeyAuth::HashKey(key)] = scopes;
}

bool PolicyStore::RemoveApiKey(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  return api_keys_.erase(ApiKeyAuth::HashKey(key)) > 0;
}

std::vector<std::string> PolicyStore::GuardrailBlocklist() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return guardrail_blocklist_;
}

void PolicyStore::SetGuardrailBlocklist(const std::vector<std::string>& blocklist) {
  std::lock_guard<std::mutex> lock(mutex_);
  guardrail_blocklist_ = blocklist;
}

int PolicyStore::RateLimitPerMinute() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return rate_limit_per_minute_;
}

void PolicyStore::SetRateLimitPerMinute(int limit) {
  std::lock_guard<std::mutex> lock(mutex_);
  rate_limit_per_minute_ = limit;
}

bool PolicyStore::Encrypt(const std::string& plaintext, std::string* output) const {
  if (!encryption_enabled_) {
    *output = plaintext;
    return true;
  }
  std::array<unsigned char, 12> nonce{};
  if (RAND_bytes(nonce.data(), nonce.size()) != 1) {
    return false;
  }
  std::vector<unsigned char> ciphertext(plaintext.size());
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    return false;
  }
  bool ok = false;
  do {
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) break;
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, nonce.size(), nullptr) != 1) break;
    if (EVP_EncryptInit_ex(ctx, nullptr, nullptr, key_.data(), nonce.data()) != 1) break;
    int out_len = 0;
    if (EVP_EncryptUpdate(ctx,
                          ciphertext.data(),
                          &out_len,
                          reinterpret_cast<const unsigned char*>(plaintext.data()),
                          plaintext.size()) != 1) {
      break;
    }
    int final_len = 0;
    if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + out_len, &final_len) != 1) break;
    out_len += final_len;
    ciphertext.resize(out_len);
    std::array<unsigned char, 16> tag{};
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, tag.size(), tag.data()) != 1) break;
    std::ostringstream result;
    result << "ENC\n";
    result << "nonce=" << HexEncode(nonce.data(), nonce.size()) << "\n";
    result << "tag=" << HexEncode(tag.data(), tag.size()) << "\n";
    result << "data=" << HexEncode(ciphertext.data(), ciphertext.size()) << "\n";
    *output = result.str();
    ok = true;
  } while (false);
  EVP_CIPHER_CTX_free(ctx);
  return ok;
}

bool PolicyStore::Decrypt(const std::string& encrypted, std::string* plaintext) const {
  if (!encryption_enabled_) {
    *plaintext = encrypted;
    return true;
  }
  std::istringstream buffer(encrypted);
  std::string header;
  if (!std::getline(buffer, header) || Trim(header) != "ENC") {
    return false;
  }
  auto parseLine = [&](const std::string& expected_prefix, std::vector<unsigned char>* out) -> bool {
    std::string line;
    if (!std::getline(buffer, line)) {
      return false;
    }
    auto eq = line.find('=');
    if (eq == std::string::npos) {
      return false;
    }
    auto key = Trim(line.substr(0, eq));
    auto value = Trim(line.substr(eq + 1));
    if (key != expected_prefix) {
      return false;
    }
    *out = HexDecode(value);
    return !out->empty();
  };
  std::vector<unsigned char> nonce;
  std::vector<unsigned char> tag;
  std::vector<unsigned char> data;
  if (!parseLine("nonce", &nonce)) return false;
  if (!parseLine("tag", &tag)) return false;
  if (!parseLine("data", &data)) return false;

  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) {
    return false;
  }
  bool ok = false;
  do {
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) break;
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, nonce.size(), nullptr) != 1) break;
    if (EVP_DecryptInit_ex(ctx, nullptr, nullptr, key_.data(), nonce.data()) != 1) break;
    std::vector<unsigned char> plain(data.size());
    int out_len = 0;
    if (EVP_DecryptUpdate(ctx, plain.data(), &out_len, data.data(), data.size()) != 1) break;
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, tag.size(), tag.data()) != 1) break;
    int final_len = 0;
    if (EVP_DecryptFinal_ex(ctx, plain.data() + out_len, &final_len) != 1) break;
    out_len += final_len;
    plain.resize(out_len);
    plaintext->assign(reinterpret_cast<char*>(plain.data()), plain.size());
    ok = true;
  } while (false);
  EVP_CIPHER_CTX_free(ctx);
  return ok;
}
}  // namespace inferflux
