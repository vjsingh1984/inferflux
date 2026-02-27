#include "policy/policy_store.h"

#include <filesystem>
#include <fstream>
#include <sstream>

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
}  // namespace

PolicyStore::PolicyStore(std::string path) : path_(std::move(path)) {}

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
  std::ifstream input(path_);
  if (!input.good()) {
    return false;
  }
  std::string line;
  std::string section;
  while (std::getline(input, line)) {
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
  std::ofstream output(path_, std::ios::trunc);
  if (!output.good()) {
    return false;
  }
  output << "[api_keys]\n";
  for (const auto& [key, scopes] : api_keys_) {
    output << key << "=" << JoinCSV(scopes) << "\n";
  }
  output << "\n[guardrail]\n";
  output << "words=" << JoinCSV(guardrail_blocklist_) << "\n";
  output << "\n[rate_limit]\n";
  output << "tokens=" << rate_limit_per_minute_ << "\n";
  return true;
}

std::vector<ApiKeyPolicy> PolicyStore::ApiKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ApiKeyPolicy> keys;
  keys.reserve(api_keys_.size());
  for (const auto& [key, scopes] : api_keys_) {
    keys.push_back({key, scopes});
  }
  return keys;
}

void PolicyStore::SetApiKey(const std::string& key, const std::vector<std::string>& scopes) {
  std::lock_guard<std::mutex> lock(mutex_);
  api_keys_[key] = scopes;
}

bool PolicyStore::RemoveApiKey(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  return api_keys_.erase(key) > 0;
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

}  // namespace inferflux
