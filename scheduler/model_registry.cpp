#include "scheduler/model_registry.h"
#include "server/logging/logger.h"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace inferflux {

ModelRegistry::ModelRegistry(std::shared_ptr<ModelRouter> router)
    : router_(std::move(router)) {}

ModelRegistry::~ModelRegistry() { Stop(); }

int ModelRegistry::LoadAndWatch(const std::filesystem::path &path,
                                int poll_interval_ms) {
  if (running_.load())
    return 0; // already watching

  path_ = path;
  poll_interval_ms_ = poll_interval_ms;

  int loaded = Reload();

  running_.store(true);
  watch_thread_ = std::thread([this] { WatchLoop(); });

  return loaded;
}

void ModelRegistry::Stop() {
  if (!running_.load())
    return;
  running_.store(false);
  if (watch_thread_.joinable()) {
    watch_thread_.join();
  }
}

std::set<std::string> ModelRegistry::ManagedIds() const {
  std::lock_guard<std::mutex> lock(managed_mutex_);
  std::set<std::string> ids;
  for (const auto &[path, id] : path_to_id_) {
    ids.insert(id);
  }
  return ids;
}

int ModelRegistry::Reload() {
  if (!std::filesystem::exists(path_)) {
    log::Warn("model_registry", "registry file not found: " + path_.string());
    return 0;
  }

  // Read file content.
  std::ifstream f(path_);
  if (!f.is_open()) {
    log::Error("model_registry",
               "cannot open registry file: " + path_.string());
    return 0;
  }
  std::ostringstream buf;
  buf << f.rdbuf();

  std::vector<RegistryEntry> entries;
  if (!ParseYaml(buf.str(), entries)) {
    return 0; // parse error already logged
  }

  return ApplyEntries(entries);
}

bool ModelRegistry::ParseYaml(const std::string &yaml_text,
                              std::vector<RegistryEntry> &out) const {
  try {
    YAML::Node root = YAML::Load(yaml_text);
    if (!root["models"] || !root["models"].IsSequence()) {
      log::Warn("model_registry",
                "registry yaml has no 'models' sequence; nothing to load");
      return true; // treat empty registry as valid
    }
    for (const auto &node : root["models"]) {
      RegistryEntry e;
      if (node["path"])
        e.path = node["path"].as<std::string>();
      if (node["id"])
        e.id = node["id"].as<std::string>();
      if (node["backend"])
        e.backend = node["backend"].as<std::string>();
      if (e.path.empty()) {
        log::Warn("model_registry", "skipping registry entry with no path");
        continue;
      }
      out.push_back(std::move(e));
    }
    return true;
  } catch (const YAML::Exception &ex) {
    log::Error("model_registry", std::string("YAML parse error: ") + ex.what());
    return false;
  }
}

int ModelRegistry::ApplyEntries(const std::vector<RegistryEntry> &entries) {
  // Build the desired path set from new entries.
  std::map<std::string, RegistryEntry> desired; // path → entry
  for (const auto &e : entries) {
    desired[e.path] = e;
  }

  std::lock_guard<std::mutex> lock(managed_mutex_);
  int net = 0;

  // Unload models whose paths are no longer in the registry.
  std::vector<std::string> to_remove;
  for (const auto &[path, assigned_id] : path_to_id_) {
    if (!desired.count(path)) {
      to_remove.push_back(path);
    }
  }
  for (const auto &path : to_remove) {
    const std::string &id = path_to_id_[path];
    if (router_->UnloadModel(id)) {
      log::Info("model_registry",
                "hot-unloaded model id=" + id + " path=" + path);
      --net;
    } else {
      log::Warn("model_registry", "failed to unload model id=" + id);
    }
    path_to_id_.erase(path);
  }

  // Load models whose paths are new.
  for (const auto &[path, entry] : desired) {
    if (path_to_id_.count(path))
      continue; // already managed

    auto assigned_id = router_->LoadModel(entry.path, entry.backend, entry.id);
    if (assigned_id.empty()) {
      log::Error("model_registry", "failed to load model path=" + entry.path);
      continue;
    }
    path_to_id_[path] = assigned_id;
    log::Info("model_registry",
              "hot-loaded model id=" + assigned_id + " path=" + path);
    ++net;
  }

  return net;
}

void ModelRegistry::WatchLoop() {
  while (running_.load()) {
    // Sleep in short intervals so Stop() is responsive.
    for (int i = 0; i < poll_interval_ms_ / 100 && running_.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!running_.load())
      break;

    try {
      auto mtime = std::filesystem::last_write_time(path_);
      if (mtime != last_mtime_) {
        last_mtime_ = mtime;
        log::Info("model_registry",
                  "registry file changed; reloading: " + path_.string());
        Reload();
      }
    } catch (const std::filesystem::filesystem_error &ex) {
      log::Warn("model_registry",
                std::string("cannot stat registry file: ") + ex.what());
    }
  }
}

} // namespace inferflux
