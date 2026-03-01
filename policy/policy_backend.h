#pragma once

#include <memory>
#include <string>
#include <vector>

namespace inferflux {

// PolicyBackend is the plugin interface for policy storage and enforcement.
// All policy implementations (native INI store, OPA, Cedar, SQL) must implement
// this interface. This abstraction ensures the core server never depends on a
// specific policy storage format.
//
// To add a new backend:
//   1. Implement this interface.
//   2. Register it via PolicyBackendRegistry (or pass to HttpServer directly).
//
// Thread safety: all methods must be safe to call concurrently.

struct PolicyKeyEntry {
  std::string key;
  std::vector<std::string> scopes;
};

class PolicyBackend {
 public:
  virtual ~PolicyBackend() = default;

  // Lifecycle
  virtual bool Load() = 0;
  virtual bool Save() const = 0;

  // API key management
  virtual std::vector<PolicyKeyEntry> ApiKeys() const = 0;
  virtual void SetApiKey(const std::string& key, const std::vector<std::string>& scopes) = 0;
  virtual bool RemoveApiKey(const std::string& key) = 0;

  // Guardrail configuration
  virtual std::vector<std::string> GuardrailBlocklist() const = 0;
  virtual void SetGuardrailBlocklist(const std::vector<std::string>& blocklist) = 0;

  // Rate limiting
  virtual int RateLimitPerMinute() const = 0;
  virtual void SetRateLimitPerMinute(int limit) = 0;

  // Identity — useful for logging and diagnostics.
  virtual std::string Name() const = 0;
};

}  // namespace inferflux
