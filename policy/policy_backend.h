#pragma once

#include <memory>
#include <optional>
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

struct RoutingPolicyEntry {
  bool allow_default_fallback{true};
  bool require_ready_backend{false};
  std::string fallback_scope{"any_compatible"};
};

// --- Focused policy interfaces (Phase D1) ---
// Each interface captures a single policy concern.  PolicyBackend inherits
// all of them for backward compatibility — existing code that accepts a
// PolicyBackend* continues to work unchanged.

/// Persistence lifecycle (load/save/identity).
class PolicyPersistence {
public:
  virtual ~PolicyPersistence() = default;
  virtual bool Load() = 0;
  virtual bool Save() const = 0;
  virtual std::string Name() const = 0;
};

/// API key management interface.
class IApiKeyPolicy {
public:
  virtual ~IApiKeyPolicy() = default;
  virtual std::vector<PolicyKeyEntry> ApiKeys() const = 0;
  virtual void SetApiKey(const std::string &key,
                         const std::vector<std::string> &scopes) = 0;
  virtual bool RemoveApiKey(const std::string &key) = 0;
};

/// Guardrail blocklist management.
class GuardrailPolicy {
public:
  virtual ~GuardrailPolicy() = default;
  virtual std::vector<std::string> GuardrailBlocklist() const = 0;
  virtual void
  SetGuardrailBlocklist(const std::vector<std::string> &blocklist) = 0;
};

/// Rate limit configuration.
class RateLimitPolicy {
public:
  virtual ~RateLimitPolicy() = default;
  virtual int RateLimitPerMinute() const = 0;
  virtual void SetRateLimitPerMinute(int limit) = 0;
};

/// Capability-aware routing policy.
class RoutingPolicyProvider {
public:
  virtual ~RoutingPolicyProvider() = default;
  virtual std::optional<RoutingPolicyEntry> RoutingPolicy() const = 0;
  virtual void SetRoutingPolicy(const RoutingPolicyEntry &policy) = 0;
  virtual void ClearRoutingPolicy() = 0;
};

/// Full policy backend — composes all focused interfaces.
/// Existing implementations (PolicyStore) inherit PolicyBackend unchanged.
class PolicyBackend : public PolicyPersistence,
                      public IApiKeyPolicy,
                      public GuardrailPolicy,
                      public RateLimitPolicy,
                      public RoutingPolicyProvider {
public:
  ~PolicyBackend() override = default;
};

} // namespace inferflux
