#include "policy/opa_client.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "net/http_client.h"

using json = nlohmann::json;

namespace inferflux {

namespace {
void ParseOPAResponse(const std::string &body, OPAResult *result) {
  try {
    auto j = json::parse(body);
    if (j.contains("allow") && j["allow"].is_boolean()) {
      result->allow = j["allow"].get<bool>();
    }
    if (j.contains("reason") && j["reason"].is_string()) {
      result->reason = j["reason"].get<std::string>();
    }
    // Also check nested result object (OPA v1 API returns {result: {allow:
    // ...}}).
    if (j.contains("result") && j["result"].is_object()) {
      auto &r = j["result"];
      if (r.contains("allow") && r["allow"].is_boolean()) {
        result->allow = r["allow"].get<bool>();
      }
      if (r.contains("reason") && r["reason"].is_string()) {
        result->reason = r["reason"].get<std::string>();
      }
    }
  } catch (const json::exception &) {
    // Leave result unchanged on parse failure.
  }
}
} // namespace

OPAClient::OPAClient(std::string endpoint) : endpoint_(std::move(endpoint)) {}

bool OPAClient::Evaluate(const std::string &prompt, OPAResult *result) const {
  if (!result) {
    return true;
  }
  result->allow = true;
  result->reason.clear();
  if (endpoint_.empty()) {
    return true;
  }
  if (endpoint_.rfind("file://", 0) == 0) {
    std::filesystem::path path = endpoint_.substr(strlen("file://"));
    if (!std::filesystem::exists(path)) {
      result->reason = "OPA file not found";
      return true;
    }
    std::ifstream input(path);
    if (!input.good()) {
      result->reason = "OPA file unreadable";
      return true;
    }
    std::string body((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
    ParseOPAResponse(body, result);
    if (!result->allow && result->reason.empty()) {
      result->reason = "OPA policy denied prompt";
    }
    return result->allow;
  }
  if (endpoint_.rfind("http://", 0) == 0) {
    HttpClient client;
    json payload = {{"input", {{"prompt", prompt}}}};
    try {
      auto http_resp = client.Post(endpoint_, payload.dump(),
                                   {{"Content-Type", "application/json"}});
      if (http_resp.status >= 200 && http_resp.status < 300) {
        ParseOPAResponse(http_resp.body, result);
        return result->allow;
      }
      result->reason = "OPA HTTP error: " + std::to_string(http_resp.status);
    } catch (const std::exception &ex) {
      result->reason = ex.what();
    }
    return true;
  }
  std::cout << "[guardrail] OPA endpoint " << endpoint_
            << " configured but unsupported scheme" << std::endl;
  result->reason = "OPA endpoint configured";
  return true;
}

} // namespace inferflux
