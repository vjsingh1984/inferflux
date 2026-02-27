#include "policy/opa_client.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "net/http_client.h"

namespace inferflux {

OPAClient::OPAClient(std::string endpoint) : endpoint_(std::move(endpoint)) {}

bool OPAClient::Evaluate(const std::string& prompt, OPAResult* result) const {
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
    std::string body((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    auto allow_pos = body.find("\"allow\"");
    if (allow_pos != std::string::npos) {
      auto true_pos = body.find("true", allow_pos);
      auto false_pos = body.find("false", allow_pos);
      if (false_pos != std::string::npos &&
          (true_pos == std::string::npos || false_pos < true_pos)) {
        result->allow = false;
      }
    }
    auto reason_pos = body.find("\"reason\"");
    if (reason_pos != std::string::npos) {
      auto colon = body.find(':', reason_pos);
      auto quote = body.find('"', colon);
      auto closing = body.find('"', quote + 1);
      if (quote != std::string::npos && closing != std::string::npos) {
        result->reason = body.substr(quote + 1, closing - quote - 1);
      }
    }
    if (!result->allow && result->reason.empty()) {
      result->reason = "OPA policy denied prompt";
    }
    return result->allow;
  }
  if (endpoint_.rfind("http://", 0) == 0) {
    HttpClient client;
    std::string payload = std::string("{\"input\":{\"prompt\":\"") + prompt + "\"}}";
    try {
      auto http_resp = client.Post(endpoint_, payload, {{"Content-Type", "application/json"}});
      if (http_resp.status >= 200 && http_resp.status < 300) {
        auto allow_pos = http_resp.body.find("\"allow\":");
        if (allow_pos != std::string::npos) {
          auto false_pos = http_resp.body.find("false", allow_pos);
          auto true_pos = http_resp.body.find("true", allow_pos);
          if (false_pos != std::string::npos &&
              (true_pos == std::string::npos || false_pos < true_pos)) {
            result->allow = false;
          }
        }
        auto reason_pos = http_resp.body.find("\"reason\":");
        if (reason_pos != std::string::npos) {
          auto colon = http_resp.body.find(':', reason_pos);
          auto quote = http_resp.body.find('"', colon + 1);
          auto closing = http_resp.body.find('"', quote + 1);
          if (colon != std::string::npos && quote != std::string::npos && closing != std::string::npos) {
            result->reason = http_resp.body.substr(quote + 1, closing - quote - 1);
          }
        }
        return result->allow;
      }
      result->reason = "OPA HTTP error: " + std::to_string(http_resp.status);
    } catch (const std::exception& ex) {
      result->reason = ex.what();
    }
    return true;
  }
  std::cout << "[guardrail] OPA endpoint " << endpoint_
            << " configured but unsupported scheme" << std::endl;
  result->reason = "OPA endpoint configured";
  return true;
}

}  // namespace inferflux
