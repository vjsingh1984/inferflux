#include "policy/opa_client.h"

#include <filesystem>
#include <fstream>
#include <iostream>

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
  std::cout << "[guardrail] OPA endpoint " << endpoint_
            << " configured but HTTP integration not yet implemented" << std::endl;
  result->reason = "OPA endpoint configured";
  return true;
}

}  // namespace inferflux
