#pragma once

#include "scheduler/request_batch.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace inferflux {

inline std::string BuildRequestDebugContext(int64_t request_id,
                                            std::string_view client_request_id,
                                            int sequence_id,
                                            uint64_t sequence_generation) {
  std::string message = "request_id=" + std::to_string(request_id);
  if (!client_request_id.empty()) {
    message += ", client_request_id=" + std::string(client_request_id);
  }
  message += ", sequence_id=" + std::to_string(sequence_id);
  message += ", sequence_generation=" + std::to_string(sequence_generation);
  return message;
}

inline std::string BuildRequestDebugContext(const InferenceRequest &request) {
  return BuildRequestDebugContext(static_cast<int64_t>(request.id),
                                  request.client_request_id,
                                  request.sequence_id,
                                  request.sequence_generation);
}

inline void AppendRequestDebugField(std::string *message, std::string_view key,
                                    std::string_view value) {
  if (!message || key.empty()) {
    return;
  }
  message->append(", ");
  message->append(key);
  message->append("=");
  message->append(value);
}

inline void AppendRequestDebugField(std::string *message, std::string_view key,
                                    const std::string &value) {
  AppendRequestDebugField(message, key, std::string_view(value));
}

inline void AppendRequestDebugField(std::string *message, std::string_view key,
                                    int value) {
  AppendRequestDebugField(message, key, std::to_string(value));
}

inline void AppendRequestDebugField(std::string *message, std::string_view key,
                                    int64_t value) {
  AppendRequestDebugField(message, key, std::to_string(value));
}

inline void AppendRequestDebugField(std::string *message, std::string_view key,
                                    uint64_t value) {
  AppendRequestDebugField(message, key, std::to_string(value));
}

inline void AppendRequestDebugField(std::string *message, std::string_view key,
                                    bool value) {
  AppendRequestDebugField(message, key, value ? "true" : "false");
}

inline void AppendRequestDebugField(std::string *message, std::string_view key,
                                    double value) {
  AppendRequestDebugField(message, key, std::to_string(value));
}

} // namespace inferflux
