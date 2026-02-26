#include "server/http/http_server.h"

#include "server/metrics/metrics.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace inferflux {

namespace {
bool SendAll(int fd, const std::string& payload) {
  const char* data = payload.c_str();
  std::size_t remaining = payload.size();
  while (remaining > 0) {
    ssize_t sent = ::send(fd, data, remaining, 0);
    if (sent <= 0) {
      return false;
    }
    data += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
  return true;
}

std::string BuildResponse(const std::string& body, int status = 200, const std::string& status_text = "OK") {
  std::string headers = "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n";
  headers += "Content-Type: application/json\r\n";
  headers += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
  return headers + body;
}

std::string EscapeJson(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          continue;
        }
        escaped.push_back(c);
    }
  }
  return escaped;
}

std::optional<std::string> ExtractJsonString(const std::string& body, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  auto key_pos = body.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  auto colon = body.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  auto value_start = body.find('"', colon);
  if (value_start == std::string::npos) {
    return std::nullopt;
  }
  ++value_start;
  std::string value;
  bool escape = false;
  for (std::size_t i = value_start; i < body.size(); ++i) {
    char c = body[i];
    if (escape) {
      value.push_back(c);
      escape = false;
      continue;
    }
    if (c == '\\') {
      escape = true;
      continue;
    }
    if (c == '"') {
      return value;
    }
    value.push_back(c);
  }
  return std::nullopt;
}

std::optional<int> ExtractJsonInt(const std::string& body, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  auto key_pos = body.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  auto colon = body.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  auto value_start = body.find_first_not_of(" \t\r\n", colon + 1);
  if (value_start == std::string::npos) {
    return std::nullopt;
  }
  std::size_t value_end = value_start;
  while (value_end < body.size() && (std::isdigit(static_cast<unsigned char>(body[value_end])) || body[value_end] == '-')) {
    ++value_end;
  }
  if (value_end == value_start) {
    return std::nullopt;
  }
  try {
    return std::stoi(body.substr(value_start, value_end - value_start));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<bool> ExtractJsonBool(const std::string& body, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  auto key_pos = body.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  auto colon = body.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  auto value_start = body.find_first_not_of(" \t\r\n", colon + 1);
  if (value_start == std::string::npos) {
    return std::nullopt;
  }
  if (body.compare(value_start, 4, "true") == 0) {
    return true;
  }
  if (body.compare(value_start, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

struct ChatMessage {
  std::string role;
  std::string content;
};

struct CompletionRequestPayload {
  std::string prompt;
  std::string model{"unknown"};
  int max_tokens{64};
  std::vector<ChatMessage> messages;
  bool stream{false};
};

std::vector<ChatMessage> ExtractMessages(const std::string& body) {
  std::vector<ChatMessage> messages;
  std::string needle = "\"messages\"";
  auto key_pos = body.find(needle);
  if (key_pos == std::string::npos) {
    return messages;
  }
  auto array_start = body.find('[', key_pos);
  if (array_start == std::string::npos) {
    return messages;
  }
  int depth = 1;
  std::size_t cursor = array_start + 1;
  std::size_t array_end = std::string::npos;
  for (; cursor < body.size(); ++cursor) {
    char c = body[cursor];
    if (c == '[') {
      ++depth;
    } else if (c == ']') {
      --depth;
      if (depth == 0) {
        array_end = cursor;
        break;
      }
    }
  }
  if (array_end == std::string::npos) {
    return messages;
  }
  std::string array_block = body.substr(array_start, array_end - array_start + 1);
  std::size_t pos = 0;
  while (true) {
    auto obj_start = array_block.find('{', pos);
    if (obj_start == std::string::npos) {
      break;
    }
    int brace_depth = 1;
    std::size_t obj_end = obj_start + 1;
    for (; obj_end < array_block.size(); ++obj_end) {
      char c = array_block[obj_end];
      if (c == '{') {
        ++brace_depth;
      } else if (c == '}') {
        --brace_depth;
        if (brace_depth == 0) {
          break;
        }
      }
    }
    if (brace_depth != 0 || obj_end >= array_block.size()) {
      break;
    }
    std::string object = array_block.substr(obj_start, obj_end - obj_start + 1);
    ChatMessage message;
    if (auto role = ExtractJsonString(object, "role")) {
      message.role = *role;
    }
    if (auto content = ExtractJsonString(object, "content")) {
      message.content = *content;
    }
    if (!message.role.empty() || !message.content.empty()) {
      messages.push_back(std::move(message));
    }
    pos = obj_end + 1;
  }
  return messages;
}

CompletionRequestPayload ParseJsonPayload(const std::string& body) {
  CompletionRequestPayload payload;
  if (body.empty()) {
    return payload;
  }
  if (auto prompt = ExtractJsonString(body, "prompt")) {
    payload.prompt = *prompt;
  }
  if (auto model = ExtractJsonString(body, "model")) {
    payload.model = *model;
  }
  if (auto max_tokens = ExtractJsonInt(body, "max_tokens")) {
    payload.max_tokens = *max_tokens;
  }
  payload.messages = ExtractMessages(body);
  if (auto stream = ExtractJsonBool(body, "stream")) {
    payload.stream = *stream;
  }
  return payload;
}

std::string FlattenMessages(const std::vector<ChatMessage>& messages) {
  std::string prompt;
  for (const auto& message : messages) {
    if (!prompt.empty()) {
      prompt.append("\n");
    }
    std::string role = message.role.empty() ? "user" : message.role;
    prompt.append(role);
    prompt.append(": ");
    prompt.append(message.content);
  }
  return prompt;
}

std::string BuildCompletionBody(const GenerateResponse& result,
                                const CompletionRequestPayload& request,
                                bool chat_mode) {
  auto now = std::chrono::system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  std::string id_prefix = chat_mode ? "chatcmpl-" : "cmpl-";
  std::string usage = "\"usage\":{\"prompt_tokens\":" + std::to_string(result.prompt_tokens) +
                      ",\"completion_tokens\":" + std::to_string(result.completion_tokens) +
                      ",\"total_tokens\":" +
                      std::to_string(result.prompt_tokens + result.completion_tokens) + "}";
  if (chat_mode) {
    return std::string("{\"id\":\"") + id_prefix + std::to_string(ts) +
           "\",\"object\":\"chat.completion\",\"created\":" + std::to_string(ts) + ",\"model\":\"" +
           EscapeJson(request.model) + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
           EscapeJson(result.completion) + "\"},\"finish_reason\":\"stop\"}]," + usage + "}";
  }
  return std::string("{\"id\":\"") + id_prefix + std::to_string(ts) +
         "\",\"object\":\"text_completion\",\"created\":" + std::to_string(ts) + ",\"model\":\"" +
         EscapeJson(request.model) + "\",\"choices\":[{\"index\":0,\"text\":\"" +
         EscapeJson(result.completion) + "\",\"finish_reason\":\"stop\"}]," + usage + "}";
}

std::string BuildErrorBody(const std::string& error) {
  return std::string("{\"error\":\"") + EscapeJson(error) + "\"}";
}

std::vector<std::string> SplitForStreaming(const std::string& text) {
  std::vector<std::string> chunks;
  std::string current;
  for (char c : text) {
    current.push_back(c);
    if (std::isspace(static_cast<unsigned char>(c))) {
      chunks.push_back(current);
      current.clear();
    }
  }
  if (!current.empty()) {
    chunks.push_back(current);
  }
  if (chunks.empty()) {
    chunks.push_back(text);
  }
  return chunks;
}

std::string BuildStreamChunk(const std::string& id,
                             const std::string& model,
                             std::time_t ts,
                             const std::string& content,
                             bool finish) {
  if (finish) {
    return std::string("data: {\"id\":\"") + id +
           "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(ts) + ",\"model\":\"" +
           EscapeJson(model) + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
  }
  return std::string("data: {\"id\":\"") + id +
         "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(ts) + ",\"model\":\"" +
         EscapeJson(model) + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
         EscapeJson(content) + "\"},\"finish_reason\":null}]}\n\n";
}
}

HttpServer::HttpServer(std::string host,
                       int port,
                       Scheduler* scheduler,
                       std::shared_ptr<ApiKeyAuth> auth,
                       MetricsRegistry* metrics,
                       OIDCValidator* oidc,
                       RateLimiter* rate_limiter,
                       Guardrail* guardrail,
                       AuditLogger* audit_logger)
    : host_(std::move(host)),
      port_(port),
      scheduler_(scheduler),
      auth_(std::move(auth)),
      metrics_(metrics),
      oidc_(oidc),
      rate_limiter_(rate_limiter),
      guardrail_(guardrail),
      audit_logger_(audit_logger) {}

HttpServer::~HttpServer() { Stop(); }

void HttpServer::Start() {
  if (running_) {
    return;
  }
  running_ = true;
  worker_ = std::thread(&HttpServer::Run, this);
}

void HttpServer::Stop() {
  if (!running_) {
    return;
  }
  running_ = false;
  if (worker_.joinable()) {
    worker_.join();
  }
}

void HttpServer::Run() {
  int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    std::perror("socket");
    return;
  }

  int opt = 1;
  ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  addr.sin_addr.s_addr = inet_addr(host_.c_str());

  if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::perror("bind");
    ::close(server_fd);
    return;
  }

  if (::listen(server_fd, 16) < 0) {
    std::perror("listen");
    ::close(server_fd);
    return;
  }

  while (running_) {
    sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
      continue;
    }
    HandleClient(client_fd);
    ::close(client_fd);
  }

  ::close(server_fd);
}

bool HttpServer::ResolveSubject(const std::string& headers, std::string* subject) const {
  bool require_auth = (auth_ && auth_->HasKeys()) || (oidc_ && oidc_->Enabled());
  if (!require_auth) {
    if (subject) {
      *subject = "anonymous";
    }
    return true;
  }
  auto pos = headers.find("Authorization:");
  if (pos == std::string::npos) {
    return false;
  }
  auto end = headers.find("\r\n", pos);
  std::string line = headers.substr(pos, end - pos);
  auto token_pos = line.find("Bearer");
  if (token_pos == std::string::npos) {
    return false;
  }
  std::string token = line.substr(token_pos + 6);
  token.erase(0, token.find_first_not_of(' '));
  token.erase(token.find_last_not_of(' ') + 1);
  if (auth_ && auth_->HasKeys() && auth_->IsAllowed(token)) {
    if (subject) {
      *subject = token;
    }
    return true;
  }
  if (oidc_ && oidc_->Enabled()) {
    std::string sub;
    if (oidc_->Validate(token, &sub)) {
      if (subject) {
        *subject = sub;
      }
      return true;
    }
  }
  return false;
}

void HttpServer::HandleClient(int client_fd) {
  char buffer[8192];
  ssize_t bytes = ::recv(client_fd, buffer, sizeof(buffer) - 1, 0);
  if (bytes <= 0) {
    return;
  }
  buffer[bytes] = '\0';
  std::string request(buffer);
  auto header_end = request.find("\r\n\r\n");
  std::string headers = header_end == std::string::npos ? request : request.substr(0, header_end);
  std::string subject = "anonymous";
  if (!ResolveSubject(headers, &subject)) {
    std::string body = R"({"error":"unauthorized"})";
    auto response = BuildResponse(body, 401, "Unauthorized");
    SendAll(client_fd, response);
    if (audit_logger_) {
      audit_logger_->Log(subject, "", "unauthorized", "missing or invalid credentials");
    }
    return;
  }
  if (rate_limiter_ && rate_limiter_->Enabled() && !rate_limiter_->Allow(subject)) {
    auto response = BuildResponse(R"({"error":"rate_limited"})", 429, "Too Many Requests");
    SendAll(client_fd, response);
    if (audit_logger_) {
      audit_logger_->Log(subject, "", "rate_limited", "token bucket exceeded");
    }
    return;
  }

  std::string body = header_end == std::string::npos ? std::string() : request.substr(header_end + 4);
  std::string path;
  auto first_line_end = headers.find("\r\n");
  std::string first_line = headers.substr(0, first_line_end);
  auto method_end = first_line.find(' ');
  auto path_end = first_line.find(' ', method_end + 1);
  std::string method = first_line.substr(0, method_end);
  path = first_line.substr(method_end + 1, path_end - method_end - 1);

  if (method == "GET" && path == "/healthz") {
    auto response = BuildResponse(R"({"status":"ok"})");
    SendAll(client_fd, response);
    return;
  }

  if (method == "GET" && path == "/metrics") {
    std::string body = metrics_ ? metrics_->RenderPrometheus() : "";
    std::string headers = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n";
    SendAll(client_fd, headers + body);
    return;
  }

  if (method == "POST" && (path == "/v1/completions" || path == "/v1/chat/completions")) {
    auto parsed = ParseJsonPayload(body);
    GenerateRequest req;
    if (!parsed.prompt.empty()) {
      req.prompt = parsed.prompt;
    } else if (!parsed.messages.empty()) {
      req.prompt = FlattenMessages(parsed.messages);
    }
    if (parsed.max_tokens > 0) {
      req.max_tokens = parsed.max_tokens;
    }
    if (req.prompt.empty()) {
      auto payload = BuildResponse(BuildErrorBody("prompt or messages are required"), 400, "Bad Request");
      SendAll(client_fd, payload);
      return;
    }
    bool chat_mode = (path == "/v1/chat/completions") || !parsed.messages.empty();
    std::string guard_reason;
    if (guardrail_ && guardrail_->Enabled() && !guardrail_->Check(req.prompt, &guard_reason)) {
      auto payload = BuildResponse(BuildErrorBody(guard_reason), 400, "Bad Request");
      SendAll(client_fd, payload);
      if (audit_logger_) {
        audit_logger_->Log(subject, parsed.model, "blocked", guard_reason);
      }
      return;
    }
    try {
      auto result = scheduler_->Generate(req);
      if (metrics_) {
        metrics_->RecordSuccess(result.prompt_tokens, result.completion_tokens);
      }
      if (audit_logger_) {
        audit_logger_->Log(subject, parsed.model, "success", "prompt_tokens=" +
                                                            std::to_string(result.prompt_tokens) +
                                                            ",completion_tokens=" +
                                                            std::to_string(result.completion_tokens));
      }
      if (parsed.stream) {
        auto now = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
        std::string id = std::string("chatcmpl-") + std::to_string(ts);
        std::string headers =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
        SendAll(client_fd, headers);
        auto chunks = SplitForStreaming(result.completion);
        for (const auto& chunk : chunks) {
          SendAll(client_fd, BuildStreamChunk(id, parsed.model, ts, chunk, false));
        }
        SendAll(client_fd, BuildStreamChunk(id, parsed.model, ts, "", true));
        SendAll(client_fd, "data: [DONE]\n\n");
      } else {
        auto payload = BuildResponse(BuildCompletionBody(result, parsed, chat_mode));
        SendAll(client_fd, payload);
      }
    } catch (const std::exception& ex) {
      if (metrics_) {
        metrics_->RecordError();
      }
      auto payload = BuildResponse(BuildErrorBody(ex.what()), 500, "Error");
      SendAll(client_fd, payload);
      if (audit_logger_) {
        audit_logger_->Log(subject, parsed.model, "error", ex.what());
      }
    }
    return;
  }

  auto response = BuildResponse(R"({"error":"not_found"})", 404, "Not Found");
  SendAll(client_fd, response);
  if (audit_logger_) {
    audit_logger_->Log(subject, "", "not_found", path);
  }
}

}  // namespace inferflux
