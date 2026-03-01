#include "server/http/http_server.h"

#include "runtime/multimodal/image_preprocessor.h"
#include "server/metrics/metrics.h"
#include "server/tracing/span.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <openssl/err.h>
#include <openssl/ssl.h>

using json = nlohmann::json;

namespace inferflux {

namespace {

// Returns the trimmed value of an HTTP header from the raw header block, or
// empty string if the header is not present. Header name is case-sensitive
// (use Title-Case, e.g. "traceparent").
std::string GetHeaderValue(const std::string &headers,
                           const std::string &name) {
  auto pos = headers.find(name + ":");
  if (pos == std::string::npos)
    return {};
  auto end = headers.find("\r\n", pos);
  std::string val =
      headers.substr(pos + name.size() + 1, end - pos - name.size() - 1);
  // Trim leading/trailing whitespace.
  auto s = val.find_first_not_of(" \t");
  auto e = val.find_last_not_of(" \t\r\n");
  return (s == std::string::npos) ? "" : val.substr(s, e - s + 1);
}

std::string BuildResponse(const std::string &body, int status = 200,
                          const std::string &status_text = "OK",
                          const std::string &extra_headers = "") {
  std::string headers =
      "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n";
  headers += "Content-Type: application/json\r\n";
  headers += "Access-Control-Allow-Origin: *\r\n";
  if (!extra_headers.empty()) {
    headers += extra_headers;
  }
  headers += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
  return headers + body;
}

struct ChatMessage {
  std::string role;
  std::string content;
};

// §2.3 — tool/function calling types.
struct ToolFunction {
  std::string name;
  std::string description;
  json parameters; // JSON Schema object (may be null/empty).
};

struct Tool {
  std::string type{"function"}; // Only "function" is supported.
  ToolFunction function;
};

struct ToolCallResult {
  bool detected{false};
  std::string call_id;
  std::string function_name;
  std::string arguments_json; // JSON-encoded arguments object.
};

namespace {

constexpr std::size_t kMaxResponseFormatBytes =
    16 * 1024; // 16 KB cap for schemas/grammars.

} // namespace

struct CompletionRequestPayload {
  std::string prompt;
  std::string model{"unknown"};
  int max_tokens{256};
  std::vector<ChatMessage> messages;
  bool stream{false};
  bool json_mode{false};   // true when response_format.type == "json_object"
  std::vector<Tool> tools; // §2.3: function definitions available to the model
  std::string first_tool_name;
  bool has_tool_schema{false};
  std::string tool_choice{"auto"}; // "auto" | "none" | "required"
  bool has_images{
      false}; // §2.2: true when any message contained image_url parts.
  std::vector<DecodedImage> images; // §2.2: decoded images in prompt order.
  bool has_response_format{false};
  std::string response_format_type;
  std::string response_format_schema;
  std::string response_format_grammar;
  std::string response_format_root{"root"};
  bool response_format_ok{true};
  std::string response_format_error;
};

CompletionRequestPayload ParseJsonPayload(const std::string &body) {
  CompletionRequestPayload payload;
  if (body.empty()) {
    return payload;
  }
  try {
    auto j = json::parse(body);
    if (j.contains("prompt") && j["prompt"].is_string()) {
      payload.prompt = j["prompt"].get<std::string>();
    }
    if (j.contains("model") && j["model"].is_string()) {
      payload.model = j["model"].get<std::string>();
    }
    if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) {
      payload.max_tokens = j["max_tokens"].get<int>();
    }
    if (j.contains("stream") && j["stream"].is_boolean()) {
      payload.stream = j["stream"].get<bool>();
    }
    // OpenAI-compatible: response_format controls structured output.
    if (j.contains("response_format") && j["response_format"].is_object()) {
      auto &rf = j["response_format"];
      if (rf.contains("type") && rf["type"].is_string()) {
        payload.has_response_format = true;
        payload.response_format_type = rf["type"].get<std::string>();
        auto type = payload.response_format_type;

        if (type == "json_object" || type == "json_schema") {
          payload.json_mode = true;
          if (rf.contains("schema")) {
            payload.response_format_schema = rf["schema"].dump();
          }
          if (rf.contains("json_schema")) {
            payload.response_format_schema = rf["json_schema"].dump();
          }
          if (payload.response_format_schema.size() > kMaxResponseFormatBytes) {
            payload.response_format_ok = false;
            payload.response_format_error =
                "response_format schema exceeds 16KB limit";
          }
          if (type == "json_schema" && payload.response_format_schema.empty()) {
            payload.response_format_ok = false;
            payload.response_format_error =
                "response_format json_schema requires a schema definition";
          }
        } else if (type == "grammar") {
          if (rf.contains("grammar") && rf["grammar"].is_string()) {
            payload.response_format_grammar = rf["grammar"].get<std::string>();
            if (payload.response_format_grammar.size() >
                kMaxResponseFormatBytes) {
              payload.response_format_ok = false;
              payload.response_format_error =
                  "response_format grammar exceeds 16KB limit";
            }
          } else {
            payload.response_format_ok = false;
            payload.response_format_error =
                "response_format grammar missing 'grammar' string";
          }
          if (rf.contains("root") && rf["root"].is_string()) {
            payload.response_format_root = rf["root"].get<std::string>();
          }
        } else if (type == "text" || type.empty()) {
          payload.has_response_format = false; // treat as default behavior.
        } else {
          payload.response_format_ok = false;
          payload.response_format_error =
              "Unsupported response_format.type '" + type + "'";
        }
      } else {
        payload.response_format_ok = false;
        payload.response_format_error = "response_format.type must be a string";
      }
    }
    if (j.contains("messages") && j["messages"].is_array()) {
      for (const auto &msg : j["messages"]) {
        ChatMessage m;
        if (msg.contains("role") && msg["role"].is_string()) {
          m.role = msg["role"].get<std::string>();
        }
        if (msg.contains("content")) {
          if (msg["content"].is_string()) {
            m.content = msg["content"].get<std::string>();
          } else if (msg["content"].is_array()) {
            // §2.2: OpenAI multipart content — extract text and image_url
            // parts.
            auto mm =
                ImagePreprocessor::ProcessContentArray(msg["content"].dump());
            m.content = mm.text;
            if (!mm.images.empty()) {
              payload.has_images = true;
              for (auto &img : mm.images) {
                payload.images.push_back(std::move(img));
              }
            }
          }
        }
        if (!m.role.empty() || !m.content.empty()) {
          payload.messages.push_back(std::move(m));
        }
      }
    }
    // §2.3: parse tool definitions.
    if (j.contains("tools") && j["tools"].is_array()) {
      payload.has_tool_schema = true;
      for (const auto &t : j["tools"]) {
        if (!t.is_object())
          continue;
        Tool tool;
        if (t.contains("type") && t["type"].is_string()) {
          tool.type = t["type"].get<std::string>();
        }
        if (t.contains("function") && t["function"].is_object()) {
          const auto &f = t["function"];
          if (f.contains("name") && f["name"].is_string()) {
            tool.function.name = f["name"].get<std::string>();
          }
          if (f.contains("description") && f["description"].is_string()) {
            tool.function.description = f["description"].get<std::string>();
          }
          if (f.contains("parameters")) {
            tool.function.parameters = f["parameters"];
          }
        }
        if (!tool.function.name.empty()) {
          if (payload.first_tool_name.empty()) {
            payload.first_tool_name = tool.function.name;
          }
          payload.tools.push_back(std::move(tool));
        }
      }
    }
    if (j.contains("tool_choice")) {
      if (j["tool_choice"].is_string()) {
        payload.tool_choice = j["tool_choice"].get<std::string>();
      } else if (j["tool_choice"].is_object()) {
        // {"type":"function","function":{"name":"..."}} — treat as "required".
        payload.tool_choice = "required";
      }
    }
  } catch (const json::exception &) {
    // Return defaults on parse failure.
  }
  return payload;
}

std::string FlattenMessages(const std::vector<ChatMessage> &messages) {
  std::string prompt;
  for (const auto &message : messages) {
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

// §2.3: builds a system-level preamble that teaches the model the tool-call
// protocol.
std::string BuildToolSystemPrompt(const std::vector<Tool> &tools) {
  std::string s = "You have access to the following tools:\n\n";
  for (const auto &t : tools) {
    s += "- " + t.function.name;
    if (!t.function.description.empty()) {
      s += ": " + t.function.description;
    }
    s += "\n";
    if (!t.function.parameters.is_null() && !t.function.parameters.empty()) {
      s += "  Parameters: " + t.function.parameters.dump() + "\n";
    }
  }
  s += "\nWhen you want to call a tool, respond with ONLY this JSON (no other "
       "text):\n";
  s +=
      "{\"tool_call\":{\"name\":\"<function_name>\",\"arguments\":{<args>}}}\n";
  return s;
}

// §2.3: multi-format tool call detection.
//
// Recognised output formats (model-native):
//   1. InferFlux preamble format  :
//   {"tool_call":{"name":"...","arguments":{...}}}
//   2. OpenAI-style / Generic JSON: {"name":"...","arguments":{...}}
//   3. Hermes / Llama-3.1 XML     :
//   <tool_call>{"name":"...","arguments":{...}}</tool_call>
//   4. Mistral                    : [TOOL_CALLS]
//   [{"name":"...","arguments":{...}}]
//
// Returns the first detected tool call or an empty result.
ToolCallResult DetectToolCall(const std::string &text) {
  ToolCallResult result;

  // Helper: populate result from a parsed JSON object that has "name" key.
  auto fill_from_obj = [&](const json &tc) -> bool {
    if (!tc.contains("name") || !tc["name"].is_string())
      return false;
    result.function_name = tc["name"].get<std::string>();
    result.call_id = "call_" + result.function_name + "_0";
    const char *args_key =
        tc.contains("arguments")
            ? "arguments"
            : (tc.contains("parameters") ? "parameters" : nullptr);
    if (args_key && tc.contains(args_key)) {
      result.arguments_json = tc[args_key].is_object()
                                  ? tc[args_key].dump()
                                  : tc[args_key].get<std::string>();
    } else {
      result.arguments_json = "{}";
    }
    result.detected = true;
    return true;
  };

  // Format 1: {"tool_call":{...}}  (InferFlux preamble convention)
  auto try_inferflux = [&](const std::string &s) -> bool {
    try {
      auto j = json::parse(s);
      if (j.contains("tool_call") && j["tool_call"].is_object())
        return fill_from_obj(j["tool_call"]);
    } catch (...) {
    }
    return false;
  };
  if (try_inferflux(text))
    return result;
  {
    auto pos = text.find("{\"tool_call\"");
    if (pos != std::string::npos && try_inferflux(text.substr(pos)))
      return result;
  }

  // Format 3: <tool_call>…</tool_call>  (Hermes / Llama-3.1)
  {
    static const std::string open_tag = "<tool_call>";
    static const std::string close_tag = "</tool_call>";
    auto a = text.find(open_tag);
    auto b = text.find(close_tag);
    if (a != std::string::npos && b != std::string::npos && b > a) {
      std::string inner =
          text.substr(a + open_tag.size(), b - a - open_tag.size());
      // Trim whitespace
      auto ws_start = inner.find_first_not_of(" \t\r\n");
      auto ws_end = inner.find_last_not_of(" \t\r\n");
      if (ws_start != std::string::npos)
        inner = inner.substr(ws_start, ws_end - ws_start + 1);
      try {
        auto j = json::parse(inner);
        if (j.is_object() && fill_from_obj(j))
          return result;
      } catch (...) {
      }
    }
  }

  // Format 4: [TOOL_CALLS] [{"name":"...","arguments":{...}}]  (Mistral)
  // Mistral models may append [/TOOL_CALLS] after the JSON array.  Parsing
  // text.substr(bracket) to end-of-string fails because nlohmann::json rejects
  // trailing non-JSON content.  Walk the bracket depth to find the matching ']'
  // and parse only that bounded substring.
  {
    static const std::string mistral_tag = "[TOOL_CALLS]";
    auto pos = text.find(mistral_tag);
    if (pos != std::string::npos) {
      auto bracket = text.find('[', pos + mistral_tag.size());
      if (bracket != std::string::npos) {
        // Find matching closing bracket, tracking nesting depth.
        std::size_t depth = 0;
        std::size_t close = std::string::npos;
        for (std::size_t i = bracket; i < text.size(); ++i) {
          if (text[i] == '[')
            ++depth;
          else if (text[i] == ']') {
            if (--depth == 0) {
              close = i;
              break;
            }
          }
        }
        if (close != std::string::npos) {
          try {
            auto arr = json::parse(text.substr(bracket, close - bracket + 1));
            if (arr.is_array() && !arr.empty() && arr[0].is_object() &&
                fill_from_obj(arr[0]))
              return result;
          } catch (...) {
          }
        }
      }
    }
  }

  // Format 2: bare {"name":"...","arguments":{...}}  (generic OpenAI-style
  // output)
  {
    auto pos = text.find("{\"name\"");
    if (pos != std::string::npos) {
      try {
        auto j = json::parse(text.substr(pos));
        if (j.is_object() && fill_from_obj(j))
          return result;
      } catch (...) {
      }
    }
  }

  return result;
}

std::string
BuildCompletionBody(const InferenceResult &result,
                    const CompletionRequestPayload &request, bool chat_mode,
                    const ToolCallResult &tool_call = ToolCallResult{}) {
  auto now = std::chrono::system_clock::now();
  auto ts =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
          .count();
  std::string id_prefix = chat_mode ? "chatcmpl-" : "cmpl-";

  json usage = {
      {"prompt_tokens", result.prompt_tokens},
      {"completion_tokens", result.completion_tokens},
      {"total_tokens", result.prompt_tokens + result.completion_tokens}};

  json j;
  j["id"] = id_prefix + std::to_string(ts);
  j["object"] = chat_mode ? "chat.completion" : "text_completion";
  j["created"] = ts;
  j["model"] = request.model;
  j["usage"] = usage;

  if (chat_mode) {
    if (tool_call.detected) {
      // §2.3: emit tool_calls array with finish_reason="tool_calls".
      json tc_arr =
          json::array({{{"id", tool_call.call_id},
                        {"type", "function"},
                        {"function",
                         {{"name", tool_call.function_name},
                          {"arguments", tool_call.arguments_json}}}}});
      j["choices"] = json::array({{{"index", 0},
                                   {"message",
                                    {{"role", "assistant"},
                                     {"content", nullptr},
                                     {"tool_calls", tc_arr}}},
                                   {"finish_reason", "tool_calls"}}});
    } else {
      j["choices"] = json::array(
          {{{"index", 0},
            {"message",
             {{"role", "assistant"}, {"content", result.completion}}},
            {"finish_reason", "stop"}}});
    }
  } else {
    j["choices"] = json::array({{{"index", 0},
                                 {"text", result.completion},
                                 {"finish_reason", "stop"}}});
  }
  return j.dump();
}

std::string BuildErrorBody(const std::string &error) {
  return json({{"error", error}}).dump();
}

std::vector<std::string> SplitForStreaming(const std::string &text) {
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

std::string BuildStreamChunk(const std::string &id, const std::string &model,
                             std::time_t ts, const std::string &content,
                             bool finish) {
  json j;
  j["id"] = id;
  j["object"] = "chat.completion.chunk";
  j["created"] = ts;
  j["model"] = model;

  if (finish) {
    j["choices"] = json::array(
        {{{"index", 0}, {"delta", json::object()}, {"finish_reason", "stop"}}});
  } else {
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", {{"content", content}}},
                                 {"finish_reason", nullptr}}});
  }
  return "data: " + j.dump() + "\n\n";
}

// §2.3: emit SSE delta sequence for a streaming tool call response.
// Sequence per OpenAI spec:
//   1. role=assistant, content=null
//   2. tool_calls[0] with id + type + function.name + empty arguments
//   3. tool_calls[0] function.arguments (full JSON string)
//   4. finish_reason=tool_calls, empty delta
std::string BuildToolCallStreamChunks(const std::string &id,
                                      const std::string &model, std::time_t ts,
                                      const ToolCallResult &tc) {
  std::string out;
  auto base = [&]() -> json {
    json j;
    j["id"] = id;
    j["object"] = "chat.completion.chunk";
    j["created"] = ts;
    j["model"] = model;
    return j;
  };

  // Chunk 1: role=assistant, content=null
  {
    json j = base();
    j["choices"] =
        json::array({{{"index", 0},
                      {"delta", {{"role", "assistant"}, {"content", nullptr}}},
                      {"finish_reason", nullptr}}});
    out += "data: " + j.dump() + "\n\n";
  }

  // Chunk 2: tool_calls[0] — id, type, function name, empty arguments
  {
    json j = base();
    json tc_delta = json::array(
        {{{"index", 0},
          {"id", tc.call_id},
          {"type", "function"},
          {"function", {{"name", tc.function_name}, {"arguments", ""}}}}});
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", {{"tool_calls", tc_delta}}},
                                 {"finish_reason", nullptr}}});
    out += "data: " + j.dump() + "\n\n";
  }

  // Chunk 3: function arguments
  if (!tc.arguments_json.empty()) {
    json j = base();
    json arg_delta = json::array(
        {{{"index", 0}, {"function", {{"arguments", tc.arguments_json}}}}});
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", {{"tool_calls", arg_delta}}},
                                 {"finish_reason", nullptr}}});
    out += "data: " + j.dump() + "\n\n";
  }

  // Chunk 4: finish_reason=tool_calls
  {
    json j = base();
    j["choices"] = json::array({{{"index", 0},
                                 {"delta", json::object()},
                                 {"finish_reason", "tool_calls"}}});
    out += "data: " + j.dump() + "\n\n";
  }

  return out;
}

std::string BuildApiKeysPayload(const std::vector<PolicyKeyEntry> &keys) {
  json arr = json::array();
  for (const auto &k : keys) {
    arr.push_back({{"key", k.key}, {"scopes", k.scopes}});
  }
  return json({{"api_keys", arr}}).dump();
}
} // namespace

HttpServer::HttpServer(std::string host, int port, Scheduler *scheduler,
                       std::shared_ptr<ApiKeyAuth> auth,
                       MetricsRegistry *metrics, OIDCValidator *oidc,
                       RateLimiter *rate_limiter, Guardrail *guardrail,
                       AuditLogger *audit_logger, PolicyBackend *policy_store,
                       std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                       TlsConfig tls_config, int num_workers)
    : host_(std::move(host)), port_(port), scheduler_(scheduler),
      auth_(std::move(auth)), metrics_(metrics), oidc_(oidc),
      rate_limiter_(rate_limiter), guardrail_(guardrail),
      audit_logger_(audit_logger), policy_store_(policy_store),
      speculative_decoder_(std::move(speculative_decoder)),
      num_workers_(num_workers > 0 ? num_workers : 4) {
  if (tls_config.enabled) {
    if (tls_config.cert_path.empty() || tls_config.key_path.empty()) {
      std::cerr << "[http] TLS enabled without cert/key; falling back to HTTP"
                << std::endl;
    } else {
      SSL_load_error_strings();
      OpenSSL_add_ssl_algorithms();
      ssl_ctx_ = SSL_CTX_new(TLS_server_method());
      if (!ssl_ctx_) {
        std::cerr << "[http] Failed to initialize TLS context" << std::endl;
      } else {
        SSL_CTX_set_ecdh_auto(ssl_ctx_, 1);
        if (SSL_CTX_use_certificate_file(ssl_ctx_, tls_config.cert_path.c_str(),
                                         SSL_FILETYPE_PEM) <= 0) {
          std::cerr << "[http] Failed to load TLS certificate: "
                    << tls_config.cert_path << std::endl;
          SSL_CTX_free(ssl_ctx_);
          ssl_ctx_ = nullptr;
        } else if (SSL_CTX_use_PrivateKey_file(ssl_ctx_,
                                               tls_config.key_path.c_str(),
                                               SSL_FILETYPE_PEM) <= 0) {
          std::cerr << "[http] Failed to load TLS key: " << tls_config.key_path
                    << std::endl;
          SSL_CTX_free(ssl_ctx_);
          ssl_ctx_ = nullptr;
        } else {
          tls_enabled_ = true;
          std::cout << "[http] TLS enabled using cert=" << tls_config.cert_path
                    << std::endl;
        }
      }
    }
  }
}

HttpServer::~HttpServer() {
  Stop();
  if (ssl_ctx_) {
    SSL_CTX_free(ssl_ctx_);
    ssl_ctx_ = nullptr;
  }
}

void HttpServer::Start() {
  if (running_) {
    return;
  }
  running_ = true;
  // Start worker threads.
  for (int i = 0; i < num_workers_; ++i) {
    workers_.emplace_back(&HttpServer::WorkerLoop, this);
  }
  accept_thread_ = std::thread(&HttpServer::Run, this);
}

void HttpServer::Stop() {
  if (!running_) {
    return;
  }
  running_ = false;
  // Close the listening socket to unblock the accept() call in Run().
  int fd = server_fd_.exchange(-1);
  if (fd >= 0) {
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }
  // Wake all worker threads.
  queue_cv_.notify_all();
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
  for (auto &w : workers_) {
    if (w.joinable()) {
      w.join();
    }
  }
  workers_.clear();
  // Drain any remaining clients in the queue.
  std::lock_guard<std::mutex> lock(queue_mutex_);
  while (!client_queue_.empty()) {
    auto session = std::move(client_queue_.front());
    client_queue_.pop();
    CloseSession(session);
  }
}

void HttpServer::WorkerLoop() {
  while (true) {
    ClientSession session;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock,
                     [this] { return !client_queue_.empty() || !running_; });
      if (!running_ && client_queue_.empty()) {
        return;
      }
      session = std::move(client_queue_.front());
      client_queue_.pop();
    }
    if (session.fd >= 0) {
      HandleClient(session);
      CloseSession(session);
    }
  }
}

void HttpServer::Run() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    std::perror("socket");
    return;
  }

  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  addr.sin_addr.s_addr = inet_addr(host_.c_str());

  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    std::perror("bind");
    ::close(fd);
    return;
  }

  if (::listen(fd, 128) < 0) {
    std::perror("listen");
    ::close(fd);
    return;
  }

  server_fd_.store(fd);

  while (running_) {
    sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd =
        ::accept(fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
    if (client_fd < 0) {
      break; // Socket closed by Stop() or error — exit loop.
    }
    if (!running_) {
      ::close(client_fd);
      break;
    }
    ClientSession session;
    session.fd = client_fd;
    if (tls_enabled_) {
      SSL *ssl = SSL_new(ssl_ctx_);
      if (!ssl) {
        ::close(client_fd);
        continue;
      }
      SSL_set_fd(ssl, client_fd);
      if (SSL_accept(ssl) != 1) {
        SSL_free(ssl);
        ::close(client_fd);
        continue;
      }
      session.ssl = ssl;
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      client_queue_.push(std::move(session));
    }
    queue_cv_.notify_one();
  }

  // If Stop() hasn't already closed the socket, close it now.
  int expected = fd;
  if (server_fd_.compare_exchange_strong(expected, -1)) {
    ::close(fd);
  }
}

bool HttpServer::ResolveSubject(const std::string &headers,
                                AuthContext *ctx) const {
  ctx->subject = "anonymous";
  ctx->scopes.clear();
  bool require_auth =
      (auth_ && auth_->HasKeys()) || (oidc_ && oidc_->Enabled());
  if (!require_auth) {
    ctx->scopes.insert("read");
    ctx->scopes.insert("generate");
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
    ctx->subject = token;
    auto scopes = auth_->Scopes(token);
    ctx->scopes.insert(scopes.begin(), scopes.end());
    return true;
  }
  if (oidc_ && oidc_->Enabled()) {
    std::string sub;
    if (oidc_->Validate(token, &sub)) {
      ctx->subject = sub.empty() ? "oidc-user" : sub;
      ctx->scopes.insert("generate");
      ctx->scopes.insert("read");
      return true;
    }
  }
  return false;
}

bool HttpServer::RequireScope(const AuthContext &ctx, const std::string &scope,
                              ClientSession &session,
                              const std::string &error_message) {
  if (ctx.scopes.find(scope) != ctx.scopes.end()) {
    return true;
  }
  auto payload =
      BuildResponse(BuildErrorBody(error_message.empty() ? "insufficient_scope"
                                                         : error_message),
                    403, "Forbidden");
  SendAll(session, payload);
  if (audit_logger_) {
    audit_logger_->Log(ctx.subject, "", "insufficient_scope", error_message);
  }
  return false;
}

void HttpServer::HandleClient(ClientSession &session) {
  // RAII guard: decrement connections and record latency on all exit paths.
  auto req_start = std::chrono::steady_clock::now();
  struct ConnectionGuard {
    MetricsRegistry *metrics;
    std::chrono::steady_clock::time_point start;
    ~ConnectionGuard() {
      if (metrics) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        double ms = std::chrono::duration<double, std::milli>(elapsed).count();
        metrics->RecordLatency(ms);
        metrics->DecrementConnections();
      }
    }
  } guard{metrics_, req_start};
  if (metrics_) {
    metrics_->IncrementConnections();
  }

  // Dynamically read the full HTTP request: headers + body based on
  // Content-Length.
  constexpr std::size_t kInitialBuf = 4096;
  constexpr std::size_t kMaxRequest = 16 * 1024 * 1024; // 16 MB hard limit
  std::string request;
  request.resize(kInitialBuf);
  std::size_t total = 0;
  std::size_t header_end_pos = std::string::npos;

  // Phase 1: read until we find the end-of-headers marker.
  while (header_end_pos == std::string::npos) {
    if (total >= request.size()) {
      if (request.size() >= kMaxRequest) {
        auto response = BuildResponse(BuildErrorBody("request_too_large"), 413,
                                      "Payload Too Large");
        SendAll(session, response);
        return;
      }
      request.resize(std::min(request.size() * 2, kMaxRequest));
    }
    ssize_t bytes = Receive(session, &request[total], request.size() - total);
    if (bytes <= 0) {
      return;
    }
    total += static_cast<std::size_t>(bytes);
    request.resize(total); // Shrink to actual for find()
    header_end_pos = request.find("\r\n\r\n");
    if (header_end_pos == std::string::npos) {
      // Grow for next iteration.
      request.resize(std::max(total + kInitialBuf, total * 2));
    }
  }

  // Phase 2: parse Content-Length and read remaining body bytes.
  std::size_t body_start = header_end_pos + 4;
  std::size_t content_length = 0;
  {
    auto cl_pos = request.find("Content-Length:");
    if (cl_pos == std::string::npos) {
      cl_pos = request.find("content-length:");
    }
    if (cl_pos != std::string::npos && cl_pos < header_end_pos) {
      auto val_start = cl_pos + 15; // strlen("Content-Length:")
      while (val_start < header_end_pos && request[val_start] == ' ') {
        ++val_start;
      }
      auto val_end = request.find("\r\n", val_start);
      if (val_end != std::string::npos) {
        try {
          content_length =
              std::stoull(request.substr(val_start, val_end - val_start));
        } catch (...) {
        }
      }
    }
  }

  if (content_length > kMaxRequest) {
    auto response = BuildResponse(BuildErrorBody("request_too_large"), 413,
                                  "Payload Too Large");
    SendAll(session, response);
    return;
  }

  std::size_t needed = body_start + content_length;
  if (needed > total) {
    request.resize(needed);
    while (total < needed) {
      ssize_t bytes = Receive(session, &request[total], needed - total);
      if (bytes <= 0) {
        return;
      }
      total += static_cast<std::size_t>(bytes);
    }
  }
  request.resize(total);

  auto header_end = header_end_pos;
  std::string headers =
      header_end == std::string::npos ? request : request.substr(0, header_end);
  std::string body = header_end == std::string::npos
                         ? std::string()
                         : request.substr(header_end + 4);
  auto first_line_end = headers.find("\r\n");
  std::string first_line = headers.substr(0, first_line_end);
  auto method_end = first_line.find(' ');
  auto path_end = first_line.find(' ', method_end + 1);
  std::string method = first_line.substr(0, method_end);
  std::string path =
      first_line.substr(method_end + 1, path_end - method_end - 1);

  // Unauthenticated health/readiness probes.
  if (method == "GET" && path == "/healthz") {
    bool ready = model_ready_.load();
    json j = {{"status", ready ? "ok" : "degraded"}, {"model_ready", ready}};
    SendAll(session, BuildResponse(j.dump(), ready ? 200 : 503,
                                   ready ? "OK" : "Service Unavailable"));
    return;
  }
  if (method == "GET" && path == "/livez") {
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    return;
  }
  if (method == "GET" && path == "/readyz") {
    PoolRole role = role_.load(std::memory_order_relaxed);
    bool ready = false;
    std::string reason;
    if (role == PoolRole::kDecode) {
      // Decode-only node: ready when a model backend is loaded AND the decode
      // worker pool is warm.  Checking only decode_pool_ready_ was wrong:
      // that flag is set at startup from pool size, so a pod would report 200
      // before weights are resident and before it can actually serve tokens.
      bool model_loaded = false;
      if (scheduler_ && scheduler_->Router()) {
        model_loaded = !scheduler_->Router()->DefaultModelId().empty();
      } else {
        model_loaded = model_ready_.load(std::memory_order_relaxed);
      }
      // Require ALL configured decode workers to be alive, not just at least
      // one.  With > 0 a 4-worker pool remains "ready" if 3 crash.
      // live == configured means every thread is in its run-loop; the RAII
      // guard in DecodeWorkerLoop decrements on any exit path so a single
      // crash immediately makes this false.
      bool pool_warm = scheduler_ &&
                       scheduler_->ConfiguredDecodeWorkers() > 0 &&
                       scheduler_->LiveDecodeWorkers() ==
                           scheduler_->ConfiguredDecodeWorkers();
      ready = model_loaded && pool_warm;
      if (!ready)
        reason =
            !model_loaded ? "no model backend loaded" : "decode pool not ready";
    } else {
      // Unified or prefill node: ready when at least one model backend is
      // loaded.
      if (scheduler_ && scheduler_->Router()) {
        ready = !scheduler_->Router()->DefaultModelId().empty();
      } else {
        ready = model_ready_.load();
      }
      if (!ready)
        reason = "no model backend loaded";
    }
    std::string role_str = (role == PoolRole::kPrefill)  ? "prefill"
                           : (role == PoolRole::kDecode) ? "decode"
                                                         : "unified";
    json body = {{"status", ready ? "ready" : "not_ready"}, {"role", role_str}};
    if (!ready)
      body["reason"] = reason;
    int status_code = ready ? 200 : 503;
    std::string status_text = ready ? "OK" : "Service Unavailable";
    SendAll(session, BuildResponse(body.dump(), status_code, status_text));
    return;
  }

  // CORS preflight.
  if (method == "OPTIONS") {
    std::string cors_headers =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Content-Length: 0\r\n\r\n";
    SendAll(session, cors_headers);
    return;
  }

  AuthContext auth_ctx;
  if (!ResolveSubject(headers, &auth_ctx)) {
    auto response =
        BuildResponse(BuildErrorBody("unauthorized"), 401, "Unauthorized");
    SendAll(session, response);
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "unauthorized",
                         "missing or invalid credentials");
    }
    return;
  }
  if (rate_limiter_ && rate_limiter_->Enabled() &&
      !rate_limiter_->Allow(auth_ctx.subject)) {
    auto response =
        BuildResponse(BuildErrorBody("rate_limited"), 429, "Too Many Requests");
    SendAll(session, response);
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "rate_limited",
                         "token bucket exceeded");
    }
    return;
  }

  if (method == "GET" && path == "/metrics") {
    if (!RequireScope(auth_ctx, "read", session, "metrics scope required")) {
      return;
    }
    std::string metrics_body = metrics_ ? metrics_->RenderPrometheus() : "";
    std::string metrics_headers =
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; "
        "version=0.0.4\r\nContent-Length: " +
        std::to_string(metrics_body.size()) + "\r\n\r\n";
    SendAll(session, metrics_headers + metrics_body);
    return;
  }

  if (method == "GET" && path == "/v1/admin/guardrails") {
    if (!guardrail_) {
      SendAll(session, BuildResponse(BuildErrorBody("guardrail_disabled"), 503,
                                     "Unavailable"));
      return;
    }
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto list = guardrail_->Blocklist();
    SendAll(session, BuildResponse(json({{"blocklist", list}}).dump()));
    return;
  }

  if (method == "PUT" && path == "/v1/admin/guardrails") {
    if (!guardrail_) {
      SendAll(session, BuildResponse(BuildErrorBody("guardrail_disabled"), 503,
                                     "Unavailable"));
      return;
    }
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    std::vector<std::string> list;
    try {
      auto j = json::parse(body);
      if (j.contains("blocklist") && j["blocklist"].is_array()) {
        list = j["blocklist"].get<std::vector<std::string>>();
      }
    } catch (const json::exception &) {
    }
    auto start = std::chrono::steady_clock::now();
    guardrail_->UpdateBlocklist(list);
    if (policy_store_) {
      policy_store_->SetGuardrailBlocklist(list);
      policy_store_->Save();
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] guardrail update applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "guardrail_update",
                         "updated blocklist");
    }
    return;
  }

  if (method == "GET" && path == "/v1/admin/rate_limit") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    int limit = rate_limiter_ ? rate_limiter_->CurrentLimit() : 0;
    SendAll(session,
            BuildResponse(json({{"tokens_per_minute", limit}}).dump()));
    return;
  }

  if (method == "PUT" && path == "/v1/admin/rate_limit") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    int value = 0;
    bool valid = false;
    try {
      auto j = json::parse(body);
      if (j.contains("tokens_per_minute") &&
          j["tokens_per_minute"].is_number_integer()) {
        value = j["tokens_per_minute"].get<int>();
        valid = true;
      }
    } catch (const json::exception &) {
    }
    if (!valid) {
      SendAll(session,
              BuildResponse(BuildErrorBody("tokens_per_minute is required"),
                            400, "Bad Request"));
      return;
    }
    auto start = std::chrono::steady_clock::now();
    if (rate_limiter_) {
      rate_limiter_->UpdateLimit(value);
    }
    if (policy_store_) {
      policy_store_->SetRateLimitPerMinute(value);
      policy_store_->Save();
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] rate-limit update applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "rate_limit_update",
                         std::to_string(value));
    }
    return;
  }

  if (method == "GET" && path == "/v1/admin/api_keys") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    if (!policy_store_) {
      SendAll(session, BuildResponse(BuildErrorBody("policy_store_disabled"),
                                     503, "Unavailable"));
      return;
    }
    auto keys = policy_store_->ApiKeys();
    SendAll(session, BuildResponse(BuildApiKeysPayload(keys)));
    return;
  }

  if (method == "POST" && path == "/v1/admin/api_keys") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    if (!policy_store_) {
      SendAll(session, BuildResponse(BuildErrorBody("policy_store_disabled"),
                                     503, "Unavailable"));
      return;
    }
    std::string key;
    std::vector<std::string> scopes;
    try {
      auto j = json::parse(body);
      if (j.contains("key") && j["key"].is_string()) {
        key = j["key"].get<std::string>();
      }
      if (j.contains("scopes") && j["scopes"].is_array()) {
        scopes = j["scopes"].get<std::vector<std::string>>();
      }
    } catch (const json::exception &) {
    }
    if (key.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("key is required"), 400,
                                     "Bad Request"));
      return;
    }
    auto start = std::chrono::steady_clock::now();
    auth_->AddKey(key, scopes);
    policy_store_->SetApiKey(key, scopes);
    policy_store_->Save();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] api-key upsert applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "api_key_upsert", key);
    }
    return;
  }

  if (method == "DELETE" && path == "/v1/admin/api_keys") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    if (!policy_store_) {
      SendAll(session, BuildResponse(BuildErrorBody("policy_store_disabled"),
                                     503, "Unavailable"));
      return;
    }
    std::string key;
    try {
      auto j = json::parse(body);
      if (j.contains("key") && j["key"].is_string()) {
        key = j["key"].get<std::string>();
      }
    } catch (const json::exception &) {
    }
    if (key.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("key is required"), 400,
                                     "Bad Request"));
      return;
    }
    auto start = std::chrono::steady_clock::now();
    auth_->RemoveKey(key);
    policy_store_->RemoveApiKey(key);
    policy_store_->Save();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "[policy] api-key removal applied in " << elapsed.count()
              << " ms" << std::endl;
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, "", "api_key_remove", key);
    }
    return;
  }

  if (method == "GET" && path == "/v1/admin/models") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    auto models = router->ListModels();
    std::string default_id = router->DefaultModelId();
    json payload;
    payload["default_model"] = default_id;
    payload["models"] = json::array();
    for (const auto &info : models) {
      payload["models"].push_back({{"id", info.id},
                                   {"path", info.path},
                                   {"backend", info.backend},
                                   {"ready", info.ready},
                                   {"default", info.id == default_id}});
    }
    SendAll(session, BuildResponse(payload.dump()));
    return;
  }

  if (method == "POST" && path == "/v1/admin/models") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    std::string path_value;
    std::string backend_hint;
    std::string requested_id;
    bool set_default = false;
    try {
      auto j = json::parse(body);
      if (j.contains("path") && j["path"].is_string()) {
        path_value = j["path"].get<std::string>();
      }
      if (j.contains("backend") && j["backend"].is_string()) {
        backend_hint = j["backend"].get<std::string>();
      }
      if (j.contains("id") && j["id"].is_string()) {
        requested_id = j["id"].get<std::string>();
      }
      if (j.contains("default")) {
        if (j["default"].is_boolean()) {
          set_default = j["default"].get<bool>();
        } else if (j["default"].is_string()) {
          std::string val = j["default"].get<std::string>();
          for (auto &ch : val) {
            ch =
                static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
          }
          set_default = (val == "true" || val == "1" || val == "yes");
        }
      }
    } catch (const json::exception &) {
    }
    if (path_value.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("path is required"), 400,
                                     "Bad Request"));
      return;
    }
    auto id = router->LoadModel(path_value, backend_hint, requested_id);
    if (id.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("load_failed"), 500,
                                     "Internal Server Error"));
      return;
    }
    if (set_default) {
      router->SetDefaultModel(id);
    }
    SendAll(session,
            BuildResponse(json({{"status", "ok"}, {"id", id}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, id, "model_load", path_value);
    }
    return;
  }

  if (method == "DELETE" && path == "/v1/admin/models") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    std::string id;
    try {
      auto j = json::parse(body);
      if (j.contains("id") && j["id"].is_string()) {
        id = j["id"].get<std::string>();
      }
    } catch (const json::exception &) {
    }
    if (id.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("id is required"), 400,
                                     "Bad Request"));
      return;
    }
    if (!router->UnloadModel(id)) {
      SendAll(session, BuildResponse(BuildErrorBody("model_not_found"), 404,
                                     "Not Found"));
      return;
    }
    SendAll(session, BuildResponse(json({{"status", "ok"}}).dump()));
    if (audit_logger_) {
      audit_logger_->Log(auth_ctx.subject, id, "model_unload",
                         "Removed via admin API");
    }
    return;
  }

  if (method == "PUT" && path == "/v1/admin/models/default") {
    if (!RequireScope(auth_ctx, "admin", session, "admin scope required")) {
      return;
    }
    auto *router = scheduler_ ? scheduler_->Router() : nullptr;
    if (!router) {
      SendAll(session, BuildResponse(BuildErrorBody("router_unavailable"), 503,
                                     "Service Unavailable"));
      return;
    }
    std::string id;
    try {
      auto j = json::parse(body);
      if (j.contains("id") && j["id"].is_string()) {
        id = j["id"].get<std::string>();
      }
    } catch (const json::exception &) {
    }
    if (id.empty()) {
      SendAll(session, BuildResponse(BuildErrorBody("id is required"), 400,
                                     "Bad Request"));
      return;
    }
    if (!router->SetDefaultModel(id)) {
      SendAll(session, BuildResponse(BuildErrorBody("model_not_found"), 404,
                                     "Not Found"));
      return;
    }
    SendAll(
        session,
        BuildResponse(json({{"status", "ok"}, {"default_model", id}}).dump()));
    return;
  }

  if (method == "POST" &&
      (path == "/v1/completions" || path == "/v1/chat/completions")) {
    if (!RequireScope(auth_ctx, "generate", session,
                      "generate scope required")) {
      return;
    }
    auto parsed = ParseJsonPayload(body);
    auto LogToolEvent = [](const std::string &line) {
      if (const char *path = std::getenv("INFERFLUX_LOG_TOOL_CALLS")) {
        std::ofstream out(path, std::ios::app);
        if (out) {
          out << line << std::endl;
        }
      }
    };
    if (!parsed.response_format_ok) {
      auto payload = BuildResponse(BuildErrorBody(parsed.response_format_error),
                                   400, "Bad Request");
      SendAll(session, payload);
      return;
    }

    InferenceRequest req;
    // Note: req.prompt is set below after the tool/template block which
    // handles both the messages chat path and the direct prompt path.
    if (parsed.max_tokens > 0) {
      req.max_tokens = parsed.max_tokens;
    }
    req.model = parsed.model;
    req.json_mode = parsed.json_mode;
    if (parsed.has_response_format) {
      req.has_response_format = true;
      req.response_format_type = parsed.response_format_type;
      req.response_format_schema = parsed.response_format_schema;
      req.response_format_grammar = parsed.response_format_grammar;
      req.response_format_root = parsed.response_format_root;
    }
    req.stream = parsed.stream;

    // §2.2: attach decoded images (populated when messages contain image_url
    // parts).
    if (parsed.has_images) {
      req.has_images = true;
      req.images = std::move(parsed.images);
      GlobalMetrics().RecordImagePreprocess(static_cast<int>(req.images.size()),
                                            0.0);
    }

    // W3C Trace Context (OBS-2): propagate trace-id from incoming traceparent
    // header.
    {
      std::string tp_header = GetHeaderValue(headers, "traceparent");
      if (!tp_header.empty()) {
        auto parent_ctx = tracing::ParseTraceparent(tp_header);
        req.trace_id = parent_ctx.trace_id;
      }
    }

    // §2.3: tool schema injection + model-native chat template formatting.
    bool use_tools = (parsed.has_tool_schema || !parsed.tools.empty()) &&
                     parsed.tool_choice != "none";
    if (use_tools && std::getenv("INFERFLUX_LOG_TOOL_CALLS")) {
      std::cout << "[tools] request provided " << parsed.tools.size()
                << " tool definition(s); choice=" << parsed.tool_choice
                << std::endl;
    }
    if (use_tools) {
      LogToolEvent("request tools=" + std::to_string(parsed.tools.size()) +
                   " choice=" + parsed.tool_choice);
    }

    // §2.3 model-native chat template path.
    // When a real model is loaded and the request came as a chat messages
    // array, format the conversation using the model's built-in template
    // (llama_chat_apply_template).  Tool definitions are injected as the first
    // system message so the model-specific role tokens wrap them correctly.
    // When no model is available (stub mode) or the template is unsupported, we
    // fall back to FlattenMessages + BuildToolSystemPrompt (existing
    // behaviour).
    bool use_native_template = false;
    if (parsed.prompt.empty() && !parsed.messages.empty()) {
      auto *router = scheduler_->Router();
      if (router) {
        auto *info = router->Resolve(parsed.model);
        if (info) {
          auto backend = router->GetBackend(info->id);
          if (backend && backend->IsReady()) {
            // Build message list: prepend tool schema as system message if
            // needed.
            std::vector<std::pair<std::string, std::string>> msgs;
            if (use_tools && !parsed.tools.empty()) {
              msgs.push_back({"system", BuildToolSystemPrompt(parsed.tools)});
            }
            for (const auto &m : parsed.messages) {
              if (!m.role.empty() || !m.content.empty()) {
                msgs.push_back({m.role, m.content});
              }
            }
            auto tmpl = backend->FormatChatMessages(
                msgs, /*add_assistant_prefix=*/true);
            if (tmpl.valid) {
              req.prompt = tmpl.prompt;
              use_native_template = true;
              LogToolEvent("native_template=true msgs=" +
                           std::to_string(msgs.size()));
            }
          }
        }
      }
    }

    if (!use_native_template) {
      // Fallback: flat concatenation + preamble injection (stub / unsupported
      // template models).
      if (!parsed.prompt.empty()) {
        req.prompt = parsed.prompt;
      } else if (!parsed.messages.empty()) {
        req.prompt = FlattenMessages(parsed.messages);
      }
      if (use_tools && !parsed.tools.empty()) {
        std::string tool_prefix = BuildToolSystemPrompt(parsed.tools);
        req.prompt =
            req.prompt.empty() ? tool_prefix : tool_prefix + "\n" + req.prompt;
      }
    }

    req.priority = static_cast<int>(auth_ctx.scopes.count("admin") ? 10 : 0);
    if (req.prompt.empty()) {
      auto payload =
          BuildResponse(BuildErrorBody("prompt or messages are required"), 400,
                        "Bad Request");
      SendAll(session, payload);
      return;
    }
    bool chat_mode =
        (path == "/v1/chat/completions") || !parsed.messages.empty();
    std::string guard_reason;
    if (guardrail_ && guardrail_->Enabled() &&
        !guardrail_->Check(req.prompt, &guard_reason)) {
      auto payload =
          BuildResponse(BuildErrorBody(guard_reason), 400, "Bad Request");
      SendAll(session, payload);
      if (audit_logger_) {
        audit_logger_->Log(auth_ctx.subject, parsed.model, "blocked",
                           guard_reason);
      }
      return;
    }
    // Build a child SpanContext for this request so downstream services can
    // correlate spans. The traceparent is emitted in the response header.
    SpanContext parent_ctx;
    parent_ctx.trace_id = req.trace_id;
    SpanContext request_ctx = tracing::ChildContext(parent_ctx);
    std::string trace_response_header;
    if (request_ctx.valid()) {
      trace_response_header =
          "traceparent: " + request_ctx.ToTraceparent() + "\r\n";
    }

    std::string stream_id;
    std::time_t stream_ts = 0;
    auto stream_mutex = std::make_shared<std::mutex>();
    auto stream_active = std::make_shared<std::atomic<bool>>(false);
    auto stream_had_chunk = std::make_shared<std::atomic<bool>>(false);
    auto stream_cancel_flag = std::make_shared<std::atomic<bool>>(false);
    // Declared here (outer scope) so they're visible in both the streaming
    // setup block and the post-Generate streaming completion block.
    auto token_buffer = std::make_shared<std::vector<std::string>>();
    bool buffer_tokens = use_tools;
    if (parsed.stream) {
      stream_ts = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
      stream_active->store(true);
      stream_id = std::string("chatcmpl-") + std::to_string(stream_ts);
      std::string stream_headers = "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: text/event-stream\r\n"
                                   "Cache-Control: no-cache\r\n"
                                   "Connection: keep-alive\r\n" +
                                   trace_response_header + "\r\n";
      if (!SendAll(session, stream_headers)) {
        return;
      }
      req.cancellation_flag = stream_cancel_flag;
      ClientSession *stream_session = &session;
      std::string stream_model = parsed.model;
      // §2.3: when tools are active we cannot stream tokens as content deltas
      // because we don't know until the full completion arrives whether the
      // model produced a tool_call JSON envelope or plain text.  Buffer all
      // tokens during generation; after Generate() returns either:
      //   a) tool_call detected  → discard buffer, emit tool_calls delta
      //   sequence b) no tool_call        → replay buffer as content deltas,
      //   then stop chunk
      // When use_tools=false the buffer is never populated and the normal
      // per-token streaming path runs unchanged.
      req.on_token = [this, stream_session, stream_mutex, stream_active,
                      stream_had_chunk, stream_cancel_flag, stream_id,
                      stream_model, stream_ts, token_buffer,
                      buffer_tokens](const std::string &chunk) {
        if (chunk.empty() || !stream_active->load()) {
          return;
        }
        if (buffer_tokens) {
          // Accumulate without sending; will be replayed or discarded below.
          std::lock_guard<std::mutex> lock(*stream_mutex);
          token_buffer->push_back(chunk);
          return;
        }
        auto pieces = SplitForStreaming(chunk);
        for (const auto &piece : pieces) {
          std::string payload = BuildStreamChunk(stream_id, stream_model,
                                                 stream_ts, piece, false);
          std::lock_guard<std::mutex> lock(*stream_mutex);
          if (!stream_active->load()) {
            return;
          }
          if (!SendAll(*stream_session, payload)) {
            stream_active->store(false);
            stream_cancel_flag->store(true);
            return;
          }
          stream_had_chunk->store(true);
        }
      };
    }

    try {
      auto result = scheduler_->Generate(std::move(req));
      if (result.no_backend) {
        if (metrics_) {
          metrics_->RecordError();
        }
        if (parsed.stream) {
          // SSE headers were already sent. Complete the stream body rather than
          // sending a second HTTP response (which would corrupt the framing).
          ToolCallResult nb_tc;
          if (use_tools &&
              (!parsed.tools.empty() || !parsed.first_tool_name.empty())) {
            std::string fname = !parsed.tools.empty()
                                    ? parsed.tools.front().function.name
                                    : parsed.first_tool_name;
            if (fname.empty())
              fname = "stub_tool";
            nb_tc.detected = true;
            nb_tc.function_name = fname;
            nb_tc.call_id = "call_stub_" + fname;
            nb_tc.arguments_json =
                json{{"reason", "no_model_available"}}.dump();
          }
          {
            std::lock_guard<std::mutex> lock(*stream_mutex);
            if (nb_tc.detected) {
              SendAll(session, BuildToolCallStreamChunks(
                                   stream_id, parsed.model, stream_ts, nb_tc));
            } else {
              SendAll(session,
                      BuildStreamChunk(stream_id, parsed.model, stream_ts,
                                       result.completion, false));
              SendAll(session, BuildStreamChunk(stream_id, parsed.model,
                                                stream_ts, "", true));
            }
            SendAll(session, "data: [DONE]\n\n");
          }
          stream_active->store(false);
        } else {
          auto payload =
              BuildResponse(BuildCompletionBody(result, parsed, chat_mode), 200,
                            "OK", trace_response_header);
          SendAll(session, payload);
        }
        if (audit_logger_) {
          audit_logger_->Log(auth_ctx.subject, parsed.model, "no_backend",
                             result.completion);
        }
        return;
      }
      if (metrics_) {
        metrics_->RecordSuccess(result.prompt_tokens, result.completion_tokens);
        metrics_->RecordSpeculative(result.speculative.total_chunks,
                                    result.speculative.accepted_chunks,
                                    result.speculative.reused_tokens);
      }
      if (audit_logger_) {
        audit_logger_->LogRequest(auth_ctx.subject, parsed.model, req.prompt,
                                  result.completion, result.prompt_tokens,
                                  result.completion_tokens);
      }
      // §2.3: detect tool call in model output (used by non-streaming
      // responses).
      ToolCallResult tool_call;
      if (use_tools) {
        bool stub_completion =
            result.no_backend ||
            result.completion.find("No model backend is loaded") !=
                std::string::npos;
        LogToolEvent("tool_state no_backend=" +
                     std::to_string(result.no_backend) + " contains_stub=" +
                     std::to_string(
                         result.completion.find("No model backend is loaded") !=
                         std::string::npos));
        if (stub_completion &&
            (!parsed.tools.empty() || !parsed.first_tool_name.empty())) {
          std::string fallback_name = !parsed.tools.empty()
                                          ? parsed.tools.front().function.name
                                          : parsed.first_tool_name;
          if (fallback_name.empty()) {
            fallback_name = "stub_tool";
          }
          json arguments = {
              {"reason", "no_model_available"},
              {"hint", "set INFERFLUX_MODEL_PATH or configure models[]"}};
          tool_call.detected = true;
          tool_call.function_name = fallback_name;
          tool_call.call_id = "call_stub_" + fallback_name;
          tool_call.arguments_json = arguments.dump();
          std::string log_line = "[tools] stub tool call for " + fallback_name;
          LogToolEvent(log_line);
          std::cout << log_line << std::endl;
          if (audit_logger_) {
            audit_logger_->Log(auth_ctx.subject, parsed.model, "tool_call_stub",
                               arguments.dump());
          }
        } else {
          tool_call = DetectToolCall(result.completion);
        }
      }
      if (parsed.stream) {
        {
          std::lock_guard<std::mutex> lock(*stream_mutex);
          if (stream_active->load()) {
            if (tool_call.detected) {
              // §2.3: emit structured tool_calls delta sequence (role → name →
              // args → finish).
              SendAll(session,
                      BuildToolCallStreamChunks(stream_id, parsed.model,
                                                stream_ts, tool_call));
            } else if (buffer_tokens && !token_buffer->empty()) {
              // Model produced plain text despite tools[] being present (no
              // tool call detected).  Replay the buffered tokens as content
              // deltas.
              for (const auto &tok : *token_buffer) {
                for (const auto &piece : SplitForStreaming(tok)) {
                  SendAll(session, BuildStreamChunk(stream_id, parsed.model,
                                                    stream_ts, piece, false));
                }
              }
              SendAll(session, BuildStreamChunk(stream_id, parsed.model,
                                                stream_ts, "", true));
            } else {
              SendAll(session, BuildStreamChunk(stream_id, parsed.model,
                                                stream_ts, "", true));
            }
            SendAll(session, "data: [DONE]\n\n");
          }
        }
        stream_active->store(false);
        return;
      } else {
        auto payload = BuildResponse(
            BuildCompletionBody(result, parsed, chat_mode, tool_call), 200,
            "OK", trace_response_header);
        SendAll(session, payload);
      }
    } catch (const std::exception &ex) {
      if (metrics_) {
        metrics_->RecordError();
      }
      auto payload = BuildResponse(BuildErrorBody(ex.what()), 500, "Error");
      SendAll(session, payload);
      if (audit_logger_) {
        audit_logger_->Log(auth_ctx.subject, parsed.model, "error", ex.what());
      }
    }
    return;
  }

  auto response = BuildResponse(BuildErrorBody("not_found"), 404, "Not Found");
  SendAll(session, response);
  if (audit_logger_) {
    audit_logger_->Log(auth_ctx.subject, "", "not_found", path);
  }
}

bool HttpServer::SendAll(ClientSession &session, const std::string &payload) {
  const char *data = payload.c_str();
  std::size_t remaining = payload.size();
  while (remaining > 0) {
    int sent = 0;
    if (session.ssl) {
      sent = SSL_write(session.ssl, data, static_cast<int>(remaining));
      if (sent <= 0) {
        int err = SSL_get_error(session.ssl, sent);
        if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
          continue;
        }
        return false;
      }
    } else {
      sent = static_cast<int>(::send(session.fd, data, remaining, 0));
      if (sent <= 0) {
        if (errno == EINTR) {
          continue;
        }
        return false;
      }
    }
    data += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
  return true;
}

ssize_t HttpServer::Receive(ClientSession &session, char *buffer,
                            std::size_t length) {
  if (session.ssl) {
    while (true) {
      int received = SSL_read(session.ssl, buffer, static_cast<int>(length));
      if (received > 0) {
        return received;
      }
      int err = SSL_get_error(session.ssl, received);
      if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
        continue;
      }
      return -1;
    }
  }
  while (true) {
    ssize_t received = ::recv(session.fd, buffer, length, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    return received;
  }
}

void HttpServer::CloseSession(ClientSession &session) {
  if (session.ssl) {
    SSL_shutdown(session.ssl);
    SSL_free(session.ssl);
    session.ssl = nullptr;
  }
  if (session.fd >= 0) {
    ::close(session.fd);
    session.fd = -1;
  }
}

} // namespace inferflux
