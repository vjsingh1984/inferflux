#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ChatMessage {
  std::string role;
  std::string content;
};

std::string EscapeJson(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::string DecodeJsonString(const std::string& value) {
  std::string out;
  bool escape = false;
  for (char c : value) {
    if (escape) {
      switch (c) {
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case '\\':
          out.push_back('\\');
          break;
        case '"':
          out.push_back('"');
          break;
        default:
          out.push_back(c);
          break;
      }
      escape = false;
    } else if (c == '\\') {
      escape = true;
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::string ExtractJsonString(const std::string& body, const std::string& key) {
  std::string needle = "\"" + key + "\":\"";
  auto pos = body.find(needle);
  if (pos == std::string::npos) {
    return {};
  }
  pos += needle.size();
  std::string out;
  bool escape = false;
  for (std::size_t i = pos; i < body.size(); ++i) {
    char c = body[i];
    if (!escape && c == '\\') {
      escape = true;
      continue;
    }
    if (!escape && c == '"') {
      break;
    }
    if (escape) {
      switch (c) {
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case '\\':
          out.push_back('\\');
          break;
        case '"':
          out.push_back('"');
          break;
        default:
          out.push_back(c);
          break;
      }
      escape = false;
    } else {
      out.push_back(c);
    }
  }
  return out;
}

int Connect(const std::string& host, int port) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    throw std::runtime_error("unable to create socket");
  }
  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = inet_addr(host.c_str());
  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(sock);
    throw std::runtime_error("unable to connect");
  }
  return sock;
}

std::string BuildChatPayload(const std::vector<ChatMessage>& messages,
                             const std::string& model,
                             int max_tokens,
                             bool stream) {
  std::ostringstream out;
  out << "{\"model\":\"" << EscapeJson(model.empty() ? "unknown" : model) << "\",\"max_tokens\":" << max_tokens
      << ",\"stream\":" << (stream ? "true" : "false") << ",\"messages\":[";
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "{\"role\":\"" << EscapeJson(messages[i].role.empty() ? "user" : messages[i].role) << "\",\"content\":\""
        << EscapeJson(messages[i].content) << "\"}";
  }
  out << "]}";
  return out.str();
}

std::string BuildCompletionPayload(const std::string& prompt,
                                   const std::string& model,
                                   int max_tokens,
                                   bool stream) {
  std::ostringstream out;
  out << "{\"prompt\":\"" << EscapeJson(prompt) << "\",\"max_tokens\":" << max_tokens << ",\"stream\":"
      << (stream ? "true" : "false");
  if (!model.empty()) {
    out << ",\"model\":\"" << EscapeJson(model) << "\"";
  }
  out << "}";
  return out.str();
}

void PrintUsage() {
  std::cout << "Usage:\n"
            << "  inferctl status [--host 127.0.0.1] [--port 8080] [--api-key KEY]\n"
            << "  inferctl completion --prompt 'Hello' [--model MODEL] [--max-tokens N] [--stream]\n"
               "                      [--host 127.0.0.1] [--port 8080] [--api-key KEY]\n"
            << "  inferctl chat --message 'user:Hello' [--message 'assistant:Hi'] [--model MODEL]\n"
               "               [--max-tokens N] [--stream] [--interactive]\n";
}

bool ReceiveStandard(int sock, bool print_raw, std::string* body_out) {
  std::string raw;
  char buf[4096];
  ssize_t read_bytes = 0;
  while ((read_bytes = ::recv(sock, buf, sizeof(buf), 0)) > 0) {
    raw.append(buf, buf + read_bytes);
  }
  if (print_raw) {
    std::cout << raw << std::endl;
  }
  if (body_out) {
    auto header_end = raw.find("\r\n\r\n");
    if (header_end != std::string::npos) {
      *body_out = raw.substr(header_end + 4);
    }
  }
  return true;
}

bool ProcessStreamChunks(std::string& buffer, std::string* accumulated) {
  bool done = false;
  std::size_t search_pos = 0;
  while (true) {
    auto data_pos = buffer.find("data:", search_pos);
    if (data_pos == std::string::npos) {
      buffer.erase(0, std::min(search_pos, buffer.size()));
      break;
    }
    auto end_pos = buffer.find("\n\n", data_pos);
    if (end_pos == std::string::npos) {
      buffer.erase(0, data_pos);
      break;
    }
    std::string chunk = buffer.substr(data_pos + 5, end_pos - (data_pos + 5));
    buffer.erase(0, end_pos + 2);
    while (!chunk.empty() && (chunk.front() == ' ' || chunk.front() == '\r')) {
      chunk.erase(chunk.begin());
    }
    while (!chunk.empty() && (chunk.back() == '\r' || chunk.back() == '\n')) {
      chunk.pop_back();
    }
    if (chunk == "[DONE]") {
      done = true;
      break;
    }
    auto content = ExtractJsonString(chunk, "content");
    if (!content.empty()) {
      std::cout << content << std::flush;
      if (accumulated) {
        accumulated->append(content);
      }
    }
  }
  return done;
}

bool ReceiveStream(int sock, std::string* accumulated) {
  std::string buffer;
  char temp[4096];
  ssize_t read_bytes = 0;
  bool headers_done = false;
  while ((read_bytes = ::recv(sock, temp, sizeof(temp), 0)) > 0) {
    buffer.append(temp, temp + read_bytes);
    if (!headers_done) {
      auto header_end = buffer.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        headers_done = true;
        buffer.erase(0, header_end + 4);
      } else {
        continue;
      }
    }
    if (ProcessStreamChunks(buffer, accumulated)) {
      break;
    }
  }
  std::cout << std::endl;
  return true;
}

bool SendRequest(const std::string& host,
                 int port,
                 const std::string& api_key,
                 const std::string& path,
                 const std::string& payload,
                 bool stream,
                 bool print_raw,
                 std::string* body_out) {
  int sock = Connect(host, port);
  std::ostringstream request;
  request << "POST " << path << " HTTP/1.1\r\n";
  request << "Host: " << host << "\r\n";
  request << "Content-Type: application/json\r\n";
  if (!api_key.empty()) {
    request << "Authorization: Bearer " << api_key << "\r\n";
  }
  request << "Content-Length: " << payload.size() << "\r\n\r\n";
  request << payload;
  auto serialized = request.str();
  ::send(sock, serialized.c_str(), serialized.size(), 0);
  bool ok = stream ? ReceiveStream(sock, body_out) : ReceiveStandard(sock, print_raw, body_out);
  ::close(sock);
  return ok;
}

void InteractiveChat(const std::string& host,
                     int port,
                     const std::string& api_key,
                     const std::string& model,
                     int max_tokens,
                     bool stream) {
  std::vector<ChatMessage> history;
  std::cout << "Interactive chat session. Type /exit to quit." << std::endl;
  std::string line;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, line)) {
      break;
    }
    if (line == "/exit") {
      break;
    }
    if (line.empty()) {
      continue;
    }
    history.push_back({"user", line});
    std::string assistant;
    if (!SendRequest(host,
                     port,
                     api_key,
                     "/v1/chat/completions",
                     BuildChatPayload(history, model, max_tokens, stream),
                     stream,
                     false,
                     &assistant)) {
      break;
    }
    if (!stream) {
      auto content = ExtractJsonString(assistant, "content");
      if (content.empty()) {
        content = ExtractJsonString(assistant, "completion");
      }
      std::cout << content << std::endl;
      assistant = content;
    }
    history.push_back({"assistant", assistant});
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }
  std::string command = argv[1];
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string prompt;
  std::string api_key;
  std::string model;
  int max_tokens = 32;
  bool stream_mode = false;
  bool interactive_mode = false;
  std::vector<ChatMessage> chat_messages;
  if (const char* env_key = std::getenv("INFERCTL_API_KEY")) {
    api_key = env_key;
  }

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--host" || arg == "-H") && i + 1 < argc) {
      host = argv[++i];
    } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--api-key" && i + 1 < argc) {
      api_key = argv[++i];
    } else if (arg == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else if ((arg == "--max-tokens" || arg == "--max_tokens") && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--stream") {
      stream_mode = true;
    } else if (arg == "--interactive") {
      interactive_mode = true;
    } else if ((arg == "--message" || arg == "-m") && i + 1 < argc) {
      std::string raw = argv[++i];
      auto colon = raw.find(':');
      ChatMessage msg;
      if (colon == std::string::npos) {
        msg.role = "user";
        msg.content = raw;
      } else {
        msg.role = raw.substr(0, colon);
        msg.content = raw.substr(colon + 1);
        if (msg.role.empty()) {
          msg.role = "user";
        }
      }
      chat_messages.push_back(std::move(msg));
    }
  }

  try {
    if (command == "status") {
      int sock = Connect(host, port);
      std::string request = "GET /healthz HTTP/1.1\r\nHost: " + host + "\r\n";
      if (!api_key.empty()) {
        request += "Authorization: Bearer " + api_key + "\r\n";
      }
      request += "\r\n";
      ::send(sock, request.c_str(), request.size(), 0);
      ReceiveStandard(sock, true, nullptr);
      ::close(sock);
      return 0;
    }
    if (command == "completion") {
      if (prompt.empty()) {
        std::cerr << "--prompt is required" << std::endl;
        return 1;
      }
      std::string body;
      SendRequest(host,
                  port,
                  api_key,
                  "/v1/completions",
                  BuildCompletionPayload(prompt, model, max_tokens, stream_mode),
                  stream_mode,
                  !stream_mode,
                  stream_mode ? nullptr : &body);
      if (!stream_mode && !interactive_mode) {
        // already printed raw response
      }
      return 0;
    }
    if (command == "chat") {
      if (interactive_mode) {
        InteractiveChat(host, port, api_key, model, max_tokens, stream_mode);
        return 0;
      }
      if (chat_messages.empty()) {
        if (prompt.empty()) {
          std::cerr << "Provide at least one --message role:text or --prompt" << std::endl;
          return 1;
        }
        chat_messages.push_back(ChatMessage{"user", prompt});
      }
      std::string body;
      SendRequest(host,
                  port,
                  api_key,
                  "/v1/chat/completions",
                  BuildChatPayload(chat_messages, model, max_tokens, stream_mode),
                  stream_mode,
                  false,
                  stream_mode ? nullptr : &body);
      if (!stream_mode) {
        std::cout << body << std::endl;
      }
      return 0;
    }
  } catch (const std::exception& ex) {
    std::cerr << "inferctl error: " << ex.what() << std::endl;
    return 1;
  }

  PrintUsage();
  return 1;
}
