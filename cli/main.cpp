#include "net/http_client.h"

#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

std::string Trim(const std::string &input) {
  auto start = input.find_first_not_of(" \t");
  auto end = input.find_last_not_of(" \t\r\n");
  if (start == std::string::npos || end == std::string::npos) {
    return "";
  }
  return input.substr(start, end - start + 1);
}

struct ChatMessage {
  std::string role;
  std::string content;
};

std::string BuildUrl(const std::string &host, int port,
                     const std::string &path) {
  return "http://" + host + ":" + std::to_string(port) + path;
}

std::map<std::string, std::string> AuthHeaders(const std::string &api_key) {
  std::map<std::string, std::string> headers;
  if (!api_key.empty()) {
    headers["Authorization"] = "Bearer " + api_key;
  }
  return headers;
}

std::filesystem::path InferfluxHome() {
  if (const char *env = std::getenv("INFERFLUX_HOME")) {
    return std::filesystem::path(env);
  }
  if (const char *home = std::getenv("HOME")) {
    return std::filesystem::path(home) / ".inferflux";
  }
  return std::filesystem::current_path() / ".inferflux";
}

std::filesystem::path DefaultConfigPath() {
  return InferfluxHome() / "config.yaml";
}

json BuildChatPayload(const std::vector<ChatMessage> &messages,
                      const std::string &model, int max_tokens, bool stream) {
  json j;
  j["model"] = model.empty() ? "unknown" : model;
  j["max_tokens"] = max_tokens;
  j["stream"] = stream;
  j["messages"] = json::array();
  for (const auto &msg : messages) {
    j["messages"].push_back({{"role", msg.role.empty() ? "user" : msg.role},
                             {"content", msg.content}});
  }
  return j;
}

json BuildCompletionPayload(const std::string &prompt, const std::string &model,
                            int max_tokens, bool stream) {
  json j;
  j["prompt"] = prompt;
  j["max_tokens"] = max_tokens;
  j["stream"] = stream;
  if (!model.empty()) {
    j["model"] = model;
  }
  return j;
}

void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  inferctl pull <owner/model-name>\n"
      << "      Download the best GGUF from HuggingFace Hub "
         "(~/.cache/inferflux/models/).\n"
      << "      Prints the local path on success.\n"
      << "  inferctl quickstart <model-id> [--profile cpu-laptop]\n"
      << "      Writes a starter config (~/.inferflux/config.yaml) and prints "
         "next steps.\n"
      << "  inferctl serve [--config ~/.inferflux/config.yaml] [--no-ui]\n"
      << "      Launch inferfluxd using the specified config (defaults to "
         "~/.inferflux/config.yaml).\n"
      << "  inferctl status [--host 127.0.0.1] [--port 8080] [--api-key KEY]\n"
      << "  inferctl completion --prompt 'Hello' [--model MODEL] [--max-tokens "
         "N] [--stream]\n"
         "                      [--host 127.0.0.1] [--port 8080] [--api-key "
         "KEY]\n"
      << "  inferctl chat --message 'user:Hello' [--message 'assistant:Hi'] "
         "[--model MODEL]\n"
         "               [--max-tokens N] [--stream] [--interactive]\n"
      << "  inferctl admin guardrails [--list | --set word1,word2] [--host ... "
         "--port ... --api-key KEY]\n"
      << "  inferctl admin rate-limit [--get | --set N] [--host ... --port ... "
         "--api-key KEY]\n"
      << "  inferctl admin models --list | --load PATH [--backend TYPE] [--id "
         "NAME] [--default]\n"
         "                       | --unload ID | --set-default ID [--host ... "
         "--port ... --api-key KEY]\n"
      << "  inferctl admin api-keys [--list | --add KEY --scopes read,admin | "
         "--remove KEY]\n";
}

bool ProcessStreamChunks(std::string &buffer, std::string *accumulated) {
  bool done = false;
  while (true) {
    auto data_pos = buffer.find("data:");
    if (data_pos == std::string::npos) {
      break;
    }
    auto end_pos = buffer.find("\n\n", data_pos);
    if (end_pos == std::string::npos) {
      buffer.erase(0, data_pos);
      break;
    }
    std::string chunk =
        Trim(buffer.substr(data_pos + 5, end_pos - (data_pos + 5)));
    buffer.erase(0, end_pos + 2);
    if (chunk == "[DONE]") {
      done = true;
      break;
    }
    try {
      auto j = json::parse(chunk);
      if (j.contains("choices") && j["choices"].is_array() &&
          !j["choices"].empty()) {
        auto &delta = j["choices"][0];
        if (delta.contains("delta") && delta["delta"].contains("content")) {
          std::string content = delta["delta"]["content"].get<std::string>();
          std::cout << content << std::flush;
          if (accumulated) {
            accumulated->append(content);
          }
        }
      }
    } catch (const json::exception &) {
    }
  }
  return done;
}

bool ReceiveStream(inferflux::HttpClient &client,
                   inferflux::HttpClient::RawConnection &conn,
                   std::string *accumulated) {
  std::string buffer;
  char temp[4096];
  ssize_t read_bytes = 0;
  bool headers_done = false;
  while ((read_bytes = client.RecvRaw(conn, temp, sizeof(temp))) > 0) {
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
  client.CloseRaw(conn);
  return true;
}

void InteractiveChat(const std::string &host, int port,
                     const std::string &api_key, const std::string &model,
                     int max_tokens, bool stream) {
  inferflux::HttpClient client;
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
    auto payload = BuildChatPayload(history, model, max_tokens, stream);
    std::string url = BuildUrl(host, port, "/v1/chat/completions");
    std::string assistant;
    if (stream) {
      auto conn =
          client.SendRaw("POST", url, payload.dump(), AuthHeaders(api_key));
      ReceiveStream(client, conn, &assistant);
    } else {
      auto resp = client.Post(url, payload.dump(), AuthHeaders(api_key));
      try {
        auto j = json::parse(resp.body);
        if (j.contains("choices") && j["choices"].is_array() &&
            !j["choices"].empty()) {
          auto &choice = j["choices"][0];
          if (choice.contains("message") &&
              choice["message"].contains("content")) {
            assistant = choice["message"]["content"].get<std::string>();
          }
        }
      } catch (const json::exception &) {
        assistant = resp.body;
      }
      std::cout << assistant << std::endl;
    }
    history.push_back({"assistant", assistant});
  }
}

int CmdQuickstart(const std::string &model_id, const std::string &profile) {
  namespace fs = std::filesystem;
  fs::path base_dir = InferfluxHome();
  fs::create_directories(base_dir);
  fs::path config_path = base_dir / "config.yaml";
  auto model_dir = base_dir / "models";
  fs::create_directories(model_dir);
  std::ofstream cfg(config_path);
  if (!cfg) {
    std::cerr << "Failed to write config at " << config_path << std::endl;
    return 1;
  }
  cfg << "# InferFlux quickstart config (profile: " << profile << ")\n"
      << "server:\n"
      << "  host: 0.0.0.0\n"
      << "  http_port: 8080\n"
      << "auth:\n"
      << "  api_keys:\n"
      << "    - dev-key-123\n"
      << "runtime:\n"
      << "  model:\n"
      << "    path: \"" << (model_dir / model_id).string() << "\"\n";
  cfg.close();
  std::cout << "Wrote config to " << config_path << "\n";
  std::cout << "Next steps:\n"
            << "  1. inferctl pull " << model_id << "\n"
            << "  2. inferctl serve --config " << config_path << "\n";
  return 0;
}

int CmdServe(const std::string &config_path, bool enable_ui) {
  namespace fs = std::filesystem;
  fs::path cfg =
      config_path.empty() ? DefaultConfigPath() : fs::path(config_path);
  if (!fs::exists(cfg)) {
    std::cerr << "Config not found: " << cfg << std::endl;
    return 1;
  }
  std::string cmd = "inferfluxd --config \"" + cfg.string() + "\"";
  if (enable_ui) {
    cmd += " --ui";
  }
  std::cout << "Launching: " << cmd << std::endl;
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::cerr << "inferfluxd exited with code " << rc << std::endl;
  }
  return rc;
}

// ---- inferctl pull helpers ----

std::string GetCacheDir() {
  const char *xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg && xdg[0] != '\0') {
    return std::string(xdg) + "/inferflux/models";
  }
  const char *home = std::getenv("HOME");
  return std::string(home ? home : "/tmp") + "/.cache/inferflux/models";
}

// Prefer quantization quality order: Q4_K_M > Q4_K_S > Q4_K > Q4_0 > Q4 >
// Q5_K_M > Q5 > Q8 > any .gguf
std::string SelectBestGguf(const std::vector<std::string> &files) {
  auto score = [](const std::string &name) -> int {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("q4_k_m") != std::string::npos)
      return 100;
    if (lower.find("q4_k_s") != std::string::npos)
      return 90;
    if (lower.find("q4_k") != std::string::npos)
      return 85;
    if (lower.find("q4_0") != std::string::npos)
      return 80;
    if (lower.find("q4") != std::string::npos)
      return 75;
    if (lower.find("q5_k_m") != std::string::npos)
      return 70;
    if (lower.find("q5") != std::string::npos)
      return 65;
    if (lower.find("q8_0") != std::string::npos)
      return 50;
    if (lower.find(".gguf") != std::string::npos)
      return 10;
    return -1;
  };
  std::string best;
  int best_score = -1;
  for (const auto &f : files) {
    int s = score(f);
    if (s > best_score) {
      best_score = s;
      best = f;
    }
  }
  return best;
}

void PrintProgress(long long downloaded, long long total) {
  if (total <= 0) {
    std::cerr << "\r  " << (downloaded / (1024 * 1024)) << " MB downloaded"
              << std::flush;
    return;
  }
  int pct = static_cast<int>(downloaded * 100 / total);
  const int kBarWidth = 40;
  int filled = pct * kBarWidth / 100;
  std::cerr << "\r[";
  for (int i = 0; i < filled; ++i)
    std::cerr << '=';
  for (int i = filled; i < kBarWidth; ++i)
    std::cerr << ' ';
  std::cerr << "] " << pct << "%  " << (downloaded / (1024 * 1024)) << "/"
            << (total / (1024 * 1024)) << " MB" << std::flush;
}

struct DownloadHeaders {
  int status{0};
  std::string location;
  long long content_length{-1};
};

DownloadHeaders ParseDownloadHeaders(const std::string &header_block) {
  DownloadHeaders h;
  std::istringstream iss(header_block);
  std::string line;
  bool first = true;
  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (first) {
      first = false;
      auto sp = line.find(' ');
      if (sp != std::string::npos) {
        try {
          h.status = std::stoi(line.substr(sp + 1));
        } catch (...) {
        }
      }
      continue;
    }
    auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    std::string key = Trim(line.substr(0, colon));
    std::string val = Trim(line.substr(colon + 1));
    std::string key_lower = key;
    std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(),
                   ::tolower);
    if (key_lower == "location")
      h.location = val;
    else if (key_lower == "content-length") {
      try {
        h.content_length = std::stoll(val);
      } catch (...) {
      }
    }
  }
  return h;
}

// Streams a GET response body to dest_path, following up to max_redirects
// redirects. Returns 0 on success, 1 on error.
int DownloadToFile(inferflux::HttpClient &client, const std::string &url,
                   const std::string &dest_path) {
  std::string current_url = url;
  const int kMaxRedirects = 5;
  for (int attempt = 0; attempt <= kMaxRedirects; ++attempt) {
    auto conn = client.SendRaw("GET", current_url, "", {});

    // Extend recv timeout to 5 minutes for large file downloads.
    if (conn.sock >= 0) {
      struct timeval tv{300, 0};
      setsockopt(conn.sock, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char *>(&tv), sizeof(tv));
    }

    // Read until we have the full response headers.
    char buf[65536];
    std::string header_buf;
    std::string leftover;
    bool found_end = false;
    while (!found_end) {
      ssize_t n = client.RecvRaw(conn, buf, sizeof(buf));
      if (n <= 0)
        break;
      header_buf.append(buf, static_cast<std::size_t>(n));
      auto pos = header_buf.find("\r\n\r\n");
      if (pos != std::string::npos) {
        leftover = header_buf.substr(pos + 4);
        header_buf.erase(pos);
        found_end = true;
      }
    }
    if (!found_end) {
      client.CloseRaw(conn);
      std::cerr << "Failed to read response headers from " << current_url
                << "\n";
      return 1;
    }

    auto hdrs = ParseDownloadHeaders(header_buf);

    if (hdrs.status == 301 || hdrs.status == 302 || hdrs.status == 307 ||
        hdrs.status == 308) {
      client.CloseRaw(conn);
      if (hdrs.location.empty()) {
        std::cerr << "Redirect with no Location header\n";
        return 1;
      }
      current_url = hdrs.location;
      std::cerr << "Following redirect...\n";
      continue;
    }

    if (hdrs.status != 200) {
      client.CloseRaw(conn);
      std::cerr << "HTTP error " << hdrs.status << " downloading "
                << current_url << "\n";
      return 1;
    }

    // Stream body to a .tmp file, then atomically rename.
    std::string tmp_path = dest_path + ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out) {
      client.CloseRaw(conn);
      std::cerr << "Cannot create file: " << tmp_path << "\n";
      return 1;
    }

    long long downloaded = 0;
    long long total = hdrs.content_length;

    if (!leftover.empty()) {
      out.write(leftover.data(), static_cast<std::streamsize>(leftover.size()));
      downloaded += static_cast<long long>(leftover.size());
      PrintProgress(downloaded, total);
    }

    while (true) {
      ssize_t n = client.RecvRaw(conn, buf, sizeof(buf));
      if (n <= 0)
        break;
      out.write(buf, n);
      downloaded += n;
      PrintProgress(downloaded, total);
    }

    client.CloseRaw(conn);
    out.close();
    std::cerr << "\n";

    if (total > 0 && downloaded < total) {
      std::filesystem::remove(tmp_path);
      std::cerr << "Download incomplete (" << downloaded << "/" << total
                << " bytes)\n";
      return 1;
    }

    std::filesystem::rename(tmp_path, dest_path);
    return 0;
  }

  std::cerr << "Too many redirects for " << url << "\n";
  return 1;
}

int CmdPull(const std::string &repo) {
  if (repo.find('/') == std::string::npos) {
    std::cerr << "Repository must be in owner/model-name format (e.g. "
                 "TheBloke/Llama-2-7B-GGUF)\n";
    return 1;
  }

  inferflux::HttpClient client;

  // Step 1: Fetch repository metadata from HuggingFace Hub API.
  std::string api_url = "https://huggingface.co/api/models/" + repo;
  std::cerr << "Fetching model info for " << repo << "...\n";
  auto resp = client.Get(api_url, {{"Accept", "application/json"}});
  if (resp.status != 200) {
    std::cerr << "HuggingFace API error " << resp.status << " for " << repo
              << "\n";
    return 1;
  }

  // Step 2: Parse siblings to collect GGUF filenames.
  std::vector<std::string> gguf_files;
  try {
    auto j = json::parse(resp.body);
    if (j.contains("siblings") && j["siblings"].is_array()) {
      for (const auto &s : j["siblings"]) {
        if (!s.contains("rfilename"))
          continue;
        std::string fname = s["rfilename"].get<std::string>();
        std::string lower = fname;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find(".gguf") != std::string::npos) {
          gguf_files.push_back(fname);
        }
      }
    }
  } catch (const json::exception &e) {
    std::cerr << "Failed to parse HuggingFace API response: " << e.what()
              << "\n";
    return 1;
  }

  if (gguf_files.empty()) {
    std::cerr << "No GGUF files found in " << repo << "\n";
    return 1;
  }

  // Step 3: Select the best-quantised file.
  std::string selected = SelectBestGguf(gguf_files);
  std::cerr << "Selected: " << selected << "\n";

  // Step 4: Resolve destination path.
  std::string cache_dir = GetCacheDir();
  std::string dest_dir = cache_dir + "/" + repo;
  std::filesystem::create_directories(dest_dir);
  std::string dest_path = dest_dir + "/" + selected;

  if (std::filesystem::exists(dest_path)) {
    std::cerr << "Already cached: " << dest_path << "\n";
    std::cout << dest_path << "\n";
    return 0;
  }

  // Step 5: Stream-download with progress.
  // HuggingFace resolve URL:
  // https://huggingface.co/{repo}/resolve/main/{filename}
  std::string download_url =
      "https://huggingface.co/" + repo + "/resolve/main/" + selected;
  std::cerr << "Downloading " << selected << "...\n";

  int rc = DownloadToFile(client, download_url, dest_path);
  if (rc != 0)
    return rc;

  std::cerr << "Saved to: " << dest_path << "\n";
  std::cout << dest_path << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
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
  int max_tokens = 256;
  bool stream_mode = false;
  bool interactive_mode = false;
  std::vector<ChatMessage> chat_messages;
  std::string quickstart_profile = "default";
  std::string serve_config_override;
  bool serve_no_ui = false;
  if (const char *env_key = std::getenv("INFERCTL_API_KEY")) {
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
    } else if ((arg == "--max-tokens" || arg == "--max_tokens") &&
               i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--stream") {
      stream_mode = true;
    } else if (arg == "--interactive") {
      interactive_mode = true;
    } else if (arg == "--profile" && i + 1 < argc) {
      quickstart_profile = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      serve_config_override = argv[++i];
    } else if (arg == "--no-ui") {
      serve_no_ui = true;
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

  // pull does not require a running server — handle before try/client block.
  if (command == "pull") {
    if (argc < 3) {
      std::cerr << "Usage: inferctl pull <owner/model-name>\n";
      return 1;
    }
    return CmdPull(argv[2]);
  }
  if (command == "quickstart") {
    if (argc < 3) {
      std::cerr
          << "Usage: inferctl quickstart <model-id> [--profile cpu-laptop]\n";
      return 1;
    }
    return CmdQuickstart(argv[2], quickstart_profile);
  }
  if (command == "serve") {
    return CmdServe(serve_config_override.empty() ? DefaultConfigPath().string()
                                                  : serve_config_override,
                    !serve_no_ui);
  }

  try {
    inferflux::HttpClient client;
    auto headers = AuthHeaders(api_key);

    if (command == "admin") {
      if (argc < 3) {
        PrintUsage();
        return 1;
      }
      if (api_key.empty()) {
        std::cerr << "--api-key is required for admin commands" << std::endl;
        return 1;
      }
      std::string target = argv[2];
      if (target == "guardrails") {
        bool list = false;
        std::string set_values;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--list")
            list = true;
          else if (arg == "--set" && i + 1 < argc)
            set_values = argv[++i];
        }
        if (list) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/guardrails"), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        if (!set_values.empty()) {
          std::vector<std::string> words;
          std::stringstream ss(set_values);
          std::string token;
          while (std::getline(ss, token, ',')) {
            auto trimmed = Trim(token);
            if (!trimmed.empty())
              words.push_back(trimmed);
          }
          auto resp = client.Put(BuildUrl(host, port, "/v1/admin/guardrails"),
                                 json({{"blocklist", words}}).dump(), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        PrintUsage();
        return 1;
      }
      if (target == "rate-limit") {
        bool get = false;
        bool set_flag = false;
        int new_limit = 0;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--get")
            get = true;
          else if (arg == "--set" && i + 1 < argc) {
            new_limit = std::stoi(argv[++i]);
            set_flag = true;
          }
        }
        if (get) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/rate_limit"), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        if (set_flag) {
          auto resp = client.Put(
              BuildUrl(host, port, "/v1/admin/rate_limit"),
              json({{"tokens_per_minute", new_limit}}).dump(), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        PrintUsage();
        return 1;
      }
      if (target == "models") {
        bool list = false;
        std::string load_path;
        std::string load_backend;
        std::string load_id;
        bool load_default = false;
        std::string unload_id;
        std::string set_default_id;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--list") {
            list = true;
          } else if (arg == "--load" && i + 1 < argc) {
            load_path = argv[++i];
          } else if (arg == "--backend" && i + 1 < argc) {
            load_backend = argv[++i];
          } else if (arg == "--id" && i + 1 < argc) {
            load_id = argv[++i];
          } else if (arg == "--default") {
            load_default = true;
          } else if (arg == "--unload" && i + 1 < argc) {
            unload_id = argv[++i];
          } else if (arg == "--set-default" && i + 1 < argc) {
            set_default_id = argv[++i];
          }
        }
        if (list) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/models"), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        if (!load_path.empty()) {
          json payload;
          payload["path"] = load_path;
          if (!load_backend.empty()) {
            payload["backend"] = load_backend;
          }
          if (!load_id.empty()) {
            payload["id"] = load_id;
          }
          if (load_default) {
            payload["default"] = true;
          }
          auto resp = client.Post(BuildUrl(host, port, "/v1/admin/models"),
                                  payload.dump(), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        if (!unload_id.empty()) {
          auto resp = client.Delete(BuildUrl(host, port, "/v1/admin/models"),
                                    json({{"id", unload_id}}).dump(), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        if (!set_default_id.empty()) {
          auto resp =
              client.Put(BuildUrl(host, port, "/v1/admin/models/default"),
                         json({{"id", set_default_id}}).dump(), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        PrintUsage();
        return 1;
      }
      if (target == "api-keys" || target == "api-key") {
        bool list = false, remove = false, add = false;
        std::string new_key, scopes_csv;
        for (int i = 3; i < argc; ++i) {
          std::string arg = argv[i];
          if (arg == "--list")
            list = true;
          else if (arg == "--add" && i + 1 < argc) {
            new_key = argv[++i];
            add = true;
          } else if (arg == "--scopes" && i + 1 < argc)
            scopes_csv = argv[++i];
          else if (arg == "--remove" && i + 1 < argc) {
            new_key = argv[++i];
            remove = true;
          }
        }
        if (list) {
          auto resp =
              client.Get(BuildUrl(host, port, "/v1/admin/api_keys"), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        if (add) {
          std::vector<std::string> scopes;
          std::stringstream ss(scopes_csv);
          std::string token;
          while (std::getline(ss, token, ',')) {
            auto trimmed = Trim(token);
            if (!trimmed.empty())
              scopes.push_back(trimmed);
          }
          auto resp = client.Post(
              BuildUrl(host, port, "/v1/admin/api_keys"),
              json({{"key", new_key}, {"scopes", scopes}}).dump(), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        if (remove) {
          auto resp = client.Delete(BuildUrl(host, port, "/v1/admin/api_keys"),
                                    json({{"key", new_key}}).dump(), headers);
          std::cout << resp.body << std::endl;
          return 0;
        }
        PrintUsage();
        return 1;
      }
      PrintUsage();
      return 1;
    }
    if (command == "status") {
      auto resp = client.Get(BuildUrl(host, port, "/healthz"), headers);
      std::cout << resp.body << std::endl;
      return 0;
    }
    if (command == "completion") {
      if (prompt.empty()) {
        std::cerr << "--prompt is required" << std::endl;
        return 1;
      }
      auto payload =
          BuildCompletionPayload(prompt, model, max_tokens, stream_mode);
      std::string url = BuildUrl(host, port, "/v1/completions");
      if (stream_mode) {
        auto conn = client.SendRaw("POST", url, payload.dump(), headers);
        ReceiveStream(client, conn, nullptr);
      } else {
        auto resp = client.Post(url, payload.dump(), headers);
        std::cout << resp.body << std::endl;
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
          std::cerr << "Provide at least one --message role:text or --prompt"
                    << std::endl;
          return 1;
        }
        chat_messages.push_back(ChatMessage{"user", prompt});
      }
      auto payload =
          BuildChatPayload(chat_messages, model, max_tokens, stream_mode);
      std::string url = BuildUrl(host, port, "/v1/chat/completions");
      if (stream_mode) {
        auto conn = client.SendRaw("POST", url, payload.dump(), headers);
        ReceiveStream(client, conn, nullptr);
      } else {
        auto resp = client.Post(url, payload.dump(), headers);
        std::cout << resp.body << std::endl;
      }
      return 0;
    }
  } catch (const std::exception &ex) {
    std::cerr << "inferctl error: " << ex.what() << std::endl;
    return 1;
  }

  PrintUsage();
  return 1;
}
