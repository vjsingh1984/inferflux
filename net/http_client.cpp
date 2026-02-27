#include "net/http_client.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sstream>
#include <stdexcept>

namespace inferflux {
namespace {
struct ParsedUrl {
  std::string host;
  std::string path;
  int port{80};
};

ParsedUrl ParseUrl(const std::string& url) {
  ParsedUrl parsed;
  auto pos = url.find("://");
  std::string remainder = pos == std::string::npos ? url : url.substr(pos + 3);
  auto slash = remainder.find('/');
  std::string host_port = slash == std::string::npos ? remainder : remainder.substr(0, slash);
  parsed.path = slash == std::string::npos ? "/" : remainder.substr(slash);
  auto colon = host_port.find(':');
  if (colon == std::string::npos) {
    parsed.host = host_port;
  } else {
    parsed.host = host_port.substr(0, colon);
    parsed.port = std::stoi(host_port.substr(colon + 1));
  }
  return parsed;
}
}

HttpResponse HttpClient::Post(const std::string& url,
                              const std::string& body,
                              const std::map<std::string, std::string>& headers) const {
  return Send("POST", url, body, headers);
}

HttpResponse HttpClient::Send(const std::string& method,
                              const std::string& url,
                              const std::string& body,
                              const std::map<std::string, std::string>& headers) const {
  auto parsed = ParseUrl(url);
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  if (getaddrinfo(parsed.host.c_str(), std::to_string(parsed.port).c_str(), &hints, &result) != 0) {
    throw std::runtime_error("failed to resolve host");
  }
  int sock = -1;
  for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
    sock = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock == -1) continue;
    if (::connect(sock, rp->ai_addr, rp->ai_addrlen) == 0) {
      break;
    }
    ::close(sock);
    sock = -1;
  }
  freeaddrinfo(result);
  if (sock == -1) {
    throw std::runtime_error("failed to connect");
  }
  std::ostringstream request;
  request << method << " " << parsed.path << " HTTP/1.1\r\n";
  request << "Host: " << parsed.host << "\r\n";
  request << "Content-Length: " << body.size() << "\r\n";
  request << "Content-Type: application/json\r\n";
  for (const auto& [key, value] : headers) {
    request << key << ": " << value << "\r\n";
  }
  request << "Connection: close\r\n\r\n";
  request << body;
  auto payload = request.str();
  ::send(sock, payload.c_str(), payload.size(), 0);
  std::string response;
  char buffer[4096];
  ssize_t read_bytes = 0;
  while ((read_bytes = ::recv(sock, buffer, sizeof(buffer), 0)) > 0) {
    response.append(buffer, buffer + read_bytes);
  }
  ::close(sock);
  HttpResponse http_response;
  auto header_end = response.find("\r\n\r\n");
  std::string header = header_end == std::string::npos ? response : response.substr(0, header_end);
  std::string body_str = header_end == std::string::npos ? std::string() : response.substr(header_end + 4);
  auto status_pos = header.find(' ');
  if (status_pos != std::string::npos) {
    http_response.status = std::stoi(header.substr(status_pos + 1));
  }
  http_response.body = body_str;
  return http_response;
}

}  // namespace inferflux
