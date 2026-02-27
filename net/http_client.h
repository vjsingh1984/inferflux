#pragma once

#include <map>
#include <string>

namespace inferflux {

struct HttpResponse {
  int status{0};
  std::string body;
};

class HttpClient {
 public:
  HttpResponse Post(const std::string& url,
                    const std::string& body,
                    const std::map<std::string, std::string>& headers = {}) const;

 private:
  HttpResponse Send(const std::string& method,
                    const std::string& url,
                    const std::string& body,
                    const std::map<std::string, std::string>& headers) const;
};

}  // namespace inferflux
