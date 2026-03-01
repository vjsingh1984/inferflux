#pragma once

#include <string>

namespace inferflux {

class WebUiRenderer {
public:
  WebUiRenderer() = default;
  std::string RenderIndex(const std::string &backend_label) const;
};

} // namespace inferflux
