#pragma once

#include <string>

namespace litehtml {

inline void replace_placeholder(std::string &target,
                                const std::string &placeholder,
                                const std::string &value) {
  std::size_t pos = target.find(placeholder);
  if (pos != std::string::npos) {
    target.replace(pos, placeholder.size(), value);
  }
}

inline std::string render_html(const std::string &html_template,
                               const std::string &css,
                               const std::string &backend_label) {
  std::string result = html_template;
  replace_placeholder(result, "{{css}}", css);
  replace_placeholder(result, "{{backend}}", backend_label);
  return result;
}

} // namespace litehtml
