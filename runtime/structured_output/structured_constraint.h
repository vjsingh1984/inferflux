#pragma once

#include <string>

namespace inferflux {

struct StructuredConstraint {
  bool has_grammar{false};
  std::string grammar;
  std::string root{"root"};
  bool require_json_object{false};
  bool has_schema{false};
  std::string schema;
};

}  // namespace inferflux
