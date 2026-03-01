#pragma once

#include <string>

#include "runtime/structured_output/structured_constraint.h"

namespace inferflux {

class StructuredOutputAdapter {
 public:
  static bool BuildConstraint(const std::string& type,
                              const std::string& schema,
                              const std::string& grammar,
                              const std::string& root,
                              StructuredConstraint* out,
                              std::string* error);
};

}  // namespace inferflux
