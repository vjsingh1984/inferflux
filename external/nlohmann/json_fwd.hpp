#pragma once

// The upstream json-schema-to-grammar helper pulls in <nlohmann/json_fwd.hpp>.
// InferFlux vendors nlohmann/json.hpp directly (v3.11.3), so expose a
// compatibility forward header that simply includes the full definition to keep
// both toolchains aligned on the same ABI.
#include "json.hpp"
