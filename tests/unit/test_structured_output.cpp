#include <catch2/catch_amalgamated.hpp>

#include "runtime/structured_output/structured_output_adapter.h"

// ---------------------------------------------------------------------------
// json_object (no schema)
// ---------------------------------------------------------------------------

TEST_CASE("StructuredOutputAdapter handles json_object without schema",
          "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_object", "", "", "root", &constraint, &error));
  REQUIRE(constraint.require_json_object);
  // The implementation falls back to {"type":"object"} and converts it.
  REQUIRE(constraint.has_grammar);
  REQUIRE(constraint.has_schema);
}

// ---------------------------------------------------------------------------
// json_schema — error paths
// ---------------------------------------------------------------------------

TEST_CASE("StructuredOutputAdapter rejects invalid schema", "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE_FALSE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", "{not json}", "", "root", &constraint, &error));
  REQUIRE_FALSE(error.empty());
}

// ---------------------------------------------------------------------------
// json_schema — basic happy path
// ---------------------------------------------------------------------------

TEST_CASE("StructuredOutputAdapter stores json_schema text", "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", "{\"type\":\"object\"}", "", "root", &constraint, &error));
  REQUIRE(constraint.has_schema);
  REQUIRE(constraint.schema.find("object") != std::string::npos);
  REQUIRE(constraint.has_grammar);
  REQUIRE(constraint.grammar.find("root") != std::string::npos);
}

TEST_CASE("StructuredOutputAdapter converts json_schema into grammar",
          "[structured]") {
  const char *schema =
      R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})";
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", schema, "", "root", &constraint, &error));
  REQUIRE(constraint.has_grammar);
  REQUIRE(constraint.grammar.find("name") != std::string::npos);
}

TEST_CASE(
    "StructuredOutputAdapter emits grammar for json_object without schema",
    "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_object", "", "", "root", &constraint, &error));
  REQUIRE(constraint.require_json_object);
  REQUIRE(constraint.has_grammar);
  REQUIRE_FALSE(constraint.grammar.empty());
}

// ---------------------------------------------------------------------------
// $defs / $ref resolution
// ---------------------------------------------------------------------------

TEST_CASE("StructuredOutputAdapter resolves $defs/$ref in schema",
          "[structured]") {
  const char *schema = R"({
    "$defs": {
      "Point": {
        "type": "object",
        "properties": {
          "x": {"type": "number"},
          "y": {"type": "number"}
        },
        "required": ["x", "y"]
      }
    },
    "type": "object",
    "properties": {
      "location": {"$ref": "#/$defs/Point"}
    },
    "required": ["location"]
  })";
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", schema, "", "root", &constraint, &error));
  REQUIRE(constraint.has_grammar);
  REQUIRE_FALSE(constraint.grammar.empty());
}

TEST_CASE("StructuredOutputAdapter resolves nested $defs", "[structured]") {
  const char *schema = R"({
    "$defs": {
      "Inner": {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"]
      },
      "Outer": {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "required": ["inner"]
      }
    },
    "type": "object",
    "properties": {"data": {"$ref": "#/$defs/Outer"}},
    "required": ["data"]
  })";
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", schema, "", "root", &constraint, &error));
  REQUIRE(constraint.has_grammar);
  REQUIRE_FALSE(constraint.grammar.empty());
}

TEST_CASE("StructuredOutputAdapter resolves array items via $ref",
          "[structured]") {
  const char *schema = R"({
    "$defs": {
      "Tag": {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"]
      }
    },
    "type": "object",
    "properties": {
      "tags": {
        "type": "array",
        "items": {"$ref": "#/$defs/Tag"}
      }
    },
    "required": ["tags"]
  })";
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", schema, "", "root", &constraint, &error));
  REQUIRE(constraint.has_grammar);
  REQUIRE_FALSE(constraint.grammar.empty());
}

// ---------------------------------------------------------------------------
// grammar passthrough
// ---------------------------------------------------------------------------

TEST_CASE("StructuredOutputAdapter grammar type passes string through",
          "[structured]") {
  const std::string raw_grammar = "root ::= [a-z]+\n";
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "grammar", "", raw_grammar, "root", &constraint, &error));
  REQUIRE(constraint.has_grammar);
  REQUIRE(constraint.grammar == raw_grammar);
  REQUIRE(constraint.root == "root");
  REQUIRE_FALSE(constraint.require_json_object);
}

TEST_CASE("StructuredOutputAdapter grammar type rejects empty grammar",
          "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE_FALSE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "grammar", "", "", "root", &constraint, &error));
  REQUIRE_FALSE(error.empty());
}

// ---------------------------------------------------------------------------
// text / empty type
// ---------------------------------------------------------------------------

TEST_CASE("StructuredOutputAdapter text type sets no constraint",
          "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "text", "", "", "root", &constraint, &error));
  REQUIRE_FALSE(constraint.has_grammar);
  REQUIRE_FALSE(constraint.has_schema);
  REQUIRE_FALSE(constraint.require_json_object);
}

TEST_CASE("StructuredOutputAdapter empty type sets no constraint",
          "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "", "", "", "root", &constraint, &error));
  REQUIRE_FALSE(constraint.has_grammar);
}

TEST_CASE("StructuredOutputAdapter rejects unknown type", "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE_FALSE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "unknown_format", "", "", "root", &constraint, &error));
  REQUIRE_FALSE(error.empty());
}
