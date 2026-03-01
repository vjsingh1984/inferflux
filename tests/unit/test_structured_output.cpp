#include <catch2/catch_amalgamated.hpp>

#include "runtime/structured_output/structured_output_adapter.h"

TEST_CASE("StructuredOutputAdapter handles json_object without schema", "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_object", "", "", "root", &constraint, &error));
  REQUIRE(constraint.require_json_object);
  REQUIRE_FALSE(constraint.has_grammar);
  REQUIRE_FALSE(constraint.has_schema);
}

TEST_CASE("StructuredOutputAdapter rejects invalid schema", "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE_FALSE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", "{not json}", "", "root", &constraint, &error));
  REQUIRE_FALSE(error.empty());
}

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

TEST_CASE("StructuredOutputAdapter converts json_schema into grammar", "[structured]") {
  const char* schema = R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})";
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_schema", schema, "", "root", &constraint, &error));
  REQUIRE(constraint.has_grammar);
  REQUIRE(constraint.grammar.find("name") != std::string::npos);
}

TEST_CASE("StructuredOutputAdapter emits grammar for json_object without schema", "[structured]") {
  inferflux::StructuredConstraint constraint;
  std::string error;
  REQUIRE(inferflux::StructuredOutputAdapter::BuildConstraint(
      "json_object", "", "", "root", &constraint, &error));
  REQUIRE(constraint.require_json_object);
  REQUIRE(constraint.has_grammar);
  REQUIRE_FALSE(constraint.grammar.empty());
}
