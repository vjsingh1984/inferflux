#include "runtime/result.h"

#include <catch2/catch_amalgamated.hpp>

#include <string>
#include <vector>

using namespace inferflux;

TEST_CASE("Result<int> success", "[result]") {
  Result<int> r = Ok(42);
  REQUIRE(r.ok());
  REQUIRE(static_cast<bool>(r));
  REQUIRE(r.value() == 42);
}

TEST_CASE("Result<int> failure", "[result]") {
  Result<int> r = Err("bad input");
  REQUIRE_FALSE(r.ok());
  REQUIRE_FALSE(static_cast<bool>(r));
  REQUIRE(r.error().message == "bad input");
  REQUIRE(r.error().category == Error::Category::kGeneric);
}

TEST_CASE("Result<int> failure with category", "[result]") {
  Result<int> r = Err("not found", Error::Category::kNotFound);
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.error().category == Error::Category::kNotFound);
}

TEST_CASE("Result<string> success", "[result]") {
  Result<std::string> r = Ok(std::string("hello"));
  REQUIRE(r.ok());
  REQUIRE(r.value() == "hello");
}

TEST_CASE("Result<void> success", "[result]") {
  Result<void> r = OkVoid();
  REQUIRE(r.ok());
  REQUIRE(static_cast<bool>(r));
}

TEST_CASE("Result<void> failure", "[result]") {
  Result<void> r = Err("write failed");
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.error().message == "write failed");
}

TEST_CASE("Result<vector> move semantics", "[result]") {
  std::vector<int> data = {1, 2, 3};
  Result<std::vector<int>> r = Ok(std::move(data));
  REQUIRE(r.ok());
  REQUIRE(r.value().size() == 3);
  REQUIRE(r.value()[2] == 3);
}

TEST_CASE("Result implicit construction from T", "[result]") {
  Result<int> r = 99;
  REQUIRE(r.ok());
  REQUIRE(r.value() == 99);
}

TEST_CASE("Result implicit construction from Error", "[result]") {
  Result<int> r = Error("oops");
  REQUIRE_FALSE(r.ok());
  REQUIRE(r.error().message == "oops");
}
