#include <catch2/catch_amalgamated.hpp>

#include "server/policy/guardrail.h"

#include <filesystem>
#include <fstream>
#include <string>

TEST_CASE("Guardrail disabled by default", "[guardrail]") {
  inferflux::Guardrail guardrail;
  REQUIRE(!guardrail.Enabled());

  std::string reason;
  REQUIRE(guardrail.Check("anything", &reason));
}

TEST_CASE("Guardrail blocklist blocks matching words", "[guardrail]") {
  inferflux::Guardrail guardrail;
  guardrail.SetBlocklist({"banned", "forbidden"});
  REQUIRE(guardrail.Enabled());

  std::string reason;
  REQUIRE(guardrail.Check("hello world", &reason));
  REQUIRE(!guardrail.Check("this is banned content", &reason));
  REQUIRE(!reason.empty());
}

TEST_CASE("Guardrail blocklist is case-insensitive", "[guardrail]") {
  inferflux::Guardrail guardrail;
  guardrail.SetBlocklist({"secret"});

  std::string reason;
  REQUIRE(!guardrail.Check("this is SECRET info", &reason));
}

TEST_CASE("Guardrail UpdateBlocklist replaces list", "[guardrail]") {
  inferflux::Guardrail guardrail;
  guardrail.SetBlocklist({"old"});
  REQUIRE(guardrail.Blocklist().size() == 1);

  guardrail.UpdateBlocklist({"new1", "new2"});
  auto bl = guardrail.Blocklist();
  REQUIRE(bl.size() == 2);

  std::string reason;
  REQUIRE(guardrail.Check("old stuff is fine now", &reason));
  REQUIRE(!guardrail.Check("new1 is blocked", &reason));
}

TEST_CASE("Guardrail OPA file-based policy denial", "[guardrail]") {
  inferflux::Guardrail guardrail;
  REQUIRE(!guardrail.Enabled());

  auto tmp_path = std::filesystem::temp_directory_path() / "inferflux_opa_test.json";
  {
    std::ofstream out(tmp_path);
    out << R"({"result":{"allow":false,"reason":"deny"}})";
  }
  guardrail.SetOPAEndpoint(std::string("file://") + tmp_path.string());
  REQUIRE(guardrail.Enabled());

  std::string reason;
  bool allowed = guardrail.Check("hello world", &reason);
  REQUIRE(!allowed);
  REQUIRE(!reason.empty());

  std::filesystem::remove(tmp_path);
}

TEST_CASE("Guardrail OPA file-based policy allow", "[guardrail]") {
  inferflux::Guardrail guardrail;

  auto tmp_path = std::filesystem::temp_directory_path() / "inferflux_opa_allow.json";
  {
    std::ofstream out(tmp_path);
    out << R"({"result":{"allow":true}})";
  }
  guardrail.SetOPAEndpoint(std::string("file://") + tmp_path.string());

  std::string reason;
  REQUIRE(guardrail.Check("hello world", &reason));

  std::filesystem::remove(tmp_path);
}
