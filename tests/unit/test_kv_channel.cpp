#include <catch2/catch_amalgamated.hpp>

#include "runtime/disaggregated/kv_channel.h"

#include <thread>
#include <vector>

TEST_CASE("KVChannel enqueues and dequeues FIFO", "[kv_channel]") {
  inferflux::disaggregated::KVChannel channel(3);
  inferflux::disaggregated::KVPacket p1;
  p1.request_id = 1;
  p1.prompt_tokens = {1, 2, 3, 4};
  inferflux::disaggregated::KVPacket p2;
  p2.request_id = 2;
  REQUIRE(channel.Enqueue(std::move(p1)));
  REQUIRE(channel.Enqueue(std::move(p2)));
  auto out1 = channel.TryDequeue();
  REQUIRE(out1.has_value());
  REQUIRE(out1->request_id == 1);
  auto out2 = channel.TryDequeue();
  REQUIRE(out2.has_value());
  REQUIRE(out2->request_id == 2);
  REQUIRE_FALSE(channel.TryDequeue().has_value());
}

TEST_CASE("KVChannel respects capacity", "[kv_channel]") {
  inferflux::disaggregated::KVChannel channel(1);
  inferflux::disaggregated::KVPacket packet;
  packet.request_id = 42;
  packet.prompt_tokens = {1};
  REQUIRE(channel.Enqueue(packet));
  inferflux::disaggregated::KVPacket another;
  another.request_id = 43;
  another.prompt_tokens = {2};
  REQUIRE_FALSE(channel.Enqueue(another));
}

TEST_CASE("KVChannel is thread-safe for simple producers/consumers",
          "[kv_channel]") {
  inferflux::disaggregated::KVChannel channel(10);
  std::vector<std::thread> producers;
  for (int i = 0; i < 5; ++i) {
    producers.emplace_back([&channel, i]() {
      inferflux::disaggregated::KVPacket pkt;
      pkt.request_id = static_cast<uint64_t>(i);
      pkt.prompt_tokens = {static_cast<uint8_t>(i),
                           static_cast<uint8_t>(i + 1)};
      channel.Enqueue(std::move(pkt));
    });
  }
  for (auto &t : producers) {
    t.join();
  }
  int seen = 0;
  while (channel.TryDequeue().has_value()) {
    ++seen;
  }
  REQUIRE(seen == 5);
}
