#include <catch2/catch_amalgamated.hpp>

#include "server/metrics/metrics.h"

#include <string>

TEST_CASE("MetricsRegistry default backend is cpu", "[metrics]") {
  inferflux::MetricsRegistry registry;
  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("backend=\"cpu\"") != std::string::npos);
}

TEST_CASE("MetricsRegistry records successes", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSuccess(10, 20);
  registry.RecordSuccess(5, 15);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_requests_total{backend=\"cpu\"} 2") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_prompt_tokens_total{backend=\"cpu\"} 15") !=
          std::string::npos);
  REQUIRE(
      output.find("inferflux_completion_tokens_total{backend=\"cpu\"} 35") !=
      std::string::npos);
}

TEST_CASE("MetricsRegistry records errors", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordError();
  registry.RecordError();
  registry.RecordError();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_errors_total{backend=\"cpu\"} 3") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry records empty generations", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordEmptyGeneration();
  registry.RecordEmptyGeneration();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_empty_generations_total{backend=\"cpu\"} 2") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry records speculative stats", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSpeculative(4, 3, 6);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_spec_chunks_total{backend=\"cpu\"} 4") !=
          std::string::npos);
  REQUIRE(
      output.find("inferflux_spec_chunks_accepted_total{backend=\"cpu\"} 3") !=
      std::string::npos);
  REQUIRE(
      output.find("inferflux_spec_tokens_reused_total{backend=\"cpu\"} 6") !=
      std::string::npos);
}

TEST_CASE("MetricsRegistry skips zero speculative stats", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSpeculative(0, 0, 0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_spec_chunks_total{backend=\"cpu\"} 0") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry SetBackend changes label", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetBackend("cuda");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("backend=\"cuda\"") != std::string::npos);
  REQUIRE(output.find("backend=\"cpu\"") == std::string::npos);
}

TEST_CASE("MetricsRegistry latency histogram records buckets", "[metrics]") {
  inferflux::MetricsRegistry registry;
  // Record 80ms — should fall in the 100ms bucket (≤100) and all larger ones.
  registry.RecordLatency(80.0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_request_duration_ms_bucket") !=
          std::string::npos);
  REQUIRE(
      output.find("inferflux_request_duration_ms_count{backend=\"cpu\"} 1") !=
      std::string::npos);
  // 80ms is above 50ms bucket but within 100ms bucket.
  REQUIRE(output.find("le=\"50\"} 0") != std::string::npos);
  REQUIRE(output.find("le=\"100\"} 1") != std::string::npos);
  REQUIRE(output.find("le=\"+Inf\"} 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry active connections gauge", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.IncrementConnections();
  registry.IncrementConnections();
  registry.DecrementConnections();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_active_connections 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry queue depth gauge", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetQueueDepth(7);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_scheduler_queue_depth 7") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry prefill/decode queue gauges", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetPrefillQueueDepth(3);
  registry.SetDecodeQueueDepth(2);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_prefill_queue_depth 3") != std::string::npos);
  REQUIRE(output.find("inferflux_decode_queue_depth 2") != std::string::npos);
}

TEST_CASE("MetricsRegistry scheduler batch limits and token-budget skips",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetSchedulerBatchLimits(8, 16384);
  registry.RecordBatchTokenBudgetSkip();
  registry.RecordBatchTokenBudgetSkip();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_scheduler_batch_limit_size 8") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_batch_limit_tokens 16384") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_batch_token_budget_skips_total{"
                      "backend=\"cpu\"} 2") != std::string::npos);
}

TEST_CASE("MetricsRegistry records scheduler iteration composition",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSchedulerIteration(/*prefill_requests=*/2,
                                    /*decode_requests=*/0,
                                    /*token_count=*/12);
  registry.RecordSchedulerIteration(/*prefill_requests=*/0,
                                    /*decode_requests=*/3,
                                    /*token_count=*/7);
  registry.RecordSchedulerIteration(/*prefill_requests=*/1,
                                    /*decode_requests=*/1,
                                    /*token_count=*/9);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_scheduler_iterations_total{backend=\"cpu\","
                      "phase=\"prefill\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_iterations_total{backend=\"cpu\","
                      "phase=\"decode\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_iterations_total{backend=\"cpu\","
                      "phase=\"mixed\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_iteration_requests_total{"
                      "backend=\"cpu\"} 7") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_iteration_tokens_total{"
                      "backend=\"cpu\"} 28") != std::string::npos);
}

TEST_CASE("MetricsRegistry records scheduler policy iteration composition",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSchedulerPolicyIteration("lpm_priority",
                                          /*prefill_requests=*/2,
                                          /*decode_requests=*/0);
  registry.RecordSchedulerPolicyIteration("lpm_priority",
                                          /*prefill_requests=*/0,
                                          /*decode_requests=*/3);
  registry.RecordSchedulerPolicyIteration("throughput_balanced",
                                          /*prefill_requests=*/1,
                                          /*decode_requests=*/1);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_scheduler_policy_iterations_total{backend="
                      "\"cpu\",policy=\"lpm_priority\",phase=\"prefill\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_policy_iterations_total{backend="
                      "\"cpu\",policy=\"lpm_priority\",phase=\"decode\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_policy_iterations_total{backend="
                      "\"cpu\",policy=\"throughput_balanced\",phase=\"mixed\"} "
                      "1") != std::string::npos);
}

TEST_CASE("MetricsRegistry records scheduler prefix-affinity probe counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordPrefixAffinityProbe(false, 0);
  registry.RecordPrefixAffinityProbe(true, 12);
  registry.RecordPrefixAffinityProbe(true, 8);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_scheduler_prefix_affinity_probes_total{"
                      "backend=\"cpu\"} 3") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_prefix_affinity_hits_total{backend="
                      "\"cpu\"} 2") != std::string::npos);
  REQUIRE(
      output.find("inferflux_scheduler_prefix_affinity_matched_tokens_total{"
                  "backend=\"cpu\"} 20") != std::string::npos);
}

TEST_CASE("MetricsRegistry records decode-step loop and prefill truncation "
          "counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordDecodeStepLoops("decode_phased", 5);
  registry.RecordDecodeStepLoops("unified_phased", 2);
  registry.RecordPrefillChunkTruncation("unified_phased", 11);
  registry.RecordPrefillChunkTruncation("unified_phased", 3);
  registry.RecordPrefillChunkTruncation("unified_step", 4);

  auto output = registry.RenderPrometheus();
  REQUIRE(
      output.find(
          "inferflux_scheduler_decode_step_loops_total{mode=\"decode_phased\"} "
          "5") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_decode_step_loops_total{mode=\""
                      "unified_phased\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_prefill_chunk_truncations_total{"
                      "mode=\"unified_phased\"} 2") != std::string::npos);
  REQUIRE(
      output.find("inferflux_scheduler_prefill_chunk_truncated_tokens_total{"
                  "mode=\"unified_phased\"} 14") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_prefill_chunk_truncations_total{"
                      "mode=\"unified_step\"} 1") != std::string::npos);
  REQUIRE(
      output.find("inferflux_scheduler_prefill_chunk_truncated_tokens_total{"
                  "mode=\"unified_step\"} 4") != std::string::npos);
}

TEST_CASE("MetricsRegistry records deferred sequence retirement metrics",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetSchedulerDeferredSequenceRetirements(2);
  registry.RecordSchedulerDeferredSequenceRetirement(17.0);
  registry.SetSchedulerDeferredSequenceRetirements(0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_scheduler_deferred_sequence_retirements{"
                      "backend=\"cpu\"} 0") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_deferred_sequence_retirements_"
                      "completed_total{backend=\"cpu\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("# HELP inferflux_scheduler_sequence_retirement_"
                      "duration_ms") != std::string::npos);
  REQUIRE(output.find("inferflux_scheduler_sequence_retirement_duration_ms_"
                      "count{backend=\"cpu\"} 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry records native down-proj operator selections",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordNativeForwardBatchSize("prefill", 1);
  registry.RecordNativeForwardBatchSize("prefill", 4);
  registry.RecordNativeForwardBatchSize("decode", 2);
  registry.RecordNativeForwardBatchSize("decode", 7);
  registry.RecordNativeFfnProjOperator("prefill", "q8_1_group_hot_q4k");
  registry.RecordNativeFfnProjOperator("prefill", "q8_1_group_row_pair_w4");
  registry.RecordNativeFfnProjOperator("prefill", "q8_1_group_v2");
  registry.RecordNativeFfnProjOperator("prefill", "q8_1_group");
  registry.RecordNativeFfnProjOperator("decode", "packed_group");
  registry.RecordNativeFfnProjOperator("decode", "fallback");
  registry.RecordNativeFfnProjGeometry("prefill", "q8_1_group_hot_q4k", "q4_k",
                                       2, 11008, 2048, 2);
  registry.RecordNativeFfnProjGeometry("prefill", "q8_1_group", "q6_k", 12,
                                       8192, 3072, 2);
  registry.RecordNativeFfnProjGeometry("decode", "packed_group", "mixed", 2,
                                       8192, 3072, 2);
  registry.RecordNativeDownProjOperator("prefill", "q8_1_gemv_v2");
  registry.RecordNativeDownProjOperator("prefill", "q8_1_gemv");
  registry.RecordNativeDownProjOperator("prefill", "q8_1_gemv_hot_fixed");
  registry.RecordNativeDownProjOperator("prefill",
                                        "q8_1_gemv_row_pair_hot_fixed");
  registry.RecordNativeDownProjOperator("prefill", "q8_1_gemv_row_pair_v2");
  registry.RecordNativeDownProjOperator("prefill", "q8_1_gemv_row_pair");
  registry.RecordNativeDownProjOperator("decode", "q8_1_gemv_row_quad");
  registry.RecordNativeDownProjOperator("decode", "packed_gemv");
  registry.RecordNativeDownProjOperator("decode", "mmq");
  registry.RecordNativeDownProjOperator("decode", "fallback");
  registry.RecordNativeDownProjGeometry("prefill", "q8_1_gemv", "q4_k", 12,
                                        3072, 8192);
  registry.RecordNativeDownProjGeometry("prefill", "q8_1_gemv_hot_fixed",
                                        "q4_k", 1, 2048, 11008);
  registry.RecordNativeDownProjGeometry("prefill",
                                        "q8_1_gemv_row_pair_hot_fixed", "q4_k",
                                        2, 2048, 11008);
  registry.RecordNativeDownProjGeometry("prefill", "q8_1_gemv_row_pair",
                                        "q4_k", 2, 2048, 11008);
  registry.RecordNativeDownProjGeometry("decode", "q8_1_gemv_row_quad",
                                        "q6_k", 7, 2048, 11008);
  registry.RecordNativeDownProjGeometry("decode", "packed_gemv", "q4_k", 1,
                                        2048, 11008);
  registry.RecordNativeDownProjGeometry("decode", "mmq", "q6_k", 2, 3072,
                                        8192);
  registry.RecordNativeRowPairSelection("prefill",
                                         "q8_1_group_row_pair_w4", 2);
  registry.RecordNativeRowPairSelection("decode", "q8_1_gemv_row_pair", 4);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("# HELP inferflux_native_forward_batch_size_total") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_native_forward_batch_size_total "
                      "counter") != std::string::npos);
  REQUIRE(output.find("inferflux_native_forward_batch_size_total{phase="
                      "\"prefill\",bucket=\"1\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_native_forward_batch_size_total{phase="
                      "\"prefill\",bucket=\"3_4\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_native_forward_batch_size_total{phase="
                      "\"decode\",bucket=\"2\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_native_forward_batch_size_total{phase="
                      "\"decode\",bucket=\"5_8\"} 1") != std::string::npos);
  REQUIRE(output.find("# HELP inferflux_native_ffn_proj_operator_total") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_native_ffn_proj_operator_total "
                      "counter") != std::string::npos);
  REQUIRE(output.find("# HELP inferflux_native_rowpair_selection_total") !=
          std::string::npos);
  REQUIRE(output.find(
              "inferflux_native_rowpair_selection_total{phase=\"prefill\",operator=\"q8_1_group_row_pair_w4\",bucket=\"2\"} 1") !=
          std::string::npos);
  REQUIRE(output.find(
              "inferflux_native_rowpair_selection_total{phase=\"decode\",operator=\"q8_1_gemv_row_pair\",bucket=\"3_4\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_group_hot_q4k\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_group_row_pair_w4\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_group_v2\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_group\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_operator_total{phase="
                      "\"decode\",operator=\"packed_group\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_operator_total{phase="
                      "\"decode\",operator=\"fallback\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("# HELP inferflux_native_down_proj_operator_total") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_native_down_proj_operator_total "
                      "counter") != std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_v2\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_hot_fixed\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_row_pair_hot_fixed\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_row_pair_v2\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_row_pair\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"decode\",operator=\"q8_1_gemv_row_quad\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"decode\",operator=\"packed_gemv\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"decode\",operator=\"mmq\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"decode\",operator=\"fallback\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_operator_total{phase="
                      "\"prefill\",operator=\"mmq\"} 0") !=
          std::string::npos);
  REQUIRE(output.find("# HELP inferflux_native_ffn_proj_geometry_total") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_native_ffn_proj_geometry_total "
                      "counter") != std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_geometry_total{phase="
                      "\"prefill\",operator=\"q8_1_group_hot_q4k\",quant=\"q4_k\","
                      "m_bucket=\"2\",n=\"11008\",n_bucket=\"8193_16384\","
                      "k=\"2048\",k_bucket=\"1025_2048\",grouped_outputs=\"2\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_geometry_total{phase="
                      "\"prefill\",operator=\"q8_1_group\",quant=\"q6_k\","
                      "m_bucket=\"9_16\",n=\"8192\",n_bucket=\"4097_8192\","
                      "k=\"3072\",k_bucket=\"2049_4096\",grouped_outputs=\"2\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_ffn_proj_geometry_total{phase="
                      "\"decode\",operator=\"packed_group\",quant=\"mixed\","
                      "m_bucket=\"2\",n=\"8192\",n_bucket=\"4097_8192\","
                      "k=\"3072\",k_bucket=\"2049_4096\",grouped_outputs=\"2\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("# HELP inferflux_native_down_proj_geometry_total") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_native_down_proj_geometry_total "
                      "counter") != std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_geometry_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv\",quant=\"q4_k\","
                      "m_bucket=\"9_16\",n=\"3072\",n_bucket=\"2049_4096\","
                      "k=\"8192\",k_bucket=\"4097_8192\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_geometry_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_hot_fixed\",quant=\"q4_k\","
                      "m_bucket=\"1\",n=\"2048\",n_bucket=\"1025_2048\","
                      "k=\"11008\",k_bucket=\"8193_16384\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_geometry_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_row_pair_hot_fixed\",quant=\"q4_k\","
                      "m_bucket=\"2\",n=\"2048\",n_bucket=\"1025_2048\","
                      "k=\"11008\",k_bucket=\"8193_16384\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_geometry_total{phase="
                      "\"prefill\",operator=\"q8_1_gemv_row_pair\",quant=\"q4_k\","
                      "m_bucket=\"2\",n=\"2048\",n_bucket=\"1025_2048\","
                      "k=\"11008\",k_bucket=\"8193_16384\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_geometry_total{phase="
                      "\"decode\",operator=\"q8_1_gemv_row_quad\",quant=\"q6_k\","
                      "m_bucket=\"5_8\",n=\"2048\",n_bucket=\"1025_2048\","
                      "k=\"11008\",k_bucket=\"8193_16384\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_geometry_total{phase="
                      "\"decode\",operator=\"packed_gemv\",quant=\"q4_k\","
                      "m_bucket=\"1\",n=\"2048\",n_bucket=\"1025_2048\","
                      "k=\"11008\",k_bucket=\"8193_16384\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_down_proj_geometry_total{phase="
                      "\"decode\",operator=\"mmq\",quant=\"q6_k\","
                      "m_bucket=\"2\",n=\"3072\",n_bucket=\"2049_4096\","
                      "k=\"8192\",k_bucket=\"4097_8192\"} 1") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry records distributed KV enqueue rejection counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordDisaggKVEnqueueRejected(false);
  registry.RecordDisaggKVEnqueueRejected(false);
  registry.RecordDisaggKVEnqueueRejected(true);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_disagg_kv_enqueue_rejections_total{"
                      "backend=\"cpu\"} 3") != std::string::npos);
  REQUIRE(output.find("inferflux_disagg_kv_enqueue_exhausted_total{"
                      "backend=\"cpu\"} 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry records distributed KV ticket lifecycle counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordDisaggKVTicketStage("enqueued");
  registry.RecordDisaggKVTicketStage("enqueued");
  registry.RecordDisaggKVTicketStage("acknowledged");
  registry.RecordDisaggKVTicketStage("committed");
  registry.RecordDisaggKVTicketStage("timed_out");
  registry.RecordDisaggKVTicketStage("committed");
  registry.RecordDisaggKVTicketStage("unknown");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("# HELP inferflux_disagg_kv_tickets_total") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_disagg_kv_tickets_total counter") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_disagg_kv_tickets_total{backend=\"cpu\","
                      "stage=\"enqueued\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_disagg_kv_tickets_total{backend=\"cpu\","
                      "stage=\"acknowledged\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_disagg_kv_tickets_total{backend=\"cpu\","
                      "stage=\"committed\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_disagg_kv_tickets_total{backend=\"cpu\","
                      "stage=\"timed_out\"} 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry tracks recoverable distributed KV timeout streak",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordDisaggKVTicketStage("timed_out");
  registry.RecordDisaggKVTicketStage("timed_out");
  REQUIRE(registry.GetDisaggKVTimeoutStreak() == 2);
  REQUIRE(registry.GetDisaggKVTimeoutDebt() == 2);

  registry.RecordDisaggKVTicketStage("acknowledged");
  REQUIRE(registry.GetDisaggKVTimeoutStreak() == 2);
  REQUIRE(registry.GetDisaggKVTimeoutDebt() == 2);

  registry.RecordDisaggKVTicketStage("committed");
  REQUIRE(registry.GetDisaggKVTimeoutStreak() == 0);
  REQUIRE(registry.GetDisaggKVTimeoutDebt() == 1);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("# HELP inferflux_disagg_kv_timeout_streak") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_disagg_kv_timeout_streak gauge") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_disagg_kv_timeout_streak{backend=\"cpu\"} 0") !=
          std::string::npos);
  REQUIRE(output.find("# HELP inferflux_disagg_kv_timeout_debt") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_disagg_kv_timeout_debt gauge") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_disagg_kv_timeout_debt{backend=\"cpu\"} 1") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry exposes distributed KV counters via getters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordDisaggKVEnqueueRejected(false);
  registry.RecordDisaggKVEnqueueRejected(true);
  registry.RecordDisaggKVTicketStage("enqueued");
  registry.RecordDisaggKVTicketStage("acknowledged");
  registry.RecordDisaggKVTicketStage("committed");
  registry.RecordDisaggKVTicketStage("timed_out");
  registry.RecordDisaggKVTicketStage("committed");

  REQUIRE(registry.GetDisaggKVEnqueueRejections() == 2);
  REQUIRE(registry.GetDisaggKVEnqueueExhausted() == 1);
  REQUIRE(registry.GetDisaggKVTicketsEnqueued() == 1);
  REQUIRE(registry.GetDisaggKVTicketsAcknowledged() == 1);
  REQUIRE(registry.GetDisaggKVTicketsCommitted() == 2);
  REQUIRE(registry.GetDisaggKVTicketsTimedOut() == 1);
  REQUIRE(registry.GetDisaggKVTimeoutDebt() == 0);
}

TEST_CASE("MetricsRegistry records per-model token counters", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordModelRoute("llama3-8b", "cuda", true);
  registry.RecordModelTokens("llama3-8b", "cuda", 12, 34);
  registry.RecordModelTokens("llama3-8b", "", 3, 6);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_model_prompt_tokens_total{model=\"llama3-8b\","
                      "backend=\"cuda\"} 15") != std::string::npos);
  REQUIRE(output.find("inferflux_model_completion_tokens_total{model=\"llama3-"
                      "8b\",backend=\"cuda\"} 40") != std::string::npos);
}

TEST_CASE("MetricsRegistry per-model counters fall back to instance backend",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetBackend("mps");
  registry.RecordModelTokens("qwen2.5-7b", "", 5, 9);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_model_prompt_tokens_total{model=\"qwen2.5-"
                      "7b\",backend=\"mps\"} 5") != std::string::npos);
  REQUIRE(output.find("inferflux_model_completion_tokens_total{model=\"qwen2.5-"
                      "7b\",backend=\"mps\"} 9") != std::string::npos);
}

TEST_CASE("MetricsRegistry records capability rejection counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordCapabilityRejection("cuda", "vision");
  registry.RecordCapabilityRejection("cuda", "vision");
  registry.RecordCapabilityRejection("cpu", "structured_output");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_capability_rejections_total{backend=\"cuda\","
                      "feature=\"vision\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_capability_rejections_total{backend=\"cpu\","
                      "feature=\"structured_output\"} 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry records backend exposure counters", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordBackendExposure("cuda", "cuda", "llama_cpp", false);
  registry.RecordBackendExposure("cuda", "cpu", "llama_cpp", true);
  registry.RecordBackendExposure("cuda", "cpu", "llama_cpp", true);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_backend_exposures_total{requested_backend="
                      "\"cuda\",exposed_backend=\"cuda\",provider="
                      "\"llama_cpp\",fallback=\"false\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_backend_exposures_total{requested_backend="
                      "\"cuda\",exposed_backend=\"cpu\",provider=\"llama_cpp\","
                      "fallback=\"true\"} 2") != std::string::npos);
}

TEST_CASE("MetricsRegistry records capability route fallback counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordCapabilityRouteFallback("cuda", "cpu", "logprobs");
  registry.RecordCapabilityRouteFallback("cuda", "cpu", "logprobs");
  registry.RecordCapabilityRouteFallback("rocm", "cpu", "backend_unavailable");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_capability_route_fallbacks_total{"
                      "from_backend=\"cuda\",to_backend=\"cpu\","
                      "feature=\"logprobs\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_capability_route_fallbacks_total{"
                      "from_backend=\"rocm\",to_backend=\"cpu\","
                      "feature=\"backend_unavailable\"} 1") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry records CUDA lane runtime metrics", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordCudaLaneSubmission(true);
  registry.RecordCudaLaneSubmission(false);
  registry.RecordCudaLaneSubmission(false);
  registry.RecordCudaLaneCompletion(true);
  registry.RecordCudaLaneCompletion(false);
  registry.RecordCudaLaneExecutionStart(true);
  registry.RecordCudaLaneExecutionStart(false);
  registry.RecordCudaLaneExecutionStop(false);
  registry.RecordCudaLaneExecutionStop(true);
  registry.SetCudaLaneQueueDepth(true, 3);
  registry.SetCudaLaneQueueDepth(false, 7);
  registry.RecordCudaLaneEnqueueReject(true);
  registry.RecordCudaLaneEnqueueReject(false);
  registry.RecordCudaLaneCollectTimeout(false);
  registry.RecordCudaLaneWorkerRestart();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_lane_submissions_total{lane=\"decode\"} "
                      "1") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_submissions_total{lane=\"prefill\"} "
                      "2") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_completions_total{lane=\"decode\"} "
                      "1") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_completions_total{lane=\"prefill\"} "
                      "1") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_queue_depth{lane=\"decode\"} 3") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_queue_depth{lane=\"prefill\"} 7") !=
          std::string::npos);
  REQUIRE(output.find(
              "inferflux_cuda_lane_enqueue_rejects_total{lane=\"decode\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_enqueue_rejects_total{lane="
                      "\"prefill\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_collect_timeouts_total{lane="
                      "\"decode\"} 0") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_collect_timeouts_total{lane="
                      "\"prefill\"} 1") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_worker_restarts_total 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_overlap_events_total 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_overlap_duration_ms_total ") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_overlap_active 0") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_inflight{lane=\"decode\"} 0") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_cuda_lane_inflight{lane=\"prefill\"} 0") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry exports selected CUDA attention kernel gauge",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetCudaAttentionKernel("fa2");

  auto output = registry.RenderPrometheus();
  REQUIRE(
      output.find("inferflux_cuda_attention_kernel_selected{kernel=\"fa3\"} "
                  "0") != std::string::npos);
  REQUIRE(
      output.find("inferflux_cuda_attention_kernel_selected{kernel=\"fa2\"} "
                  "1") != std::string::npos);
  REQUIRE(output.find(
              "inferflux_cuda_attention_kernel_selected{kernel=\"standard\"} "
              "0") != std::string::npos);
}

TEST_CASE(
    "MetricsRegistry records CUDA attention kernel fallbacks and switches",
    "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordCudaAttentionKernelFallback("fa3", "fa2", "fa3_unavailable");
  registry.RecordCudaAttentionKernelFallback("fa3", "fa2", "fa3_unavailable");
  registry.RecordCudaAttentionKernelSwitch("standard", "fa2");
  registry.RecordCudaAttentionKernelSwitch("fa2", "fa2");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_cuda_attention_kernel_fallbacks_total{"
                      "requested=\"fa3\",selected=\"fa2\","
                      "reason=\"fa3_unavailable\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_cuda_attention_kernel_switches_total{"
                      "from_kernel=\"standard\",to_kernel=\"fa2\"} 1") !=
          std::string::npos);
}
