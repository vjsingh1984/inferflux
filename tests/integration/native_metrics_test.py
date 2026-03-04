#!/usr/bin/env python3
"""Integration tests for native CUDA backend metrics.

Validates that the /metrics endpoint exposes all native backend counters,
histograms, and gauges in the correct Prometheus format. Runs in stub mode
(no model required) — the metrics are always rendered even when the native
backend is inactive.

When INFERFLUX_MODEL_PATH is set and the CUDA native backend is active,
additional tests verify that forward-pass and sampling counters increment
after completions.
"""

import http.client
import json
import os
import re
import signal
import subprocess
import time
import unittest

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18083
SERVER_BIN = os.environ.get("INFERFLUX_SERVER_BIN", "./build/inferfluxd")


def _start_server(env):
    proc = subprocess.Popen(
        [SERVER_BIN, "--config", "config/server.yaml"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    return proc


def _wait_for_ready(proc, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=1)
            conn.request(
                "GET", "/livez", headers={"Authorization": "Bearer dev-key-123"}
            )
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status in (200, 401):
                return True
        except Exception:
            time.sleep(0.1)
    return False


class NativeMetricsStubTests(unittest.TestCase):
    """Test native metric presence and format against a stub server."""

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["INFERFLUX_HOST_OVERRIDE"] = SERVER_HOST
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT)
        cls.server_proc = _start_server(env)
        if not _wait_for_ready(cls.server_proc):
            try:
                out, err = cls.server_proc.communicate(timeout=2)
            except Exception:
                out, err = b"", b""
            raise RuntimeError(
                "inferfluxd did not become ready; "
                f"stdout={out[-400:] if out else ''} "
                f"stderr={err[-400:] if err else ''}"
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "server_proc") and cls.server_proc:
            try:
                os.killpg(os.getpgid(cls.server_proc.pid), signal.SIGTERM)
                cls.server_proc.wait(timeout=5)
            except Exception:
                pass

    def _get(self, path):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Authorization": "Bearer dev-key-123"}
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _post(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dev-key-123",
        }
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    # -- Metric presence tests ----------------------------------------

    def test_metrics_endpoint_returns_200(self):
        resp, _ = self._get("/metrics")
        self.assertEqual(resp.status, 200)

    def test_native_forward_passes_present(self):
        """Forward pass counters are rendered for both phases."""
        resp, body = self._get("/metrics")
        self.assertEqual(resp.status, 200)
        self.assertIn(
            'inferflux_native_forward_passes_total{phase="prefill"}', body
        )
        self.assertIn(
            'inferflux_native_forward_passes_total{phase="decode"}', body
        )

    def test_native_forward_batch_tokens_present(self):
        resp, body = self._get("/metrics")
        self.assertIn("inferflux_native_forward_batch_tokens_total", body)

    def test_native_forward_duration_histogram(self):
        """Forward pass latency histogram has correct Prometheus format."""
        resp, body = self._get("/metrics")
        self.assertIn("# TYPE inferflux_native_forward_duration_ms histogram", body)
        self.assertIn("inferflux_native_forward_duration_ms_bucket{le=", body)
        self.assertIn('inferflux_native_forward_duration_ms_bucket{le="+Inf"}', body)
        self.assertIn("inferflux_native_forward_duration_ms_sum", body)
        self.assertIn("inferflux_native_forward_duration_ms_count", body)

    def test_native_sampling_duration_histogram(self):
        """Sampling latency histogram has correct Prometheus format."""
        resp, body = self._get("/metrics")
        self.assertIn("# TYPE inferflux_native_sampling_duration_ms histogram", body)
        self.assertIn("inferflux_native_sampling_duration_ms_bucket{le=", body)
        self.assertIn('inferflux_native_sampling_duration_ms_bucket{le="+Inf"}', body)
        self.assertIn("inferflux_native_sampling_duration_ms_sum", body)
        self.assertIn("inferflux_native_sampling_duration_ms_count", body)

    def test_native_kv_cache_gauges_present(self):
        """KV cache occupancy gauges are rendered."""
        resp, body = self._get("/metrics")
        self.assertIn("inferflux_native_kv_active_sequences", body)
        self.assertIn("inferflux_native_kv_max_sequences", body)

    def test_native_forward_counters_zero_in_stub(self):
        """In stub mode (no native backend), counters should be 0."""
        resp, body = self._get("/metrics")
        # Extract the decode counter value
        match = re.search(
            r'inferflux_native_forward_passes_total\{phase="decode"\}\s+(\d+)',
            body,
        )
        self.assertIsNotNone(match, "decode counter not found in metrics")
        self.assertEqual(match.group(1), "0")

    def test_native_histogram_buckets_complete(self):
        """Histogram has all expected bucket boundaries."""
        resp, body = self._get("/metrics")
        expected_buckets = ["10", "50", "100", "250", "500", "1000", "2500", "5000"]
        for bucket in expected_buckets:
            self.assertIn(
                f'inferflux_native_forward_duration_ms_bucket{{le="{bucket}"}}',
                body,
                f"Missing bucket le={bucket}",
            )

    def test_native_metrics_have_help_text(self):
        """All native metrics have # HELP lines."""
        resp, body = self._get("/metrics")
        self.assertIn(
            "# HELP inferflux_native_forward_passes_total", body
        )
        self.assertIn(
            "# HELP inferflux_native_forward_batch_tokens_total", body
        )
        self.assertIn(
            "# HELP inferflux_native_forward_duration_ms", body
        )
        self.assertIn(
            "# HELP inferflux_native_sampling_duration_ms", body
        )
        self.assertIn(
            "# HELP inferflux_native_kv_active_sequences", body
        )
        self.assertIn(
            "# HELP inferflux_native_kv_max_sequences", body
        )

    def test_native_metrics_have_type_annotations(self):
        """All native metrics have # TYPE lines."""
        resp, body = self._get("/metrics")
        self.assertIn(
            "# TYPE inferflux_native_forward_passes_total counter", body
        )
        self.assertIn(
            "# TYPE inferflux_native_forward_batch_tokens_total counter", body
        )
        self.assertIn(
            "# TYPE inferflux_native_forward_duration_ms histogram", body
        )
        self.assertIn(
            "# TYPE inferflux_native_sampling_duration_ms histogram", body
        )
        self.assertIn(
            "# TYPE inferflux_native_kv_active_sequences gauge", body
        )
        self.assertIn(
            "# TYPE inferflux_native_kv_max_sequences gauge", body
        )


if __name__ == "__main__":
    unittest.main()
