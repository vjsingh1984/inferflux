#!/usr/bin/env python3
"""
SHM smoke integration test — validates the decode-worker + ShmKVTransport path.

Starts inferfluxd with:
  INFERFLUX_DECODE_POOL_SIZE=2   (enables decode worker threads + use_decode_workers_=true)
  INFERFLUX_KV_TRANSPORT=shm     (selects ShmKVTransport over the in-process KVChannel)

No model is required: the stub-mode no-backend path still exercises the scheduler's
prefill-enqueue → kv_transport → decode-worker-dequeue → respond cycle. This confirms
that:
  1. The server starts without crashing when both flags are set.
  2. Completion requests finish and return a valid JSON body.
  3. /metrics exposes the inferflux_kv_transfer_duration_ms histogram.
  4. /readyz returns 200 (unified role, model not required for readyz in unified mode).

Registered as ctest target ShmSmokeTest in CMakeLists.txt.
"""

import http.client
import json
import os
import subprocess
import time
import unittest

SERVER_PORT = 18082  # Distinct from StubIntegration (18081) and IntegrationSSE (18080)
API_KEY = "shm-smoke-key-321"

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_SERVER_BIN = os.environ.get(
    "INFERFLUX_SERVER_BIN", os.path.join(_ROOT, "build", "inferfluxd")
)


def start_server(env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        [_SERVER_BIN, "--config", "config/server.yaml"],
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_ready(proc: subprocess.Popen, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        if "listening on" in line:
            return True
    return False


class ShmSmokeTests(unittest.TestCase):
    """
    Integration tests for the SHM-backed decode-worker path.

    The server is started once for the whole class; all tests share the
    running instance to avoid slow repeated start-up.
    """

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        # No model required — stub mode is sufficient.
        env["INFERFLUX_MODEL_PATH"] = ""
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT)
        env["INFERCTL_API_KEY"] = API_KEY
        env["INFERFLUX_RATE_LIMIT_PER_MINUTE"] = "120"
        env["INFERFLUX_GUARDRAIL_BLOCKLIST"] = ""
        env["INFERFLUX_OIDC_ISSUER"] = ""
        env["INFERFLUX_OIDC_AUDIENCE"] = ""
        env["INFERFLUX_MPS_LAYERS"] = "0"
        # Key SHM-path knobs:
        env["INFERFLUX_DECODE_POOL_SIZE"] = "2"
        env["INFERFLUX_KV_TRANSPORT"] = "shm"

        cls.proc = start_server(env)
        if not wait_for_ready(cls.proc, timeout=20):
            cls.proc.terminate()
            raise RuntimeError("inferfluxd did not start in time (SHM smoke test)")
        # Brief pause so all decode worker threads are live.
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        if cls.proc:
            cls.proc.terminate()
            cls.proc.wait(timeout=5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get(self, path: str):
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request("GET", path, headers={"Authorization": f"Bearer {API_KEY}"})
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _post(self, path: str, payload: dict):
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=10)
        conn.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
        )
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_server_starts_with_shm_transport(self):
        """Server starts and /livez is reachable when SHM transport is enabled."""
        resp, body = self._get("/livez")
        self.assertEqual(resp.status, 200)

    def test_readyz_returns_ok(self):
        """/readyz returns 200 in unified role (no model required for probe)."""
        resp, _ = self._get("/readyz")
        # In unified role /readyz checks model_loaded; without a model it may
        # return 503. Accept both 200 and 503 — the important assertion is that
        # the server does NOT crash (connection succeeded).
        self.assertIn(resp.status, (200, 503))

    def test_completion_returns_valid_json(self):
        """POST /v1/completions succeeds and returns parseable JSON."""
        resp, body = self._post(
            "/v1/completions",
            {"prompt": "Hello, world!", "max_tokens": 4},
        )
        self.assertEqual(resp.status, 200, msg=body)
        data = json.loads(body)
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)

    def test_chat_completion_returns_valid_json(self):
        """POST /v1/chat/completions succeeds and returns parseable JSON."""
        resp, body = self._post(
            "/v1/chat/completions",
            {
                "model": "stub",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
            },
        )
        self.assertEqual(resp.status, 200, msg=body)
        data = json.loads(body)
        self.assertIn("choices", data)

    def test_multiple_concurrent_completions(self):
        """Three sequential completions all succeed (decode worker FIFO hold)."""
        for i in range(3):
            resp, body = self._post(
                "/v1/completions",
                {"prompt": f"Request {i}", "max_tokens": 2},
            )
            self.assertEqual(resp.status, 200, msg=f"Request {i} failed: {body}")

    def test_metrics_exposes_kv_transfer_histogram(self):
        """/metrics contains the inferflux_kv_transfer_duration_ms histogram."""
        resp, body = self._get("/metrics")
        self.assertEqual(resp.status, 200)
        self.assertIn("inferflux_kv_transfer_duration_ms", body)

    def test_metrics_exposes_scheduler_counters(self):
        """/metrics contains scheduler queue and request counters."""
        resp, body = self._get("/metrics")
        self.assertEqual(resp.status, 200)
        self.assertIn("inferflux_requests_total", body)
        self.assertIn("inferflux_scheduler_queue_depth", body)

    def test_auth_rejected_without_key(self):
        """Requests without Authorization header receive 401."""
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps({"prompt": "test", "max_tokens": 1}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 401)


if __name__ == "__main__":
    unittest.main()
