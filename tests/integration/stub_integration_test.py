#!/usr/bin/env python3
"""
Stub-mode integration tests — run without INFERFLUX_MODEL_PATH.

These tests verify the server's auth, health, metrics, guardrail, rate-limit,
and JSON-mode layers without requiring a loaded model. They form the CI smoke
suite and are registered as StubIntegration in CMakeLists.txt.
"""

import http.client
import json
import os
import subprocess
import tempfile
import time
import unittest

SERVER_PORT = 18081  # Different from sse_metrics_test (18080) to avoid conflicts.
API_KEY = "stub-test-key-456"
# Dedicated key used only by test_rate_limit_enforced so its exhausted bucket
# does not affect other tests (rate limiter tracks per-key buckets).
RATE_LIMIT_KEY = "rate-limit-exhaustion-key-789"


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_SERVER_BIN = os.environ.get("INFERFLUX_SERVER_BIN", os.path.join(_ROOT, "build", "inferfluxd"))


def start_server(env):
    proc = subprocess.Popen(
        [_SERVER_BIN, "--config", "config/server.yaml"],
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_ready(proc, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        if "listening on" in line:
            return True
    return False


class StubIntegrationTests(unittest.TestCase):
    """Tests that pass without a loaded model."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        env = os.environ.copy()
        env["INFERFLUX_MODEL_PATH"] = ""          # explicitly no model
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT)
        env["INFERCTL_API_KEY"] = API_KEY
        env["INFERFLUX_API_KEYS"] = RATE_LIMIT_KEY  # separate key for rate-limit test
        env["INFERFLUX_RATE_LIMIT_PER_MINUTE"] = "30"
        env["INFERFLUX_GUARDRAIL_BLOCKLIST"] = "forbidden"
        env["INFERFLUX_AUDIT_LOG"] = os.path.join(cls.tmpdir.name, "audit.jsonl")
        env["INFERFLUX_OIDC_ISSUER"] = ""
        env["INFERFLUX_OIDC_AUDIENCE"] = ""
        env["INFERFLUX_MPS_LAYERS"] = "0"
        cls.proc = start_server(env)
        if not wait_for_ready(cls.proc):
            cls.proc.terminate()
            raise RuntimeError("Server did not start in time")
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        if cls.proc:
            cls.proc.terminate()
            cls.proc.wait(timeout=5)
        cls.tmpdir.cleanup()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get(self, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        hdrs = {"Authorization": f"Bearer {API_KEY}"}
        if headers:
            hdrs.update(headers)
        conn.request("GET", path, headers=hdrs)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _post(self, path, payload, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        if headers:
            hdrs.update(headers)
        conn.request("POST", path, body=json.dumps(payload), headers=hdrs)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    # ── health probes ─────────────────────────────────────────────────────────

    def test_livez_always_200(self):
        """GET /livez must return 200 regardless of model state."""
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request("GET", "/livez")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)

    def test_healthz_returns_json(self):
        """GET /healthz must return JSON with a status field."""
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        self.assertIn(resp.status, (200, 503))
        data = json.loads(body)
        self.assertIn("status", data)

    def test_readyz_without_model_is_not_200(self):
        """/readyz should signal not-ready when no model is loaded."""
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request("GET", "/readyz")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        # 503 (degraded) expected — no model loaded.
        self.assertNotEqual(resp.status, 200)

    # ── authentication ────────────────────────────────────────────────────────

    def test_missing_auth_header_rejected(self):
        """POST without Authorization header must be rejected."""
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps({"prompt": "hi", "max_tokens": 4}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertIn(resp.status, (401, 403))

    def test_wrong_api_key_rejected(self):
        """POST with a bad API key must be rejected."""
        resp, _ = self._post(
            "/v1/completions",
            {"prompt": "hi", "max_tokens": 4},
            headers={"Authorization": "Bearer totally-wrong-key"},
        )
        self.assertIn(resp.status, (401, 403))

    # ── metrics ───────────────────────────────────────────────────────────────

    def test_metrics_endpoint_200(self):
        """GET /metrics must return 200 with Prometheus text."""
        resp, body = self._get("/metrics")
        self.assertEqual(resp.status, 200)
        self.assertIn("inferflux_requests_total", body)

    def test_metrics_contains_histogram(self):
        """Metrics output must include the latency histogram."""
        resp, body = self._get("/metrics")
        self.assertEqual(resp.status, 200)
        self.assertIn("inferflux_request_duration_ms_bucket", body)
        self.assertIn('le="+Inf"', body)

    def test_metrics_contains_queue_depth(self):
        """Metrics output must include the queue depth gauge."""
        resp, body = self._get("/metrics")
        self.assertEqual(resp.status, 200)
        self.assertIn("inferflux_scheduler_queue_depth", body)

    # ── guardrails ────────────────────────────────────────────────────────────

    def test_guardrail_blocks_blocklisted_word(self):
        """POST with a prompt containing a blocked keyword must be rejected."""
        resp, body = self._post(
            "/v1/completions",
            {"prompt": "the password is forbidden", "max_tokens": 4},
        )
        self.assertEqual(resp.status, 400)
        self.assertIn("Blocked", body)

    def test_guardrail_passes_clean_prompt(self):
        """POST with a clean prompt must not be blocked by guardrails."""
        resp, body = self._post(
            "/v1/completions",
            {"prompt": "tell me about cats", "max_tokens": 4},
        )
        # Without a model we expect 200 with the no-backend stub message.
        self.assertEqual(resp.status, 200)
        data = json.loads(body)
        self.assertIn("choices", data)

    # ── no-backend stub ───────────────────────────────────────────────────────

    def test_no_backend_returns_200_with_stub_message(self):
        """Without a model loaded the server must still return 200 with a stub completion."""
        resp, body = self._post(
            "/v1/completions",
            {"prompt": "what is 2+2?", "max_tokens": 8},
        )
        self.assertEqual(resp.status, 200)
        data = json.loads(body)
        self.assertIn("choices", data)
        text = data["choices"][0]["text"]
        self.assertIn("No model backend", text)

    def test_no_backend_chat_completions_stub(self):
        """Chat completions endpoint also returns 200 with stub message when no model."""
        resp, body = self._post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4},
        )
        self.assertEqual(resp.status, 200)
        data = json.loads(body)
        self.assertIn("choices", data)

    # ── JSON mode (§2.1) ──────────────────────────────────────────────────────

    def test_json_mode_response_is_valid_json(self):
        """response_format: json_object must wrap the completion in valid JSON."""
        resp, body = self._post(
            "/v1/completions",
            {
                "prompt": "give me a JSON object",
                "max_tokens": 16,
                "response_format": {"type": "json_object"},
            },
        )
        self.assertEqual(resp.status, 200)
        outer = json.loads(body)
        # The completion text itself must be valid JSON.
        completion_text = outer["choices"][0]["text"]
        inner = json.loads(completion_text)  # raises if not valid JSON
        self.assertIsInstance(inner, (dict, list))

    # ── CORS ─────────────────────────────────────────────────────────────────

    def test_options_preflight_returns_cors_headers(self):
        """OPTIONS must return CORS headers for browser compatibility."""
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request("OPTIONS", "/v1/completions")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        # RFC 7231 allows 200 or 204 for OPTIONS preflight.
        self.assertIn(resp.status, (200, 204))
        headers_lower = {k.lower(): v for k, v in resp.getheaders()}
        self.assertIn("access-control-allow-origin", headers_lower)

    # ── rate limiting ─────────────────────────────────────────────────────────

    def test_rate_limit_enforced(self):
        """After exhausting the per-minute limit the server must return 429."""
        payload = {"prompt": "ping", "max_tokens": 4}
        # Use a dedicated key so exhausting its bucket does not affect other tests.
        # The limit is 30/min, so within 35 requests we must see a 429.
        rl_headers = {"Authorization": f"Bearer {RATE_LIMIT_KEY}"}
        got_429 = False
        for _ in range(35):
            resp, body = self._post("/v1/completions", payload, headers=rl_headers)
            if resp.status == 429:
                self.assertIn("rate_limited", body)
                got_429 = True
                break
        self.assertTrue(got_429, "Expected a 429 response within 35 requests")

    # ── tool calling (§2.3) ───────────────────────────────────────────────────

    def test_tool_calling_request_does_not_break_server(self):
        """POST with tools array must return a valid completion response."""
        resp, body = self._post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "max_tokens": 8,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        # Without a model the server returns a stub completion — structure must be valid.
        self.assertEqual(resp.status, 200)
        data = json.loads(body)
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)

    def test_tool_choice_none_skips_tool_injection(self):
        """tool_choice=none must bypass tool injection and return a normal completion."""
        resp, body = self._post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "do_something", "description": "Does something"},
                    }
                ],
                "tool_choice": "none",
            },
        )
        self.assertEqual(resp.status, 200)
        data = json.loads(body)
        self.assertIn("choices", data)
        # With tool_choice=none the response must NOT contain tool_calls.
        msg = data["choices"][0].get("message", {})
        self.assertNotIn("tool_calls", msg)

    def test_streaming_tool_call_emits_tool_calls_deltas(self):
        """stream=true + tools must emit structured tool_calls deltas, not raw JSON tokens.

        Verifies the §2.3 streaming delta sequence:
          1. A chunk with delta.role="assistant" and content=null
          2. A chunk with delta.tool_calls[0].function.name set
          3. A chunk with finish_reason="tool_calls" (not "stop")
        """
        resp, body = self._post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "max_tokens": 8,
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        self.assertEqual(resp.status, 200)
        # Parse SSE chunks: lines starting with "data: " contain JSON payloads.
        chunks = []
        for line in body.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        self.assertGreater(len(chunks), 0, "Expected at least one SSE chunk")

        # Collect delta fields across all chunks.
        found_role_chunk = any(
            chunk.get("choices", [{}])[0].get("delta", {}).get("role") == "assistant"
            for chunk in chunks
        )
        found_tool_calls_delta = any(
            "tool_calls" in chunk.get("choices", [{}])[0].get("delta", {})
            for chunk in chunks
        )
        finish_reasons = [
            chunk.get("choices", [{}])[0].get("finish_reason")
            for chunk in chunks
        ]
        self.assertTrue(found_role_chunk, "Expected a delta with role=assistant")
        self.assertTrue(found_tool_calls_delta, "Expected a delta with tool_calls")
        self.assertIn("tool_calls", finish_reasons, "Expected finish_reason=tool_calls")
        self.assertNotIn("stop", finish_reasons, "finish_reason must not be 'stop' for tool calls")

        # Verify the function name appears in one of the tool_calls delta chunks.
        names_found = []
        for chunk in chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            for tc in delta.get("tool_calls", []):
                fn = tc.get("function", {}).get("name", "")
                if fn:
                    names_found.append(fn)
        self.assertTrue(
            any("get_weather" in n for n in names_found),
            f"Expected 'get_weather' in tool_calls delta; got: {names_found}",
        )

    # ── SSE streaming framing (covers IntegrationSSE gap) ────────────────────

    def _parse_sse_chunks(self, body):
        """Return parsed JSON objects from SSE body (skips [DONE] sentinel)."""
        chunks = []
        for line in body.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        return chunks

    def test_sse_streaming_response_is_properly_framed(self):
        """stream=true without tools must return text/event-stream with valid delta chunks.

        This test covers the SSE framing gap that IntegrationSSE validates with a real
        model.  In stub mode the scheduler returns a completion without a backend; the
        no_backend streaming path (fixed alongside §2.3) now correctly emits:
          1.  Content delta chunk
          2.  Stop chunk (finish_reason="stop")
          3.  data: [DONE] sentinel
        rather than a second HTTP response that would corrupt the framing.
        """
        resp, body = self._post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Hello, world"}],
                "max_tokens": 8,
                "stream": True,
            },
        )
        self.assertEqual(resp.status, 200)
        content_type = resp.getheader("Content-Type", "")
        self.assertIn("text/event-stream", content_type,
                      f"Expected text/event-stream; got: {content_type}")
        self.assertIn("data: [DONE]", body, "SSE sentinel missing from streaming response")

        chunks = self._parse_sse_chunks(body)
        self.assertGreater(len(chunks), 0, "Expected at least one SSE data chunk")

        finish_reasons = [c.get("choices", [{}])[0].get("finish_reason") for c in chunks]
        self.assertIn("stop", finish_reasons, "Expected finish_reason=stop in stream")

        # Every chunk must carry a 'choices' array with a 'delta'.
        for i, chunk in enumerate(chunks):
            self.assertIn("choices", chunk, f"chunk[{i}] missing 'choices'")
            self.assertIn("delta", chunk["choices"][0], f"chunk[{i}]['choices'][0] missing 'delta'")

    def test_sse_streaming_content_delta_carries_text(self):
        """Content delta chunks must contain non-empty text (stub completion text)."""
        resp, body = self._post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Ping"}],
                "max_tokens": 4,
                "stream": True,
            },
        )
        self.assertEqual(resp.status, 200)
        chunks = self._parse_sse_chunks(body)
        content_chunks = [
            c for c in chunks
            if c.get("choices", [{}])[0].get("delta", {}).get("content")
        ]
        self.assertGreater(len(content_chunks), 0,
                           "Expected at least one chunk with non-empty delta.content")

    def test_sse_streaming_server_remains_healthy_after_stream(self):
        """Server must continue serving /livez after a streaming response completes."""
        self._post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "test"}], "max_tokens": 4, "stream": True},
        )
        # /livez is always 200 regardless of model state (unlike /healthz which
        # returns 503 when no model is loaded).
        resp, _ = self._get("/livez")
        self.assertEqual(resp.status, 200)

    def test_sse_streaming_metrics_endpoint_still_reachable(self):
        """Metrics endpoint must remain responsive after a streaming request.

        In stub mode, no_backend streaming calls RecordError() (not RecordSuccess),
        so inferflux_errors_total is the counter that increments.  The test verifies
        the metrics endpoint stays reachable and that at least one known counter line
        appears in the output — without asserting exact delta counts that depend on
        test execution order.
        """
        # Fire a streaming request.
        self._post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "metrics check"}], "max_tokens": 4, "stream": True},
        )
        resp, body = self._get("/metrics")
        self.assertEqual(resp.status, 200)
        # inferflux_errors_total is the counter incremented by the no_backend path.
        self.assertIn("inferflux_errors_total", body,
                      "inferflux_errors_total counter missing from /metrics after streaming request")


if __name__ == "__main__":
    unittest.main()
