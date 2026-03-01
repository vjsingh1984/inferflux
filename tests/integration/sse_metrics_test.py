#!/usr/bin/env python3

import http.client
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest


SERVER_PORT = 18080
BASE_URL = f"127.0.0.1:{SERVER_PORT}"


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_SERVER_BIN = os.environ.get("INFERFLUX_SERVER_BIN", os.path.join(_ROOT, "build", "inferfluxd"))


def start_server(env):
    proc = subprocess.Popen(
        [
            _SERVER_BIN,
            "--config",
            "config/server.yaml",
        ],
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_line(proc, needle, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if line == "":
            time.sleep(0.2)
            continue
        if needle in line:
            return True
    return False


class InferFluxIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        env = os.environ.copy()
        env["INFERFLUX_MODEL_PATH"] = env.get("INFERFLUX_MODEL_PATH", "")
        env["INFERFLUX_RATE_LIMIT_PER_MINUTE"] = "2"
        env["INFERFLUX_GUARDRAIL_BLOCKLIST"] = "secret"
        env["INFERFLUX_AUDIT_LOG"] = os.path.join(cls.tmpdir.name, "audit.log")
        env["INFERFLUX_MPS_LAYERS"] = "0"
        env["INFERCTL_API_KEY"] = "dev-key-123"
        env["INFERFLUX_OIDC_ISSUER"] = ""
        env["INFERFLUX_OIDC_AUDIENCE"] = ""
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT)
        cls.proc = start_server(env)
        ready = wait_for_line(cls.proc, "listening on")
        if not ready:
            raise RuntimeError("Server did not start in time")
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        if cls.proc:
            cls.proc.terminate()
            cls.proc.wait(timeout=5)
        cls.tmpdir.cleanup()

    def http_get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        conn.request("GET", path, headers={"Authorization": "Bearer dev-key-123"})
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def http_post(self, path, payload, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        hdrs = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        if headers:
            hdrs.update(headers)
        conn.request("POST", path, body=json.dumps(payload), headers=hdrs)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def test_metrics_endpoint(self):
        resp, body = self.http_get("/metrics")
        self.assertEqual(resp.status, 200)
        self.assertIn("inferflux_requests_total", body)

    def test_guardrail_blocks_prompt(self):
        payload = {"prompt": "this contains secret", "max_tokens": 8}
        resp, body = self.http_post("/v1/completions", payload)
        self.assertEqual(resp.status, 400)
        self.assertIn("Blocked content keyword", body)

    def test_rate_limit(self):
        payload = {"prompt": "hello", "max_tokens": 8}
        for _ in range(2):
            resp, _ = self.http_post("/v1/completions", payload)
            self.assertEqual(resp.status, 200)
        resp, body = self.http_post("/v1/completions", payload)
        self.assertEqual(resp.status, 429)
        self.assertIn("rate_limited", body)

    def test_sse_streaming(self):
        conn = http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=5)
        payload = {
            "model": "tinyllama",
            "max_tokens": 8,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("POST", "/v1/chat/completions", body=json.dumps(payload), headers=headers)
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        body = resp.read().decode()
        conn.close()
        self.assertIn("data:", body)
        self.assertIn("[DONE]", body)


if __name__ == "__main__":
    unittest.main()
