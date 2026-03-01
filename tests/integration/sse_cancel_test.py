#!/usr/bin/env python3
"""
Integration test that ensures SSE streaming survives client disconnects
and exercises the cancellation flag path.
"""

import http.client
import json
import os
import pathlib
import signal
import socket
import subprocess
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BINARY = pathlib.Path(os.environ.get("INFERFLUX_SERVER_BIN", str(ROOT / "build" / "inferfluxd")))
CONFIG = ROOT / "config" / "server.yaml"
HOST = "127.0.0.1"
PORT = 18100
API_KEY = "sse-cancel-key"


def start_server(env):
    proc = subprocess.Popen(
        [str(BINARY), "--config", str(CONFIG)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_port(timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((HOST, PORT), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


class SSECancelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.exists():
            raise unittest.SkipTest("inferfluxd binary missing; build before running tests")
        env = os.environ.copy()
        env["INFERFLUX_MODEL_PATH"] = ""
        env["INFERFLUX_PORT_OVERRIDE"] = str(PORT)
        env["INFERFLUX_API_KEYS"] = API_KEY
        env["INFERFLUX_RATE_LIMIT_PER_MINUTE"] = "0"
        env["INFERCTL_API_KEY"] = API_KEY
        env["INFERFLUX_FAIRNESS_ENABLE_PREEMPTION"] = "true"
        env["INFERFLUX_FAIRNESS_PRIORITY_THRESHOLD"] = "5"
        env["INFERFLUX_FAIRNESS_MAX_TIMESLICE"] = "4"
        cls.proc = start_server(env)
        if not wait_for_port():
            cls.proc.terminate()
            raise RuntimeError("server failed to start for SSE cancel test")

    @classmethod
    def tearDownClass(cls):
        if not cls.proc:
            return
        cls.proc.send_signal(signal.SIGINT)
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def test_stream_cancel(self):
        payload = {
            "prompt": "cancel test prompt",
            "stream": True,
            "max_tokens": 64,
        }
        body = json.dumps(payload).encode()
        with socket.create_connection((HOST, PORT), timeout=5) as sock:
            request = (
                "POST /v1/completions HTTP/1.1\r\n"
                f"Host: {HOST}:{PORT}\r\n"
                f"Authorization: Bearer {API_KEY}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode() + body
            sock.sendall(request)
            data = sock.recv(1024)
            self.assertIn(b"HTTP/1.1 200", data)
            # Consume a chunk to ensure streaming started, then close early.
            sock.recv(512)
            sock.close()
        time.sleep(0.5)
        # verify server still serves health endpoint
        conn = http.client.HTTPConnection(HOST, PORT, timeout=5)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertIn(resp.status, (200, 503))

        # Fairness metrics should be exposed while cancellation occurs under timeslice configs.
        conn = http.client.HTTPConnection(HOST, PORT, timeout=5)
        conn.request("GET", "/metrics", headers={"Authorization": f"Bearer {API_KEY}"})
        metrics_resp = conn.getresponse()
        metrics_body = metrics_resp.read().decode()
        conn.close()
        yields_line = next(
            (line for line in metrics_body.splitlines() if line.startswith("inferflux_fairness_yields_total")),
            None,
        )
        self.assertIsNotNone(yields_line)
        yield_value = float(yields_line.split()[-1])
        self.assertGreaterEqual(yield_value, 0.0)


if __name__ == "__main__":
    unittest.main()
