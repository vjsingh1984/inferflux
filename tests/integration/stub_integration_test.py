#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import time
import unittest
import http.client
import socket

class StubIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We assume the binary is in the build directory and we are in the project root.
        cls.server_proc = subprocess.Popen(
            ["./build/inferfluxd", "--config", "config/server.yaml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        time.sleep(2) # Initial wait

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'server_proc') and cls.server_proc:
            try:
                os.killpg(os.getpgid(cls.server_proc.pid), signal.SIGTERM)
                cls.server_proc.wait(timeout=5)
            except:
                pass

    def _post(self, path, data):
        conn = http.client.HTTPConnection("localhost", 8080, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def test_health(self):
        # /healthz might require auth depending on config
        conn = http.client.HTTPConnection("localhost", 8080, timeout=5)
        headers = {"Authorization": "Bearer dev-key-123"}
        conn.request("GET", "/healthz", headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        conn.close()

    def test_stub_completion(self):
        resp, body = self._post("/v1/completions", {"model": "default", "prompt": "hi"})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

    def test_cache_warm(self):
        resp, body = self._post("/v1/admin/cache/warm", {"tokens": [1, 2, 3], "block_table": [100]})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

if __name__ == "__main__":
    unittest.main()
