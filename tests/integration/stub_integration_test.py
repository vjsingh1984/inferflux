#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import time
import unittest
import http.client

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18081

class StubIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["INFERFLUX_HOST_OVERRIDE"] = SERVER_HOST
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT)
        # We assume the binary is in the build directory and we are in the project root.
        cls.server_proc = subprocess.Popen(
            ["./build/inferfluxd", "--config", "config/server.yaml"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        deadline = time.time() + 20.0
        ready = False
        while time.time() < deadline:
            if cls.server_proc.poll() is not None:
                break
            try:
                conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=1)
                conn.request("GET", "/livez", headers={"Authorization": "Bearer dev-key-123"})
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status in (200, 401):
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        if not ready:
            try:
                out, err = cls.server_proc.communicate(timeout=1)
            except Exception:
                out, err = ("", "")
            raise RuntimeError(
                "inferfluxd did not become ready in time; "
                f"stdout={out[-400:] if out else ''} stderr={err[-400:] if err else ''}"
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'server_proc') and cls.server_proc:
            try:
                os.killpg(os.getpgid(cls.server_proc.pid), signal.SIGTERM)
                cls.server_proc.wait(timeout=5)
            except:
                pass

    def _post(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _put(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("PUT", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _get(self, path):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Authorization": "Bearer dev-key-123"}
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def test_health(self):
        # /healthz might require auth depending on config
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=5)
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

    def test_admin_routing_policy(self):
        resp, body = self._get("/v1/admin/routing")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertIn("allow_default_fallback", payload)
        self.assertIn("require_ready_backend", payload)
        self.assertIn("fallback_scope", payload)

        resp, body = self._put("/v1/admin/routing", {"fallback_scope": "same_path_only"})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

        resp, body = self._get("/v1/admin/routing")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("fallback_scope"), "same_path_only")

        # Restore default scope so this test does not affect other flows.
        resp, body = self._put("/v1/admin/routing", {"fallback_scope": "any_compatible"})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

if __name__ == "__main__":
    unittest.main()
