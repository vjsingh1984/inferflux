#!/usr/bin/env python3

import http.server
import importlib.util
import json
import socketserver
import threading
import unittest
from pathlib import Path


def _load_driver_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "benchmark_request_driver.py"
    spec = importlib.util.spec_from_file_location("benchmark_request_driver", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if self.path == "/v1/completions":
            assert payload.get("prompt") == "ping"
            if payload.get("stream"):
                body = (
                    'data: {"choices":[{"text":"ready"}]}\n\n'
                    'data: {"choices":[{"text":" stream"}]}\n\n'
                    'data: {"usage":{"completion_tokens":2}}\n\n'
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps(
                {"choices": [{"text": "ready steady"}], "usage": {"completion_tokens": 2}}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/v1/chat/completions":
            messages = payload.get("messages", [])
            assert isinstance(messages, list)
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "ping"
            if payload.get("stream"):
                body = (
                    'data: {"choices":[{"delta":{"content":"ready"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":" stream"}}]}\n\n'
                    'data: {"usage":{"completion_tokens":2}}\n\n'
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps(
                {"choices": [{"message": {"content": "ready steady"}}],
                 "usage": {"completion_tokens": 2}}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/generate":
            if payload.get("stream"):
                body = (
                    json.dumps({"response": "ready"}) + "\n" +
                    json.dumps({"response": " stream"}) + "\n" +
                    json.dumps({"done": True}) + "\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps({"response": "ready steady"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


class BenchmarkRequestDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _ThreadedTCPServer(("127.0.0.1", 0), _RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_openai_non_streaming_request(self):
        result = driver.request_openai(
            f"{self.base_url}/v1/completions",
            "stub-model",
            "ping",
            4,
            "req-1",
            "",
            inferflux_style=False,
            stream=False,
        )
        self.assertEqual(result["text"], "ready steady")
        self.assertEqual(result["tokens"], 2)

    def test_openai_streaming_request(self):
        result = driver.request_openai(
            f"{self.base_url}/v1/completions",
            "stub-model",
            "ping",
            4,
            "req-2",
            "",
            inferflux_style=False,
            stream=True,
        )
        self.assertEqual(result["text"], "ready stream")
        self.assertEqual(result["tokens"], 2)

    def test_openai_chat_non_streaming_request(self):
        result = driver.request_openai(
            f"{self.base_url}/v1/chat/completions",
            "stub-model",
            "ping",
            4,
            "req-2b",
            "",
            inferflux_style=False,
            stream=False,
        )
        self.assertEqual(result["text"], "ready steady")
        self.assertEqual(result["tokens"], 2)

    def test_openai_chat_streaming_request(self):
        result = driver.request_openai(
            f"{self.base_url}/v1/chat/completions",
            "stub-model",
            "ping",
            4,
            "req-2c",
            "",
            inferflux_style=False,
            stream=True,
        )
        self.assertEqual(result["text"], "ready stream")
        self.assertEqual(result["tokens"], 2)

    def test_ollama_non_streaming_request(self):
        result = driver.request_ollama(
            f"{self.base_url}/api/generate",
            "stub-model",
            "ping",
            4,
            "req-3",
            stream=False,
        )
        self.assertEqual(result["text"], "ready steady")
        self.assertEqual(result["tokens"], 2)

    def test_ollama_streaming_request(self):
        result = driver.request_ollama(
            f"{self.base_url}/api/generate",
            "stub-model",
            "ping",
            4,
            "req-4",
            stream=True,
        )
        self.assertEqual(result["text"], "ready stream")
        self.assertEqual(result["tokens"], 2)


if __name__ == "__main__":
    unittest.main()
