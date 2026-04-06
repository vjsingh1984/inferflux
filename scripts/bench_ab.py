#!/usr/bin/env python3
"""Apples-to-apples backend comparison: inferflux_cuda vs llama_cpp_cuda.

Measures both HTTP non-streaming and SSE streaming throughput through
identical API calls. Starts each backend on a separate port, runs
the same prompt suite, and reports side-by-side results.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get(
    "MODEL_PATH",
    str(ROOT / "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"),
)
BUILD_DIR = os.environ.get("BUILD_DIR", str(ROOT / "build-cuda-opt/Release"))
SERVER_BIN = os.path.join(BUILD_DIR, "inferfluxd.exe")
API_KEY = "dev-key-123"
NUM_RUNS = int(os.environ.get("NUM_RUNS", "5"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
PROMPT = os.environ.get("PROMPT", "Explain quantum computing in one paragraph.")

BACKENDS = {
    "inferflux_cuda": {"port": 18091, "prefer_inferflux": True, "allow_fallback": False},
    "llama_cpp_cuda": {"port": 18090, "prefer_inferflux": False, "allow_fallback": True},
}


def write_config(backend: str, port: int, prefer: bool, fallback: bool) -> str:
    temp_dir = Path(tempfile.gettempdir()) / "inferflux_bench_ab"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"bench_{backend}.yaml"
    gpu_layers = 99 if backend == "llama_cpp_cuda" else 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""server:
  host: "127.0.0.1"
  http_port: {port}
  max_concurrent: 128
  enable_metrics: true
models:
  - id: bench-model
    path: "{MODEL}"
    format: gguf
    backend: {backend}
    default: true
runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: true
    gpu_layers: {gpu_layers}
  backend_exposure:
    prefer_inferflux: {"true" if prefer else "false"}
    allow_llama_cpp_fallback: {"true" if fallback else "false"}
  scheduler:
    max_batch_size: 32
    max_batch_tokens: 16384
  paged_kv:
    cpu_pages: 4096
    eviction: lru
auth:
  api_keys:
    - key: {API_KEY}
      scopes: [generate, read, admin]
logging:
  level: warning
  format: text
""")
    return str(path)


def start_server(backend: str, cfg: dict) -> subprocess.Popen:
    config_path = write_config(backend, cfg["port"], cfg["prefer_inferflux"], cfg["allow_fallback"])
    env = os.environ.copy()
    env["INFERFLUX_DISABLE_STARTUP_ADVISOR"] = "true"
    env["INFERFLUX_CUDA_KV_MAX_BATCH"] = "8"
    env["INFERFLUX_CUDA_KV_MAX_SEQ"] = "2048"
    # Explicitly clear any env overrides to ensure config takes effect
    env.pop("INFERFLUX_BACKEND_PREFER_INFERFLUX", None)
    env.pop("INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK", None)
    proc = subprocess.Popen(
        [SERVER_BIN, "--config", config_path],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Wait for ready
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{cfg['port']}/healthz", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    stderr = ""
    if proc.stderr is not None:
        try:
            stderr = proc.stderr.read().decode(errors="replace").strip()
        except Exception:
            stderr = ""
    proc.kill()
    detail = f": {stderr}" if stderr else ""
    raise RuntimeError(f"{backend} failed to start{detail}")


def get_model_name(port: int) -> str:
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    models = data.get("data", [])
    if models:
        return models[0]["id"]
    return "bench-model"


def bench_non_streaming(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    """Single non-streaming request. Returns {tokens, ms, tok_s}."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    elapsed = (time.perf_counter() - t0) * 1000
    ct = result.get("usage", {}).get("completion_tokens", 0)
    return {"tokens": ct, "ms": elapsed, "tok_s": ct / (elapsed / 1000) if elapsed > 0 else 0}


def bench_streaming(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    """Single SSE streaming request. Returns {tokens, ttft_ms, decode_ms, decode_tok_s, e2e_tok_s}."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    tokens = 0
    first_t = last_t = None
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        buf = b""
        for chunk in iter(lambda: resp.read(1), b""):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        d = json.loads(line[6:])
                        content = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            t = time.perf_counter()
                            if first_t is None:
                                first_t = t
                            last_t = t
                            tokens += 1
                    except Exception:
                        pass
    end = time.perf_counter()
    ttft = (first_t - start) * 1000 if first_t else 0
    decode_ms = (last_t - first_t) * 1000 if first_t and last_t and tokens > 1 else 1
    decode_tok = max(tokens - 1, 1)
    total_ms = (end - start) * 1000
    return {
        "tokens": tokens,
        "ttft_ms": ttft,
        "decode_tok": decode_tok,
        "decode_ms": decode_ms,
        "decode_tok_s": decode_tok / (decode_ms / 1000),
        "e2e_tok_s": tokens / (total_ms / 1000) if total_ms > 0 else 0,
    }


def main():
    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT[:60]}...")
    print(f"Max tokens: {MAX_TOKENS}, Runs: {NUM_RUNS}")
    print()

    results = {}
    procs = {}

    for backend, cfg in BACKENDS.items():
        print(f"Starting {backend} on port {cfg['port']}...", end=" ", flush=True)
        try:
            procs[backend] = start_server(backend, cfg)
            model_name = get_model_name(cfg["port"])
            print(f"OK (model={model_name})")

            # Warmup
            for _ in range(3):
                try:
                    bench_non_streaming(cfg["port"], model_name, "Hi", 8)
                except Exception:
                    pass

            # Non-streaming benchmark
            ns_results = []
            for i in range(NUM_RUNS):
                r = bench_non_streaming(cfg["port"], model_name, PROMPT, MAX_TOKENS)
                ns_results.append(r)

            # Streaming benchmark
            st_results = []
            for i in range(NUM_RUNS):
                r = bench_streaming(cfg["port"], model_name, PROMPT, MAX_TOKENS)
                st_results.append(r)

            results[backend] = {"non_streaming": ns_results, "streaming": st_results}

        except Exception as e:
            print(f"FAILED: {e}")
            results[backend] = None
        finally:
            if backend in procs:
                procs[backend].terminate()
                try:
                    procs[backend].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    procs[backend].kill()
            time.sleep(1)

    # Print results
    print()
    print("=" * 72)
    print(f"{'METRIC':<30} {'inferflux_cuda':>18} {'llama_cpp_cuda':>18}")
    print("=" * 72)

    for mode in ["non_streaming", "streaming"]:
        mode_label = "HTTP Non-Streaming" if mode == "non_streaming" else "SSE Streaming"
        print(f"\n--- {mode_label} ---")

        for backend in BACKENDS:
            if results.get(backend) is None:
                continue
            data = results[backend][mode]

            if mode == "non_streaming":
                avg_tok_s = sum(r["tok_s"] for r in data) / len(data)
                avg_ms = sum(r["ms"] for r in data) / len(data)
                avg_tok = sum(r["tokens"] for r in data) / len(data)
                print(f"  {backend}: {avg_tok:.0f} tok in {avg_ms:.0f}ms = {avg_tok_s:.1f} tok/s")
            else:
                avg_ttft = sum(r["ttft_ms"] for r in data) / len(data)
                avg_decode = sum(r["decode_tok_s"] for r in data) / len(data)
                avg_e2e = sum(r["e2e_tok_s"] for r in data) / len(data)
                print(f"  {backend}: TTFT={avg_ttft:.0f}ms, decode={avg_decode:.1f} tok/s, e2e={avg_e2e:.1f} tok/s")

    # Side-by-side summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for backend in BACKENDS:
        if results.get(backend) is None:
            print(f"  {backend}: FAILED")
            continue
        ns = results[backend]["non_streaming"]
        st = results[backend]["streaming"]
        ns_avg = sum(r["tok_s"] for r in ns) / len(ns)
        decode_avg = sum(r["decode_tok_s"] for r in st) / len(st)
        ttft_avg = sum(r["ttft_ms"] for r in st) / len(st)
        print(f"  {backend}:")
        print(f"    Non-streaming:  {ns_avg:.1f} tok/s")
        print(f"    Pure decode:    {decode_avg:.1f} tok/s")
        print(f"    TTFT:           {ttft_avg:.0f} ms")


if __name__ == "__main__":
    main()
