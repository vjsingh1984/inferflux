#!/usr/bin/env python3
"""Backend Comparison Benchmark: native CUDA vs llama.cpp CUDA

Starts inferfluxd with each backend, runs identical prompts at temperature=0,
and compares:
  1. Throughput (tok/s, latency percentiles)
  2. Response similarity (exact match, token overlap, Jaccard)

Usage:
  python3 scripts/benchmark_backend_comparison.py \
      --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

  # With custom settings
  python3 scripts/benchmark_backend_comparison.py \
      --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
      --num-requests 20 --max-tokens 64 --concurrency 4
"""

import argparse
import http.client
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Configuration
# ============================================================================

BACKENDS = ["cuda_llama_cpp", "cuda_native"]

TEST_PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the capital of France? Answer in one word.",
    "Translate 'hello world' to Spanish.",
    "List three prime numbers greater than 10.",
]

API_KEY = "dev-key-123"
BASE_PORT = 18090


# ============================================================================
# ANSI colors
# ============================================================================

class C:
    R = "\033[0;31m"
    G = "\033[0;32m"
    Y = "\033[1;33m"
    B = "\033[0;34m"
    BOLD = "\033[1m"
    NC = "\033[0m"


def log(level: str, msg: str):
    colors = {"INFO": C.B, "OK": C.G, "WARN": C.Y, "ERR": C.R}
    print(f"{colors.get(level, '')}{f'[{level}]':>8}{C.NC} {msg}")


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class RequestResult:
    prompt: str
    response: str
    latency_s: float
    completion_tokens: int
    success: bool
    error: str = ""


@dataclass
class BackendResults:
    backend: str
    results: List[RequestResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def successful(self) -> List[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def total_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.successful)

    @property
    def tok_per_sec(self) -> float:
        if self.total_time_s <= 0:
            return 0.0
        return self.total_tokens / self.total_time_s

    @property
    def avg_latency(self) -> float:
        s = self.successful
        if not s:
            return 0.0
        return sum(r.latency_s for r in s) / len(s)

    def latency_percentile(self, p: float) -> float:
        s = sorted(r.latency_s for r in self.successful)
        if not s:
            return 0.0
        idx = min(int(math.ceil(p / 100.0 * len(s))) - 1, len(s) - 1)
        return s[max(0, idx)]


# ============================================================================
# Server management
# ============================================================================

def write_config(model_path: str, backend: str, port: int, tmpdir: str) -> str:
    config = {
        "server": {"host": "127.0.0.1", "http_port": port,
                    "max_concurrent": 128, "enable_metrics": True},
        "models": [{"id": "bench-model", "path": os.path.abspath(model_path),
                     "format": "gguf", "backend": backend, "default": True}],
        "runtime": {
            "backend_priority": ["cuda", "cpu"],
            "cuda": {"enabled": True,
                     "attention": {"kernel": "auto"},
                     "flash_attention": {"enabled": True},
                     "phase_overlap": {"enabled": True}},
            "backend_exposure": {"prefer_native": backend == "cuda_native",
                                 "allow_llama_cpp_fallback": True},
            "scheduler": {"max_batch_size": 32, "max_batch_tokens": 16384,
                          "min_batch_size": 1, "batch_accumulation_ms": 2},
            "paged_kv": {"cpu_pages": 4096, "eviction": "lru"},
        },
        "auth": {"api_keys": [{"key": API_KEY,
                                "scopes": ["generate", "read", "admin"]}],
                 "rate_limit_per_minute": 600},
        "guardrails": {"blocklist": []},
        "logging": {"level": "warning", "format": "text"},
    }
    path = os.path.join(tmpdir, f"config_{backend}.yaml")
    try:
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    except ImportError:
        # Fallback: write minimal YAML manually
        with open(path, "w") as f:
            f.write(json.dumps(config, indent=2))
            # inferfluxd accepts JSON as config too
    return path


def start_server(server_bin: str, config_path: str, port: int,
                 backend: str, tmpdir: str) -> subprocess.Popen:
    log("INFO", f"Starting {backend} server on port {port}...")
    env = os.environ.copy()
    env["INFERFLUX_PORT_OVERRIDE"] = str(port)
    env["INFERCTL_API_KEY"] = API_KEY

    log_path = os.path.join(tmpdir, f"server_{backend}.log")
    log_handle = open(log_path, "w")

    proc = subprocess.Popen(
        [server_bin, "--config", config_path],
        stdout=log_handle, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setsid, text=True,
    )

    # Wait for readiness
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            log_handle.close()
            with open(log_path) as f:
                tail = f.read()[-1000:]
            raise RuntimeError(
                f"{backend} server exited (code={proc.returncode}):\n{tail}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/livez",
                         headers={"Authorization": f"Bearer {API_KEY}"})
            resp = conn.getresponse()
            conn.close()
            if resp.status == 200:
                log("OK", f"{backend} server ready (PID {proc.pid})")
                log_handle.close()
                return proc
        except Exception:
            pass
        time.sleep(0.5)

    log_handle.close()
    stop_server(proc)
    with open(log_path) as f:
        tail = f.read()[-1000:]
    raise RuntimeError(f"{backend} server did not become ready:\n{tail}")


def stop_server(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


# ============================================================================
# HTTP client
# ============================================================================

def send_completion(host: str, port: int, prompt: str,
                    max_tokens: int, model: str) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        conn = http.client.HTTPConnection(host, port, timeout=120)
        conn.request("POST", "/v1/completions",
                     body=json.dumps(payload),
                     headers={"Content-Type": "application/json",
                              "Authorization": f"Bearer {API_KEY}"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        conn.close()
    except Exception as exc:
        return RequestResult(prompt=prompt, response="", success=False,
                             latency_s=time.perf_counter() - start,
                             completion_tokens=0, error=str(exc))

    latency = time.perf_counter() - start
    if resp.status != 200:
        return RequestResult(prompt=prompt, response="", success=False,
                             latency_s=latency, completion_tokens=0,
                             error=f"HTTP {resp.status}: {body[:200]}")

    try:
        data = json.loads(body)
        text = ""
        if "choices" in data and data["choices"]:
            text = data["choices"][0].get("text", "")
        tokens = int(data.get("usage", {}).get("completion_tokens", 0))
    except Exception:
        text = body
        tokens = 0

    return RequestResult(prompt=prompt, response=text.strip(),
                         latency_s=latency, completion_tokens=tokens,
                         success=True)


# ============================================================================
# Workload runner
# ============================================================================

def run_workload(port: int, prompts: List[str], max_tokens: int,
                 num_requests: int, concurrency: int,
                 backend: str) -> BackendResults:
    import concurrent.futures

    results = BackendResults(backend=backend)
    # Build request list: cycle through prompts
    requests = []
    for i in range(num_requests):
        requests.append(prompts[i % len(prompts)])

    # Warmup
    log("INFO", f"  Warmup (2 requests)...")
    for i in range(min(2, num_requests)):
        r = send_completion("127.0.0.1", port, requests[i], max_tokens,
                            "bench-model")
        if not r.success:
            log("WARN", f"  Warmup failed: {r.error}")

    log("INFO", f"  Running {num_requests} requests (concurrency={concurrency})...")
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for prompt in requests:
            futures.append(pool.submit(
                send_completion, "127.0.0.1", port, prompt, max_tokens,
                "bench-model"))

        for fut in concurrent.futures.as_completed(futures):
            results.results.append(fut.result())

    results.total_time_s = time.perf_counter() - start
    return results


# ============================================================================
# Similarity analysis
# ============================================================================

def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for similarity."""
    import re
    return re.findall(r'\w+', text.lower())


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def token_overlap(a: List[str], b: List[str]) -> float:
    """Fraction of tokens in common (order-independent)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return 2.0 * len(sa & sb) / (len(sa) + len(sb))


def compute_similarity(results_a: BackendResults,
                       results_b: BackendResults) -> Dict:
    """Compare responses from two backends for the same prompts."""
    # Build prompt -> response maps
    map_a: Dict[str, str] = {}
    map_b: Dict[str, str] = {}
    for r in results_a.successful:
        if r.prompt not in map_a:
            map_a[r.prompt] = r.response
    for r in results_b.successful:
        if r.prompt not in map_b:
            map_b[r.prompt] = r.response

    common_prompts = set(map_a.keys()) & set(map_b.keys())
    if not common_prompts:
        return {"error": "No common prompts with successful responses"}

    exact_matches = 0
    jaccard_scores = []
    overlap_scores = []
    comparisons = []

    for prompt in sorted(common_prompts):
        resp_a = map_a[prompt]
        resp_b = map_b[prompt]

        exact = resp_a == resp_b
        if exact:
            exact_matches += 1

        toks_a = tokenize_simple(resp_a)
        toks_b = tokenize_simple(resp_b)
        jac = jaccard_similarity(toks_a, toks_b)
        ovl = token_overlap(toks_a, toks_b)
        jaccard_scores.append(jac)
        overlap_scores.append(ovl)

        comparisons.append({
            "prompt": prompt[:60],
            "exact_match": exact,
            "jaccard": jac,
            "overlap": ovl,
            "response_a": resp_a[:80],
            "response_b": resp_b[:80],
        })

    return {
        "num_compared": len(common_prompts),
        "exact_match_rate": exact_matches / len(common_prompts),
        "mean_jaccard": sum(jaccard_scores) / len(jaccard_scores),
        "mean_token_overlap": sum(overlap_scores) / len(overlap_scores),
        "comparisons": comparisons,
    }


# ============================================================================
# Metrics collection
# ============================================================================

def fetch_metrics(port: int) -> str:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/metrics",
                     headers={"Authorization": f"Bearer {API_KEY}"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        conn.close()
        return body if resp.status == 200 else ""
    except Exception:
        return ""


def extract_metric(metrics_text: str, name: str,
                   labels: Optional[Dict[str, str]] = None) -> float:
    import re
    total = 0.0
    pattern = re.compile(
        r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+'
        r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$')
    label_re = re.compile(r'([a-zA-Z_]\w*)="((?:\\.|[^"])*)"')

    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m or m.group(1) != name:
            continue
        parsed = dict(label_re.findall(m.group(2) or ""))
        if labels and not all(parsed.get(k) == v for k, v in labels.items()):
            continue
        try:
            total += float(m.group(3))
        except ValueError:
            pass
    return total


# ============================================================================
# Report
# ============================================================================

def print_report(all_results: Dict[str, BackendResults],
                 similarity: Dict, args):
    print()
    print(f"{C.BOLD}{'=' * 72}{C.NC}")
    print(f"{C.BOLD}  Backend Comparison Report{C.NC}")
    print(f"{C.BOLD}{'=' * 72}{C.NC}")
    print()
    print(f"  Model:       {args.model}")
    print(f"  Requests:    {args.num_requests} (concurrency={args.concurrency})")
    print(f"  Max tokens:  {args.max_tokens}")
    print()

    # Throughput table
    print(f"{C.BOLD}  Throughput{C.NC}")
    print(f"  {'Backend':<20} {'Tok/s':>8} {'Avg(ms)':>10} {'P50(ms)':>10} "
          f"{'P95(ms)':>10} {'P99(ms)':>10} {'OK/Total':>10}")
    print(f"  {'-' * 78}")

    for backend in BACKENDS:
        r = all_results.get(backend)
        if not r:
            print(f"  {backend:<20} {'SKIPPED':>8}")
            continue
        ok = len(r.successful)
        total = len(r.results)
        print(f"  {backend:<20} {r.tok_per_sec:>8.1f} "
              f"{r.avg_latency * 1000:>10.0f} "
              f"{r.latency_percentile(50) * 1000:>10.0f} "
              f"{r.latency_percentile(95) * 1000:>10.0f} "
              f"{r.latency_percentile(99) * 1000:>10.0f} "
              f"{ok}/{total}:>10")
    print()

    # Speedup
    native = all_results.get("cuda_native")
    llama = all_results.get("cuda_llama_cpp")
    if native and llama and llama.tok_per_sec > 0:
        ratio = native.tok_per_sec / llama.tok_per_sec
        color = C.G if ratio >= 0.9 else C.Y if ratio >= 0.5 else C.R
        print(f"  {C.BOLD}Speedup (native / llama.cpp):{C.NC} "
              f"{color}{ratio:.2f}x{C.NC}")
        print()

    # Similarity table
    if "error" not in similarity:
        print(f"{C.BOLD}  Response Similarity{C.NC}")
        print(f"  Prompts compared:   {similarity['num_compared']}")
        emr = similarity['exact_match_rate']
        color = C.G if emr >= 0.8 else C.Y if emr >= 0.5 else C.R
        print(f"  Exact match rate:   {color}{emr:.1%}{C.NC}")
        print(f"  Mean Jaccard:       {similarity['mean_jaccard']:.3f}")
        print(f"  Mean token overlap: {similarity['mean_token_overlap']:.3f}")
        print()

        print(f"  {'Prompt':<40} {'Match':>6} {'Jaccard':>8} {'Overlap':>8}")
        print(f"  {'-' * 66}")
        for comp in similarity["comparisons"]:
            match_str = f"{C.G}YES{C.NC}" if comp["exact_match"] else f"{C.R}NO{C.NC}"
            print(f"  {comp['prompt']:<40} {match_str:>15} "
                  f"{comp['jaccard']:>8.3f} {comp['overlap']:>8.3f}")

        # Show divergent responses
        divergent = [c for c in similarity["comparisons"]
                     if not c["exact_match"]]
        if divergent:
            print()
            print(f"{C.BOLD}  Divergent Responses (first 3){C.NC}")
            for comp in divergent[:3]:
                print(f"  Prompt: {comp['prompt']}")
                print(f"    llama.cpp: {comp['response_a']}")
                print(f"    native:    {comp['response_b']}")
                print()
    else:
        print(f"  {C.Y}Similarity: {similarity['error']}{C.NC}")

    print(f"{'=' * 72}")


def save_results_json(all_results: Dict[str, BackendResults],
                      similarity: Dict, output_path: str):
    data = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backends": {}}
    for backend, r in all_results.items():
        data["backends"][backend] = {
            "total_time_s": r.total_time_s,
            "tok_per_sec": r.tok_per_sec,
            "avg_latency_ms": r.avg_latency * 1000,
            "p50_latency_ms": r.latency_percentile(50) * 1000,
            "p95_latency_ms": r.latency_percentile(95) * 1000,
            "p99_latency_ms": r.latency_percentile(99) * 1000,
            "total_tokens": r.total_tokens,
            "successful": len(r.successful),
            "total": len(r.results),
            "responses": [{"prompt": rr.prompt, "response": rr.response[:200],
                           "latency_s": rr.latency_s,
                           "completion_tokens": rr.completion_tokens,
                           "success": rr.success}
                          for rr in r.results],
        }
    data["similarity"] = {k: v for k, v in similarity.items()
                          if k != "comparisons"}
    if "comparisons" in similarity:
        data["similarity"]["comparisons"] = [
            {k: v for k, v in c.items()
             if k not in ("response_a", "response_b")}
            for c in similarity["comparisons"]]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    log("INFO", f"Results saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare native CUDA and llama.cpp CUDA backends",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--server-bin", default="./build/inferfluxd",
                        help="Path to inferfluxd binary")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of requests per backend")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max completion tokens per request")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent request count")
    parser.add_argument("--output", default="",
                        help="JSON output file (default: auto-generated)")
    parser.add_argument("--backends", nargs="+", default=BACKENDS,
                        choices=BACKENDS,
                        help="Backends to benchmark")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Custom prompts (overrides built-in set)")
    args = parser.parse_args()

    if not Path(args.server_bin).exists():
        log("ERR", f"Server binary not found: {args.server_bin}")
        log("INFO", "Build with: cmake --build build -j")
        return 1

    if not Path(args.model).exists():
        log("ERR", f"Model not found: {args.model}")
        return 1

    prompts = args.prompts or TEST_PROMPTS

    tmpdir = tempfile.mkdtemp(prefix="inferflux_bench_")
    log("INFO", f"Temp dir: {tmpdir}")

    all_results: Dict[str, BackendResults] = {}
    servers: Dict[str, subprocess.Popen] = {}

    try:
        for idx, backend in enumerate(args.backends):
            port = BASE_PORT + idx
            print()
            log("INFO", f"=== Benchmarking: {backend} ===")

            config_path = write_config(args.model, backend, port, tmpdir)

            try:
                proc = start_server(args.server_bin, config_path, port,
                                    backend, tmpdir)
                servers[backend] = proc
            except RuntimeError as exc:
                log("ERR", f"Failed to start {backend}: {exc}")
                all_results[backend] = BackendResults(backend=backend)
                continue

            try:
                results = run_workload(port, prompts, args.max_tokens,
                                       args.num_requests, args.concurrency,
                                       backend)
                all_results[backend] = results

                ok = len(results.successful)
                log("OK", f"  {backend}: {ok}/{len(results.results)} OK, "
                    f"{results.tok_per_sec:.1f} tok/s, "
                    f"avg {results.avg_latency * 1000:.0f}ms")

                # Collect metrics
                metrics = fetch_metrics(port)
                if metrics:
                    native_fwd = extract_metric(
                        metrics, "inferflux_native_forward_passes_total")
                    fused_gemv = extract_metric(
                        metrics, "inferflux_fused_dequant_dispatches_total")
                    if native_fwd > 0:
                        log("INFO", f"  Native forward passes: {native_fwd:.0f}")
                    if fused_gemv > 0:
                        log("INFO", f"  Fused dequant dispatches: {fused_gemv:.0f}")

            finally:
                log("INFO", f"Stopping {backend} server...")
                stop_server(proc)
                time.sleep(1)  # Let port free up

    except KeyboardInterrupt:
        log("WARN", "Interrupted")
    finally:
        for backend, proc in servers.items():
            stop_server(proc)

    # Similarity analysis
    similarity = {}
    if "cuda_llama_cpp" in all_results and "cuda_native" in all_results:
        similarity = compute_similarity(
            all_results["cuda_llama_cpp"], all_results["cuda_native"])
    elif len(all_results) == 1:
        similarity = {"error": "Only one backend ran — cannot compare"}
    else:
        similarity = {"error": "No backends completed successfully"}

    print_report(all_results, similarity, args)

    # Save JSON
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            "gguf_benchmark_results",
            f"backend_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_results_json(all_results, similarity, output_path)

    # Exit code: 0 if both ran
    if all(len(all_results.get(b, BackendResults(b)).successful) > 0
           for b in args.backends):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
