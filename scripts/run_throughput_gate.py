#!/usr/bin/env python3
"""InferFlux throughput gate runner.

Runs a concurrent completion workload against an InferFlux endpoint, reads
Prometheus counters from /metrics, and enforces configurable throughput floors.
"""

import argparse
import concurrent.futures
import http.client
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')


@dataclass
class RequestResult:
  success: bool
  latency_s: float
  completion_tokens: int
  error: str = ""


@dataclass
class MetricsSnapshot:
  completion_tokens_global: float
  completion_tokens_model: float
  decode_lane_submissions: float
  prefill_lane_submissions: float
  cuda_lane_overlap_events: float
  cuda_lane_overlap_duration_ms: float
  cuda_attention_fallback_events: float
  selected_attention_kernel: str
  native_forward_prefill: float
  native_forward_decode: float
  native_forward_batch_tokens: float


@dataclass
class BackendExposureSnapshot:
  model_id: str
  requested_backend: str
  exposed_backend: str
  provider: str
  fallback: bool
  fallback_reason: str


@dataclass
class ManagedServer:
  process: subprocess.Popen
  log_path: str


@dataclass(frozen=True)
class GpuProfileDefaults:
  require_cuda_lanes: bool
  require_cuda_overlap: bool
  min_cuda_overlap_duration_ms: float
  max_cuda_attention_fallbacks: float
  expect_cuda_attention_kernel: str
  mixed_prompt_workload: bool
  unique_prompts: bool
  target_total_prefill_tokens: int


GPU_PROFILE_DEFAULTS: Dict[str, GpuProfileDefaults] = {
    "ada_rtx_4000": GpuProfileDefaults(
        require_cuda_lanes=True,
        require_cuda_overlap=True,
        min_cuda_overlap_duration_ms=5.0,
        max_cuda_attention_fallbacks=1.0,
        expect_cuda_attention_kernel="",
        mixed_prompt_workload=True,
        unique_prompts=True,
        target_total_prefill_tokens=4096,
    ),
}


def _parse_labels(raw: str) -> Dict[str, str]:
  labels: Dict[str, str] = {}
  if not raw:
    return labels
  for match in LABEL_RE.finditer(raw):
    key = match.group(1)
    value = match.group(2).replace('\\"', '"').replace("\\\\", "\\")
    labels[key] = value
  return labels


def read_metric(metrics_text: str, metric_name: str,
                labels: Optional[Dict[str, str]] = None) -> float:
  total = 0.0
  for line in metrics_text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    match = METRIC_LINE_RE.match(line)
    if not match:
      continue
    name, raw_labels, raw_value = match.groups()
    if name != metric_name:
      continue
    parsed_labels = _parse_labels(raw_labels or "")
    if labels:
      if not all(parsed_labels.get(k) == v for k, v in labels.items()):
        continue
    try:
      total += float(raw_value)
    except ValueError:
      continue
  return total


def selected_attention_kernel(metrics_text: str) -> str:
  for kernel in ("fa3", "fa2", "standard"):
    value = read_metric(
        metrics_text,
        "inferflux_cuda_attention_kernel_selected",
        {"kernel": kernel},
    )
    if value >= 0.5:
      return kernel
  return "unknown"


def http_get(host: str, port: int, path: str, timeout_s: float,
             api_key: str) -> Tuple[int, str]:
  conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
  headers = {}
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
  conn.request("GET", path, headers=headers)
  response = conn.getresponse()
  body = response.read().decode("utf-8", errors="replace")
  status = response.status
  conn.close()
  return status, body


def http_post_json(host: str, port: int, path: str, payload: Dict,
                   timeout_s: float, api_key: str) -> Tuple[int, str]:
  conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
  headers = {"Content-Type": "application/json"}
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
  conn.request("POST", path, body=json.dumps(payload), headers=headers)
  response = conn.getresponse()
  body = response.read().decode("utf-8", errors="replace")
  status = response.status
  conn.close()
  return status, body


def wait_for_server(host: str, port: int, timeout_s: float, api_key: str,
                    proc: Optional[subprocess.Popen] = None) -> None:
  deadline = time.time() + timeout_s
  last_error = "timeout waiting for /livez"
  while time.time() < deadline:
    if proc is not None and proc.poll() is not None:
      raise RuntimeError(f"server exited before readiness check (code={proc.returncode})")
    try:
      status, _ = http_get(host, port, "/livez", timeout_s=1.0, api_key=api_key)
      if status == 200:
        return
      last_error = f"/livez returned status={status}"
    except Exception as exc:  # noqa: BLE001
      last_error = str(exc)
    time.sleep(0.1)
  raise RuntimeError(last_error)


def tail_file(path: str, max_bytes: int = 2048) -> str:
  try:
    with open(path, "rb") as handle:
      handle.seek(0, os.SEEK_END)
      size = handle.tell()
      handle.seek(max(0, size - max_bytes), os.SEEK_SET)
      return handle.read().decode("utf-8", errors="replace")
  except Exception:  # noqa: BLE001
    return ""


def start_server(args: argparse.Namespace) -> ManagedServer:
  if not args.server_bin:
    raise ValueError("server_bin must be set to launch a managed server")
  if not args.config:
    raise ValueError("--config is required when --server-bin is provided")

  env = os.environ.copy()
  env["INFERFLUX_HOST_OVERRIDE"] = args.host
  env["INFERFLUX_PORT_OVERRIDE"] = str(args.port)
  for pair in args.server_env:
    if "=" not in pair:
      raise ValueError(f"invalid --server-env value '{pair}' (expected KEY=VALUE)")
    key, value = pair.split("=", 1)
    key = key.strip()
    if not key:
      raise ValueError(f"invalid --server-env value '{pair}' (empty key)")
    env[key] = value

  if args.server_log_path:
    log_path = args.server_log_path
    log_handle = open(log_path, "w", encoding="utf-8")
  else:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="inferflux_tp_gate_", suffix=".log",
        delete=False)
    log_path = handle.name
    log_handle = handle

  proc = subprocess.Popen(
      [args.server_bin, "--config", args.config],
      stdout=log_handle,
      stderr=subprocess.STDOUT,
      env=env,
      preexec_fn=os.setsid,
      text=True,
  )

  try:
    wait_for_server(args.host, args.port, args.startup_timeout_sec,
                    args.api_key, proc)
  except Exception as exc:  # noqa: BLE001
    stop_server(ManagedServer(process=proc, log_path=log_path))
    log_tail = tail_file(log_path)
    raise RuntimeError(
        f"managed server failed to become ready: {exc}\n"
        f"server_log_tail:\n{log_tail}"
    ) from exc
  finally:
    log_handle.flush()
    log_handle.close()

  return ManagedServer(process=proc, log_path=log_path)


def stop_server(managed: ManagedServer) -> None:
  proc = managed.process
  if proc.poll() is not None:
    return
  try:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
  except ProcessLookupError:
    return
  except Exception:  # noqa: BLE001
    try:
      proc.terminate()
    except Exception:
      return

  try:
    proc.wait(timeout=8)
  except subprocess.TimeoutExpired:
    try:
      os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
      proc.kill()
    proc.wait(timeout=5)


def build_request_payload(args: argparse.Namespace, request_index: int) -> Dict:
  prompt = args.prompt
  if args.mixed_prompt_workload:
    repeat = args.long_prompt_repeat
    if request_index % 2 == 1:
      repeat = args.short_prompt_repeat
    repeat = max(1, repeat)
    # Keep generated prompt size below conservative per-request prefill budget
    # to avoid tripping backend n_batch assertions during stress runs.
    words_per_chunk = max(1, len(prompt.split()))
    target_total_prefill_tokens = max(24, args.target_total_prefill_tokens)
    per_request_budget = max(
        24, target_total_prefill_tokens // max(1, args.concurrency))
    if words_per_chunk * repeat > per_request_budget:
      repeat = max(1, per_request_budget // words_per_chunk)
    prompt = ((prompt + " ") * repeat).strip()
  if args.unique_prompts:
    prompt = f"{prompt} [req-{request_index}]"

  if args.endpoint == "/v1/chat/completions":
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
  else:
    payload = {
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
  if args.model:
    payload["model"] = args.model
  return payload


def run_single_request(args: argparse.Namespace, request_index: int) -> RequestResult:
  payload = build_request_payload(args, request_index)
  start = time.perf_counter()
  try:
    status, body = http_post_json(
        args.host,
        args.port,
        args.endpoint,
        payload,
        timeout_s=args.request_timeout_sec,
        api_key=args.api_key,
    )
  except Exception as exc:  # noqa: BLE001
    return RequestResult(
        success=False,
        latency_s=time.perf_counter() - start,
        completion_tokens=0,
        error=f"request exception: {exc}",
    )

  latency = time.perf_counter() - start
  if status != 200:
    clipped = body[:240].replace("\n", " ")
    return RequestResult(
        success=False,
        latency_s=latency,
        completion_tokens=0,
        error=f"http {status}: {clipped}",
    )

  completion_tokens = 0
  try:
    data = json.loads(body)
    usage = data.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
  except Exception:
    completion_tokens = 0

  return RequestResult(
      success=True,
      latency_s=latency,
      completion_tokens=max(0, completion_tokens),
  )


def run_warmup(args: argparse.Namespace) -> None:
  for idx in range(args.warmup_requests):
    result = run_single_request(args, idx)
    if not result.success:
      raise RuntimeError(f"warmup request failed: {result.error}")


def fetch_metrics_snapshot(args: argparse.Namespace) -> MetricsSnapshot:
  status, body = http_get(args.host, args.port, args.metrics_path,
                          timeout_s=5.0, api_key=args.api_key)
  if status != 200:
    raise RuntimeError(f"{args.metrics_path} returned HTTP {status}")

  completion_labels: Dict[str, str] = {}
  if args.backend:
    completion_labels["backend"] = args.backend
  completion_total_global = read_metric(body, "inferflux_completion_tokens_total",
                                        completion_labels)
  completion_total_model = 0.0
  if args.model:
    model_labels = {"model": args.model}
    if args.backend:
      model_labels["backend"] = args.backend
    completion_total_model = read_metric(
        body, "inferflux_model_completion_tokens_total", model_labels)

  decode_lane_submissions = read_metric(
      body, "inferflux_cuda_lane_submissions_total", {"lane": "decode"})
  prefill_lane_submissions = read_metric(
      body, "inferflux_cuda_lane_submissions_total", {"lane": "prefill"})
  overlap_events = read_metric(body, "inferflux_cuda_lane_overlap_events_total")
  overlap_duration_ms = read_metric(
      body, "inferflux_cuda_lane_overlap_duration_ms_total")
  attention_fallbacks = read_metric(
      body, "inferflux_cuda_attention_kernel_fallbacks_total")
  native_fwd_prefill = read_metric(
      body, "inferflux_native_forward_passes_total", {"phase": "prefill"})
  native_fwd_decode = read_metric(
      body, "inferflux_native_forward_passes_total", {"phase": "decode"})
  native_fwd_batch_tokens = read_metric(
      body, "inferflux_native_forward_batch_tokens_total")
  return MetricsSnapshot(
      completion_tokens_global=completion_total_global,
      completion_tokens_model=completion_total_model,
      decode_lane_submissions=decode_lane_submissions,
      prefill_lane_submissions=prefill_lane_submissions,
      cuda_lane_overlap_events=overlap_events,
      cuda_lane_overlap_duration_ms=overlap_duration_ms,
      cuda_attention_fallback_events=attention_fallbacks,
      selected_attention_kernel=selected_attention_kernel(body),
      native_forward_prefill=native_fwd_prefill,
      native_forward_decode=native_fwd_decode,
      native_forward_batch_tokens=native_fwd_batch_tokens,
  )


def fetch_backend_exposure(args: argparse.Namespace) -> BackendExposureSnapshot:
  status, body = http_get(args.host, args.port, "/v1/models",
                          timeout_s=5.0, api_key=args.api_key)
  if status != 200:
    raise RuntimeError(f"/v1/models returned HTTP {status}")
  try:
    payload = json.loads(body)
  except json.JSONDecodeError as exc:
    raise RuntimeError(f"/v1/models returned invalid JSON: {exc}") from exc

  data = payload.get("data", [])
  if not isinstance(data, list) or not data:
    raise RuntimeError("/v1/models did not return any models")

  selected: Optional[Dict] = None
  if args.model:
    for entry in data:
      if isinstance(entry, dict) and str(entry.get("id", "")) == args.model:
        selected = entry
        break
    if selected is None:
      raise RuntimeError(
          f"model '{args.model}' not found in /v1/models response")
  else:
    ready = [m for m in data if isinstance(m, dict) and bool(m.get("ready", True))]
    candidates = ready if ready else [m for m in data if isinstance(m, dict)]
    if args.backend:
      backend_filtered: List[Dict] = []
      for entry in candidates:
        exposure = entry.get("backend_exposure", {})
        if not isinstance(exposure, dict):
          exposure = {}
        exposed = str(exposure.get("exposed_backend", entry.get("backend", "")))
        if exposed == args.backend:
          backend_filtered.append(entry)
      if backend_filtered:
        candidates = backend_filtered
    if not candidates:
      raise RuntimeError("/v1/models response contains no valid model entries")
    selected = candidates[0]

  if selected is None:
    raise RuntimeError("failed to resolve model exposure from /v1/models")

  exposure = selected.get("backend_exposure", {})
  if not isinstance(exposure, dict):
    exposure = {}
  requested_backend = str(
      exposure.get("requested_backend", selected.get("backend", "")))
  exposed_backend = str(exposure.get("exposed_backend", selected.get("backend", "")))
  provider = str(exposure.get("provider", "unknown"))
  fallback = bool(exposure.get("fallback", False))
  fallback_reason = str(exposure.get("fallback_reason", ""))
  return BackendExposureSnapshot(
      model_id=str(selected.get("id", "")),
      requested_backend=requested_backend,
      exposed_backend=exposed_backend,
      provider=provider,
      fallback=fallback,
      fallback_reason=fallback_reason,
  )


def percentile(values: List[float], p: float) -> float:
  if not values:
    return 0.0
  sorted_values = sorted(values)
  rank = max(0, min(len(sorted_values) - 1,
                    int(math.ceil((p / 100.0) * len(sorted_values)) - 1)))
  return sorted_values[rank]


def run_workload(args: argparse.Namespace) -> Tuple[List[RequestResult], float]:
  start = time.perf_counter()
  results: List[RequestResult] = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    futures = [
        pool.submit(run_single_request, args, args.warmup_requests + idx)
        for idx in range(args.requests)
    ]
    for future in concurrent.futures.as_completed(futures):
      results.append(future.result())
  elapsed = time.perf_counter() - start
  return results, elapsed


def apply_gpu_profile(args: argparse.Namespace) -> argparse.Namespace:
  if args.gpu_profile == "none":
    if args.target_total_prefill_tokens <= 0:
      args.target_total_prefill_tokens = 320
    return args
  defaults = GPU_PROFILE_DEFAULTS.get(args.gpu_profile)
  if defaults is None:
    if args.target_total_prefill_tokens <= 0:
      args.target_total_prefill_tokens = 320
    return args

  args.require_cuda_lanes = defaults.require_cuda_lanes
  args.require_cuda_overlap = defaults.require_cuda_overlap
  if args.min_cuda_overlap_duration_ms < 0.0:
    args.min_cuda_overlap_duration_ms = defaults.min_cuda_overlap_duration_ms
  if args.max_cuda_attention_fallbacks < 0.0:
    args.max_cuda_attention_fallbacks = defaults.max_cuda_attention_fallbacks
  if not args.expect_cuda_attention_kernel:
    args.expect_cuda_attention_kernel = defaults.expect_cuda_attention_kernel
  args.mixed_prompt_workload = (
      args.mixed_prompt_workload or defaults.mixed_prompt_workload)
  args.unique_prompts = args.unique_prompts or defaults.unique_prompts
  if args.target_total_prefill_tokens <= 0:
    args.target_total_prefill_tokens = defaults.target_total_prefill_tokens
  else:
    args.target_total_prefill_tokens = max(args.target_total_prefill_tokens,
                                           defaults.target_total_prefill_tokens)
  if args.backend == "":
    args.backend = "cuda"
  return args


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run InferFlux throughput gate.")
  parser.add_argument("--gpu-profile", default="none",
                      choices=["none", "ada_rtx_4000"])
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=18081)
  parser.add_argument("--api-key", default="dev-key-123")
  parser.add_argument("--endpoint", default="/v1/completions",
                      choices=["/v1/completions", "/v1/chat/completions"])
  parser.add_argument("--model", default="")
  parser.add_argument("--prompt", default="Explain why throughput guardrails matter.")
  parser.add_argument("--max-tokens", type=int, default=64)
  parser.add_argument("--concurrency", type=int, default=8)
  parser.add_argument("--requests", type=int, default=96)
  parser.add_argument("--warmup-requests", type=int, default=8)
  parser.add_argument("--mixed-prompt-workload", action="store_true")
  parser.add_argument("--short-prompt-repeat", type=int, default=8)
  parser.add_argument("--long-prompt-repeat", type=int, default=64)
  parser.add_argument("--target-total-prefill-tokens", type=int, default=320)
  parser.add_argument("--unique-prompts", action="store_true")
  parser.add_argument("--request-timeout-sec", type=float, default=30.0)
  parser.add_argument("--startup-timeout-sec", type=float, default=40.0)
  parser.add_argument("--metrics-path", default="/metrics")
  parser.add_argument("--backend", default="")
  parser.add_argument("--min-completion-tok-per-sec", type=float, default=0.0)
  parser.add_argument("--min-req-per-sec", type=float, default=0.0)
  parser.add_argument("--min-success-rate", type=float, default=1.0)
  parser.add_argument("--require-cuda-lanes", action="store_true")
  parser.add_argument("--require-cuda-overlap", action="store_true")
  parser.add_argument("--require-backend-provider", default="any",
                      choices=["any", "native", "universal"])
  parser.add_argument("--require-no-backend-fallback", action="store_true")
  parser.add_argument("--min-cuda-overlap-duration-ms", type=float, default=-1.0)
  parser.add_argument("--max-cuda-attention-fallbacks", type=float, default=-1.0)
  parser.add_argument("--expect-cuda-attention-kernel",
                      choices=["fa3", "fa2", "standard"])
  parser.add_argument("--require-native-forward-passes", action="store_true",
                      help="Require native forward pass counters > 0.")
  parser.add_argument("--require-metrics", action="store_true")
  parser.add_argument("--server-bin", default="")
  parser.add_argument("--config", default="")
  parser.add_argument("--server-env", action="append", default=[],
                      help="Additional server env override (KEY=VALUE).")
  parser.add_argument("--server-log-path", default="")
  args = parser.parse_args()

  if args.concurrency <= 0:
    parser.error("--concurrency must be > 0")
  if args.requests <= 0:
    parser.error("--requests must be > 0")
  if args.max_tokens <= 0:
    parser.error("--max-tokens must be > 0")
  if args.warmup_requests < 0:
    parser.error("--warmup-requests must be >= 0")
  if args.short_prompt_repeat <= 0:
    parser.error("--short-prompt-repeat must be > 0")
  if args.long_prompt_repeat <= 0:
    parser.error("--long-prompt-repeat must be > 0")
  if args.target_total_prefill_tokens <= 0:
    parser.error("--target-total-prefill-tokens must be > 0")
  if not (0.0 <= args.min_success_rate <= 1.0):
    parser.error("--min-success-rate must be in [0.0, 1.0]")
  if args.server_bin and not args.config:
    parser.error("--config is required with --server-bin")
  return apply_gpu_profile(args)


def main() -> int:
  args = parse_args()
  managed_server: Optional[ManagedServer] = None
  backend_exposure: Optional[BackendExposureSnapshot] = None
  backend_exposure_error = ""
  before = MetricsSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "unknown", 0.0, 0.0, 0.0)
  after = MetricsSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "unknown", 0.0, 0.0, 0.0)
  metrics_before_ok = False
  metrics_after_ok = False

  try:
    if args.server_bin:
      managed_server = start_server(args)
    else:
      wait_for_server(args.host, args.port, args.startup_timeout_sec,
                      args.api_key, proc=None)

    if args.warmup_requests > 0:
      run_warmup(args)

    try:
      backend_exposure = fetch_backend_exposure(args)
    except Exception as exc:  # noqa: BLE001
      backend_exposure_error = str(exc)
      if args.require_backend_provider != "any":
        raise RuntimeError(
            f"required backend exposure check failed: {backend_exposure_error}") from exc
      print(f"[warn] backend exposure unavailable: {backend_exposure_error}",
            file=sys.stderr)

    try:
      before = fetch_metrics_snapshot(args)
      metrics_before_ok = True
    except Exception as exc:  # noqa: BLE001
      if args.require_metrics:
        raise
      print(f"[warn] metrics unavailable before run: {exc}", file=sys.stderr)

    results, elapsed = run_workload(args)

    try:
      after = fetch_metrics_snapshot(args)
      metrics_after_ok = True
    except Exception as exc:  # noqa: BLE001
      if args.require_metrics:
        raise
      print(f"[warn] metrics unavailable after run: {exc}", file=sys.stderr)

    metrics_available = metrics_before_ok and metrics_after_ok

    successes = [r for r in results if r.success]
    errors = [r for r in results if not r.success]
    total_completion_usage = sum(r.completion_tokens for r in successes)
    success_rate = len(successes) / float(len(results))
    req_per_sec = len(successes) / elapsed if elapsed > 0 else 0.0
    latencies_ms = [r.latency_s * 1000.0 for r in successes]

    completion_delta = 0.0
    decode_lane_delta = 0.0
    prefill_lane_delta = 0.0
    overlap_events_delta = 0.0
    overlap_duration_delta_ms = 0.0
    attention_fallback_delta = 0.0
    native_fwd_prefill_delta = 0.0
    native_fwd_decode_delta = 0.0
    native_fwd_batch_tokens_delta = 0.0
    token_source = "usage"
    if metrics_available:
      model_delta = after.completion_tokens_model - before.completion_tokens_model
      global_delta = after.completion_tokens_global - before.completion_tokens_global
      if args.model and model_delta <= 0.0 and global_delta > 0.0:
        # Fallback: model label may not match runtime model ID on some setups.
        completion_delta = max(0.0, global_delta)
      elif args.model:
        completion_delta = max(0.0, model_delta)
      else:
        completion_delta = max(0.0, global_delta)
      decode_lane_delta = max(0.0, after.decode_lane_submissions -
                              before.decode_lane_submissions)
      prefill_lane_delta = max(0.0, after.prefill_lane_submissions -
                               before.prefill_lane_submissions)
      overlap_events_delta = max(
          0.0, after.cuda_lane_overlap_events - before.cuda_lane_overlap_events)
      overlap_duration_delta_ms = max(
          0.0, after.cuda_lane_overlap_duration_ms -
          before.cuda_lane_overlap_duration_ms)
      attention_fallback_delta = max(
          0.0, after.cuda_attention_fallback_events -
          before.cuda_attention_fallback_events)
      native_fwd_prefill_delta = max(
          0.0, after.native_forward_prefill - before.native_forward_prefill)
      native_fwd_decode_delta = max(
          0.0, after.native_forward_decode - before.native_forward_decode)
      native_fwd_batch_tokens_delta = max(
          0.0, after.native_forward_batch_tokens - before.native_forward_batch_tokens)
      if completion_delta > 0:
        token_source = "metrics"
    completion_tokens = completion_delta if token_source == "metrics" else float(
        total_completion_usage)
    completion_tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

    summary = {
        "requests_total": len(results),
        "requests_success": len(successes),
        "requests_failed": len(errors),
        "success_rate": round(success_rate, 4),
        "elapsed_sec": round(elapsed, 3),
        "req_per_sec": round(req_per_sec, 3),
        "completion_tokens_total": round(completion_tokens, 2),
        "completion_tok_per_sec": round(completion_tok_per_sec, 3),
        "token_source": token_source,
        "latency_ms_p50": round(percentile(latencies_ms, 50.0), 2),
        "latency_ms_p95": round(percentile(latencies_ms, 95.0), 2),
        "metrics_available": metrics_available,
        "cuda_decode_lane_submissions_delta": round(decode_lane_delta, 2),
        "cuda_prefill_lane_submissions_delta": round(prefill_lane_delta, 2),
        "cuda_lane_overlap_events_delta": round(overlap_events_delta, 2),
        "cuda_lane_overlap_duration_ms_delta": round(overlap_duration_delta_ms, 3),
        "cuda_attention_fallback_events_delta": round(attention_fallback_delta, 2),
        "cuda_attention_kernel_selected": after.selected_attention_kernel,
        "native_forward_prefill_delta": round(native_fwd_prefill_delta, 2),
        "native_forward_decode_delta": round(native_fwd_decode_delta, 2),
        "native_forward_batch_tokens_delta": round(native_fwd_batch_tokens_delta, 2),
        "backend_provider": backend_exposure.provider if backend_exposure else "",
        "backend_exposed": backend_exposure.exposed_backend if backend_exposure else "",
        "backend_requested": backend_exposure.requested_backend if backend_exposure else "",
        "backend_fallback": backend_exposure.fallback if backend_exposure else False,
        "backend_fallback_reason": backend_exposure.fallback_reason if backend_exposure else "",
        "backend_model_id": backend_exposure.model_id if backend_exposure else "",
        "thresholds": {
            "gpu_profile": args.gpu_profile,
            "mixed_prompt_workload": args.mixed_prompt_workload,
            "short_prompt_repeat": args.short_prompt_repeat,
            "long_prompt_repeat": args.long_prompt_repeat,
            "target_total_prefill_tokens": args.target_total_prefill_tokens,
            "min_completion_tok_per_sec": args.min_completion_tok_per_sec,
            "min_req_per_sec": args.min_req_per_sec,
            "min_success_rate": args.min_success_rate,
            "require_cuda_lanes": args.require_cuda_lanes,
            "require_cuda_overlap": args.require_cuda_overlap,
            "require_backend_provider": args.require_backend_provider,
            "require_no_backend_fallback": args.require_no_backend_fallback,
            "min_cuda_overlap_duration_ms": args.min_cuda_overlap_duration_ms,
            "max_cuda_attention_fallbacks": args.max_cuda_attention_fallbacks,
            "expect_cuda_attention_kernel": args.expect_cuda_attention_kernel
            or "",
            "require_native_forward_passes": args.require_native_forward_passes,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    failures: List[str] = []
    if success_rate < args.min_success_rate:
      failures.append(
          f"success rate {success_rate:.4f} below floor {args.min_success_rate:.4f}")
    if completion_tok_per_sec < args.min_completion_tok_per_sec:
      failures.append(
          f"completion tok/s {completion_tok_per_sec:.3f} below floor "
          f"{args.min_completion_tok_per_sec:.3f}")
    if req_per_sec < args.min_req_per_sec:
      failures.append(
          f"req/s {req_per_sec:.3f} below floor {args.min_req_per_sec:.3f}")
    if args.require_backend_provider != "any":
      if backend_exposure is None:
        failures.append(
            "backend exposure missing from /v1/models while provider requirement "
            f"is '{args.require_backend_provider}'"
            + (f" ({backend_exposure_error})" if backend_exposure_error else ""))
      elif backend_exposure.provider != args.require_backend_provider:
        failures.append(
            f"backend provider '{backend_exposure.provider}' != required "
            f"'{args.require_backend_provider}'")
    if args.require_no_backend_fallback:
      if backend_exposure is None:
        failures.append(
            "backend exposure missing from /v1/models while fallback-free "
            "provider path is required")
      elif backend_exposure.fallback:
        reason = (
            f" ({backend_exposure.fallback_reason})"
            if backend_exposure.fallback_reason else "")
        failures.append(
            "backend exposure indicates fallback=true but no fallback is "
            f"required{reason}")
    if args.require_cuda_lanes:
      if prefill_lane_delta <= 0:
        failures.append(
            "cuda prefill lane activity missing (prefill submissions delta must be > 0)")
      # Zero completion tokens means all requests ended at prefill-time EOS.
      # In that case decode-lane submissions are legitimately absent.
      if completion_tokens > 0 and decode_lane_delta <= 0:
        failures.append(
            "cuda decode lane activity missing (decode submissions delta must be > 0 when completion tokens were generated)")
    if args.require_cuda_overlap and overlap_events_delta <= 0:
      failures.append(
          "cuda lane overlap missing (overlap events delta must be > 0)")
    if (args.min_cuda_overlap_duration_ms >= 0.0 and
        overlap_duration_delta_ms < args.min_cuda_overlap_duration_ms):
      failures.append(
          f"cuda overlap duration {overlap_duration_delta_ms:.3f} ms "
          f"below min {args.min_cuda_overlap_duration_ms:.3f} ms")
    if (args.max_cuda_attention_fallbacks >= 0.0 and
        attention_fallback_delta > args.max_cuda_attention_fallbacks):
      failures.append(
          f"cuda attention fallback events {attention_fallback_delta:.3f} "
          f"exceed max {args.max_cuda_attention_fallbacks:.3f}")
    if args.expect_cuda_attention_kernel:
      if after.selected_attention_kernel != args.expect_cuda_attention_kernel:
        failures.append(
            f"selected attention kernel '{after.selected_attention_kernel}' "
            f"!= expected '{args.expect_cuda_attention_kernel}'")
    if args.require_native_forward_passes:
      native_total = native_fwd_prefill_delta + native_fwd_decode_delta
      if native_total <= 0:
        failures.append(
            "native forward pass counters are 0 (expected > 0 with native backend)")

    if errors:
      sample_errors = [r.error for r in errors[:5]]
      print("[errors] sample request failures:", file=sys.stderr)
      for msg in sample_errors:
        print(f"  - {msg}", file=sys.stderr)

    if failures:
      print("[throughput-gate] FAILED", file=sys.stderr)
      for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
      return 1

    print("[throughput-gate] PASSED")
    return 0
  finally:
    if managed_server is not None:
      stop_server(managed_server)
      print(f"[throughput-gate] server log: {managed_server.log_path}")


if __name__ == "__main__":
  sys.exit(main())
