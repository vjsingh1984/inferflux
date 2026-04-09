#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Dict, List


def load_prompts(prompt_suite_path: str, single_prompt: str, num_requests: int) -> List[str]:
    if single_prompt:
        return [single_prompt for _ in range(num_requests)]
    suite = json.loads(Path(prompt_suite_path).read_text(encoding="utf-8"))
    prompts = [entry.get("prompt", "") for entry in suite.get("prompts", []) if entry.get("prompt")]
    if not prompts:
        raise RuntimeError("prompt suite did not contain any prompts")
    return [prompts[i % len(prompts)] for i in range(num_requests)]


def _extract_openai_text(choice: Dict) -> str:
    if not isinstance(choice, dict):
        return ""
    text = choice.get("text")
    if isinstance(text, str) and text:
        return text
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
    return ""


def _parse_openai_stream(stdout: str) -> Dict:
    pieces: List[str] = []
    usage_tokens = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        payload = json.loads(data)
        choices = payload.get("choices", [])
        if choices:
            pieces.append(_extract_openai_text(choices[0]))
        usage = payload.get("usage", {})
        if isinstance(usage, dict):
            usage_tokens = max(usage_tokens, int(usage.get("completion_tokens", 0) or 0))
    text = "".join(pieces).strip()
    tokens = usage_tokens or (len(text.split()) if text else 0)
    return {"text": text, "tokens": tokens}


def request_openai(endpoint: str, model: str, prompt: str, max_tokens: int,
                   request_id: str, api_key: str, inferflux_style: bool,
                   stream: bool) -> Dict:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    if endpoint.rstrip("/").endswith("/v1/chat/completions"):
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    cmd = [
        "curl",
        "-sf",
        "-X",
        "POST",
        endpoint,
        "-H",
        "Content-Type: application/json",
    ]
    if api_key:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]
    if inferflux_style:
        cmd += ["-H", f"x-inferflux-request-id: {request_id}"]
    cmd += ["-d", json.dumps(payload)]
    started = time.time_ns()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "curl request failed")
    latency_ms = (time.time_ns() - started) // 1_000_000
    if stream:
        parsed = _parse_openai_stream(proc.stdout)
        text = parsed["text"]
        tokens = parsed["tokens"]
    else:
        body = json.loads(proc.stdout)
        text = _extract_openai_text(body.get("choices", [{}])[0]).strip()
        tokens = body.get("usage", {}).get("completion_tokens", 0)
        if not tokens and text:
            tokens = len(text.split())
    return {
        "request_id": request_id,
        "text": text,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }


def _parse_ollama_stream(stdout: str) -> Dict:
    pieces: List[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        piece = payload.get("response", "")
        if isinstance(piece, str) and piece:
            pieces.append(piece)
    text = "".join(pieces).strip()
    return {"text": text, "tokens": len(text.split()) if text else 0}


def request_ollama(endpoint: str, model: str, prompt: str, max_tokens: int,
                   request_id: str, stream: bool) -> Dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": max_tokens, "temperature": 0.0},
        "stream": stream,
    }
    cmd = [
        "curl",
        "-sf",
        "-X",
        "POST",
        endpoint,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ]
    started = time.time_ns()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "curl request failed")
    latency_ms = (time.time_ns() - started) // 1_000_000
    if stream:
        parsed = _parse_ollama_stream(proc.stdout)
        text = parsed["text"]
        tokens = parsed["tokens"]
    else:
        body = json.loads(proc.stdout)
        msg = body.get("message", {})
        text = msg.get("content", body.get("response", "")).strip()
        tokens = body.get("eval_count", len(text.split()))
    return {
        "request_id": request_id,
        "text": text,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }


def write_error(path: Path, marker: str) -> None:
    path.write_text(marker, encoding="utf-8")


def worker(index: int, prompt: str, args: argparse.Namespace) -> Dict:
    output_path = Path(args.output_dir) / f"req_{index}.json"
    request_id = f"bench-c{args.concurrency}-{index}"
    try:
        if args.backend_kind == "ollama":
            result = request_ollama(args.endpoint, args.model, prompt, args.max_tokens,
                                    request_id, args.stream)
        else:
            result = request_openai(
                args.endpoint,
                args.model,
                prompt,
                args.max_tokens,
                request_id,
                args.api_key,
                inferflux_style=(args.backend_kind == "inferflux"),
                stream=args.stream,
            )
        output_path.write_text(json.dumps(result), encoding="utf-8")
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        write_error(output_path, "ERROR")
        return {}
    except Exception:
        write_error(output_path, "PARSE_ERROR")
        return {}


def run_warmup(prompts: List[str], args: argparse.Namespace) -> None:
    # Warmup absorbs CUDA JIT kernel compilation (can take 30-60s on first
    # run after a CUDA toolkit upgrade or clean build).  We send up to 4
    # sequential warmup requests with a generous timeout to ensure the model
    # is fully compiled before timed benchmarks begin.
    warmup_count = min(4, len(prompts))
    for i in range(warmup_count):
        request_id = f"warmup-{i}"
        try:
            if args.backend_kind == "ollama":
                request_ollama(args.endpoint, args.model, prompts[i],
                               min(args.max_tokens, 16),
                               request_id, False)
            else:
                request_openai(
                    args.endpoint,
                    args.model,
                    prompts[i],
                    min(args.max_tokens, 16),  # Short warmup responses
                    request_id,
                    args.api_key,
                    inferflux_style=(args.backend_kind == "inferflux"),
                    stream=False,
                )
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent benchmark request driver")
    parser.add_argument("--backend-kind", choices=["inferflux", "openai", "ollama"],
                        required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt-suite")
    parser.add_argument("--single-prompt", default="")
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    if not args.single_prompt and not args.prompt_suite:
        raise SystemExit("--prompt-suite is required when --single-prompt is not set")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompt_suite, args.single_prompt, args.num_requests)
    if not args.skip_warmup:
        run_warmup(prompts, args)
        time.sleep(1)

    results: List[Dict] = []
    wall_started = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, i, prompt, args)
                   for i, prompt in enumerate(prompts)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    wall_time_ms = (time.perf_counter_ns() - wall_started) // 1_000_000

    success = [result for result in results if result]
    summary = {
        "success_count": len(success),
        "total_tokens": sum(int(result["tokens"]) for result in success),
        "total_latency_ms": sum(int(result["latency_ms"]) for result in success),
        "latencies": [int(result["latency_ms"]) for result in success],
        "wall_time_ms": int(wall_time_ms),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
