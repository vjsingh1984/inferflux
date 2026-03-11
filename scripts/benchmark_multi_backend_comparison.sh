#!/usr/bin/env bash
#
# Multi-Backend Comparison Benchmark
#
# Benchmarks cuda_native, cuda_llama_cpp, Ollama, and LM Studio backends.
# Measures throughput, latency percentiles, and GPU memory consumption across
# multiple concurrency levels to generate scaling curves.
#
# Usage:
#   ./scripts/benchmark_multi_backend_comparison.sh [model.gguf]
#   ./scripts/benchmark_multi_backend_comparison.sh models/qwen2.5-3b-instruct-q4_k_m.gguf
#   CONCURRENCY_LEVELS="1,2,4,8" ./scripts/benchmark_multi_backend_comparison.sh model.gguf
#
# Environment Variables:
#   OLLAMA_HOST           - Ollama server URL (default: http://192.168.1.20:11434)
#   LMSTUDIO_HOST         - LM Studio server URL (default: http://192.168.1.20:1234)
#   LMSTUDIO_MODEL        - LM Studio model id (default: auto-discover first /v1/models entry)
#   CONCURRENCY_LEVELS    - Comma-separated concurrency levels (default: 1,2,4,8,16)
#   NUM_REQUESTS          - Requests per concurrency level (default: 32)
#   MAX_TOKENS            - Max tokens per request (default: 64)
#   OUTPUT_DIR            - Results directory (default: ./multi_backend_benchmark_results)
#   SKIP_OLLAMA           - Skip Ollama benchmark (default: false)
#   SKIP_LMSTUDIO         - Skip LM Studio benchmark (default: false)
#   BUILD_DIR             - Build directory (default: auto-detect ./build or ./build-cuda)
#   PORT_NATIVE           - Port for cuda_native (default: 18090)
#   PORT_LLAMA            - Port for cuda_llama_cpp (default: 18091)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

DEFAULT_MODEL="models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH="${1:-${MODEL_PATH:-$DEFAULT_MODEL}}"
BUILD_DIR="${BUILD_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./multi_backend_benchmark_results}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,2,4,8,16}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
MAX_TOKENS="${MAX_TOKENS:-64}"
API_KEY="${API_KEY:-dev-key-123}"

# Backend ports
PORT_NATIVE="${PORT_NATIVE:-18090}"
PORT_LLAMA="${PORT_LLAMA:-18091}"
OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.1.20:11434}"
LMSTUDIO_HOST="${LMSTUDIO_HOST:-http://192.168.1.20:1234}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-}"

# Skip Ollama?
SKIP_OLLAMA="${SKIP_OLLAMA:-false}"
SKIP_LMSTUDIO="${SKIP_LMSTUDIO:-false}"

# Logging
log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
header()   { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

resolve_build_dir() {
    if [ -n "$BUILD_DIR" ]; then
        echo "$BUILD_DIR"
        return 0
    fi

    if [ -f "./build/inferfluxd" ]; then
        echo "./build"
        return 0
    fi

    if [ -f "./build-cuda/inferfluxd" ]; then
        echo "./build-cuda"
        return 0
    fi

    echo "./build"
}

# ============================================================================
# GPU Memory Measurement
# ============================================================================

gpu_mem_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_mem_total_mb() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_name() {
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
}

gpu_utilization() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# ============================================================================
# Test Prompts (deterministic, temperature=0)
# ============================================================================

PROMPTS=(
    "Explain what a hash table is in two sentences."
    "Write a Python function that returns the nth Fibonacci number."
    "What is the capital of France? Answer in one word."
    "Translate 'hello world' to Spanish."
    "List three prime numbers greater than 10."
)

# ============================================================================
# InferFlux Server Management
# ============================================================================

write_inferflux_config() {
    local backend=$1 port=$2 config_file=$3
    cat > "$config_file" <<EOF
server:
  host: "127.0.0.1"
  http_port: $port
  max_concurrent: 128
  enable_metrics: true

models:
  - id: bench-model
    path: "$(realpath "$MODEL_PATH")"
    format: gguf
    backend: $backend
    default: true

runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: true
    phase_overlap:
      enabled: true
  backend_exposure:
    prefer_native: $([ "$backend" = "cuda_native" ] && echo "true" || echo "false")
    allow_llama_cpp_fallback: $([ "$backend" = "cuda_native" ] && echo "false" || echo "true")
  scheduler:
    max_batch_size: 32
    max_batch_tokens: 16384
    min_batch_size: 1
    batch_accumulation_ms: 2
  disaggregated:
    prefill_pool_size: ${INFERFLUX_SCHED_PREFILL_POOL_SIZE:-1}
    decode_pool_size: ${INFERFLUX_SCHED_DECODE_POOL_SIZE:-0}
  paged_kv:
    cpu_pages: 4096
    eviction: lru

auth:
  api_keys:
    - key: $API_KEY
      scopes: [generate, read, admin]
  rate_limit_per_minute: 600

guardrails:
  blocklist: []

logging:
  level: warning
  format: text
EOF
}

start_inferflux_server() {
    local backend=$1 port=$2
    local config_file="$OUTPUT_DIR/config_${backend}.yaml"
    local log_file="$OUTPUT_DIR/server_${backend}.log"

    write_inferflux_config "$backend" "$port" "$config_file"

    log "Starting $backend on port $port..."

    # Set environment variables for native backend
    local kv_batch="${INFERFLUX_NATIVE_KV_MAX_BATCH:-16}"
    local kv_seq="${INFERFLUX_NATIVE_KV_MAX_SEQ:-2048}"
    local strict="$([ "$backend" = "cuda_native" ] && echo "1" || echo "0")"

    INFERFLUX_PORT_OVERRIDE=$port INFERCTL_API_KEY=$API_KEY \
        INFERFLUX_LOG_LEVEL=warning \
        INFERFLUX_NATIVE_KV_MAX_BATCH=$kv_batch \
        INFERFLUX_NATIVE_KV_MAX_SEQ=$kv_seq \
        INFERFLUX_NATIVE_CUDA_STRICT=$strict \
        INFERFLUX_NATIVE_DISABLE_PARITY_DELEGATE=$strict \
        "$BUILD_DIR/inferfluxd" --config "$config_file" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$OUTPUT_DIR/${backend}.pid"

    # Wait for readiness (max 60s)
    local waited=0
    while [ $waited -lt 60 ]; do
        if ! kill -0 $pid 2>/dev/null; then
            log_err "$backend server exited early. Last log lines:"
            tail -20 "$log_file"
            return 1
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$port/livez" >/dev/null 2>&1; then
            log_ok "$backend ready (PID $pid)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_err "$backend did not start in 60s. Last log lines:"
    tail -20 "$log_file"
    kill $pid 2>/dev/null || true
    return 1
}

stop_inferflux_server() {
    local backend=$1
    local pidfile="$OUTPUT_DIR/${backend}.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping $backend (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

reset_benchmark_artifacts() {
    local backend=$1 concurrency=$2
    rm -rf "$OUTPUT_DIR/responses_${backend}/c${concurrency}"
    rm -f "$OUTPUT_DIR/stats_${backend}_c${concurrency}.json" \
          "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt" \
          "$OUTPUT_DIR/admin_cache_${backend}_c${concurrency}.json"
}

capture_inferflux_admin_cache_snapshot() {
    local backend=$1 port=$2 concurrency=$3
    local snapshot_file="$OUTPUT_DIR/admin_cache_${backend}_c${concurrency}.json"
    local stats_file="$OUTPUT_DIR/stats_${backend}_c${concurrency}.json"

    if ! curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/v1/admin/cache" > "$snapshot_file" 2>/dev/null; then
        log_warn "  Failed to capture admin cache snapshot for $backend @ c=$concurrency"
        return 0
    fi

    python3 - "$snapshot_file" "$stats_file" <<'PYEOF'
import json, sys

snapshot_path, stats_path = sys.argv[1], sys.argv[2]
with open(snapshot_path) as f:
    snapshot = json.load(f)
with open(stats_path) as f:
    stats = json.load(f)

memory = snapshot.get("memory", {})
stats["cache_snapshot"] = {
    "size": snapshot.get("size", 0),
    "capacity": snapshot.get("capacity", 0),
    "hits": snapshot.get("hits", 0),
    "misses": snapshot.get("misses", 0),
    "hit_rate": snapshot.get("hit_rate", 0.0),
    "partial_hits": snapshot.get("partial_hits", 0),
    "matched_tokens": snapshot.get("matched_tokens", 0),
    "kv_reuse_count": snapshot.get("kv_reuse_count", 0),
    "kv_reuse_tokens": snapshot.get("kv_reuse_tokens", 0),
}
stats["memory_snapshot"] = memory

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

native_model = memory.get("native_model", {})
native_kv = memory.get("native_kv", {})
paged_kv = memory.get("paged_kv", {})

print("native_model_reserved_bytes=" + str(native_model.get("reserved_bytes", 0)))
print("native_model_in_use_bytes=" + str(native_model.get("in_use_bytes", 0)))
print("native_kv_active_bytes=" + str(native_kv.get("active_bytes", 0)))
print("native_kv_prefix_retained_bytes=" +
      str(native_kv.get("prefix_retained_bytes", 0)))
print("paged_kv_used_bytes=" + str(paged_kv.get("used_bytes", 0)))
print("paged_kv_prefix_retained_bytes=" +
      str(paged_kv.get("prefix_retained_bytes", 0)))
PYEOF
}

# ============================================================================
# Ollama Availability Check
# ============================================================================

check_ollama_available() {
    log "Checking Ollama availability at $OLLAMA_HOST..."

    if ! curl -sf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
        log_warn "Ollama not available at $OLLAMA_HOST"
        log_warn "Set OLLAMA_HOST=http://your-host:port or SKIP_OLLAMA=true"
        return 1
    fi

    log_ok "Ollama is available"
    return 0
}

check_lmstudio_available() {
    log "Checking LM Studio availability at $LMSTUDIO_HOST..."

    local models_json
    models_json=$(curl -sf "$LMSTUDIO_HOST/v1/models" 2>/dev/null) || {
        log_warn "LM Studio not available at $LMSTUDIO_HOST"
        log_warn "Set LMSTUDIO_HOST=http://your-host:port or SKIP_LMSTUDIO=true"
        return 1
    }

    if [ -z "$LMSTUDIO_MODEL" ]; then
        LMSTUDIO_MODEL=$(printf '%s' "$models_json" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    print(models[0].get('id', '') if models else '')
except Exception:
    print('')
")
    fi

    if [ -z "$LMSTUDIO_MODEL" ]; then
        log_warn \"LM Studio responded but no model id could be discovered from /v1/models\"
        log_warn \"Set LMSTUDIO_MODEL explicitly\"
        return 1
    fi

    log_ok "LM Studio is available (model=$LMSTUDIO_MODEL)"
    return 0
}

# ============================================================================
# Request Runner
# ============================================================================

send_inferflux_request() {
    local port=$1 prompt=$2 max_tokens=$3 output_file=$4 request_id=$5

    mkdir -p "$(dirname "$output_file")"

    local start_ns=$(date +%s%N)
    local prompt_json
    prompt_json=$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

    local response
    response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -H "x-inferflux-request-id: $request_id" \
        -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$max_tokens,\"temperature\":0.0}" \
        --max-time 120 2>/dev/null) || {
        echo "ERROR" > "$output_file"
        return 1
    }

    local end_ns=$(date +%s%N)
    local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    local text
    text=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
text = d.get('choices', [{}])[0].get('text', '')
tokens = d.get('usage', {}).get('completion_tokens', 0)
print(json.dumps({'request_id': '$request_id', 'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

send_ollama_request() {
    local prompt=$1 max_tokens=$2 output_file=$3 request_id=$4

    mkdir -p "$(dirname "$output_file")"

    local start_ns=$(date +%s%N)

    local response
    response=$(curl -sf -X POST "${OLLAMA_HOST}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${OLLAMA_MODEL:-qwen2.5:3b}\",\"prompt\":$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),\"options\":{\"num_predict\":$max_tokens,\"temperature\":0.0},\"stream\":false}" \
        --max-time 120 2>/dev/null) || {
        echo "ERROR" > "$output_file"
        return 1
    }

    local end_ns=$(date +%s%N)
    local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    local text
    text=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
text = d.get('response', '')
tokens = len(text.split())  # Ollama doesn't return token count
print(json.dumps({'request_id': '$request_id', 'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

send_lmstudio_request() {
    local prompt=$1 max_tokens=$2 output_file=$3 request_id=$4

    mkdir -p "$(dirname "$output_file")"

    local start_ns=$(date +%s%N)
    local prompt_json
    prompt_json=$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

    local response
    response=$(curl -sf -X POST "${LMSTUDIO_HOST}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${LMSTUDIO_MODEL}\",\"prompt\":$prompt_json,\"max_tokens\":$max_tokens,\"temperature\":0.0}" \
        --max-time 120 2>/dev/null) || {
        echo "ERROR" > "$output_file"
        return 1
    }

    local end_ns=$(date +%s%N)
    local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    local text
    text=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
text = d.get('choices', [{}])[0].get('text', '')
tokens = d.get('usage', {}).get('completion_tokens', 0)
print(json.dumps({'request_id': '$request_id', 'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

# ============================================================================
# Benchmark Runner
# ============================================================================

run_benchmark() {
    local backend=$1 concurrency=$2 port_or_url=$3
    local backend_kind=$4

    local results_dir="$OUTPUT_DIR/responses_${backend}/c${concurrency}"
    mkdir -p "$results_dir"

    # Warmup
    log "  Warmup (2 requests)..."
    for i in 0 1; do
        local prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        case "$backend_kind" in
            ollama)
                send_ollama_request "$prompt" "$MAX_TOKENS" "/dev/null" "warmup-$i" || true
                ;;
            lmstudio)
                send_lmstudio_request "$prompt" "$MAX_TOKENS" "/dev/null" "warmup-$i" || true
                ;;
            *)
                send_inferflux_request "$port_or_url" "$prompt" "$MAX_TOKENS" "/dev/null" "warmup-$i" || true
                ;;
        esac
    done

    # Measure GPU memory after warmup
    sleep 1
    local mem_loaded=$(gpu_mem_mb)

    # Run benchmark
    log "  Running $NUM_REQUESTS requests (concurrency=$concurrency)..."
    local start_time=$(date +%s%N)

    # Track peak memory and GPU utilization
    local mem_peak=$mem_loaded
    local gpu_util_peak=0
    (
        while true; do
            local m=$(gpu_mem_mb)
            local u=$(gpu_utilization)
            echo "$m $u"
            sleep 0.2
        done
    ) > "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt" &
    local monitor_pid=$!

    # Launch requests with concurrency limit
    local pids=()
    for i in $(seq 0 $((NUM_REQUESTS - 1))); do
        local prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        local outfile="$results_dir/req_${i}.json"
        local request_id="bench-c${concurrency}-${i}"

        case "$backend_kind" in
            ollama)
                send_ollama_request "$prompt" "$MAX_TOKENS" "$outfile" "$request_id" &
                ;;
            lmstudio)
                send_lmstudio_request "$prompt" "$MAX_TOKENS" "$outfile" "$request_id" &
                ;;
            *)
                send_inferflux_request "$port_or_url" "$prompt" "$MAX_TOKENS" "$outfile" "$request_id" &
                ;;
        esac
        pids+=($!)

        # Enforce concurrency limit
        if [ ${#pids[@]} -ge $concurrency ]; then
            wait "${pids[0]}" 2>/dev/null || true
            pids=("${pids[@]:1}")
        fi
    done

    # Wait for remaining
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    local end_time=$(date +%s%N)
    local total_ms=$(( (end_time - start_time) / 1000000 ))

    # Stop memory monitor
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true

    # Compute peak memory and GPU utilization
    if [ -f "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt" ]; then
        mem_peak=$(awk '{print $1}' "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt" | sort -rn | head -1)
        gpu_util_peak=$(awk '{print $2}' "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt" | sort -rn | head -1)
    fi

    # Aggregate results
    local total_tokens=0
    local total_latency=0
    local success_count=0
    local latencies=""

    for f in "$results_dir"/req_*.json; do
        [ -f "$f" ] || continue
        local content=$(cat "$f")
        if [ "$content" = "ERROR" ] || [ "$content" = "PARSE_ERROR" ]; then
            continue
        fi

        local tokens=$(echo "$content" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens',0))" 2>/dev/null || echo 0)
        local lat=$(echo "$content" | python3 -c "import json,sys; print(json.load(sys.stdin).get('latency_ms',0))" 2>/dev/null || echo 0)

        total_tokens=$((total_tokens + tokens))
        total_latency=$((total_latency + lat))
        success_count=$((success_count + 1))
        latencies="$latencies $lat"
    done

    # Compute stats
    local tok_per_sec=0
    if [ $total_ms -gt 0 ]; then
        tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')")
    fi

    local avg_latency=0
    local p50=0 p95=0 p99=0
    if [ $success_count -gt 0 ]; then
        avg_latency=$((total_latency / success_count))
        local sorted_lats=$(echo $latencies | tr ' ' '\n' | sort -n)
        p50=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.50+0.5){print}")
        p95=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.95+0.5){print}")
        p99=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.99+0.5){print}")
    fi

    # Store results
    cat > "$OUTPUT_DIR/stats_${backend}_c${concurrency}.json" <<EOF
{
    "backend": "$backend",
    "concurrency": $concurrency,
    "tok_per_sec": $tok_per_sec,
    "avg_latency_ms": $avg_latency,
    "p50_latency_ms": ${p50:-0},
    "p95_latency_ms": ${p95:-0},
    "p99_latency_ms": ${p99:-0},
    "total_tokens": $total_tokens,
    "total_time_ms": $total_ms,
    "success_count": $success_count,
    "total_requests": $NUM_REQUESTS,
    "gpu_mem_loaded_mb": $mem_loaded,
    "gpu_mem_peak_mb": ${mem_peak:-0},
    "gpu_util_peak_percent": ${gpu_util_peak:-0}
}
EOF

    local memory_summary=""
    if [ "$backend_kind" = "inferflux" ]; then
        memory_summary=$(capture_inferflux_admin_cache_snapshot \
            "$backend" "$port_or_url" "$concurrency" 2>/dev/null || true)
    fi

    log_ok "  $backend @ c=$concurrency: $success_count/$NUM_REQUESTS OK, ${tok_per_sec} tok/s, avg ${avg_latency}ms, GPU ${gpu_util_peak}%"
    if [ -n "${memory_summary:-}" ]; then
        log "  Memory snapshot:"
        printf '%s\n' "$memory_summary" | sed 's/^/    /'
    fi

    if [ $success_count -lt $NUM_REQUESTS ]; then
        return 1
    fi
}

# ============================================================================
# Main Benchmark Loop
# ============================================================================

main() {
    BUILD_DIR=$(resolve_build_dir)

    header "Multi-Backend Comparison Benchmark"
    echo "  Model:       $MODEL_PATH"
    echo "  GPU:         $(gpu_name)"
    echo "  GPU Memory:  $(gpu_mem_total_mb) MB total"
    echo "  Requests:    $NUM_REQUESTS per concurrency level"
    echo "  Max tokens:  $MAX_TOKENS"
    echo "  Concurrency: $CONCURRENCY_LEVELS"
    echo "  Build dir:   $BUILD_DIR"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    # Validate model
    if [ ! -f "$MODEL_PATH" ]; then
        log_err "Model not found: $MODEL_PATH"
        exit 1
    fi

    # Validate inferfluxd
    if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
        log_err "inferfluxd not found at $BUILD_DIR/inferfluxd"
        log "Build with: cmake -S . -B $BUILD_DIR -DENABLE_CUDA=ON && cmake --build $BUILD_DIR -j"
        exit 1
    fi

    # Check nvidia-smi
    if ! nvidia-smi >/dev/null 2>&1; then
        log_err "nvidia-smi not found — CUDA GPU required"
        exit 1
    fi

    local mem_baseline=$(gpu_mem_mb)
    log "GPU baseline memory: ${mem_baseline} MB"

    # Convert comma-separated concurrency levels to array
    IFS=',' read -ra CONCURRENCY_ARRAY <<< "$CONCURRENCY_LEVELS"

    # Backend configurations
    declare -A BACKEND_PORTS
    declare -A BACKEND_KIND
    declare -A BACKEND_AVAILABLE

    BACKEND_PORTS[cuda_native]=$PORT_NATIVE
    BACKEND_KIND[cuda_native]=inferflux

    BACKEND_PORTS[cuda_llama_cpp]=$PORT_LLAMA
    BACKEND_KIND[cuda_llama_cpp]=inferflux

    BACKEND_PORTS[ollama]="$OLLAMA_HOST"
    BACKEND_KIND[ollama]=ollama

    BACKEND_PORTS[lmstudio]="$LMSTUDIO_HOST"
    BACKEND_KIND[lmstudio]=lmstudio

    for backend in cuda_native cuda_llama_cpp ollama lmstudio; do
        for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
            reset_benchmark_artifacts "$backend" "$concurrency"
        done
    done

    # Check Ollama availability
    if [ "$SKIP_OLLAMA" = "true" ]; then
        log_warn "Skipping Ollama benchmark (SKIP_OLLAMA=true)"
        BACKEND_AVAILABLE[ollama]=false
    elif check_ollama_available; then
        BACKEND_AVAILABLE[ollama]=true
    else
        BACKEND_AVAILABLE[ollama]=false
    fi

    if [ "$SKIP_LMSTUDIO" = "true" ]; then
        log_warn "Skipping LM Studio benchmark (SKIP_LMSTUDIO=true)"
        BACKEND_AVAILABLE[lmstudio]=false
    elif check_lmstudio_available; then
        BACKEND_AVAILABLE[lmstudio]=true
    else
        BACKEND_AVAILABLE[lmstudio]=false
    fi

    # Start InferFlux servers
    for backend in cuda_native cuda_llama_cpp; do
        local port=${BACKEND_PORTS[$backend]}
        echo ""
        header "Starting: $backend"
        if start_inferflux_server "$backend" "$port"; then
            BACKEND_AVAILABLE[$backend]=true
        else
            log_err "Failed to start $backend"
            BACKEND_AVAILABLE[$backend]=false
        fi
    done

    # Run benchmarks for all backends and concurrency levels
    for backend in cuda_native cuda_llama_cpp ollama lmstudio; do
        if [ "${BACKEND_AVAILABLE[$backend]}" != "true" ]; then
            continue
        fi

        for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
            echo ""
            header "Benchmarking: $backend @ concurrency=$concurrency"

            local port_or_url="${BACKEND_PORTS[$backend]}"
            local backend_kind="${BACKEND_KIND[$backend]}"

            if ! run_benchmark "$backend" "$concurrency" "$port_or_url" "$backend_kind"; then
                log_warn "  Some requests failed for $backend @ c=$concurrency"
            fi

            # Let GPU memory settle between runs
            sleep 2
        done
    done

    # Stop servers
    echo ""
    log "Stopping servers..."
    stop_inferflux_server cuda_native 2>/dev/null || true
    stop_inferflux_server cuda_llama_cpp 2>/dev/null || true
    sleep 2

    # Generate comparison report
    echo ""
    header "Scaling Analysis Report"
    generate_report

    # Save combined JSON
    save_combined_results

    log_ok "Benchmark complete! Results saved to $OUTPUT_DIR"
}

# ============================================================================
# Report Generation
# ============================================================================

generate_report() {
    python3 - "$OUTPUT_DIR" "$CONCURRENCY_LEVELS" <<'PYEOF'
import json, os, sys

output_dir = sys.argv[1]
concurrency_levels = sys.argv[2].split(',')

backends = ['cuda_native', 'cuda_llama_cpp', 'ollama', 'lmstudio']

# Load all results
results = {}
for backend in backends:
    results[backend] = {}
    for c in concurrency_levels:
        stats_file = os.path.join(output_dir, f"stats_{backend}_c{c}.json")
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                results[backend][int(c)] = json.load(f)

# Print header
print()
print(f"{'Backend':<20} {'C':<4} {'Tok/s':>10} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'GPU Mem':>10} {'GPU %':>8}")
print("-" * 92)

# Print results sorted by concurrency
for c in sorted([int(x) for x in concurrency_levels]):
    for backend in backends:
        if c in results[backend]:
            r = results[backend][c]
            print(f"{backend:<20} {c:<4} {r['tok_per_sec']:>10.1f} "
                  f"{r['avg_latency_ms']:>10.0f} {r['p50_latency_ms']:>10.0f} "
                  f"{r['p95_latency_ms']:>10.0f} {r['gpu_mem_peak_mb']:>10.0f} "
                  f"{r['gpu_util_peak_percent']:>8.0f}")
    print()

# Print scaling efficiency (speedup from c=1)
print()
print("Scaling Efficiency (Speedup from c=1):")
print("-" * 60)
for backend in backends:
    if 1 in results[backend]:
        baseline = results[backend][1]['tok_per_sec']
        if baseline > 0:
            print(f"{backend:<20}: ", end='')
            for c in sorted([int(x) for x in concurrency_levels if int(x) > 1]):
                if c in results[backend]:
                    speedup = results[backend][c]['tok_per_sec'] / baseline
                    efficiency = (speedup / c) * 100
                    print(f"c={c}={speedup:.2f}x ({efficiency:.0f}%)  ", end='')
            print()

# Print memory efficiency
print()
print("Memory Efficiency:")
print("-" * 60)
for backend in backends:
    if 1 in results[backend]:
        baseline_mem = results[backend][1]['gpu_mem_peak_mb']
        print(f"{backend:<20}: ", end='')
        for c in sorted([int(x) for x in concurrency_levels if int(x) > 1]):
            if c in results[backend]:
                mem = results[backend][c]['gpu_mem_peak_mb']
                mem_ratio = mem / baseline_mem if baseline_mem > 0 else 0
                print(f"c={c}={mem:.0f}MB ({mem_ratio:.2f}x)  ", end='')
        print()

PYEOF
}

# ============================================================================
# Save Combined Results
# ============================================================================

save_combined_results() {
    python3 - "$OUTPUT_DIR" "$CONCURRENCY_LEVELS" "$MODEL_PATH" <<'PYEOF'
import json, os, sys, time

output_dir = sys.argv[1]
concurrency_levels = sys.argv[2].split(',')
model_path = sys.argv[3]

backends = ['cuda_native', 'cuda_llama_cpp', 'ollama', 'lmstudio']

combined = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'model': model_path,
    'num_requests': int(os.environ.get('NUM_REQUESTS', 32)),
    'max_tokens': int(os.environ.get('MAX_TOKENS', 64)),
    'concurrency_levels': [int(c) for c in concurrency_levels],
    'backends': {}
}

for backend in backends:
    combined['backends'][backend] = {}
    for c in concurrency_levels:
        stats_file = os.path.join(output_dir, f"stats_{backend}_c{c}.json")
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                combined['backends'][backend][int(c)] = json.load(f)

output_file = os.path.join(output_dir, f"combined_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Combined results saved to: {output_file}")

# Also generate CSV for easy plotting
csv_file = os.path.join(output_dir, f"scaling_curves_{time.strftime('%Y%m%d_%H%M%S')}.csv")
with open(csv_file, 'w') as f:
    f.write("backend,concurrency,tok_per_sec,avg_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,gpu_mem_peak_mb,gpu_util_peak_percent,")
    f.write("native_model_reserved_bytes,native_model_in_use_bytes,native_kv_active_bytes,native_kv_prefix_retained_bytes,")
    f.write("paged_kv_used_bytes,paged_kv_prefix_retained_bytes\n")
    for backend in backends:
        for c in concurrency_levels:
            ci = int(c)
            if ci in combined['backends'].get(backend, {}):
                r = combined['backends'][backend][ci]
                memory = r.get('memory_snapshot', {})
                native_model = memory.get('native_model', {})
                native_kv = memory.get('native_kv', {})
                paged_kv = memory.get('paged_kv', {})
                f.write(f"{backend},{c},{r['tok_per_sec']},{r['avg_latency_ms']},{r['p50_latency_ms']},")
                f.write(f"{r['p95_latency_ms']},{r['p99_latency_ms']},{r['gpu_mem_peak_mb']},{r['gpu_util_peak_percent']},")
                f.write(f"{native_model.get('reserved_bytes', 0)},{native_model.get('in_use_bytes', 0)},")
                f.write(f"{native_kv.get('active_bytes', 0)},{native_kv.get('prefix_retained_bytes', 0)},")
                f.write(f"{paged_kv.get('used_bytes', 0)},{paged_kv.get('prefix_retained_bytes', 0)}\n")

print(f"CSV for plotting saved to: {csv_file}")

PYEOF
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    stop_inferflux_server cuda_native 2>/dev/null || true
    stop_inferflux_server cuda_llama_cpp 2>/dev/null || true
}

trap cleanup EXIT

# ============================================================================
# Entry Point
# ============================================================================

main "$@"
