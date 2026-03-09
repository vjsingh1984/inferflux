#!/usr/bin/env bash
#
# GGUF Backend Comparison Benchmark
#
# Measures REAL throughput, memory, and response similarity between
# llama.cpp CUDA and native CUDA backends using an existing GGUF model.
#
# Usage:
#   ./scripts/run_gguf_comparison_benchmark.sh
#   MODEL_PATH=models/other.gguf ./scripts/run_gguf_comparison_benchmark.sh
#   NUM_REQUESTS=20 MAX_TOKENS=64 ./scripts/run_gguf_comparison_benchmark.sh
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration (override via env)
MODEL_PATH="${MODEL_PATH:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf}"
BUILD_DIR="${BUILD_DIR:-./build-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-./gguf_benchmark_results}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"
MAX_TOKENS="${MAX_TOKENS:-32}"
CONCURRENCY="${CONCURRENCY:-4}"
API_KEY="${API_KEY:-dev-key-123}"
QUANTIZE_TO="${QUANTIZE_TO:-}"  # Set to e.g. "q4_k_m" to convert safetensors → GGUF
PORT_LLAMA=18090
PORT_NATIVE=18091

# llama.cpp tools (from submodule)
LLAMA_CONVERT="external/llama.cpp/convert_hf_to_gguf.py"
LLAMA_QUANTIZE="external/llama.cpp/build/bin/llama-quantize"

# Prompts (deterministic, temperature=0)
PROMPTS=(
    "Explain what a hash table is in two sentences."
    "Write a Python function that returns the nth Fibonacci number."
    "What is the capital of France? Answer in one word."
    "Translate hello world to Spanish."
    "List three prime numbers greater than 10."
)

log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
header()   { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

# ============================================================================
# GPU memory measurement (real nvidia-smi)
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

# ============================================================================
# Server management
# ============================================================================
write_config() {
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

start_server() {
    local backend=$1 port=$2
    local config_file="$OUTPUT_DIR/config_${backend}.yaml"
    local log_file="$OUTPUT_DIR/server_${backend}.log"

    write_config "$backend" "$port" "$config_file"

    log "Starting $backend on port $port..."

    # Right-size KV cache for native backend to reduce GPU memory overhead
    local kv_batch=${INFERFLUX_NATIVE_KV_MAX_BATCH:-16}
    local kv_seq=${INFERFLUX_NATIVE_KV_MAX_SEQ:-2048}

    INFERFLUX_PORT_OVERRIDE=$port INFERCTL_API_KEY=$API_KEY \
        INFERFLUX_NATIVE_KV_MAX_BATCH=$kv_batch \
        INFERFLUX_NATIVE_KV_MAX_SEQ=$kv_seq \
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

stop_server() {
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

# ============================================================================
# Request runner
# ============================================================================
send_request() {
    local port=$1 prompt=$2 max_tokens=$3 output_file=$4

    local start_ns=$(date +%s%N)

    local response
    response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],\"max_tokens\":$max_tokens,\"temperature\":0.0}" \
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
text = d.get('choices', [{}])[0].get('message', {}).get('content', '')
tokens = d.get('usage', {}).get('completion_tokens', 0)
print(json.dumps({'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

run_benchmark() {
    local backend=$1 port=$2
    local results_dir="$OUTPUT_DIR/responses_${backend}"
    mkdir -p "$results_dir"

    # Warmup (2 requests, sequential)
    log "  Warmup..."
    for i in 0 1; do
        local prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        send_request "$port" "$prompt" "$MAX_TOKENS" "/dev/null" || true
    done

    # Measure GPU memory after warmup (model loaded)
    sleep 1
    local mem_loaded=$(gpu_mem_mb)

    # Run benchmark
    log "  Running $NUM_REQUESTS requests (concurrency=$CONCURRENCY)..."
    local start_time=$(date +%s%N)

    # Track peak memory during benchmark
    local mem_peak=$mem_loaded
    (
        while true; do
            local m=$(gpu_mem_mb)
            echo "$m"
            sleep 0.2
        done
    ) > "$OUTPUT_DIR/mem_trace_${backend}.txt" &
    local monitor_pid=$!

    # Launch requests with concurrency limit
    local completed=0
    local pids=()
    for i in $(seq 0 $((NUM_REQUESTS - 1))); do
        local prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        local outfile="$results_dir/req_${i}.json"

        send_request "$port" "$prompt" "$MAX_TOKENS" "$outfile" &
        pids+=($!)

        # Enforce concurrency limit
        if [ ${#pids[@]} -ge $CONCURRENCY ]; then
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

    # Compute peak memory
    if [ -f "$OUTPUT_DIR/mem_trace_${backend}.txt" ]; then
        mem_peak=$(sort -rn "$OUTPUT_DIR/mem_trace_${backend}.txt" | head -1)
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
        # Compute percentiles
        local sorted_lats=$(echo $latencies | tr ' ' '\n' | sort -n)
        p50=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.50+0.5){print}")
        p95=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.95+0.5){print}")
        p99=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.99+0.5){print}")
    fi

    # Store results in file for later use
    cat > "$OUTPUT_DIR/stats_${backend}.json" <<EOF
{
    "backend": "$backend",
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
    "gpu_mem_peak_mb": ${mem_peak:-0}
}
EOF

    log_ok "  $backend: $success_count/$NUM_REQUESTS OK, ${tok_per_sec} tok/s, avg ${avg_latency}ms, mem ${mem_loaded}→${mem_peak} MB"
}

# ============================================================================
# Response similarity comparison
# ============================================================================
compare_responses() {
    header "Response Similarity Analysis"

    python3 - "$OUTPUT_DIR" <<'PYEOF'
import json, os, re, sys

results_dir = sys.argv[1]
backends = ["cuda_llama_cpp", "cuda_native"]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0

def overlap(a, b):
    sa, sb = set(a), set(b)
    total = len(sa) + len(sb)
    return 2.0 * len(sa & sb) / total if total else 1.0

# Load responses per backend
responses = {}
for backend in backends:
    resp_dir = os.path.join(results_dir, f"responses_{backend}")
    if not os.path.isdir(resp_dir):
        print(f"  No responses for {backend}")
        continue
    responses[backend] = {}
    for f in sorted(os.listdir(resp_dir)):
        if not f.endswith(".json"):
            continue
        idx = int(f.split("_")[1].split(".")[0])
        try:
            with open(os.path.join(resp_dir, f)) as fh:
                data = json.load(fh)
            responses[backend][idx] = data.get("text", "")
        except:
            pass

if len(responses) < 2:
    print("  Cannot compare — need both backends")
    sys.exit(0)

# Compare matching request indices
a_resp = responses.get(backends[0], {})
b_resp = responses.get(backends[1], {})
common = sorted(set(a_resp.keys()) & set(b_resp.keys()))

if not common:
    print("  No matching requests to compare")
    sys.exit(0)

exact = 0
jaccards = []
overlaps = []
comparisons = []

for idx in common:
    ta, tb = a_resp[idx], b_resp[idx]
    is_exact = ta == tb
    if is_exact:
        exact += 1
    toks_a, toks_b = tokenize(ta), tokenize(tb)
    j = jaccard(toks_a, toks_b)
    o = overlap(toks_a, toks_b)
    jaccards.append(j)
    overlaps.append(o)
    comparisons.append((idx, is_exact, j, o, ta[:60], tb[:60]))

n = len(common)
print(f"  Compared: {n} request pairs")
print(f"  Exact match rate: {exact}/{n} ({100*exact/n:.0f}%)")
print(f"  Mean Jaccard:     {sum(jaccards)/n:.3f}")
print(f"  Mean overlap:     {sum(overlaps)/n:.3f}")
print()

fmt = "  {:>4}  {:>6}  {:>8}  {:>8}  {:<30}  {:<30}"
print(fmt.format("Req#", "Match", "Jaccard", "Overlap",
                  backends[0][:30], backends[1][:30]))
print("  " + "-" * 96)
for idx, match, j, o, ta, tb in comparisons:
    m = "\033[0;32mYES\033[0m" if match else "\033[0;31mNO\033[0m"
    print(f"  {idx:>4}  {m:>15}  {j:>8.3f}  {o:>8.3f}  {ta:<30}  {tb:<30}")

# Save comparison JSON
comp_data = {
    "num_compared": n,
    "exact_match_rate": exact / n,
    "mean_jaccard": sum(jaccards) / n,
    "mean_overlap": sum(overlaps) / n,
}
with open(os.path.join(results_dir, "similarity.json"), "w") as f:
    json.dump(comp_data, f, indent=2)

PYEOF
}

# ============================================================================
# Final report
# ============================================================================
print_report() {
    header "Backend Comparison Report"

    echo ""
    echo "  Model:       $MODEL_PATH"
    echo "  GPU:         $(gpu_name)"
    echo "  GPU Memory:  $(gpu_mem_total_mb) MB total"
    echo "  Requests:    $NUM_REQUESTS (concurrency=$CONCURRENCY)"
    echo "  Max tokens:  $MAX_TOKENS"
    echo ""

    printf "  ${BOLD}%-20s %8s %10s %10s %10s %10s %12s %12s${NC}\n" \
        "Backend" "Tok/s" "Avg(ms)" "P50(ms)" "P95(ms)" "P99(ms)" "GPU Load" "GPU Peak"
    printf "  %-20s %8s %10s %10s %10s %10s %12s %12s\n" \
        "--------------------" "--------" "----------" "----------" "----------" "----------" "------------" "------------"

    for backend in cuda_native cuda_llama_cpp; do
        local stats_file="$OUTPUT_DIR/stats_${backend}.json"
        if [ ! -f "$stats_file" ]; then
            printf "  %-20s %8s\n" "$backend" "SKIPPED"
            continue
        fi

        python3 -c "
import json
with open('$stats_file') as f:
    s = json.load(f)
print(f\"  {s['backend']:<20} {s['tok_per_sec']:>8} {s['avg_latency_ms']:>10} \"
      f\"{s['p50_latency_ms']:>10} {s['p95_latency_ms']:>10} {s['p99_latency_ms']:>10} \"
      f\"{s['gpu_mem_loaded_mb']:>10} MB {s['gpu_mem_peak_mb']:>10} MB\")
"
    done

    echo ""

    # Speedup ratio
    if [ -f "$OUTPUT_DIR/stats_cuda_llama_cpp.json" ] && [ -f "$OUTPUT_DIR/stats_cuda_native.json" ]; then
        python3 -c "
import json
with open('$OUTPUT_DIR/stats_cuda_llama_cpp.json') as f:
    llama = json.load(f)
with open('$OUTPUT_DIR/stats_cuda_native.json') as f:
    native = json.load(f)
if llama['tok_per_sec'] > 0:
    ratio = native['tok_per_sec'] / llama['tok_per_sec']
    color = '\033[0;32m' if ratio >= 0.9 else '\033[1;33m' if ratio >= 0.5 else '\033[0;31m'
    print(f\"  Speedup (native / llama.cpp): {color}{ratio:.2f}x\033[0m\")
    mem_saved = llama['gpu_mem_peak_mb'] - native['gpu_mem_peak_mb']
    if mem_saved > 0:
        print(f\"  GPU memory savings: \033[0;32m{mem_saved} MB\033[0m\")
    elif mem_saved < 0:
        print(f\"  GPU memory overhead: \033[1;33m{-mem_saved} MB\033[0m\")
"
    fi

    echo ""
}

# ============================================================================
# Safetensors → GGUF conversion (uses llama.cpp tools from external/)
# ============================================================================
convert_safetensors_to_gguf() {
    local model_dir=$1
    local quant=${2:-q4_k_m}

    header "Converting safetensors → GGUF ($quant)"

    if [ ! -f "$LLAMA_CONVERT" ]; then
        log_err "convert_hf_to_gguf.py not found at $LLAMA_CONVERT"
        log "  Run: git submodule update --init --recursive"
        return 1
    fi

    local model_name=$(basename "$model_dir")
    local fp16_gguf="$OUTPUT_DIR/${model_name}-f16.gguf"
    local quant_gguf="$OUTPUT_DIR/${model_name}-${quant}.gguf"

    # Step 1: Convert to FP16 GGUF
    if [ -f "$fp16_gguf" ]; then
        log "  FP16 GGUF already exists: $fp16_gguf"
    else
        log "  Converting $model_dir → FP16 GGUF..."
        python3 "$LLAMA_CONVERT" "$model_dir" \
            --outfile "$fp16_gguf" --outtype f16 2>&1 | tail -5

        if [ ! -f "$fp16_gguf" ]; then
            log_err "FP16 conversion failed"
            return 1
        fi
        log_ok "  Created: $fp16_gguf ($(du -h "$fp16_gguf" | cut -f1))"
    fi

    # Step 2: Quantize
    if [ "$quant" = "f16" ]; then
        echo "$fp16_gguf"
        return 0
    fi

    if [ -f "$quant_gguf" ]; then
        log "  Quantized GGUF already exists: $quant_gguf"
    else
        if [ ! -f "$LLAMA_QUANTIZE" ]; then
            log_err "llama-quantize not found at $LLAMA_QUANTIZE"
            log "  Build llama.cpp: cd external/llama.cpp && cmake -B build && cmake --build build -j"
            return 1
        fi

        log "  Quantizing to $quant..."
        "$LLAMA_QUANTIZE" "$fp16_gguf" "$quant_gguf" "$quant" 2>&1 | tail -3

        if [ ! -f "$quant_gguf" ]; then
            log_err "Quantization failed"
            return 1
        fi
        log_ok "  Created: $quant_gguf ($(du -h "$quant_gguf" | cut -f1))"
    fi

    echo "$quant_gguf"
}

# ============================================================================
# Main
# ============================================================================
main() {
    header "GGUF Backend Comparison Benchmark"
    echo "  All measurements are REAL — no simulated or expected values."
    echo ""

    # Handle safetensors → GGUF conversion
    if [ -d "$MODEL_PATH" ] || [[ "$MODEL_PATH" == *.safetensors ]] || [ -n "$QUANTIZE_TO" ]; then
        if [ -d "$MODEL_PATH" ]; then
            mkdir -p "$OUTPUT_DIR"
            local quant="${QUANTIZE_TO:-q4_k_m}"
            local converted
            converted=$(convert_safetensors_to_gguf "$MODEL_PATH" "$quant")
            if [ $? -ne 0 ] || [ -z "$converted" ]; then
                log_err "Conversion failed"
                exit 1
            fi
            MODEL_PATH="$converted"
            log "  Using converted model: $MODEL_PATH"
        else
            log_err "MODEL_PATH is not a directory for safetensors conversion: $MODEL_PATH"
            exit 1
        fi
    fi

    # Validate
    if [ ! -f "$MODEL_PATH" ]; then
        log_err "Model not found: $MODEL_PATH"
        echo "Set MODEL_PATH=path/to/model.gguf"
        echo "Or point to safetensors dir: MODEL_PATH=models/my-model-safetensors QUANTIZE_TO=q4_k_m"
        exit 1
    fi

    # Auto-build if binary is missing or stale (older than any source file).
    local needs_build=false
    if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
        needs_build=true
    else
        # Check if any source file is newer than the binary.
        local newest_src
        newest_src=$(find runtime server scheduler model cli net io policy \
            -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' 2>/dev/null \
            | xargs stat --format='%Y' 2>/dev/null | sort -rn | head -1)
        local bin_mtime
        bin_mtime=$(stat --format='%Y' "$BUILD_DIR/inferfluxd" 2>/dev/null || echo 0)
        if [ -n "$newest_src" ] && [ "$newest_src" -gt "$bin_mtime" ]; then
            needs_build=true
        fi
    fi
    if $needs_build; then
        log "Building $BUILD_DIR/inferfluxd (CUDA enabled)..."
        cmake -S . -B "$BUILD_DIR" -DENABLE_CUDA=ON >/dev/null 2>&1 || {
            log_err "cmake configure failed"; exit 1; }
        cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5 || {
            log_err "cmake build failed"; exit 1; }
        log "Build complete: $BUILD_DIR/inferfluxd"
    fi
    if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
        log_err "inferfluxd not found at $BUILD_DIR/inferfluxd after build attempt"
        exit 1
    fi

    if ! nvidia-smi >/dev/null 2>&1; then
        log_err "nvidia-smi not found — CUDA GPU required"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    local mem_baseline=$(gpu_mem_mb)
    log "GPU baseline memory: ${mem_baseline} MB"
    log "GPU: $(gpu_name), $(gpu_mem_total_mb) MB"

    # Benchmark each backend
    for backend in cuda_native cuda_llama_cpp; do
        local port=$PORT_LLAMA
        [ "$backend" = "cuda_native" ] && port=$PORT_NATIVE

        echo ""
        header "Benchmarking: $backend"

        if ! start_server "$backend" "$port"; then
            log_err "Failed to start $backend — skipping"
            continue
        fi

        run_benchmark "$backend" "$port"
        stop_server "$backend"

        # Let GPU memory settle
        sleep 3
    done

    # Compare
    echo ""
    compare_responses
    echo ""
    print_report

    # Save combined JSON
    python3 -c "
import json, os, glob
combined = {'timestamp': '$(date -Iseconds)', 'model': '$MODEL_PATH',
            'gpu': '$(gpu_name)', 'num_requests': $NUM_REQUESTS,
            'max_tokens': $MAX_TOKENS, 'concurrency': $CONCURRENCY,
            'backends': {}}
for backend in ['cuda_llama_cpp', 'cuda_native']:
    sf = '$OUTPUT_DIR/stats_' + backend + '.json'
    if os.path.exists(sf):
        with open(sf) as f:
            combined['backends'][backend] = json.load(f)
sf = '$OUTPUT_DIR/similarity.json'
if os.path.exists(sf):
    with open(sf) as f:
        combined['similarity'] = json.load(f)
outf = '$OUTPUT_DIR/comparison_$(date +%Y%m%d_%H%M%S).json'
with open(outf, 'w') as f:
    json.dump(combined, f, indent=2)
print(f'Results saved to: {outf}')
"

    log_ok "Benchmark complete!"
}

# Cleanup on exit
cleanup() {
    stop_server cuda_llama_cpp 2>/dev/null || true
    stop_server cuda_native 2>/dev/null || true
}
trap cleanup EXIT

main "$@"
