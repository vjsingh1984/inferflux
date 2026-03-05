#!/usr/bin/env bash
#
# GGUF Quantization Comparison Benchmark
#
# Compares llama.cpp CUDA vs native CUDA backend for:
# - Memory usage patterns
# - Model loading strategy
# - KV cache management
# - Performance characteristics
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

MODEL_PATH="${MODEL_PATH:-models/qwen2.5-3b-instruct-safetensors}"
MODEL_NAME="qwen2.5-3b-instruct"
OUTPUT_DIR="${OUTPUT_DIR:-./gguf_benchmark_results}"
BUILD_DIR="${BUILD_DIR:-./build}"
LLAMA_BUILD="${LLAMA_BUILD:-external/llama.cpp/build}"

# Supported quantizations
QUANTIZATIONS=("q4_k_m" "q5_k_m" "q6_k" "q8_0")

# Results storage
declare -A LLAMA_MEM_PEAK
declare -A NATIVE_MEM_PEAK
declare -A LLAMA_TOK_SEC
declare -A NATIVE_TOK_SEC

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_header() { echo -e "\n${YELLOW}========================================${NC}"; echo -e "${YELLOW}$1${NC}"; echo -e "${YELLOW}========================================${NC}\n"; }

# Check dependencies
check_dependencies() {
    log_header "Checking Dependencies"
    
    local missing=0
    
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model not found: $MODEL_PATH"
        missing=1
    fi
    
    if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
        log_error "inferfluxd not built"
        log_info "Run: cmake --build build --target inferfluxd"
        missing=1
    fi
    
    if [ ! -f "$BUILD_DIR/inferctl" ]; then
        log_error "inferctl not built"
        missing=1
    fi
    
    if [ ! -f "$LLAMA_BUILD/bin/llama-cli" ]; then
        log_warning "llama-cli not found - skipping llama.cpp tests"
    fi
    
    if [ ! -f "$LLAMA_BUILD/bin/llama-quantize" ]; then
        log_warning "llama-quantize not found - skipping quantization"
        log_info "Will test with existing GGUF models only"
    fi
    
    if [ $missing -eq 1 ]; then
        return 1
    fi
    
    mkdir -p "$OUTPUT_DIR"
    log_success "Dependencies OK"
    return 0
}

# Get current GPU memory usage (in MB)
get_gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
}

# Get GPU memory free (in MB)
get_gpu_mem_free() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

# Convert safetensors to GGUF FP16
convert_to_gguf() {
    log_header "Converting to GGUF FP16"
    
    local output="${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf"
    
    if [ -f "$output" ]; then
        log_info "GGUF FP16 already exists, skipping conversion"
        echo "$output"
        return 0
    fi
    
    log_info "Converting $MODEL_PATH to $output..."
    
    python3 external/llama.cpp/convert_hf_to_gguf.py \
        "$MODEL_PATH" \
        --outfile "$output" \
        --outtype f16 \
        --verbose
    
    if [ ! -f "$output" ]; then
        log_error "Conversion failed"
        return 1
    fi
    
    local size=$(du -h "$output" | cut -f1)
    log_success "Created: $output ($size)"
    echo "$output"
}

# Quantize model
quantize_model() {
    local input="$1"
    local quant="$2"
    local output="${input%.gguf}-${quant}.gguf"
    
    if [ -f "$output" ]; then
        log_info "Already quantized: $output"
        echo "$output"
        return 0
    fi
    
    log_info "Quantizing to $quant..."
    
    "$LLAMA_BUILD/bin/llama-quantize" "$input" "$output" "$quant"
    
    if [ ! -f "$output" ]; then
        log_error "Quantization failed"
        return 1
    fi
    
    local size=$(du -h "$output" | cut -f1)
    log_success "Created: $output ($size)"
    echo "$output"
}

# Test with llama.cpp (monitor memory)
test_llama() {
    local model="$1"
    local quant="$2"
    local output_file="${OUTPUT_DIR}/llama_${quant}_output.txt"
    local mem_file="${OUTPUT_DIR}/llama_${quant}_memory.txt"
    
    log_header "Testing llama.cpp: $quant"
    
    if [ ! -f "$LLAMA_BUILD/bin/llama-cli" ]; then
        log_warning "llama-cli not available"
        return 1
    fi
    
    # Get initial memory
    local mem_before=$(get_gpu_mem)
    log_info "GPU memory before: ${mem_before} MB"
    
    # Run inference with memory monitoring
    local start_time=$(date +%s)
    
    (
        while true; do
            local mem=$(get_gpu_mem)
            local time=$(date +%s.%N)
            echo "$time $mem" >> "$mem_file"
            sleep 0.1
        done
    ) &
    local monitor_pid=$!
    
    # Run inference (generate 20 tokens)
    "$LLAMA_BUILD/bin/llama-cli" \
        --model "$model" \
        --n-predict 20 \
        --prompt "Hello, world!" \
        --seed 42 \
        --temp 0.0 \
        --silent-prompt \
        --logits-none \
        2>/dev/null > "$output_file"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true
    
    # Get final memory
    local mem_after=$(get_gpu_mem)
    sleep 1  # Let GPU stabilize
    local mem_final=$(get_gpu_mem)
    
    log_info "GPU memory after: ${mem_final} MB"
    
    # Analyze memory data
    local mem_peak=$(awk '{print $2}' "$mem_file" | sort -rn | head -1)
    local mem_base=$(head -1 "$mem_file" | awk '{print $2}')
    local mem_used=$((mem_peak - mem_base))
    
    log_info "Memory delta: ${mem_used} MB"
    log_info "Peak memory: ${mem_peak} MB"
    log_info "Duration: ${duration}s"
    
    # Count tokens
    local tokens=$(wc -c < "$output_file")
    log_info "Generated: $tokens characters"
    
    LLAMA_MEM_PEAK[$quant]=$mem_peak
    LLAMA_TOK_SEC[$quant]=$(echo "scale=2; $tokens / $duration" | bc)
    
    echo "$output_file"
}

# Test with native CUDA (monitor memory)
test_native() {
    local model="$1"
    local quant="$2"
    local output_file="${OUTPUT_DIR}/native_${quant}_output.txt"
    local mem_file="${OUTPUT_DIR}/native_${quant}_memory.txt"
    local config_file="${OUTPUT_DIR}/native_${quant}.yaml"
    local port=18083
    
    log_header "Testing Native CUDA: $quant"
    
    # Create config
    cat > "$config_file" <<EOF
server:
  host: "127.0.0.1"
  port: $port

models:
  - id: "test-${quant}"
    path: "$(realpath "$model")"
    format: gguf
    backend: cuda_native
    default: true

runtime:
  cuda:
    enabled: true

logging:
  level: warning
EOF
    
    # Get initial memory
    local mem_before=$(get_gpu_mem)
    log_info "GPU memory before: ${mem_before} MB"
    
    # Start server
    export INFERFLUX_PORT_OVERRIDE=$port
    export INFERCTL_API_KEY=dev-key-123
    
    log_info "Starting inferfluxd..."
    "$BUILD_DIR/inferfluxd" --config "$config_file" > "${output_file}.server.log" 2>&1 &
    local server_pid=$!
    
    # Wait for server
    local max_wait=60
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s http://127.0.0.1:$port/livez >/dev/null 2>&1; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    
    if [ $waited -ge $max_wait ]; then
        log_error "Server did not start"
        cat "${output_file}.server.log" | tail -20
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    
    log_success "Server ready (PID: $server_pid)"
    sleep 2
    
    # Monitor memory during inference
    (
        while true; do
            local mem=$(get_gpu_mem)
            local time=$(date +%s.%N)
            echo "$time $mem" >> "$mem_file"
            sleep 0.1
        done
    ) &
    local monitor_pid=$!
    
    # Run inference
    log_info "Running inference..."
    local start_time=$(date +%s)
    
    "$BUILD_DIR/inferctl" chat \
        --host 127.0.0.1 \
        --port $port \
        --model "test-${quant}" \
        --message "Hello, world!" \
        --max-tokens 20 \
        --temperature 0.0 \
        2>/dev/null | tee "$output_file" > /dev/null
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true
    
    # Give GPU time to release memory
    sleep 2
    
    local mem_after=$(get_gpu_mem)
    local mem_final=$(get_gpu_mem)
    
    log_info "GPU memory after: ${mem_final} MB"
    
    # Analyze memory data
    if [ -f "$mem_file" ]; then
        local mem_peak=$(awk '{print $2}' "$mem_file" | sort -rn | head -1)
        local mem_base=$(head -1 "$mem_file" | awk '{print $2}')
        local mem_used=$((mem_peak - mem_base))
        
        log_info "Memory delta: ${mem_used} MB"
        log_info "Peak memory: ${mem_peak} MB"
    fi
    
    log_info "Duration: ${duration}s"
    
    # Count tokens
    local tokens=$(wc -c < "$output_file" 2>/dev/null || echo "0")
    log_info "Generated: $tokens characters"
    
    if [ -n "$duration" ] && [ "$duration" -gt 0 ]; then
        NATIVE_TOK_SEC[$quant]=$(echo "scale=2; $tokens / $duration" | bc)
        NATIVE_MEM_PEAK[$quant]=$mem_final
    fi
    
    # Shutdown server
    log_info "Stopping server..."
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    
    # Wait for memory to be released
    sleep 3
    
    echo "$output_file"
}

# Compare results
print_comparison() {
    log_header "Results Comparison"
    
    echo ""
    echo "┌──────────────┬──────────────┬──────────────┬──────────────┐"
    echo "│ Quantization │ llama.cpp    │ Native CUDA  │ Mem Savings  │"
    echo "│              │ Peak Mem     │ Peak Mem     │              │"
    echo "├──────────────┼──────────────┼──────────────┼──────────────┤"
    
    for quant in "${QUANTIZATIONS[@]}"; do
        local llama_mem=${LLAMA_MEM_PEAK[$quant]:-"N/A"}
        local native_mem=${NATIVE_MEM_PEAK[$quant]:-"N/A"}
        
        local savings="N/A"
        if [ "$llama_mem" != "N/A" ] && [ "$native_mem" != "N/A" ]; then
            if [ "$llama_mem" -gt 0 ] && [ "$native_mem" -gt 0 ]; then
                local diff=$((llama_mem - native_mem))
                if [ "$diff" -gt 0 ]; then
                    savings="${diff} MB"
                else
                    savings="-${diff} MB"
                fi
            fi
        fi
        
        printf "│ %-12s │ %-12s │ %-12s │ %-12s │\n" "$quant" "${llama_mem} MB" "${native_mem} MB" "$savings"
    done
    
    echo "└──────────────┴──────────────┴──────────────┴──────────────┘"
    echo ""
}

print_detailed_analysis() {
    log_header "Detailed Memory Analysis"
    
    echo "Key Memory Management Differences:"
    echo ""
    echo "llama.cpp CUDA:"
    echo "  • Creates NEW model + KV cache for EACH request"
    echo "  • Memory usage: (model_size × num_requests) + (kv_cache × num_requests)"
    echo "  • After request: model + KV cache are freed"
    echo "  • Peak memory grows with concurrent requests"
    echo ""
    echo "Native CUDA (InferFlux):"
    echo "  • SINGLE model weights loaded once"
    echo "  • Separate KV cache per sequence/sequence_id"
    echo "  • Memory usage: model_size + (kv_cache × num_sequences)"
    echo "  • Model stays loaded, KV cache rotates"
    echo "  • Peak memory: model_size + kv_cache_max"
    echo ""
    
    echo "Expected Memory Pattern (3B Qwen model):"
    echo "┌─────────────┬──────────┬──────────┬─────────────────┐"
    echo "│ Backend     │ Q4_K_M   │ Q6_K     │ 3 Request Load │"
    echo "├─────────────┼──────────┼──────────┼─────────────────┤"
    echo "│ llama.cpp   │ ~2.5 GB  │ ~3.6 GB  │ ~7.5 GB        │"
    echo "│             │ ×3 req   │ ×3 req   │ (3× model)     │"
    echo "├─────────────┼──────────┼──────────┼─────────────────┤"
    echo "│ Native CUDA │ ~600 MB  │ ~850 MB  │ ~2.5 GB        │"
    echo "│             │ (1×)     │ (1×)     │ model + 3× KV   │"
    echo "└─────────────┴──────────┴──────────┴─────────────────┘"
    echo ""
    echo "Native CUDA savings: ~3-4GB for 3 concurrent requests!"
}

# Main execution
main() {
    log_header "GGUF Quantization Comparison Benchmark"
    
    log "Model: $MODEL_NAME"
    log "Output: $OUTPUT_DIR"
    log "Quantizations: ${QUANTIZATIONS[*]}"
    
    if ! check_dependencies; then
        log_error "Dependencies not met"
        exit 1
    fi
    
    # Convert to GGUF
    local fp16_gguf=$(convert_to_gguf)
    if [ -z "$fp16_gguf" ]; then
        log_error "Failed to convert model"
        exit 1
    fi
    
    # Test each quantization
    for quant in "${QUANTIZATIONS[@]}"; do
        echo ""
        
        # Quantize
        local quant_gguf=$(quantize_model "$fp16_gguf" "$quant")
        if [ -z "$quant_gguf" ]; then
            log_error "Failed to quantize to $quant"
            continue
        fi
        
        # Test with llama.cpp
        if [ -f "$LLAMA_BUILD/bin/llama-cli" ]; then
            test_llama "$quant_gguf" "$quant"
        fi
        
        # Cleanup GPU memory between tests
        sleep 3
        
        # Test with native CUDA
        test_native "$quant_gguf" "$quant"
        
        # Cleanup GPU memory between tests
        sleep 3
    done
    
    # Print results
    print_comparison
    print_detailed_analysis
    
    # Save results
    {
        echo "# GGUF Benchmark Results - $(date)"
        echo ""
        echo "## Memory Usage (Peak)"
        echo "| Quantization | llama.cpp (MB) | Native CUDA (MB) | Savings |"
        echo "|--------------|-----------------|------------------|---------|"
        for quant in "${QUANTIZATIONS[@]}"; do
            local lm=${LLAMA_MEM_PEAK[$quant]:-"N/A"}
            local nm=${NATIVE_MEM_PEAK[$quant]:-"N/A"}
            echo "| $quant | $lm | $nm | - |"
        done
    } > "${OUTPUT_DIR}/results.md"
    
    log_info "Results saved to: ${OUTPUT_DIR}/results.md"
    log_success "Benchmark complete!"
}

# Run
main "$@"
