#!/bin/bash
# Profile InferFlux backends with Nsight Systems

set -e

BACKEND="${1:-native}"
CONFIG="${2:-config/server.cuda.yaml}"
OUTPUT_DIR="/tmp/inferflux_profiles"
DURATION_SEC="${3:-30}"

mkdir -p "$OUTPUT_DIR"

echo "=== Profiling InferFlux Backend: $BACKEND ==="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR/$BACKEND/"
echo "Duration: ${DURATION_SEC}s"
echo ""

# Set backend exposure policy for profile mode
env_vars=""
if [ "$BACKEND" = "native" ]; then
    env_vars="INFERFLUX_BACKEND_PREFER_NATIVE=1"
elif [ "$BACKEND" = "llamacpp" ]; then
    env_vars="INFERFLUX_BACKEND_PREFER_NATIVE=0"
else
    echo "Unknown backend: $BACKEND (use 'native' or 'llamacpp')"
    exit 1
fi

# Kill any existing server
pkill -9 -f inferfluxd 2>/dev/null || true
sleep 2

# Start server in background
echo "Starting server..."
env $env_vars ./build/inferfluxd --config "$CONFIG" > /tmp/profile_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:8080/healthz > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        break
    fi
    sleep 1
done

# Warmup request
echo "Sending warmup request..."
curl -s -X POST http://127.0.0.1:8080/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer dev-key-123" \
    -d '{"prompt": "Warmup", "max_tokens": 10}' > /dev/null

# Start profiling
echo ""
echo "Starting Nsight Systems profiling for ${DURATION_SEC}s..."
echo "Running workload..."

# Run nsys profiling in background
nsys profile -t cuda,nvtx -o "$OUTPUT_DIR/${BACKEND}_profile" \
    --force-overwrite=true \
    --duration=$DURATION_SEC \
    --capture-range=nvtx \
    --capture-range-end=stop \
    --capture-range-end=none \
    python3 scripts/run_throughput_gate.py \
        --port 8080 \
        --gpu-profile ada_rtx_4000 \
        --backend cuda \
        --requests 24 \
        --min-completion-tok-per-sec 10.0 \
    > /tmp/profile_output.log 2>&1 &

NSYS_PID=$!

# Wait for profiling to complete
sleep $((DURATION_SEC + 10))

# Check if nsys is still running
if kill -0 $NSYS_PID 2>/dev/null; then
    echo "Waiting for profiling to complete..."
    wait $NSYS_PID || true
fi

# Stop server
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
pkill -9 -f inferfluxd 2>/dev/null || true

# Generate stats
echo ""
echo "Generating statistics..."
if nsys stats "$OUTPUT_DIR/${BACKEND}_profile".qdrep > "$OUTPUT_DIR/${BACKEND}_stats.txt" 2>&1; then
    echo "✓ Stats saved to: $OUTPUT_DIR/${BACKEND}_stats.txt"
fi

echo ""
echo "=== Profile Complete ==="
echo "Results: $OUTPUT_DIR/${BACKEND}_profile.qdrep"
echo "Stats:  $OUTPUT_DIR/${BACKEND}_stats.txt"
echo ""
echo "View with: nsys gui $OUTPUT_DIR/${BACKEND}_profile.qdrep"
echo ""
echo "Key metrics:"
echo "  GPU Time: $(grep -oP 'GPU Time: \K[\d.]+(?= ms)' "$OUTPUT_DIR/${BACKEND}_stats.txt" 2>/dev/null || echo 'N/A') ms"
echo "  CPU Time: $(grep -oP 'CPU Time: \K[\d.]+(?= ms)' "$OUTPUT_DIR/${BACKEND}_stats.txt" 2>/dev/null || echo 'N/A') ms"
echo "  GPU Util: $(grep -oP 'GPU Util: \K[\d.]+' "$OUTPUT_DIR/${BACKEND}_stats.txt" 2>/dev/null || echo 'N/A')%"
