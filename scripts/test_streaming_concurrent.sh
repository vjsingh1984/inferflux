#!/bin/bash
# Quick test: Streaming vs Non-Streaming Concurrent Throughput
# Verifies that streaming mode fixes the concurrent throughput issue

set -e

API_KEY="dev-key-123"
PORT="8080"
NUM_REQUESTS=8
MAX_TOKENS=50

echo "========================================"
echo "Streaming vs Non-Streaming Test"
echo "========================================"
echo "Requests: $NUM_REQUESTS"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Test 1: Non-streaming (current benchmark)
echo "Test 1: Non-Streaming (stream: false)"
echo "----------------------------------------"

START=$(date +%s)

for i in $(seq 1 $NUM_REQUESTS); do
    (
        RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Count to 5 (request $i)\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": false
            }")

        if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
            echo "  [Request $i] SUCCESS"
        else
            echo "  [Request $i] FAILED"
        fi
    ) &
done

wait

END=$(date +%s)
NON_STREAM_TIME=$((END - START))
NON_STREAM_TOK_PER_SEC=$(awk "BEGIN {printf \"%.2f\", ($NUM_REQUESTS * $MAX_TOKENS) / $NON_STREAM_TIME}")

echo "  Time: ${NON_STREAM_TIME}s"
echo "  Throughput: ${NON_STREAM_TOK_PER_SEC} tok/s"
echo ""

# Wait a bit between tests
sleep 3

# Test 2: Streaming (proposed fix)
echo "Test 2: Streaming (stream: true)"
echo "----------------------------------------"

START=$(date +%s)

for i in $(seq 1 $NUM_REQUESTS); do
    (
        # Streaming request - capture all output
        curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Count to 5 (request $i)\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"stream\": true
            }" > /tmp/req_$i.txt

        # Count lines in streaming response (each chunk is a line)
        LINES=$(wc -l < /tmp/req_$i.txt)
        if [ $LINES -gt 0 ]; then
            echo "  [Request $i] SUCCESS ($LINES chunks)"
        else
            echo "  [Request $i] FAILED"
        fi
    ) &
done

wait

END=$(date +%s)
STREAM_TIME=$((END - START))
STREAM_TOK_PER_SEC=$(awk "BEGIN {printf \"%.2f\", ($NUM_REQUESTS * $MAX_TOKENS) / $STREAM_TIME}")

echo "  Time: ${STREAM_TIME}s"
echo "  Throughput: ${STREAM_TOK_PER_SEC} tok/s"
echo ""

# Summary
echo "========================================"
echo "Results Summary"
echo "========================================"

SPEEDUP=$(awk "BEGIN {printf \"%.2fx\", $NON_STREAM_TOK_PER_SEC / $STREAM_TOK_PER_SEC}")

if [ "$STREAM_TOK_PER_SEC" -gt "$NON_STREAM_TOK_PER_SEC" ]; then
    echo "Streaming is ${SPEEDUP}x FASTER ✅"
else
    echo "Streaming is ${SPEEDUP}x SLOWER ❌"
fi

echo ""
echo "Non-Streaming: ${NON_STREAM_TOK_PER_SEC} tok/s (${NON_STREAM_TIME}s)"
echo "Streaming:     ${STREAM_TOK_PER_SEC} tok/s (${STREAM_TIME}s)"
echo ""

# Expected: Streaming should be 10-20x faster
# Non-Streaming: ~2 tok/s (392s for 8 requests)
# Streaming: ~35-50 tok/s (~10-15s for 8 requests)
