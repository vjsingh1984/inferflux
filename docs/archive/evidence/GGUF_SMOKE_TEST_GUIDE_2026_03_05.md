# GGUF Quantization Smoke Test Guide

**Date**: 2026-03-04  
**Purpose**: Validate GGUF quantization support with native CUDA backend

## Overview

This guide explains how to run smoke tests for each supported GGUF quantization type (Q4_K_M, Q5_K_M, Q6_K, Q8_0) using both llama.cpp CUDA backend and the new native CUDA backend.

## Prerequisites

### Required

1. **Built InferFlux binaries**:
   ```bash
   cmake --build build --target inferfluxd
   cmake --build build --target inferctl
   ```

2. **GGUF quantized models** - Either:
   - Pre-quantized GGUF models (downloaded from HuggingFace)
   - Convert from safetensors using llama.cpp tools

### Optional (for llama.cpp comparison)

3. **Built llama.cpp**:
   ```bash
   cd external/llama.cpp
   cmake -B build -DLLAMA_CUDA=ON
   cmake --build build --target llama-cli
   cmake --build build --target llama-quantize
   ```

4. **Python with pyyaml**:
   ```bash
   pip install pyyaml
   ```

## Quick Start

### Method 1: Test with Pre-Quantized GGUF Models (Fastest)

If you already have GGUF models, use the Python smoke test:

```bash
# Download some GGUF models to a directory
mkdir -p ~/.inferflux/models/qwen-gguf
# Copy your GGUF files here (q4_k_m, q5_k_m, q6_k, q8_0 variants)

# Run the smoke test
python3 scripts/test_gguf_native_smoke.py \
    --model-dir ~/.inferflux/models/qwen-gguf \
    --num-tokens 20
```

**Expected Output**:
```
========================================
GGUF Native CUDA Smoke Test
========================================
[INFO] Model directory: /home/user/.inferflux/models/qwen-gguf
[INFO] Found q4_k_m: qwen2.5-0.5b-instruct-q4_k_m.gguf
[INFO] Found q5_k_m: qwen2.5-0.5b-instruct-q5_k_m.gguf
...
[SUCCESS] ✅ q4_k_m: SUCCESS (245ms)
[SUCCESS] ✅ q5_k_m: SUCCESS (267ms)
[SUCCESS] ✅ q6_k: SUCCESS (312ms)
[SUCCESS] ✅ q8_0: SUCCESS (198ms)
```

### Method 2: Full Test with Conversion (llama.cpp Required)

Convert safetensors → GGUF → Test with both backends:

```bash
# Set model path (safetensors format)
export MODEL_PATH=/path/to/qwen2.5-0.5b-instruct

# Run full smoke test
./scripts/test_gguf_quantization_smoke.sh \
    --model-path "$MODEL_PATH" \
    --num-tokens 20
```

This will:
1. Convert safetensors → GGUF FP16
2. Quantize to each type (Q4_K_M, Q5_K_M, Q6_K, Q8_0)
3. Test with llama.cpp CUDA
4. Test with native CUDA
5. Compare outputs

## Smoke Test Scripts

### 1. Python Smoke Test (Recommended)

**File**: `scripts/test_gguf_native_smoke.py`

**Features**:
- ✅ Tests native CUDA backend only
- ✅ No conversion required (uses existing GGUF models)
- ✅ Faster testing
- ✅ Clean Python implementation
- ✅ Easy to extend

**Usage**:
```bash
python3 scripts/test_gguf_native_smoke.py [OPTIONS]

Options:
  --model-dir DIR     Directory containing GGUF models
  --inferfluxd PATH   Path to inferfluxd binary
  --inferctl PATH    Path to inferctl binary
  --num-tokens N     Number of tokens to generate (default: 10)
  --prompt TEXT      Test prompt (default: "Hello, how are you?")
```

**Example**:
```bash
python3 scripts/test_gguf_native_smoke.py \
    --model-dir ~/.inferflux/models \
    --num-tokens 50 \
    --prompt "What is machine learning?"
```

### 2. Bash Smoke Test (Full Pipeline)

**File**: `scripts/test_gguf_quantization_smoke.sh`

**Features**:
- ✅ Tests both llama.cpp and native CUDA
- ✅ Converts safetensors → GGUF → Quantized
- ✅ Compares outputs between backends
- ✅ Measures performance comparison

**Usage**:
```bash
./scripts/test_gguf_quantization_smoke.sh [OPTIONS]

Options:
  --model-path PATH   Path to safetensors model
  --num-tokens N      Number of tokens to generate (default: 10)
  --prompt TEXT       Test prompt
```

**Environment Variables**:
```bash
export MODEL_PATH=/path/to/model
export LLAMA_CPP_DIR=./external/llama.cpp
export NUM_TOKENS=20
export TEST_PROMPT="Test prompt here"
```

## Getting Test Models

### Option 1: Download Pre-Quantized GGUF Models

**Qwen2.5 0.5B (Small, Fast for Testing)**:
```bash
# Create models directory
mkdir -p ~/.inferflux/models/qwen-gguf
cd ~/.inferflux/models/qwen-gguf

# Download from HuggingFace (TheBloke provides good quantized models)
# Example URLs (replace with actual model URLs):
wget https://huggingface.co/TheBloke/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
wget https://huggingface.co/TheBloke/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q5_k_m.gguf
wget https://huggingface.co/TheBloke/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q6_k.gguf
wget https://huggingface.co/TheBloke/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

### Option 2: Convert Your Own Models

**Using llama.cpp tools**:
```bash
# 1. Convert safetensors/HF to GGUF FP16
python3 external/llama.cpp/convert_hf_to_gguf.py \
    /path/to/qwen-model \
    --outfile qwen-f16.gguf \
    --outtype f16

# 2. Quantize to each type
external/llama.cpp/llama-quantize qwen-f16.gguf qwen-q4_k_m.gguf q4_k_m
external/llama.cpp/llama-quantize qwen-f16.gguf qwen-q5_k_m.gguf q5_k_m
external/llama.cpp/llama-quantize qwen-f16.gguf qwen-q6_k.gguf q6_k
external/llama.cpp/llama-quantize qwen-f16.gguf qwen-q8_0.gguf q8_0
```

## Understanding Test Results

### Success Criteria

Each quantization type should:
1. ✅ Load successfully in native CUDA backend
2. ✅ Generate output (non-empty text)
3. ✅ Complete without errors
4. ✅ Complete in reasonable time (< 5 seconds for 10 tokens)

### Sample Successful Output

```
========================================
Test Results Summary
========================================

Quantization Type | Duration  | Output Len | Status
------------------|-----------|------------|---------
q4_k_m            | 245ms     | 52         | SUCCESS
q5_k_m            | 267ms     | 51         | SUCCESS
q6_k              | 312ms     | 53         | SUCCESS
q8_0              | 198ms     | 50         | SUCCESS

[INFO] Tests passed: 4 / 4
[SUCCESS] ✅ All quantization types passed!
```

### Troubleshooting Failures

#### "SERVER_FAILED" Status
**Symptom**: Server won't start

**Solutions**:
- Check server logs: `cat /tmp/gguf_native_smoke/server_<quant>.log`
- Verify CUDA is available: `nvidia-smi`
- Check if port is in use: `lsof -i :18083`

#### "INFERENCE_FAILED" Status
**Symptom**: Inference command fails

**Solutions**:
- Verify model format: `file model.gguf` (should show "GGUF")
- Check model compatibility (must be GGUF, not safetensors)
- Verify inferctl can connect: `inferctl --help`

#### "EMPTY_OUTPUT" Status
**Symptom**: Output is too short

**Solutions**:
- Model may not support the quantization type
- Check if model is a chat model (not base model)
- Try a longer prompt

## Performance Expectations

### Relative Performance (Qwen2.5 0.5B, RTX 4090)

| Quantization | File Size | Memory | Latency (10 tokens) | Quality |
|--------------|-----------|--------|---------------------|---------|
| Q4_K_M | ~300 MB | ~600 MB | ~250ms | Good |
| Q5_K_M | ~350 MB | ~700 MB | ~270ms | Very Good |
| Q6_K | ~400 MB | ~800 MB | ~310ms | Excellent |
| Q8_0 | ~600 MB | ~1.2 GB | ~200ms | Near FP16 |

**Notes**:
- Q8_0 is fastest (less dequantization overhead) but largest
- Q4_K_M is most compressed but slower quality
- Q6_K offers best quality/size tradeoff

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: GGUF Quantization Tests

on: [push, pull_request]

jobs:
  smoke-test:
    runs-on: [self-hosted, gpu]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build InferFlux
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build --target inferfluxd
          cmake --build build --target inferctl
      
      - name: Download Test Models
        run: |
          wget https://example.com/qwen-q4_k_m.gguf -O models/
          wget https://example.com/qwen-q8_0.gguf -O models/
      
      - name: Run Smoke Tests
        run: |
          python3 scripts/test_gguf_native_smoke.py \
              --model-dir models/ \
              --num-tokens 20
```

## Advanced Usage

### Custom Test Prompt

```bash
python3 scripts/test_gguf_native_smoke.py \
    --model-dir ~/.inferflux/models \
    --prompt "Explain quantum computing in simple terms" \
    --num-tokens 100
```

### Test Specific Quantization

Modify the script to test only specific types:
```python
SUPPORTED_QUANTIZATIONS = ["q4_k_m", "q8_0"]  # Only test these
```

### Performance Benchmarking

Run with larger token counts to measure throughput:
```bash
python3 scripts/test_gguf_native_smoke.py \
    --model-dir ~/.inferflux/models \
    --num-tokens 500
```

Calculate tokens/second:
```python
tokens_per_sec = NUM_TOKENS / (duration_ms / 1000.0)
print(f"Throughput: {tokens_per_sec:.1f} tok/s")
```

## Cleanup

```bash
# Remove test artifacts
rm -rf /tmp/gguf_native_smoke
rm -rf /tmp/gguf_quant_tests
```

## Next Steps

After smoke tests pass:

1. ✅ **Integration Tests**: Test with real workloads
2. ✅ **Performance Benchmarks**: Compare with llama.cpp baseline
3. ✅ **Phase Overlap Tests**: Verify phase overlap works with quantized models
4. ✅ **Production Validation**: Test with user-facing applications

## See Also

- [GGUF Implementation Guide](GGUF_NATIVE_KERNEL_IMPLEMENTATION.md)
- [Quantization Reference](GGUF_QUANTIZATION_REFERENCE.md)
- [GGUF Complete Implementation Snapshot](archive/evidence/GGUF_COMPLETE_IMPLEMENTATION_2026_03_04.md)
- [Archive Index](ARCHIVE_INDEX.md)
