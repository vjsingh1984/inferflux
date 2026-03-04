# Configuration Reference

Complete guide to configuring InferFlux with visual examples.

## Quick Reference

```mermaid
mindmap
  root((InferFlux Config))
    Server
      host
      port
      tls
    Models
      path
      format
      backend
      quantization
    Runtime
      backends
      scheduler
      cache
      parallelism
    Auth
      api_keys
      oidc
      rate_limits
    Observability
      metrics
      logging
      tracing
    Policy
      guardrails
      opa
```

## File Locations

| Environment | Config Path |
|-------------|-------------|
| Default | `~/.inferflux/config.yaml` |
| System | `/etc/inferflux/config.yaml` |
| Custom | `--config /path/to/config.yaml` |
| Env override | `INFERFLUX_*` variables |

## Configuration Hierarchy

```mermaid
graph TD
    A[Request] --> B{Config Source}
    B --> C1[1. Environment Variables]
    B --> C2[2. Command Line Flags]
    B --> C3[3. Config File]
    B --> C4[4. Defaults]

    C1 --> D[Merged Config]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E[Validation]
    E --> F[Startup Advisor]
    F --> G[Active Config]

    style B fill:#feca57
    style D fill:#4ecdc4
    style F fill:#ff6b6b
    style G fill:#1dd1a1
```

**Priority:** Env vars > CLI flags > Config file > Defaults

## Server Configuration

```yaml
server:
  host: 0.0.0.0              # Bind address (0.0.0.0 = all interfaces)
  http_port: 8080            # HTTP port for API
  max_concurrent: 1024        # Max concurrent requests
  enable_metrics: true        # Prometheus /metrics endpoint
  enable_tracing: false       # OpenTelemetry tracing (future)
```

### Network Diagram

```mermaid
graph LR
    A[Clients] --> B[Server:8080]
    B --> C[Scheduler]
    B --> D[Metrics: /metrics]
    B --> E[Health: /healthz]

    style B fill:#ff6b6b
    style C fill:#4ecdc4
    style D fill:#45b7d1
    style E fill:#96ceb4
```

## Model Configuration

### Single Model

```yaml
models:
  - id: llama3-8b
    path: /models/llama3-8b.gguf
    format: gguf                # gguf, safetensors, hf, auto
    backend: cuda                # cuda, cpu, mps, rocm, cuda_native, cuda_universal
    default: true
```

### Multi-Model

```yaml
models:
  - id: llama3-8b
    path: /models/llama3-8b.gguf
    format: gguf
    backend: cuda_universal
    default: true

  - id: qwen2.5-3b
    path: /models/qwen2.5-3b/
    format: safetensors
    backend: cuda_native
    default: false
```

### Model Selection Flow

```mermaid
flowchart TD
    A[Request] --> B{Model ID Specified?}
    B -->|Yes| C[Load Specific Model]
    B -->|No| D{Default Model Set?}
    D -->|Yes| E[Load Default Model]
    D -->|No| F[Capability Routing]

    C --> G{Backend Available?}
    E --> G
    F --> H[Find Compatible Backend]

    G -->|Yes| I[Execute with Backend]
    G -->|No| J[Backend Fallback]

    H --> I
    J --> I

    I --> K[Return Response]

    style A fill:#ff6b6b
    style G fill:#feca57
    style J fill:#ff9ff3
    style K fill:#1dd1a1
```

### Format Options

| Format | Description | Backend | When to Use |
|--------|-------------|---------|------------|
| `gguf` | llama.cpp format | cuda_universal, cpu | Best compatibility |
| `safetensors` | HuggingFace format | cuda_native | Native CUDA inference |
| `hf` | HuggingFace URI | Auto-detected | `hf://org/repo` |
| `auto` | Auto-detect | Auto-detected | Unknown format |

### Backend Options

| Backend | Description | Formats | Hardware |
|---------|-------------|---------|----------|
| `cuda` | Auto-select CUDA | All | NVIDIA GPUs |
| `cuda_native` | Native implementation | safetensors | NVIDIA GPUs |
| `cuda_universal` | llama.cpp CUDA | gguf | NVIDIA GPUs |
| `cpu` | CPU inference | All | Any CPU |
| `mps` | Metal Performance Shaders | gguf | Apple Silicon |
| `rocm` | AMD HIP | gguf | AMD GPUs |

#### Native CUDA Kernels

**When Native Kernels Are Used:**

Native CUDA kernels are automatically enabled for safetensors models when using `cuda_native` backend.

```mermaid
graph TB
    A[Load Model] --> B{Format?}

    B -->|Safetensors| C[Use Native Kernels]
    B -->|GGUF| D[Use llama.cpp]
    B -->|Auto| E{Detect Format}

    E -->|Safetensors| C
    E -->|GGUF| D

    C --> F[Custom CUDA Kernels]
    C --> G[FlashAttention-2]
    C --> H[cuBLAS GEMM]
    C --> I[GPU KV Cache]

    style C fill:#4ecdc4
    style F fill:#feca57
    style G fill:#ff9ff3
```

**Automatic Detection:**

When `INFERFLUX_NATIVE_CUDA_EXECUTOR` is not set, InferFlux auto-detects safetensors models and uses native kernels:

```yaml
models:
  - id: qwen2.5-3b
    path: models/qwen2.5-3b-safetensors/
    format: auto  # Detected as safetensors → native kernels
    backend: cuda_native
```

**Manual Override:**

```bash
# Force native kernels (for safetensors)
export INFERFLUX_NATIVE_CUDA_EXECUTOR=native_kernel

# Force llama.cpp delegate
export INFERFLUX_NATIVE_CUDA_EXECUTOR=delegate
```

**Native Kernel Components:**

| Component | Description | Performance |
|-----------|-------------|-------------|
| **FlashAttention-2** | Optimized attention kernel | 2.2x speedup (SM 8.0+) |
| **cuBLAS GEMM** | NVIDIA optimized matrix multiply | Peak throughput |
| **GPU KV Cache** | On-GPU KV cache storage | Zero-copy access |
| **Custom CUDA kernels** | RMSNorm, RoPE, FFN | End-to-end optimization |
| **NVTX annotations** | Profiling support | Nsight Systems compatible |

**Metrics:**

Native kernels report Prometheus metrics:

```prometheus
# Forward pass timing
inferflux_native_forward_passes_total{phase="prefill|decode"}
inferflux_native_forward_duration_ms{phase="prefill|decode"}

# Sampling
inferflux_native_sampling_duration_ms

# KV cache
inferflux_native_kv_active_sequences
inferflux_native_kv_max_sequences

# Throughput
inferflux_native_forward_batch_tokens_total
```

**When to Use Native Kernels:**

✅ **Use native kernels for:**
- Safetensors models (BF16, FP16)
- Maximum performance on NVIDIA GPUs
- Models trained with FlashAttention
- Production deployments

❌ **Use llama.cpp (cuda_universal) for:**
- GGUF quantized models (Q4, Q8, etc.)
- Models with custom llama.cpp optimizations
- Compatibility testing

**Example Configuration:**

```yaml
models:
  # Native kernels for safetensors
  - id: qwen2.5-3b-bf16
    path: models/qwen2.5-3b-safetensors/
    backend: cuda_native
    format: auto

  # llama.cpp for GGUF
  - id: tinyllama-quantized
    path: models/tinyllama-1.1b.gguf
    backend: cuda
    format: gguf
```

## Runtime Configuration

### Backend Priority

```yaml
runtime:
  backend_priority: [cuda, cpu]  # Try CUDA first, fallback to CPU
```

**Selection Flow:**

```mermaid
flowchart LR
    A[Backend Request] --> B{Priority List}
    B --> C1[1. cuda]
    B --> C2[2. cpu]

    C1 --> D1{CUDA Available?}
    D1 -->|Yes| E1[Select CUDA]
    D1 -->|No| C2

    C2 --> D2{CPU Available?}
    D2 -->|Yes| E2[Select CPU]
    D2 -->|No| F[Fail]

    style B fill:#feca57
    style E1 fill:#1dd1a1
    style E2 fill:#1dd1a1
    style F fill:#ff6b6b
```

### CUDA Configuration

```yaml
runtime:
  cuda:
    enabled: true

    # Attention kernel selection
    attention:
      kernel: auto              # auto, fa2, fa3, standard

    # FlashAttention configuration
    flash_attention:
      enabled: true             # Enable FA2 (requires SM 8.0+)
      tile_size: 128            # FA2 tile size (64-256)

    # Phase overlap for mixed batches
    phase_overlap:
      enabled: true             # Enable prefill/decode overlap
      min_prefill_tokens: 256   # Min tokens before overlap
      prefill_replica: false    # Dual-context overlap (memory intensive)
```

### FlashAttention Decision Tree

```mermaid
flowchart TD
    A[CUDA Request] --> B{GPU SM Version}
    B -->|SM < 8.0| C[Standard Attention]
    B -->|SM >= 8.0| D{Flash Attention Enabled?}

    D -->|No| E[Startup Advisor Recommends FA2]
    D -->|Yes| F{Kernel Selection}

    F -->|auto| G[Auto-select FA2]
    F -->|fa2| H[Use FA2]
    F -->|fa3| I[Use FA3]
    F -->|standard| J[Use Standard]

    style A fill:#ff6b6b
    style E fill:#96ceb4
    style H fill:#1dd1a1
    style I fill:#1dd1a1
    style J fill:#96ceb4
```

### Scheduler Configuration

```yaml
runtime:
  scheduler:
    max_batch_size: 16           # Max concurrent requests
    max_batch_tokens: 8192        # Max tokens per batch
    min_batch_size: 1             # Min batch before flushing
    batch_accumulation_ms: 5      # Wait time to accumulate larger batches
```

### Batch Size Recommendation

```mermaid
xychart-beta
    title "Recommended Batch Size by Model Size (20GB VRAM)"
    x-axis ["1-3B", "7-8B", "13-14B", "30B+"]
    y-axis "Max Batch Size" 0 --> 64
    bar [32, 20, 12, 6]
```

| Model Size | max_batch_size | max_batch_tokens | cpu_pages |
|-------------|----------------|-------------------|-----------|
| 1-3B | 24-32 | 8192-16384 | 256-512 |
| 7-8B | 16-24 | 8192 | 512-1024 |
| 13-14B | 12-16 | 8192 | 1536-2048 |
| 30B+ | 6-12 | 4096-8192 | 2048-3072 |

### KV Cache Configuration

```yaml
runtime:
  paged_kv:
    cpu_pages: 256               # Number of KV cache pages
    eviction: lru                # Eviction policy: lru or clock
```

### KV Cache Sizing

```mermaid
graph TB
    subgraph "KV Cache Memory Calculation"
        A[Model Params] --> B[Hidden Size]
        B --> C[Num Layers]
        C --> D[Num Heads]
        D --> E[Head Dim]

        E --> F[KV Size per Token]
        F --> G[Max Context Length]
        G --> H[KV Size per Request]

        H --> I[cpu_pages]
        I --> J[Total KV Memory]
    end

    style A fill:#ff6b6b
    style I fill:#feca57
    style J fill:#1dd1a1
```

**Formula:** `KV_memory = cpu_pages × page_size × 2 (K+V) × 2 bytes (FP16)`

### Tensor Parallelism

```yaml
runtime:
  tensor_parallel: 1             # Number of GPUs for model sharding
```

### Multi-GPU Decision

```mermaid
flowchart TD
    A[Model Loaded] --> B{Model Size}
    B -->|< 7B| C[TP = 1]
    B -->|7B - 14B| D{GPU Count}
    B -->|> 14B| E{GPU Count}

    D -->|1 GPU| C
    D -->|2+ GPUs| F[TP = GPU Count]

    E -->|1-2 GPUs| G[TP = 1 or 2]
    E -->|3+ GPUs| H[TP = GPU Count or Less]

    style B fill:#feca57
    style C fill:#1dd1a1
    style F fill:#1dd1a1
    style G fill:#1dd1a1
    style H fill:#1dd1a1
```

## Authentication Configuration

### API Key Authentication

```yaml
auth:
  api_keys:
    - key: dev-key-123            # Development key
      scopes:                     # Permissions
        - generate                # - generate: Create completions
        - read                    # - read: View models
        - admin                   # - admin: Full access

    - key: prod-key-abc           # Production key
      scopes:
        - generate
        - read                    # No admin access
```

### Permission Matrix

```mermaid
graph TD
    subgraph "Scopes and Permissions"
        A[API Key] --> B{Scopes}

        B --> C1[generate]
        B --> C2[read]
        B --> C3[admin]

        C1 --> D1[POST /v1/completions]
        C1 --> D2[POST /v1/chat/completions]

        C2 --> D3[GET /v1/models]
        C2 --> D4[GET /metrics]

        C3 --> D5[POST /admin/*]
        C3 --> D6[All other endpoints]
    end

    style A fill:#ff6b6b
    style B fill:#feca57
    style C1 fill:#4ecdc4
    style C2 fill:#45b7d1
    style C3 fill:#96ceb4
```

### OIDC Authentication

```yaml
auth:
  oidc_issuer: https://auth.example.com
  oidc_audience: inferflux
```

### Auth Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant I as InferFlux
    participant O as OIDC Provider

    C->>I: Request with Bearer Token
    I->>I: Validate RS256 Signature
    I->>I: Verify Claims
    I->>I: Check Audience
    I-->>C: Allow/Deny Request
```

### Rate Limiting

```yaml
auth:
  rate_limit_per_minute: 120     # Per-API-key limit
```

## Observability Configuration

### Metrics

```yaml
server:
  enable_metrics: true            # Enable Prometheus endpoint
```

### Available Metrics

```mermaid
mindmap
  root((Prometheus Metrics))
    scheduler
      batch_size
      queue_depth
      latency
      throughput
    backend
      cuda_tokens_per_second
      cuda_forward_duration
      cuda_kv_cache utilization
      cuda_lane_submissions
    model
      model_load_duration
      model_switches
    http
      request_duration
      request_duration_by_path
      auth_failures
```

### Structured Logging

```yaml
logging:
  level: info                    # debug, info, warn, error
  format: json                   # json or text
  audit_log: logs/audit.log      # Audit trail location
```

### Log Format

```json
{
  "timestamp": "2026-03-04T15:30:45.123Z",
  "level": "INFO",
  "component": "server",
  "message": "Request completed",
  "duration_ms": 523,
  "model": "llama3-8b",
  "tokens": 128
}
```

## Policy Configuration

### Guardrails

```yaml
guardrails:
  blocklist:                    # Blocked words/phrases
    - secret
    - confidential
    - password
```

### OPA Integration

```yaml
guardrails:
  opa_endpoint: http://opa:8181/v1/data
```

### Policy Enforcement Flow

```mermaid
flowchart TD
    A[Request] --> B{Auth Check}
    B -->|Fail| C[401 Unauthorized]
    B -->|Pass| D{Guardrails Check}

    D -->|Blocked| E[400 Bad Request]
    D -->|Pass| F{OPA Policy?}

    F -->|Deny| G[403 Forbidden]
    F -->|Allow| H[Process Request]

    style A fill:#ff6b6b
    style C fill:#ff6b6b
    style E fill:#ff6b6b
    style G fill:#ff6b6b
    style H fill:#1dd1a1
```

## Environment Variables

### Variable Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `INFERFLUX_MODEL_PATH` | Default model path | `/models/llama3.gguf` |
| `INFERFLUX_MODELS` | Multi-model config | `id=m1,path=/p1.gguf;id=m2,path=/p2.gguf` |
| `INFERFLUX_NATIVE_CUDA_EXECUTOR` | CUDA executor mode | `native_kernel` for safetensors |
| `INFERFLUX_BACKEND_PRIORITY` | Backend selection order | `cuda,cpu` |
| `INFERFLUX_DISABLE_STARTUP_ADVISOR` | Disable advisor | `true` |
| `INFERCTL_API_KEY` | Client API key | `dev-key-123` |
| `INFERFLUX_POLICY_PASSPHRASE` | Policy encryption | `secret-key` |

### Config Precedence

```mermaid
graph TD
    A[Config Value] --> B{Env Var Set?}
    B -->|Yes| C[Use Env Var]
    B -->|No| D{Config File Set?}
    D -->|Yes| E[Use Config File]
    D -->|No| F[Use Default]

    style B fill:#feca57
    style C fill:#1dd1a1
    style E fill:#1dd1a1
    style F fill:#96ceb4
```

## Complete Example

```yaml
server:
  host: 0.0.0.0
  http_port: 8080
  max_concurrent: 1024
  enable_metrics: true

models:
  - id: llama3-8b
    path: /models/llama3-8b.Q4_K_M.gguf
    format: gguf
    backend: cuda_universal
    default: true

runtime:
  backend_priority: [cuda, cpu]

  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: true
      tile_size: 128
    phase_overlap:
      enabled: true
      min_prefill_tokens: 256

  scheduler:
    max_batch_size: 32
    max_batch_tokens: 8192
    min_batch_size: 1
    batch_accumulation_ms: 5

  tensor_parallel: 1

  paged_kv:
    cpu_pages: 512
    eviction: lru

auth:
  api_keys:
    - key: prod-key-abc
      scopes: [generate, read]
  rate_limit_per_minute: 120

logging:
  level: info
  format: json
  audit_log: logs/audit.log

guardrails:
  blocklist:
    - secret
    - confidential
```

## Troubleshooting Configs

### Common Issues

```mermaid
flowchart TD
    A[Config Issue] --> B{Problem}
    B -->|Model not loading| C[Check path and format]
    B -->|CUDA not working| D[Check cuda.enabled]
    B -->|Slow performance| E[Check batch_size and KV pages]
    B -->|Auth failures| F[Check API key and scopes]

    C --> G[Verify file exists]
    D --> H[Verify GPU available]
    E --> I[Run startup advisor]
    F --> J[Check key is hashed]

    style A fill:#ff6b6b
    style G fill:#1dd1a1
    style H fill:#1dd1a1
    style I fill:#1dd1a1
    style J fill:#1dd1a1
```

### Validation Command

```bash
# Check config syntax
./build/inferfluxd --config config/server.yaml --validate

# Run with startup advisor
./build/inferfluxd --config config/server.yaml 2>&1 | grep -A 20 "Startup Recommendations"
```

---

**Next:** [Performance Tuning](PERFORMANCE_TUNING.md) | [Admin Guide](AdminGuide.md) | [Competitive Positioning](COMPETITIVE_POSITIONING.md)
