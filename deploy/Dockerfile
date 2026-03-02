# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libyaml-cpp-dev \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Initialise submodules (llama.cpp lives at external/llama.cpp)
RUN git submodule update --init --recursive

# CPU-only Release build. Pass -DENABLE_CUDA=ON for NVIDIA GPU images.
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=OFF \
      -DENABLE_MTMD=OFF \
      -DENABLE_SBOM=OFF \
      -DENABLE_COVERAGE=OFF \
    && cmake --build build -j"$(nproc)"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libyaml-cpp0.8 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries from builder.
COPY --from=builder /src/build/inferfluxd  /usr/local/bin/inferfluxd
COPY --from=builder /src/build/inferctl   /usr/local/bin/inferctl

# Default configuration directory and model cache.
RUN mkdir -p /etc/inferflux /var/cache/inferflux/models

# Ship a minimal default config. Users override via bind-mount or env vars.
COPY config/server.yaml /etc/inferflux/server.yaml

# Prometheus metrics port (8080 default).
EXPOSE 8080

# Models can be mounted at /models; set INFERFLUX_MODEL_PATH accordingly.
VOLUME ["/models"]

ENTRYPOINT ["/usr/local/bin/inferfluxd"]
CMD ["--config", "/etc/inferflux/server.yaml"]
