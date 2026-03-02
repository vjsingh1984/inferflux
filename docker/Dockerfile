# syntax=docker/dockerfile:1.4

FROM ubuntu:22.04 AS builder
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git libssl-dev libyaml-cpp-dev python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cmake -S . -B build -DENABLE_WEBUI=ON && \
    cmake --build build -j$(nproc)

FROM ubuntu:22.04
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libssl3 libyaml-cpp0.7 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/build/inferfluxd .
COPY --from=builder /src/build/inferctl .
EXPOSE 8080

ENTRYPOINT ["./inferfluxd", "--config", "/app/config/server.yaml", "--ui"]
