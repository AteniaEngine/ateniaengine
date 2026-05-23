# Atenia Engine — CPU-only CLI image
#
# Build:
#   docker build -t atenia:latest .
#
# Run (no model required):
#   docker run --rm atenia:latest doctor
#   docker run --rm atenia:latest capabilities --json
#
# Run with a local models directory mounted in:
#   docker run --rm -v "$(pwd)/models:/models" -v "$(pwd):/work" \
#       atenia:latest generate --model /models/<your-model> --prompt "Hello"
#
# The image is CPU-only by design: `build.rs` auto-falls-back when no CUDA
# toolkit is present, so the produced binary uses the engine's vendor-agnostic
# matmul paths. GPU acceleration still requires `cargo install` on a host with
# CUDA + driver; this image is the zero-setup entry point.

# ---- build stage ---------------------------------------------------
FROM rust:1.85-slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
           build-essential \
           pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cargo build --release --bin atenia \
    && strip target/release/atenia

# ---- runtime stage -------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
           ca-certificates \
           libgcc-s1 \
           libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/atenia /usr/local/bin/atenia

# Conventional mount points. Users supply them via `-v` at run time.
VOLUME ["/models"]
WORKDIR /work

ENTRYPOINT ["atenia"]
CMD ["--help"]
