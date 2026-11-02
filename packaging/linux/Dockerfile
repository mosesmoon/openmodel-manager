# Build environment for the Linux packages (AppImage, deb).
# Ubuntu 22.04 keeps the glibc requirement low (2.35), so the AppImage runs on
# most current distributions.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config curl wget file ca-certificates xz-utils git \
      libwebkit2gtk-4.1-dev libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev patchelf \
 && rm -rf /var/lib/apt/lists/*

# Toolchains live in /opt and are writable, so the build can run as the host user.
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:/opt/node/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
 && chmod -R a+rwX /opt/rustup /opt/cargo

# Node.js 24.22.1 or newer: earlier releases are affected by CVE-2026-111497 (Node2Shell).
ARG NODE_VERSION=24.22.1
RUN cd /tmp \
 && curl -fsSLO https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz \
 && curl -fsSL https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt | grep " node-v${NODE_VERSION}-linux-x64.tar.xz\$" | sha256sum -c - \
 && mkdir -p /opt/node && tar -xJf node-v${NODE_VERSION}-linux-x64.tar.xz -C /opt/node --strip-components=1 \
 && rm node-v${NODE_VERSION}-linux-x64.tar.xz
