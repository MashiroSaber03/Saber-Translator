# syntax=docker/dockerfile:1
# Community-maintained self-hosting recipe (see packaging/web/README.md#docker); the official
# web build is the `web` job in .github/workflows/release.yml (packaging/web/package.sh).
#
# Source of truth for the toolchain: release.yml and ci.yml build with the latest *stable* Rust
# (dtolnay/rust-toolchain@stable), not a pinned version. This default is the stable release CI
# used when it was last bumped; raise it when CI's stable moves on. It must stay >= the
# workspace `rust-version` in Cargo.toml.
ARG RUST_VERSION=1.99
ARG NGINX_VERSION=1.28-alpine

FROM rust:${RUST_VERSION}-slim-bookworm AS builder
# Must match `TRUNK_VERSION` in the `web` job of .github/workflows/release.yml.
ARG TRUNK_VERSION=0.21.14
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add wasm32-unknown-unknown
RUN set -eu; \
    case "$(uname -m)" in \
        x86_64) arch=x86_64 ;; \
        aarch64) arch=aarch64 ;; \
        *) echo 'The web builder supports amd64 and arm64' >&2; exit 1 ;; \
    esac; \
    asset="trunk-${arch}-unknown-linux-musl.tar.gz"; \
    url="https://github.com/trunk-rs/trunk/releases/download/v${TRUNK_VERSION}"; \
    cd /tmp; \
    curl -fsSL --retry 3 -A Photocraft-dev "$url/$asset" -o "$asset"; \
    curl -fsSL --retry 3 -A Photocraft-dev "$url/$asset.sha256" -o "$asset.sha256"; \
    printf '%s  %s\n' "$(cat "$asset.sha256")" "$asset" | sha256sum -c -; \
    tar -xzf "$asset" -C /usr/local/bin trunk; \
    rm "$asset" "$asset.sha256"; \
    trunk --version

WORKDIR /src
COPY . .
WORKDIR /src/apps/photocraft-web
# Docker serving has no static-host per-file size cap. Thin LTO uses less build
# memory than the release zip's fat LTO; retain the profile's other optimizations.
ARG CARGO_PROFILE_WASM_RELEASE_LTO=thin
ARG CARGO_BUILD_JOBS=1
# Optional build provenance shown in About and `--version` (crates/engine/src/build_info.rs).
# .git is not in the build context, so pass it in, e.g.
#   --build-arg PHOTOCRAFT_BUILD_SHA=$(git rev-parse HEAD) --build-arg PHOTOCRAFT_BUILD_DATE=$(date -u +%F)
# Left empty, the build reports itself as a dev build.
ARG PHOTOCRAFT_BUILD_SHA=
ARG PHOTOCRAFT_BUILD_DATE=
# index.html selects the wasm-release profile, HEIF support and wasm-opt -Oz.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    --mount=type=cache,target=/root/.cache/trunk \
    trunk build --release --locked \
    && test -s /src/dist/web/index.html \
    && gzip -9 -k /src/dist/web/*.wasm /src/dist/web/*.js /src/dist/web/index.html

FROM nginx:${NGINX_VERSION} AS runtime
COPY packaging/web/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /src/dist/web/ /usr/share/nginx/html/
COPY LICENSE-MIT LICENSE-APACHE NOTICE ATTRIBUTION.md /usr/share/nginx/html/
COPY assets/fonts/OFL-*.txt assets/icons/LICENSE-*.txt assets/dict/LICENSE-*.txt \
    crates/ui-egui/src/i18n/LICENSE-translations.txt /usr/share/nginx/html/licenses/
# Bypass the stock entrypoint: this configuration needs no runtime rewriting.
USER nginx
EXPOSE 8080
ENTRYPOINT []
CMD ["nginx", "-g", "daemon off;"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
