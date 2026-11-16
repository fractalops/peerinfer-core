#!/usr/bin/env bash
# Builds the engine for the browser into web/src/wasm.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p peerinfer-core-web --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/peerinfer_core_web.wasm \
  --target web --out-dir web/src/wasm
