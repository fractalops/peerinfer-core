# peerinfer-core

peerinfer-core runs language models on the GPU with WebGPU. The same Rust code runs in the browser, compiled to WebAssembly, and natively through wgpu. Native runs are tested on Metal only.

It runs Llama-style and Qwen3 models in the MLC `q4f16_1` format: 4-bit weights in groups of 32 with f16 scales. It generates tokens greedily or samples them with temperature and top-p. With the same seed and context, it generates the same tokens.

## Install

You need Rust. For the browser build you also need the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` at the version in `Cargo.lock`. On macOS, also install LLVM (`brew install llvm`).

On a laptop, build the command-line tool:

```
cargo build --release -p peerinfer-bench
```

## Start

Download a model in MLC `q4f16_1` format, for example with the Hugging Face CLI:

```
hf download mlc-ai/SmolLM2-360M-Instruct-q4f16_1-MLC --local-dir fixtures/smollm2-360m-q4f16_1
```

Measure its speed on this machine:

```
./target/release/peerinfer-bench speed fixtures/smollm2-360m-q4f16_1
```

See the [reference](docs/reference.md) for the library, the browser build, correctness checks and the bench page.

## License

MIT or Apache-2.0. Model weights keep their upstream licenses.
