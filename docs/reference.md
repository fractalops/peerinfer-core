# Reference

## Supported models

- Architecture: Llama-style decoder models with grouped-query attention, rotary position embedding (rotate-half) and SwiGLU. `model_type` must be `llama`.
- Weights: MLC `q4f16_1` (`mlc-chat-config.json`, `ndarray-cache.json` and `params_shard_*.bin`). Each linear layer has `q_weight` (u32, 8 values each, low bits first) and `q_scale` (f16, one per group of 32). A value is `(q - 7) * scale`.
- Tied embeddings only: the token embedding is also the output layer.
- Head size 64.
- Context: up to 2,048 tokens. A forward pass takes up to 256 tokens; longer prompts run in parts.

## Library

`peerinfer_core::Engine` runs one model on one GPU.

| Method | Description |
|---|---|
| `Engine::new(&weights)` | Requests a GPU and uploads the weights. |
| `forward(&tokens)` | Appends tokens to the context and returns the next token. |
| `generate(max, on_tokens)` | Generates up to `max` tokens after the token from `forward`. `on_tokens` receives tokens in batches and returns `false` to stop. |
| `set_sampling(Some(Sampling { temperature, top_p, seed }))` | Samples instead of picking the most likely token. `None` restores greedy decoding. |
| `reset()` | Clears the context. |
| `last_logits()` | Returns the logits of the last forward pass. |

`Weights::read_dir(path)` reads a model directory on native targets.

### Stages

A model can run as stages on different engines, each with a contiguous range of layers. The stages give the same logits and tokens as one engine with all layers.

| Method | Description |
|---|---|
| `Engine::new_stage(&weights, a..b)` | Uploads only layers `a..b`. The stage that starts at layer 0 embeds tokens. The stage that ends at the last layer applies the final norm and the LM head and picks the token. Both load the token embedding, which is also the LM head. |
| `run(Input::Tokens(&tokens))` | Runs the first stage. |
| `run(Input::Hidden(&hidden))` | Runs a later stage on the output of the stage before it. |
| `is_first()`, `is_last()`, `layers()` | Describe the stage. |

`run` returns `Output::Token(t)` from the last stage and `Output::Hidden(values)` from other stages: `hidden_size` f32 values per token, in token order. Each stage keeps the KV cache for its own layers, so all stages must see the same tokens in the same order. Call `reset()` on every stage to start a new context. Set sampling on the last stage; the seed and the token position decide each draw.

`WeightIndex::shards_for(&(a..b), total_layers)` lists the shard files that a stage needs, and `Weights::read_dir_stage(path, Some(a..b))` reads only those. MLC shard files group tensors by name, so a stage usually needs more files than its layers fill.

During `generate`, the next token stays on the GPU. The CPU queues later steps while the GPU runs earlier ones, and reads tokens back in batches of 16. When `on_tokens` returns `false`, the context ends after the last token it received.

Sampling uses Philox4x32-10 with the seed as key and the token position as counter.

## Browser build

Build the WebAssembly module into `web/src/wasm`:

```
./scripts/build-wasm.sh
```

The `Engine` class in JavaScript has `Engine.load(chatConfig, ndarrayCache, shards)`, `forward(tokens)`, `generate(max)`, `generateStream(max, onTokens)`, `setSampling(temperature, topP, seed)` and `reset()`. A temperature of 0 selects greedy decoding. `generateStream` calls `onTokens` with each batch of tokens as a `Uint32Array` and stops when it returns `false`.

For stages, `Engine.shardsFor(chatConfig, ndarrayCache, start, end)` returns the indexes of the shard files to fetch, and `Engine.loadStage(chatConfig, ndarrayCache, shards, start, end)` loads them; pass empty arrays for the other files. `runTokens(tokens)` and `runHidden(hidden)` return the next token as a number on the last stage and the hidden states as a `Float32Array` on other stages. `isFirst`, `isLast` and `hiddenSize` describe the stage.

The engine needs WebGPU with its default limits. It needs no optional features; `profile` uses timestamp queries when the GPU has them. Each kernel uses at most 5 storage buffers.

## peerinfer-bench

| Command | Description |
|---|---|
| `speed MODEL_DIR [PROMPT] [GEN] [RUNS]` | Prompt and generation speed, greedy. Defaults: 128, 128, 5. Prints the median. |
| `speed-sample MODEL_DIR [PROMPT] [GEN] [RUNS]` | Generation speed with temperature 0.7 and top-p 0.9. |
| `check MODEL_DIR REF_DIR` | Compares the logits at each step with a reference run and checks that `generate` returns the reference tokens. |
| `split-check MODEL_DIR REF_DIR CUTS` | Runs the model whole and as stages split at the layers in `CUTS` (for example `16` or `8,24`). Requires the same logits and tokens at every step, greedy and with sampling. |
| `sample-check MODEL_DIR` | Compares 4,000 GPU draws with an exact CPU computation of the same distribution. |
| `profile MODEL_DIR [TOKENS]` | GPU time per kernel for one forward pass. Needs timestamp queries. |

Other programs that use the GPU change the results. Close browsers and other GPU programs before measuring.

## Reference runs

`reference/reference.py` runs the same model on the CPU with NumPy and writes the tokens and logits that `check` reads:

```
uv run --with numpy --with tokenizers reference/reference.py MODEL_DIR --steps 12 --out-dir fixtures/ref
```

`reference/onnx_to_mlc.py` converts a Llama ONNX export with 4-bit `MatMulNBits` weights to `q4f16_1`. The 4-bit weights are quantized again, so the values differ slightly from the ONNX file.

## Bench page

`web/` is a page that measures the engine in the browser next to other browser runtimes with the same model. Start it after the browser build:

```
cd web
npm install
npm run dev
```

The page serves `../fixtures` at `/fixtures/`. The 135M comparison imports code from a peerinfer checkout next to this repository. Query options: `model=135m`, `temperature`, `top_p`, `prompt`, `gen` and `runs`. The page does not watch files; restart the server after changes.
