import init, { Engine } from "./wasm/peerinfer_core_web.js";

const MODEL = "SmolLM2-360M-Instruct-q4f16_1-MLC";
const params = new URLSearchParams(location.search);
// ?model=135m runs SmolLM2 135M converted from the ONNX baseline's weights.
const DIR =
  params.get("model") === "135m" ? "/fixtures/smollm2-135m-q4f16_1/" : "/fixtures/smollm2-360m-q4f16_1/";
const ONNX_DIR = "/fixtures/smollm2-135m-onnx/";
const PROMPT = Number(params.get("prompt") ?? 128);
const GEN = Number(params.get("gen") ?? 128);
const RUNS = Number(params.get("runs") ?? 5);
// ?temperature=0.7&top_p=0.9 samples; the default is greedy decoding.
const TEMPERATURE = Number(params.get("temperature") ?? 0);
const TOP_P = Number(params.get("top_p") ?? 1);

const log = (s) => {
  document.querySelector("#log").textContent += s + "\n";
  console.log(s);
};
const median = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];

async function runOurs() {
  const t0 = performance.now();
  await init();
  const text = (f) => fetch(DIR + f).then((r) => r.text());
  const [chat, cache] = await Promise.all([text("mlc-chat-config.json"), text("ndarray-cache.json")]);
  const shards = await Promise.all(
    JSON.parse(cache).records.map((s) =>
      fetch(DIR + s.dataPath).then(async (r) => new Uint8Array(await r.arrayBuffer())),
    ),
  );
  const engine = await Engine.load(chat, cache, shards);
  const load = (performance.now() - t0) / 1000;
  log(`peerinfer-core on ${engine.adapter()}: load ${load.toFixed(2)} s`);
  const prompt = Array.from({ length: PROMPT }, (_, i) => 1000 + ((i * 37) % 20000));
  const prefill = [];
  const decode = [];
  for (let run = 0; run <= RUNS; run++) {
    engine.setSampling(TEMPERATURE, TOP_P, run);
    engine.reset();
    const a = performance.now();
    await engine.forward(Uint32Array.from(prompt));
    const b = performance.now();
    const out = await engine.generate(GEN);
    const c = performance.now();
    if (out.length !== GEN) throw new Error(`generated ${out.length} tokens`);
    if (run === 0) continue;
    prefill.push(PROMPT / ((b - a) / 1000));
    decode.push(GEN / ((c - b) / 1000));
  }
  return {
    engine: "peerinfer-core",
    temperature: TEMPERATURE,
    top_p: TOP_P,
    load,
    prefill: median(prefill),
    decode: median(decode),
  };
}

async function runWebLLM() {
  const webllm = await import("@mlc-ai/web-llm");
  const t0 = performance.now();
  const engine = await webllm.CreateMLCEngine(MODEL);
  const load = (performance.now() - t0) / 1000;
  log(`WebLLM: load ${load.toFixed(2)} s (includes download on first run)`);
  // About PROMPT tokens including the chat template.
  const words = Array.from({ length: PROMPT - 12 }, (_, i) => ["river", "stone", "light", "paper"][i % 4]).join(" ");
  const prefill = [];
  const decode = [];
  let promptTokens = 0;
  const reported = [];
  for (let run = 0; run <= RUNS; run++) {
    await engine.resetChat();
    // Wall-clock timing from the stream, measured the same way as ours:
    // prefill ends at the first token, decode covers the remaining tokens.
    const start = performance.now();
    let first = 0;
    let usage;
    const stream = await engine.chat.completions.create({
      messages: [{ role: "user", content: words }],
      max_tokens: GEN,
      temperature: TEMPERATURE,
      top_p: TOP_P,
      seed: run,
      ignore_eos: true,
      stream: true,
      stream_options: { include_usage: true },
    });
    for await (const chunk of stream) {
      if (!first && chunk.choices?.[0]?.delta?.content) first = performance.now();
      if (chunk.usage) usage = chunk.usage;
    }
    const end = performance.now();
    if (run === 0) continue;
    promptTokens = usage.prompt_tokens;
    prefill.push(usage.prompt_tokens / ((first - start) / 1000));
    decode.push((usage.completion_tokens - 1) / ((end - first) / 1000));
    reported.push(usage.extra.decode_tokens_per_s);
  }
  return {
    engine: "WebLLM 0.2.88",
    temperature: TEMPERATURE,
    top_p: TOP_P,
    load,
    promptTokens,
    prefill: median(prefill),
    decode: median(decode),
    decodeReported: median(reported),
  };
}

// The product's ONNX Runtime path: the same graph rewrite, the same token
// selection, and the same loop as peerinfer's model worker. The KV cache
// stays on the GPU; logits come back to the CPU every token.
async function runOrt() {
  const ort = await import("onnxruntime-web/webgpu");
  const { default: wasmUrl } = await import(
    "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url"
  );
  const { wholeModel } = await import("../../../peerinfer/web/src/runtime/partition.js");
  const { selectToken } = await import("../../../peerinfer/web/src/runtime/generation.ts");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { wasm: wasmUrl };
  const t0 = performance.now();
  const config = await fetch(ONNX_DIR + "config.json").then((r) => r.json());
  const bytes = new Uint8Array(await fetch(ONNX_DIR + "onnx/model_q4.onnx").then((r) => r.arrayBuffer()));
  ort.env.webgpu.adapter = await navigator.gpu.requestAdapter();
  const layers = config.num_hidden_layers;
  const present = Object.fromEntries(
    Array.from({ length: layers }, (_, i) => [`present.${i}.key`, `present.${i}.value`])
      .flat()
      .map((n) => [n, "gpu-buffer"]),
  );
  const session = await ort.InferenceSession.create(wholeModel(bytes), {
    executionProviders: ["webgpu"],
    preferredOutputLocation: present,
    graphOptimizationLevel: "all",
  });
  const load = (performance.now() - t0) / 1000;
  log(`ONNX Runtime: load ${load.toFixed(2)} s`);
  const kv = config.num_key_value_heads;
  const hd = config.hidden_size / config.num_attention_heads;
  let cache = {};
  let past = 0;
  const step = async (tokens) => {
    const n = tokens.length;
    const feeds = {};
    for (const name of session.inputNames) {
      if (name === "input_ids") feeds[name] = new ort.Tensor("int64", BigInt64Array.from(tokens, BigInt), [1, n]);
      else if (name === "attention_mask")
        feeds[name] = new ort.Tensor("int64", new BigInt64Array(past + n).fill(1n), [1, past + n]);
      else if (name === "position_ids")
        feeds[name] = new ort.Tensor("int64", BigInt64Array.from(tokens, (_, i) => BigInt(past + i)), [1, n]);
      else feeds[name] = cache[name] ?? new ort.Tensor("float32", new Float32Array(), [1, kv, 0, hd]);
    }
    const out = await session.run(feeds);
    let logits;
    for (const [name, value] of Object.entries(out)) {
      if (name.startsWith("present.")) {
        const input = name.replace("present.", "past_key_values.");
        cache[input]?.dispose();
        cache[input] = value;
      } else {
        logits = (await value.getData()).slice(-config.vocab_size);
        value.dispose();
      }
    }
    for (const [name, value] of Object.entries(feeds)) if (!name.startsWith("past_key_values.")) value.dispose();
    past += n;
    return selectToken(logits, TEMPERATURE);
  };
  const prompt = Array.from({ length: PROMPT }, (_, i) => 1000 + ((i * 37) % 20000));
  const prefill = [];
  const decode = [];
  for (let run = 0; run <= RUNS; run++) {
    for (const t of Object.values(cache)) t.dispose();
    cache = {};
    past = 0;
    const a = performance.now();
    let next = await step(prompt);
    const b = performance.now();
    for (let i = 0; i < GEN; i++) next = await step([next]);
    const c = performance.now();
    if (run === 0) continue;
    prefill.push(PROMPT / ((b - a) / 1000));
    decode.push(GEN / ((c - b) / 1000));
  }
  await session.release();
  return {
    engine: `ONNX Runtime ${ort.env.versions.web}`,
    temperature: TEMPERATURE,
    load,
    prefill: median(prefill),
    decode: median(decode),
  };
}

async function bench(fn) {
  try {
    const r = await fn();
    log(JSON.stringify(r));
    (window.results ??= []).push(r);
  } catch (e) {
    log(`error: ${e?.message ?? e}`);
    (window.results ??= []).push({ error: String(e?.message ?? e) });
  }
}

document.querySelector("#ours").onclick = () => bench(runOurs);
document.querySelector("#webllm").onclick = () => bench(runWebLLM);
window.bench = { ours: () => bench(runOurs), webllm: () => bench(runWebLLM), ort: () => bench(runOrt) };
