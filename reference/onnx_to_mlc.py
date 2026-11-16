"""Converts a Llama ONNX export with 4-bit MatMulNBits weights to MLC q4f16_1.

The engine reads only MLC q4f16_1. This lets it run the same model as an
ONNX baseline. The 4-bit weights are dequantized and quantized again, so
the values differ slightly from the ONNX file; the size per weight is the
same. The f32 embedding is quantized too, as MLC does.

    uv run --with numpy reference/onnx_to_mlc.py DUMP_DIR CONFIG_JSON OUT_DIR

DUMP_DIR holds the initializers written by onnx-dump.mjs (index.json and
one raw file per tensor).

q4f16_1 encoding, from mlc-llm python/mlc_llm/quantization/
group_quantization.py: scale = max|w| / 7 per group of 32,
q = clip(round(w / scale + 7), 0, 14), 8 values per uint32, low bits first.
"""

import json
import sys
from pathlib import Path

import numpy as np

GROUP = 32


def load(dump: Path) -> dict[str, np.ndarray]:
    types = {1: np.float32, 2: np.uint8, 7: np.int64, 10: np.float16}
    out = {}
    for t in json.loads((dump / "index.json").read_text()):
        if t["type"] not in types:
            continue
        a = np.fromfile(dump / t["file"], dtype=types[t["type"]])
        out[t["name"]] = a.reshape(t["dims"]) if t["dims"] else a
    return out


def matmul_nbits(t: dict, name: str, n: int, k: int) -> np.ndarray:
    """Dequantizes a MatMulNBits weight to W[n, k]: low nibble first,
    zero point 8 when the model has no zero-point tensor."""
    assert f"{name}.MatMul.weight_zp" not in t, "zero points are not supported"
    packed = t[f"{name}.MatMul.weight_Q4"].reshape(n, k // 2)
    q = np.empty((n, k), dtype=np.float32)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    scales = t[f"{name}.MatMul.weight_scales"].reshape(n, k // GROUP)
    return (q - 8.0) * np.repeat(scales, GROUP, axis=1)


def quantize(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k = w.shape
    g = w.reshape(n, k // GROUP, GROUP)
    scale = (np.abs(g).max(axis=2) / 7.0).astype(np.float16)
    s = scale.astype(np.float32)[:, :, None]
    q = np.clip(np.round(np.divide(g, s, out=np.zeros_like(g), where=s != 0) + 7), 0, 14)
    q = q.astype(np.uint32).reshape(n, k // 8, 8)
    packed = np.zeros((n, k // 8), dtype=np.uint32)
    for j in range(8):
        packed |= q[:, :, j] << np.uint32(4 * j)
    return packed, scale


def main() -> None:
    dump, config_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cfg = json.loads(config_path.read_text())
    t = load(dump)
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    hq, hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    d = h // hq

    tensors: list[tuple[str, np.ndarray]] = []

    def linear(name: str, w: np.ndarray) -> None:
        q, s = quantize(w)
        tensors.append((f"{name}.q_weight", q))
        tensors.append((f"{name}.q_scale", s))

    linear("model.embed_tokens", t["model.embed_tokens.weight"].astype(np.float32))
    for i in range(cfg["num_hidden_layers"]):
        o, m = f"model.layers.{i}.attn", f"model.layers.{i}.mlp"
        p = f"model.layers.{i}"
        tensors.append((f"{p}.input_layernorm.weight", t[f"{p}.input_layernorm.weight"].astype(np.float16)))
        tensors.append(
            (f"{p}.post_attention_layernorm.weight", t[f"{p}.post_attention_layernorm.weight"].astype(np.float16))
        )
        qkv = np.concatenate(
            [
                matmul_nbits(t, f"{o}.q_proj", hq * d, h),
                matmul_nbits(t, f"{o}.k_proj", hkv * d, h),
                matmul_nbits(t, f"{o}.v_proj", hkv * d, h),
            ]
        )
        linear(f"{p}.self_attn.qkv_proj", qkv)
        linear(f"{p}.self_attn.o_proj", matmul_nbits(t, f"{o}.o_proj", h, hq * d))
        gate_up = np.concatenate(
            [matmul_nbits(t, f"{m}.gate_proj", inter, h), matmul_nbits(t, f"{m}.up_proj", inter, h)]
        )
        linear(f"{p}.mlp.gate_up_proj", gate_up)
        linear(f"{p}.mlp.down_proj", matmul_nbits(t, f"{m}.down_proj", h, inter))
    tensors.append(("model.norm.weight", t[f"model.layers.{cfg['num_hidden_layers']}.final_norm_layernorm.weight"].astype(np.float16)))

    out.mkdir(parents=True, exist_ok=True)
    records, blob = [], bytearray()
    for name, a in tensors:
        raw = np.ascontiguousarray(a).tobytes()
        dtype = {np.dtype(np.uint32): "uint32", np.dtype(np.float16): "float16"}[a.dtype]
        records.append(
            {"name": name, "shape": list(a.shape), "dtype": dtype, "format": "raw", "nbytes": len(raw), "byteOffset": len(blob)}
        )
        blob += raw
    (out / "params_shard_0.bin").write_bytes(blob)
    (out / "ndarray-cache.json").write_text(
        json.dumps({"metadata": {}, "records": [{"dataPath": "params_shard_0.bin", "format": "raw-shard", "nbytes": len(blob), "records": records}]})
    )
    (out / "mlc-chat-config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization": "q4f16_1",
                "model_config": {
                    "hidden_size": h,
                    "intermediate_size": inter,
                    "num_attention_heads": hq,
                    "num_key_value_heads": hkv,
                    "num_hidden_layers": cfg["num_hidden_layers"],
                    "head_dim": d,
                    "vocab_size": cfg["vocab_size"],
                    "rms_norm_eps": cfg["rms_norm_eps"],
                    "position_embedding_base": cfg["rope_theta"],
                    "tie_word_embeddings": cfg["tie_word_embeddings"],
                },
            }
        )
    )
    print(f"wrote {len(tensors)} tensors, {len(blob) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
