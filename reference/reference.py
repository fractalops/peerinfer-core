"""Reference forward pass for Llama and Qwen3 models in MLC q4f16_1 format.

Runs on the CPU with NumPy. Used to check the GPU engine.

    uv run --with numpy --with tokenizers reference/reference.py MODEL_DIR \
        --prompt "..." --steps 16 --out-dir fixtures/ref

Writes tokens.json and logits.f32 (little-endian, [steps, vocab]) to --out-dir.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_mlc(model_dir: Path) -> dict[str, np.ndarray]:
    cache = json.loads((model_dir / "ndarray-cache.json").read_text())
    tensors = {}
    for shard in cache["records"]:
        data = (model_dir / shard["dataPath"]).read_bytes()
        for rec in shard["records"]:
            dtype = {"uint32": np.uint32, "float16": np.float16, "float32": np.float32}[rec["dtype"]]
            start = rec["byteOffset"]
            arr = np.frombuffer(data[start : start + rec["nbytes"]], dtype=dtype)
            tensors[rec["name"]] = arr.reshape(rec["shape"])
    return tensors


def dequantize(q_weight: np.ndarray, q_scale: np.ndarray) -> np.ndarray:
    """q4f16_1: 8 unsigned 4-bit values per u32, low bits first; value = (q - 7) * scale."""
    n, k8 = q_weight.shape
    shifts = np.arange(8, dtype=np.uint32) * 4
    q = (q_weight[:, :, None] >> shifts) & 0xF
    q = q.reshape(n, k8 * 8).astype(np.float32) - 7.0
    scale = np.repeat(q_scale.astype(np.float32), 32, axis=1)
    return q * scale


class Model:
    """Llama and Qwen3 decoders. Weights stay quantized and are dequantized
    one layer at a time, so a 4B model fits in memory. Keeps a KV cache."""

    def __init__(self, model_dir: Path):
        chat = json.loads((model_dir / "mlc-chat-config.json").read_text())
        cfg = chat["model_config"]
        self.cfg = cfg
        self.qwen3 = chat["model_type"] == "qwen3"
        self.base = cfg.get("position_embedding_base") or cfg["rope_theta"]
        self.t = load_mlc(model_dir)
        self.qkv_name = "c_attn" if self.qwen3 else "qkv_proj"
        self.reset()

    def reset(self) -> None:
        n = self.cfg["num_hidden_layers"]
        self.k_cache: list[np.ndarray | None] = [None] * n
        self.v_cache: list[np.ndarray | None] = [None] * n
        self.pos = 0

    def lin(self, name: str) -> np.ndarray:
        return dequantize(self.t[f"{name}.q_weight"], self.t[f"{name}.q_scale"])

    def vec(self, name: str) -> np.ndarray:
        return self.t[name].astype(np.float32)

    def rms(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        eps = self.cfg["rms_norm_eps"]
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

    def rope(self, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """x: [T, H, D]. Rotates the two halves of each head (rotate_half)."""
        d = x.shape[-1]
        inv = 1.0 / (self.base ** (np.arange(0, d, 2, dtype=np.float64) / d))
        ang = pos[:, None].astype(np.float64) * inv[None, :]
        cos = np.cos(ang).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(np.float32)[:, None, :]
        a, b = x[..., : d // 2], x[..., d // 2 :]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    def forward(self, tokens: list[int]) -> np.ndarray:
        """Appends `tokens` to the context and returns the last position's logits."""
        c = self.cfg
        hq, hkv = c["num_attention_heads"], c["num_key_value_heads"]
        d = c.get("head_dim") or c["hidden_size"] // hq
        n = len(tokens)
        pos = np.arange(self.pos, self.pos + n)
        total = self.pos + n
        ids = np.array(tokens)
        x = dequantize(self.t["model.embed_tokens.q_weight"][ids], self.t["model.embed_tokens.q_scale"][ids])
        mask = np.triu(np.full((n, total), -np.inf, dtype=np.float32), self.pos + 1)
        for i in range(c["num_hidden_layers"]):
            p = f"model.layers.{i}"
            h = self.rms(x, self.vec(f"{p}.input_layernorm.weight")) @ self.lin(f"{p}.self_attn.{self.qkv_name}").T
            q = h[:, : hq * d].reshape(n, hq, d)
            k = h[:, hq * d : (hq + hkv) * d].reshape(n, hkv, d)
            v = h[:, (hq + hkv) * d :].reshape(n, hkv, d)
            if self.qwen3:
                q = self.rms(q, self.vec(f"{p}.self_attn.q_norm.weight"))
                k = self.rms(k, self.vec(f"{p}.self_attn.k_norm.weight"))
            q, k = self.rope(q, pos), self.rope(k, pos)
            if self.k_cache[i] is not None:
                k = np.concatenate([self.k_cache[i], k])
                v = np.concatenate([self.v_cache[i], v])
            self.k_cache[i], self.v_cache[i] = k, v
            rep = hq // hkv
            kr, vr = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
            s = np.einsum("thd,shd->hts", q, kr) / np.sqrt(d) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = np.einsum("hts,shd->thd", s, vr).reshape(n, hq * d)
            x = x + a @ self.lin(f"{p}.self_attn.o_proj").T
            gu = self.rms(x, self.vec(f"{p}.post_attention_layernorm.weight")) @ self.lin(f"{p}.mlp.gate_up_proj").T
            g, u = gu[:, : c["intermediate_size"]], gu[:, c["intermediate_size"] :]
            x = x + ((g / (1 + np.exp(-g))) * u) @ self.lin(f"{p}.mlp.down_proj").T
        self.pos = total
        last = self.rms(x[-1:], self.vec("model.norm.weight"))
        # The tied LM head, in vocabulary chunks to bound memory.
        qw, qs = self.t["model.embed_tokens.q_weight"], self.t["model.embed_tokens.q_scale"]
        return np.concatenate(
            [(last @ dequantize(qw[r : r + 16384], qs[r : r + 16384]).T)[0] for r in range(0, qw.shape[0], 16384)]
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    text = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt = tok.encode(text).ids
    model = Model(args.model_dir)

    step_logits = []
    generated: list[int] = []
    step = list(prompt)
    for _ in range(args.steps):
        logits = model.forward(step)
        step_logits.append(logits)
        generated.append(int(np.argmax(logits)))
        step = [generated[-1]]
    print(json.dumps({"prompt_tokens": prompt, "generated": generated}))
    print(repr(tok.decode(generated)))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "tokens.json").write_text(json.dumps({"prompt": prompt, "generated": generated}))
        np.stack(step_logits).astype("<f4").tofile(args.out_dir / "logits.f32")


if __name__ == "__main__":
    main()
