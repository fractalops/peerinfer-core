"""Reference forward pass for SmolLM2 in MLC q4f16_1 format.

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
    def __init__(self, model_dir: Path):
        cfg = json.loads((model_dir / "mlc-chat-config.json").read_text())["model_config"]
        self.cfg = cfg
        t = load_mlc(model_dir)

        def lin(name: str) -> np.ndarray:
            return dequantize(t[f"{name}.q_weight"], t[f"{name}.q_scale"])

        self.embed = lin("model.embed_tokens")
        self.norm = t["model.norm.weight"].astype(np.float32)
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}"
            self.layers.append(
                dict(
                    ln1=t[f"{p}.input_layernorm.weight"].astype(np.float32),
                    ln2=t[f"{p}.post_attention_layernorm.weight"].astype(np.float32),
                    qkv=lin(f"{p}.self_attn.qkv_proj"),
                    o=lin(f"{p}.self_attn.o_proj"),
                    gate_up=lin(f"{p}.mlp.gate_up_proj"),
                    down=lin(f"{p}.mlp.down_proj"),
                )
            )

    def rms(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        eps = self.cfg["rms_norm_eps"]
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

    def rope(self, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """x: [T, H, D]. Rotates the two halves of each head (rotate_half)."""
        d = x.shape[-1]
        inv = 1.0 / (self.cfg["position_embedding_base"] ** (np.arange(0, d, 2, dtype=np.float64) / d))
        ang = pos[:, None].astype(np.float64) * inv[None, :]
        cos = np.cos(ang).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(np.float32)[:, None, :]
        a, b = x[..., : d // 2], x[..., d // 2 :]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    def forward(self, tokens: list[int]) -> np.ndarray:
        """Returns logits for every position, shape [T, vocab]."""
        c = self.cfg
        hq, hkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        n = len(tokens)
        pos = np.arange(n)
        x = self.embed[np.array(tokens)]
        mask = np.triu(np.full((n, n), -np.inf, dtype=np.float32), 1)
        for L in self.layers:
            h = self.rms(x, L["ln1"]) @ L["qkv"].T
            q = h[:, : hq * d].reshape(n, hq, d)
            k = h[:, hq * d : (hq + hkv) * d].reshape(n, hkv, d)
            v = h[:, (hq + hkv) * d :].reshape(n, hkv, d)
            q, k = self.rope(q, pos), self.rope(k, pos)
            rep = hq // hkv
            k = np.repeat(k, rep, axis=1)
            v = np.repeat(v, rep, axis=1)
            s = np.einsum("thd,shd->hts", q, k) / np.sqrt(d) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = np.einsum("hts,shd->thd", s, v).reshape(n, hq * d)
            x = x + a @ L["o"].T
            gu = self.rms(x, L["ln2"]) @ L["gate_up"].T
            g, u = gu[:, : c["intermediate_size"]], gu[:, c["intermediate_size"] :]
            x = x + ((g / (1 + np.exp(-g))) * u) @ L["down"].T
        return self.rms(x, self.norm) @ self.embed.T


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

    tokens = list(prompt)
    step_logits = []
    for _ in range(args.steps):
        logits = model.forward(tokens)[-1]
        step_logits.append(logits)
        tokens.append(int(np.argmax(logits)))
    generated = tokens[len(prompt) :]
    print(json.dumps({"prompt_tokens": prompt, "generated": generated}))
    print(repr(tok.decode(generated)))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "tokens.json").write_text(json.dumps({"prompt": prompt, "generated": generated}))
        np.stack(step_logits).astype("<f4").tofile(args.out_dir / "logits.f32")


if __name__ == "__main__":
    main()
