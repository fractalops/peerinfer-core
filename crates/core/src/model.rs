//! Model configuration and MLC `q4f16_1` weight files.

use std::{collections::HashMap, ops::Range};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

/// The model families the engine runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Llama,
    /// Llama with RMSNorm on each query and key head before rotary embedding.
    Qwen3,
}

/// Decoder shape, read from `mlc-chat-config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// Rotary embedding base: `position_embedding_base`, or `rope_theta` in
    /// Qwen3 configs.
    #[serde(alias = "rope_theta")]
    pub position_embedding_base: f32,
    pub tie_word_embeddings: bool,
    #[serde(skip, default = "default_arch")]
    pub arch: Arch,
}

fn default_arch() -> Arch {
    Arch::Llama
}

impl Config {
    pub fn from_mlc_chat_config(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct File {
            quantization: String,
            model_type: String,
            model_config: Config,
        }
        let file: File = serde_json::from_str(json).context("parse mlc-chat-config.json")?;
        let arch = match file.model_type.as_str() {
            "llama" => Arch::Llama,
            "qwen3" => Arch::Qwen3,
            other => bail!("unsupported model type {other}"),
        };
        ensure!(
            file.quantization == "q4f16_1",
            "unsupported quantization {}",
            file.quantization
        );
        let mut c = file.model_config;
        c.arch = arch;
        ensure!(c.tie_word_embeddings, "only tied embeddings are supported");
        ensure!(
            c.num_attention_heads.is_multiple_of(c.num_key_value_heads),
            "attention heads must be a multiple of key-value heads"
        );
        ensure!(
            c.head_dim == 64 || c.head_dim == 128,
            "unsupported head size {}",
            c.head_dim
        );
        for k in [c.hidden_size, c.intermediate_size] {
            ensure!(
                k % 32 == 0,
                "dimension {k} is not a multiple of the group size 32"
            );
        }
        Ok(c)
    }

    /// Name of the fused query, key and value projection.
    pub fn qkv_name(&self) -> &'static str {
        match self.arch {
            Arch::Llama => "qkv_proj",
            Arch::Qwen3 => "c_attn",
        }
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn qkv_dim(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
}

/// One tensor as stored in the shards: raw little-endian bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    U32,
    F16,
    F32,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shard: usize,
    pub offset: usize,
    pub nbytes: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Index of the tensors in `ndarray-cache.json`.
#[derive(Debug, Clone)]
pub struct WeightIndex {
    pub shards: Vec<String>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl WeightIndex {
    pub fn from_ndarray_cache(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct File {
            records: Vec<Shard>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Shard {
            data_path: String,
            records: Vec<Record>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Record {
            name: String,
            shape: Vec<usize>,
            dtype: String,
            nbytes: usize,
            byte_offset: usize,
        }
        let file: File = serde_json::from_str(json).context("parse ndarray-cache.json")?;
        let mut shards = Vec::new();
        let mut tensors = HashMap::new();
        for (i, shard) in file.records.into_iter().enumerate() {
            shards.push(shard.data_path);
            for r in shard.records {
                let dtype = match r.dtype.as_str() {
                    "uint32" => DType::U32,
                    "float16" => DType::F16,
                    "float32" => DType::F32,
                    other => bail!("tensor {} has unsupported type {other}", r.name),
                };
                tensors.insert(
                    r.name,
                    TensorInfo {
                        shard: i,
                        offset: r.byte_offset,
                        nbytes: r.nbytes,
                        shape: r.shape,
                        dtype,
                    },
                );
            }
        }
        Ok(Self { shards, tensors })
    }

    /// True when a stage running `layers` of a model with `total` layers
    /// needs tensor `name`.
    pub fn stage_needs(name: &str, layers: &Range<usize>, total: usize) -> bool {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            return rest
                .split('.')
                .next()
                .and_then(|l| l.parse::<usize>().ok())
                .is_some_and(|l| layers.contains(&l));
        }
        let (first, last) = (layers.start == 0, layers.end == total);
        if name.starts_with("model.embed_tokens.") {
            return first || last;
        }
        name.starts_with("model.norm.") && last
    }

    /// Indexes of the shard files that hold the tensors for `layers`.
    pub fn shards_for(&self, layers: &Range<usize>, total: usize) -> Vec<usize> {
        let mut shards: Vec<usize> = self
            .tensors
            .iter()
            .filter(|(name, _)| Self::stage_needs(name, layers, total))
            .map(|(_, t)| t.shard)
            .collect();
        shards.sort_unstable();
        shards.dedup();
        shards
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }
}

/// Weight files held in memory.
pub struct Weights {
    pub config: Config,
    pub index: WeightIndex,
    pub shards: Vec<Vec<u8>>,
}

impl Weights {
    /// Returns the bytes of one tensor after checking its shape and type.
    pub fn tensor(&self, name: &str, dtype: DType, shape: &[usize]) -> Result<&[u8]> {
        let t = self.index.get(name)?;
        ensure!(
            t.dtype == dtype,
            "tensor {name} has type {:?}, expected {dtype:?}",
            t.dtype
        );
        ensure!(
            t.shape == shape,
            "tensor {name} has shape {:?}, expected {shape:?}",
            t.shape
        );
        let shard = self
            .shards
            .get(t.shard)
            .filter(|s| !s.is_empty())
            .with_context(|| {
                format!("tensor {name} is in shard {}, which is not loaded", t.shard)
            })?;
        shard
            .get(t.offset..t.offset + t.nbytes)
            .with_context(|| format!("tensor {name} is outside its shard"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_dir(dir: &std::path::Path) -> Result<Self> {
        Self::read_dir_stage(dir, None)
    }

    /// Reads only the shard files that `layers` needs; others stay empty.
    /// With `None`, reads every shard.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_dir_stage(dir: &std::path::Path, layers: Option<Range<usize>>) -> Result<Self> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .with_context(|| format!("read {}", dir.join(name).display()))
        };
        let config = Config::from_mlc_chat_config(&read("mlc-chat-config.json")?)?;
        let index = WeightIndex::from_ndarray_cache(&read("ndarray-cache.json")?)?;
        let needed = layers.map(|l| index.shards_for(&l, config.num_hidden_layers));
        let shards = index
            .shards
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if needed.as_ref().is_some_and(|n| !n.contains(&i)) {
                    return Ok(Vec::new());
                }
                std::fs::read(dir.join(s)).with_context(|| format!("read {s}"))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            config,
            index,
            shards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::WeightIndex;

    #[test]
    fn stage_needs_its_layers_and_ends() {
        let needs = |name: &str, layers: std::ops::Range<usize>| {
            WeightIndex::stage_needs(name, &layers, 30)
        };
        assert!(needs("model.layers.3.mlp.down_proj.q_weight", 0..15));
        assert!(!needs("model.layers.15.mlp.down_proj.q_weight", 0..15));
        assert!(needs("model.layers.15.mlp.down_proj.q_weight", 15..30));
        // "layers.1" must not match "layers.10".
        assert!(!needs("model.layers.10.self_attn.o_proj.q_scale", 1..2));
        // The embedding is the input of the first stage and the LM head of the last.
        assert!(needs("model.embed_tokens.q_weight", 0..15));
        assert!(needs("model.embed_tokens.q_weight", 15..30));
        assert!(!needs("model.embed_tokens.q_weight", 10..20));
        assert!(needs("model.norm.weight", 15..30));
        assert!(!needs("model.norm.weight", 0..15));
    }
}
