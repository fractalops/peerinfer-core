//! Llama forward pass on wgpu.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use wgpu::util::DeviceExt;

use crate::model::{Config, DType, Weights};

/// Largest context the attention kernel supports.
pub const MAX_CONTEXT: usize = 2048;

/// Largest number of tokens in one forward pass. Longer prompts run in chunks.
pub const MAX_CHUNK: usize = 256;

const ROWS_PER_WORKGROUP: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Step {
    tokens: u32,
    pos0: u32,
    _pad: [u32; 2],
}

struct Pass {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    /// Workgroups in x; y is the token count unless `fixed_y` is set.
    groups_x: u32,
    fixed_y: Option<u32>,
}

/// Runs one model on one GPU device.
pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: Config,
    step: wgpu::Buffer,
    tokens: wgpu::Buffer,
    logits: wgpu::Buffer,
    next: wgpu::Buffer,
    next_read: wgpu::Buffer,
    logits_read: wgpu::Buffer,
    passes: Vec<Pass>,
    head: Vec<Pass>,
    /// Number of positions already in the KV cache.
    pos: usize,
    adapter_info: wgpu::AdapterInfo,
}

impl Engine {
    /// Requests a GPU device and uploads the weights.
    pub async fn new(weights: &Weights) -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no GPU adapter")?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("peerinfer"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .context("request GPU device")?;
        Self::with_device(device, queue, adapter_info, weights)
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Number of tokens in the KV cache.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Clears the KV cache.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
        weights: &Weights,
    ) -> Result<Self> {
        let c = weights.config.clone();
        ensure!(
            c.head_dim <= 64,
            "head size {} is larger than 64",
            c.head_dim
        );
        let (h, inter, qkv_dim) = (c.hidden_size, c.intermediate_size, c.qkv_dim());

        let storage = |label: &str, bytes: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let upload = |name: &str, dtype: DType, shape: &[usize]| -> Result<wgpu::Buffer> {
            let bytes = weights.tensor(name, dtype, shape)?;
            let mut padded = bytes.to_vec();
            padded.resize(bytes.len().next_multiple_of(4), 0);
            Ok(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: &padded,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            )
        };
        let linear = |name: &str, rows: usize, k: usize| -> Result<(wgpu::Buffer, wgpu::Buffer)> {
            Ok((
                upload(&format!("{name}.q_weight"), DType::U32, &[rows, k / 8])?,
                upload(&format!("{name}.q_scale"), DType::F16, &[rows, k / 32])?,
            ))
        };

        let f = std::mem::size_of::<f32>();
        let x = storage("x", MAX_CHUNK * h * f);
        let xn = storage("xn", MAX_CHUNK * h * f);
        let qkv = storage("qkv", MAX_CHUNK * qkv_dim * f);
        let attn = storage("attn", MAX_CHUNK * c.q_dim() * f);
        let act = storage("act", MAX_CHUNK * inter * f);
        let last = storage("last", h * f);
        let logits = storage("logits", c.vocab_size * f);
        let next = storage("next", 4);
        let tokens = storage("tokens", MAX_CHUNK * 4);
        let step = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step"),
            size: std::mem::size_of::<Step>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read = |label: &str, bytes: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let next_read = read("next_read", 4);
        let logits_read = read("logits_read", c.vocab_size * f);

        let mut kernels = Kernels::new(&device);
        let hq = c.num_attention_heads as f64;
        let hkv = c.num_key_value_heads as f64;
        let d = c.head_dim as f64;
        let attn_consts = [("HQ", hq), ("HKV", hkv), ("D", d)];
        let rope_consts = [
            ("HQ", hq),
            ("HKV", hkv),
            ("D", d),
            ("BASE", c.position_embedding_base as f64),
        ];
        let norm_consts = [("K", h as f64), ("EPS", c.rms_norm_eps as f64)];
        let mm = |n: usize, k: usize, mode: u32| {
            [("N", n as f64), ("K", k as f64), ("MODE", mode as f64)]
        };
        let mm_groups = |n: usize| (n as u32).div_ceil(ROWS_PER_WORKGROUP);

        let (embed_w, embed_s) = linear("model.embed_tokens", c.vocab_size, h)?;
        let mut passes = vec![kernels.pass(
            &device,
            "embed",
            &[("K", h as f64)],
            &[&tokens, &embed_w, &embed_s, &x],
            ((h / 8) as u32).div_ceil(64),
            None,
        )?];

        for l in 0..c.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let ln1 = upload(&format!("{p}.input_layernorm.weight"), DType::F16, &[h])?;
            let ln2 = upload(
                &format!("{p}.post_attention_layernorm.weight"),
                DType::F16,
                &[h],
            )?;
            let (wqkv, sqkv) = linear(&format!("{p}.self_attn.qkv_proj"), qkv_dim, h)?;
            let (wo, so) = linear(&format!("{p}.self_attn.o_proj"), h, c.q_dim())?;
            let (wgu, sgu) = linear(&format!("{p}.mlp.gate_up_proj"), 2 * inter, h)?;
            let (wd, sd) = linear(&format!("{p}.mlp.down_proj"), h, inter)?;
            let kc = storage("k_cache", MAX_CONTEXT * c.kv_dim() * f);
            let vc = storage("v_cache", MAX_CONTEXT * c.kv_dim() * f);
            let heads = (c.num_attention_heads + c.num_key_value_heads) as u32;

            passes.push(kernels.pass(
                &device,
                "rmsnorm",
                &norm_consts,
                &[&x, &ln1, &xn, &step],
                1,
                None,
            )?);
            passes.push(kernels.pass(
                &device,
                "matmul",
                &mm(qkv_dim, h, 0),
                &[&xn, &wqkv, &sqkv, &qkv],
                mm_groups(qkv_dim),
                None,
            )?);
            passes.push(kernels.pass(
                &device,
                "rope_kv",
                &rope_consts,
                &[&qkv, &kc, &vc, &step],
                heads,
                None,
            )?);
            passes.push(kernels.pass(
                &device,
                "attention",
                &attn_consts,
                &[&qkv, &kc, &vc, &attn, &step],
                c.num_attention_heads as u32,
                None,
            )?);
            passes.push(kernels.pass(
                &device,
                "matmul",
                &mm(h, c.q_dim(), 1),
                &[&attn, &wo, &so, &x],
                mm_groups(h),
                None,
            )?);
            passes.push(kernels.pass(
                &device,
                "rmsnorm",
                &norm_consts,
                &[&x, &ln2, &xn, &step],
                1,
                None,
            )?);
            passes.push(kernels.pass(
                &device,
                "matmul",
                &mm(inter, h, 2),
                &[&xn, &wgu, &sgu, &act],
                mm_groups(inter),
                None,
            )?);
            passes.push(kernels.pass(
                &device,
                "matmul",
                &mm(h, inter, 1),
                &[&act, &wd, &sd, &x],
                mm_groups(h),
                None,
            )?);
        }

        let norm = upload("model.norm.weight", DType::F16, &[h])?;
        let last_norm = [
            ("K", h as f64),
            ("EPS", c.rms_norm_eps as f64),
            ("LAST", 1.0),
        ];
        let head = vec![
            kernels.pass(
                &device,
                "rmsnorm",
                &last_norm,
                &[&x, &norm, &last, &step],
                1,
                Some(1),
            )?,
            kernels.pass(
                &device,
                "matmul",
                &mm(c.vocab_size, h, 0),
                &[&last, &embed_w, &embed_s, &logits],
                mm_groups(c.vocab_size),
                Some(1),
            )?,
            kernels.pass(
                &device,
                "argmax",
                &[("V", c.vocab_size as f64)],
                &[&logits, &next],
                1,
                Some(1),
            )?,
        ];

        Ok(Self {
            device,
            queue,
            config: c,
            step,
            tokens,
            logits,
            next,
            next_read,
            logits_read,
            passes,
            head,
            pos: 0,
            adapter_info,
        })
    }

    /// Appends `tokens` to the context and returns the most likely next token.
    pub async fn forward(&mut self, tokens: &[u32]) -> Result<u32> {
        ensure!(!tokens.is_empty(), "no tokens");
        ensure!(
            self.pos + tokens.len() <= MAX_CONTEXT,
            "context is longer than {MAX_CONTEXT} tokens"
        );
        for &t in tokens {
            ensure!(
                (t as usize) < self.config.vocab_size,
                "token {t} is outside the vocabulary"
            );
        }
        let chunks: Vec<&[u32]> = tokens.chunks(MAX_CHUNK).collect();
        let last = chunks.len() - 1;
        for (i, chunk) in chunks.into_iter().enumerate() {
            self.run(chunk, i == last);
        }
        let bytes = read_buffer(&self.device, &self.next_read).await?;
        Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
    }

    /// Returns the logits of the last forward pass.
    pub async fn last_logits(&self) -> Result<Vec<f32>> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.logits, 0, &self.logits_read, 0, None);
        self.queue.submit([enc.finish()]);
        let bytes = read_buffer(&self.device, &self.logits_read).await?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    fn run(&mut self, chunk: &[u32], with_head: bool) {
        let n = chunk.len() as u32;
        self.queue.write_buffer(
            &self.step,
            0,
            bytemuck::bytes_of(&Step {
                tokens: n,
                pos0: self.pos as u32,
                _pad: [0; 2],
            }),
        );
        self.queue
            .write_buffer(&self.tokens, 0, bytemuck::cast_slice(chunk));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            let passes = self
                .passes
                .iter()
                .chain(if with_head { &self.head[..] } else { &[] });
            for p in passes {
                cp.set_pipeline(&p.pipeline);
                cp.set_bind_group(0, &p.bind, &[]);
                cp.dispatch_workgroups(p.groups_x, p.fixed_y.unwrap_or(n), 1);
            }
        }
        if with_head {
            enc.copy_buffer_to_buffer(&self.next, 0, &self.next_read, 0, None);
        }
        self.queue.submit([enc.finish()]);
        self.pos += chunk.len();
    }
}

/// Copies a mapped buffer to memory and unmaps it.
async fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = futures_channel::oneshot::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("wait for the GPU")?;
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    rx.await
        .context("buffer mapping was cancelled")?
        .context("map buffer")?;
    let bytes = buffer
        .slice(..)
        .get_mapped_range()
        .context("read mapped buffer")?
        .to_vec();
    buffer.unmap();
    Ok(bytes)
}

/// Compiled shader modules and pipelines, keyed by kernel and constants.
struct Kernels {
    modules: HashMap<&'static str, wgpu::ShaderModule>,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Self {
        let sources: [(&'static str, &str); 6] = [
            ("embed", include_str!("kernels/embed.wgsl")),
            ("rmsnorm", include_str!("kernels/rmsnorm.wgsl")),
            ("matmul", include_str!("kernels/matmul.wgsl")),
            ("rope_kv", include_str!("kernels/rope_kv.wgsl")),
            ("attention", include_str!("kernels/attention.wgsl")),
            ("argmax", include_str!("kernels/argmax.wgsl")),
        ];
        let modules = sources
            .into_iter()
            .map(|(name, src)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
                (name, module)
            })
            .collect();
        Self {
            modules,
            pipelines: HashMap::new(),
        }
    }

    fn pass(
        &mut self,
        device: &wgpu::Device,
        kernel: &'static str,
        constants: &[(&str, f64)],
        buffers: &[&wgpu::Buffer],
        groups_x: u32,
        fixed_y: Option<u32>,
    ) -> Result<Pass> {
        let key = format!("{kernel}{constants:?}");
        let module = self
            .modules
            .get(kernel)
            .with_context(|| format!("unknown kernel {kernel}"))?;
        let pipeline = self
            .pipelines
            .entry(key)
            .or_insert_with(|| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(kernel),
                    layout: None,
                    module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                    cache: None,
                })
            })
            .clone();
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(Pass {
            pipeline,
            bind,
            groups_x,
            fixed_y,
        })
    }
}
