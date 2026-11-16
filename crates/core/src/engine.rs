//! Llama forward pass on wgpu.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use wgpu::util::DeviceExt;

use crate::model::{Config, DType, Weights};

/// Largest context the attention kernel supports.
pub const MAX_CONTEXT: usize = 2048;

/// Largest number of tokens in one forward pass. Longer prompts run in chunks.
pub const MAX_CHUNK: usize = 256;

const ROWS_PER_WORKGROUP: u32 = 32;

/// Decode steps queued before their tokens are read back.
const GENERATE_BATCH: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Step {
    tokens: u32,
    pos0: u32,
    _pad: [u32; 2],
}

/// Tiled variant of a matmul pass for forward passes over many tokens.
struct Tiled {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    reduce: wgpu::ComputePipeline,
    reduce_bind: wgpu::BindGroup,
    rows: u32,
    splits: u32,
}

/// Target number of workgroups for a tiled matmul over one token tile.
const TILED_TARGET_WORKGROUPS: usize = 240;

/// Number of K splits for an `n` x `k` matmul.
fn k_splits(n: usize, k: usize) -> usize {
    let tiles = n.div_ceil(TILE as usize);
    let max = (k / 32 / 4).max(1);
    TILED_TARGET_WORKGROUPS.div_ceil(tiles).clamp(1, max)
}

impl Pass {
    fn dispatch(&self, cp: &mut wgpu::ComputePass<'_>, n: usize) {
        if self.tiled_only && n < TILED_MIN_TOKENS {
            return;
        }
        match &self.tiled {
            Some(t) if n >= TILED_MIN_TOKENS => {
                cp.set_pipeline(&t.pipeline);
                cp.set_bind_group(0, &t.bind, &[]);
                cp.dispatch_workgroups(t.rows.div_ceil(TILE), (n as u32).div_ceil(TILE), t.splits);
                cp.set_pipeline(&t.reduce);
                cp.set_bind_group(0, &t.reduce_bind, &[]);
                cp.dispatch_workgroups((n as u32 * t.rows).div_ceil(256), 1, 1);
            }
            _ => {
                cp.set_pipeline(&self.pipeline);
                cp.set_bind_group(0, &self.bind, &[]);
                cp.dispatch_workgroups(self.groups_x, self.fixed_y.unwrap_or(n as u32), 1);
            }
        }
    }
}

/// Forward passes with at least this many tokens use the tiled matmul.
const TILED_MIN_TOKENS: usize = 4;
const TILE: u32 = 32;

struct Pass {
    name: String,
    tiled: Option<Tiled>,
    /// Runs only in passes that use the tiled matmul (RMSNorm, which the
    /// one-token matmul does itself).
    tiled_only: bool,
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
    history: wgpu::Buffer,
    next_read: wgpu::Buffer,
    history_read: [wgpu::Buffer; 2],
    logits_read: wgpu::Buffer,
    passes: Vec<Pass>,
    head: Vec<Pass>,
    /// Number of positions already in the KV cache.
    pos: usize,
    adapter_info: wgpu::AdapterInfo,
    timestamps: bool,
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
        let optional = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("peerinfer"),
                required_features: optional,
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
        let device_has_timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let c = weights.config.clone();
        ensure!(c.head_dim == 64, "head size {} is not 64", c.head_dim);
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
        // Token at each position, written by argmax.
        let history = storage("history", (MAX_CONTEXT + 1) * 4);
        let tokens = storage("tokens", MAX_CHUNK * 4);
        // Partial sums for the tiled matmuls: splits x tokens x output width.
        let part_len = [
            (qkv_dim, h, qkv_dim),
            (h, c.q_dim(), h),
            (inter, h, 2 * inter),
            (h, inter, h),
        ]
        .iter()
        .map(|&(n, k, width)| k_splits(n, k) * MAX_CHUNK * width)
        .max()
        .unwrap();
        let part = storage("part", part_len * f);
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
        let history_read = [
            read("history_read_0", GENERATE_BATCH * 4),
            read("history_read_1", GENERATE_BATCH * 4),
        ];
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
            passes.last_mut().unwrap().tiled_only = true;
            passes.push(kernels.matmul(
                &device,
                qkv_dim,
                h,
                0,
                [&xn, &wqkv, &sqkv, &qkv],
                &step,
                &part,
                Some((&x, &ln1, c.rms_norm_eps)),
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
            passes.push(kernels.matmul(
                &device,
                h,
                c.q_dim(),
                1,
                [&attn, &wo, &so, &x],
                &step,
                &part,
                None,
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
            passes.last_mut().unwrap().tiled_only = true;
            passes.push(kernels.matmul(
                &device,
                inter,
                h,
                2,
                [&xn, &wgu, &sgu, &act],
                &step,
                &part,
                Some((&x, &ln2, c.rms_norm_eps)),
                None,
            )?);
            passes.push(kernels.matmul(
                &device,
                h,
                inter,
                1,
                [&act, &wd, &sd, &x],
                &step,
                &part,
                None,
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
            kernels.matmul(
                &device,
                c.vocab_size,
                h,
                0,
                [&last, &embed_w, &embed_s, &logits],
                &step,
                &part,
                None,
                Some(1),
            )?,
            kernels.pass(
                &device,
                "argmax",
                &[("V", c.vocab_size as f64)],
                &[&logits, &tokens, &history, &step],
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
            history,
            next_read,
            history_read,
            logits_read,
            passes,
            head,
            pos: 0,
            adapter_info,
            timestamps: device_has_timestamps,
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
            self.queue
                .write_buffer(&self.tokens, 0, bytemuck::cast_slice(chunk));
            let mut enc = self.encode_step(chunk.len(), i == last);
            if i == last {
                enc.copy_buffer_to_buffer(
                    &self.history,
                    4 * self.pos as u64,
                    &self.next_read,
                    0,
                    4,
                );
            }
            self.queue.submit([enc.finish()]);
        }
        let bytes = read_buffer(&self.device, &self.next_read, None).await?;
        Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
    }

    /// Generates up to `max` tokens greedily. The first input is the token
    /// returned by the last [`Engine::forward`] call, which must come first.
    ///
    /// Tokens stay on the GPU between steps, so the CPU queues later steps
    /// while the GPU runs earlier ones. `on_tokens` receives the new tokens
    /// in batches and returns `false` to stop. The context then ends after
    /// the last token it received.
    pub async fn generate(
        &mut self,
        max: usize,
        mut on_tokens: impl FnMut(&[u32]) -> bool,
    ) -> Result<()> {
        ensure!(self.pos > 0, "generate needs a forward pass first");
        ensure!(
            self.pos + max <= MAX_CONTEXT,
            "context is longer than {MAX_CONTEXT} tokens"
        );
        // Position of the first token not yet passed to on_tokens.
        let mut accepted = self.pos;
        let mut queued = 0;
        let mut pending: Option<(usize, usize, wgpu::SubmissionIndex)> = None;
        let mut slot = 0;
        loop {
            let mut current = None;
            if queued < max {
                let n = GENERATE_BATCH.min(max - queued);
                let first = self.pos + 1;
                let mut last_index = None;
                for i in 0..n {
                    let mut enc = self.encode_step(1, true);
                    if i == n - 1 {
                        enc.copy_buffer_to_buffer(
                            &self.history,
                            4 * first as u64,
                            &self.history_read[slot],
                            0,
                            4 * n as u64,
                        );
                    }
                    last_index = Some(self.queue.submit([enc.finish()]));
                }
                queued += n;
                current = Some((slot, n, last_index.unwrap()));
                slot ^= 1;
            }
            let Some((s, n, index)) = pending.take() else {
                if current.is_none() {
                    break;
                }
                pending = current;
                continue;
            };
            let bytes = read_buffer(&self.device, &self.history_read[s], Some(index)).await?;
            let tokens: &[u32] = bytemuck::cast_slice(&bytes[..4 * n]);
            accepted += n;
            if !on_tokens(tokens) {
                if let Some((_, _, index)) = current {
                    // Let the queued steps finish before the buffers are reused.
                    wait(&self.device, index)?;
                }
                self.pos = accepted;
                return Ok(());
            }
            pending = current;
        }
        Ok(())
    }

    /// Returns the logits of the last forward pass.
    pub async fn last_logits(&self) -> Result<Vec<f32>> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.logits, 0, &self.logits_read, 0, None);
        self.queue.submit([enc.finish()]);
        let bytes = read_buffer(&self.device, &self.logits_read, None).await?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Runs one decode step with each kernel in its own compute pass and
    /// returns GPU time in milliseconds per kernel, summed by name.
    /// Needs timestamp queries; returns an empty list without them.
    pub async fn profile(&mut self, tokens: &[u32]) -> Result<Vec<(String, f64)>> {
        let n = tokens.len();
        ensure!(
            (1..=MAX_CHUNK).contains(&n),
            "profile takes 1 to {MAX_CHUNK} tokens"
        );
        if !self.timestamps {
            return Ok(Vec::new());
        }
        let passes: Vec<&Pass> = self.passes.iter().chain(&self.head).collect();
        let count = 2 * passes.len() as u32;
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profile"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let bytes = 8 * count as u64;
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &self.step,
            0,
            bytemuck::bytes_of(&Step {
                tokens: n as u32,
                pos0: self.pos as u32,
                _pad: [0; 2],
            }),
        );
        self.queue
            .write_buffer(&self.tokens, 0, bytemuck::cast_slice(tokens));
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, p) in passes.iter().enumerate() {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &set,
                    beginning_of_pass_write_index: Some(2 * i as u32),
                    end_of_pass_write_index: Some(2 * i as u32 + 1),
                }),
            });
            p.dispatch(&mut cp, n);
        }
        enc.resolve_query_set(&set, 0..count, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, None);
        self.queue.submit([enc.finish()]);
        let raw = read_buffer(&self.device, &read, None).await?;
        let ticks: &[u64] = bytemuck::cast_slice(&raw);
        let period = self.queue.get_timestamp_period() as f64;
        let mut totals: Vec<(String, f64)> = Vec::new();
        for (i, p) in passes.iter().enumerate() {
            let ms = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period / 1e6;
            match totals.iter_mut().find(|(n, _)| *n == p.name) {
                Some((_, t)) => *t += ms,
                None => totals.push((p.name.clone(), ms)),
            }
        }
        self.pos += n;
        Ok(totals)
    }

    /// Encodes one forward pass over `n` tokens already in the token buffer.
    fn encode_step(&mut self, n: usize, with_head: bool) -> wgpu::CommandEncoder {
        self.queue.write_buffer(
            &self.step,
            0,
            bytemuck::bytes_of(&Step {
                tokens: n as u32,
                pos0: self.pos as u32,
                _pad: [0; 2],
            }),
        );
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            let passes = self
                .passes
                .iter()
                .chain(if with_head { &self.head[..] } else { &[] });
            for p in passes {
                p.dispatch(&mut cp, n);
            }
        }
        self.pos += n;
        enc
    }
}

/// Copies a mapped buffer to memory and unmaps it.
/// Waits for `index`, or for all work when it is `None`.
async fn read_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    index: Option<wgpu::SubmissionIndex>,
) -> Result<Vec<u8>> {
    let (tx, rx) = futures_channel::oneshot::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    match index {
        Some(i) => wait(device, i)?,
        None => wait_all(device)?,
    }
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

/// Blocks until `index` completes on native targets. Does nothing on the
/// web, where the browser completes work by itself.
fn wait(device: &wgpu::Device, index: wgpu::SubmissionIndex) -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .context("wait for the GPU")?;
    #[cfg(target_arch = "wasm32")]
    let _ = (device, index);
    Ok(())
}

fn wait_all(device: &wgpu::Device) -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("wait for the GPU")?;
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    Ok(())
}

/// Compiled shader modules and pipelines, keyed by kernel and constants.
struct Kernels {
    modules: HashMap<&'static str, wgpu::ShaderModule>,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Self {
        let sources: [(&'static str, &str); 8] = [
            ("matmul_tiled", include_str!("kernels/matmul_tiled.wgsl")),
            ("matmul_reduce", include_str!("kernels/matmul_reduce.wgsl")),
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
        let name = match kernel {
            "matmul" => format!(
                "matmul N={} K={} mode={}",
                constants[0].1, constants[1].1, constants[2].1
            ),
            _ => kernel.to_string(),
        };
        Ok(Pass {
            name,
            tiled: None,
            tiled_only: false,
            pipeline,
            bind,
            groups_x,
            fixed_y,
        })
    }
}

impl Kernels {
    /// A matmul pass with both the one-token and the tiled pipeline.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &mut self,
        device: &wgpu::Device,
        n: usize,
        k: usize,
        mode: u32,
        bufs: [&wgpu::Buffer; 4],
        step: &wgpu::Buffer,
        part: &wgpu::Buffer,
        norm: Option<(&wgpu::Buffer, &wgpu::Buffer, f32)>,
        fixed_y: Option<u32>,
    ) -> Result<Pass> {
        let consts = [("N", n as f64), ("K", k as f64), ("MODE", mode as f64)];
        // The one-token kernel normalizes the raw input itself when asked.
        let (one_x, gain, norm_consts) = match norm {
            Some((raw, gain, eps)) => (raw, gain, [("NORM", 1.0), ("EPS", eps as f64)]),
            None => (bufs[0], bufs[1], [("NORM", 0.0), ("EPS", 0.0)]),
        };
        let one_consts: Vec<(&str, f64)> = consts.iter().chain(&norm_consts).copied().collect();
        let mut pass = self.pass(
            device,
            "matmul",
            &one_consts,
            &[one_x, bufs[1], bufs[2], bufs[3], gain],
            (n as u32).div_ceil(ROWS_PER_WORKGROUP),
            fixed_y,
        )?;
        if fixed_y.is_none() {
            let [x, w, s, y] = bufs;
            let splits = k_splits(n, k);
            let tconsts = [
                ("N", n as f64),
                ("K", k as f64),
                ("MODE", mode as f64),
                ("SPLITS", splits as f64),
            ];
            let t = self.pass(
                device,
                "matmul_tiled",
                &tconsts,
                &[x, w, s, part, step],
                0,
                None,
            )?;
            let rconsts = [
                ("N", n as f64),
                ("MODE", mode as f64),
                ("SPLITS", splits as f64),
            ];
            let r = self.pass(device, "matmul_reduce", &rconsts, &[part, y, step], 0, None)?;
            pass.tiled = Some(Tiled {
                pipeline: t.pipeline,
                bind: t.bind,
                reduce: r.pipeline,
                reduce_bind: r.bind,
                rows: n as u32,
                splits: splits as u32,
            });
        }
        Ok(pass)
    }
}
