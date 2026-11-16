// Adds the SPLITS partial sums from matmul_tiled.wgsl and writes y.
// MODE 0 stores, MODE 1 adds to y (residual), MODE 2 applies SwiGLU:
// y[t, n] = silu(gate) * up. One thread per output value.

override N: u32;
override MODE: u32 = 0u;
override SPLITS: u32 = 1u;

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> step: Step;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tokens = step.tokens;
    let i = gid.x;
    if (i >= tokens * N) {
        return;
    }
    let t = i / N;
    let n = i % N;
    let width = select(N, 2u * N, MODE == 2u);
    var a = 0.0;
    var b = 0.0;
    for (var z = 0u; z < SPLITS; z++) {
        let o = (z * tokens + t) * width + n;
        a += part[o];
        if (MODE == 2u) {
            b += part[o + N];
        }
    }
    if (MODE == 0u) {
        y[i] = a;
    } else if (MODE == 1u) {
        y[i] = y[i] + a;
    } else {
        y[i] = a / (1.0 + exp(-a)) * b;
    }
}
