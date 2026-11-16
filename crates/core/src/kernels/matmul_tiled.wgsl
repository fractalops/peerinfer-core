// Partial sums of x[t, k] * W[n, k] for many tokens, W in q4f16_1 format.
// Workgroup z handles one of SPLITS ranges of K and writes
// part[z, t, n] (and part[z, t, N + n] for the up rows with SwiGLU).
// matmul_reduce.wgsl adds the partial sums and applies MODE.
// Each workgroup computes a tile of 32 tokens x 32 rows. Each of the 64
// threads keeps 4 tokens x 4 rows in registers. Each K step covers one
// quantization group (32 values), dequantized once into workgroup memory.
// MODE 2 (SwiGLU) also computes the up rows n + N.
//
// Workgroup tiles hold whole vec4 values and are only written as whole
// vec4 values: element [k * 8 + i] holds tokens (or rows) 4i .. 4i + 3.

override N: u32;
override K: u32;
override MODE: u32 = 0u;
override SPLITS: u32 = 1u;

const TILE: u32 = 32u;

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> s: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<uniform> step: Step;

var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 256>;
var<workgroup> us: array<vec4<f32>, 256>;

fn scale(i: u32) -> f32 {
    let p = unpack2x16float(s[i >> 1u]);
    return select(p.x, p.y, (i & 1u) == 1u);
}

fn nibble(q: u32, j: u32) -> f32 {
    return f32((q >> (4u * j)) & 15u) - 7.0;
}

// Dequantizes word `wi` of group `g` for the 4 rows of tile row group
// `rg` (W rows offset by `off`) into 8 vec4 values.
fn load_rows(n0: u32, rg: u32, off: u32, g: u32, wi: u32, up: bool) {
    var q = vec4<u32>();
    var sc = vec4<f32>();
    for (var i = 0u; i < 4u; i++) {
        let row = min(n0 + rg * 4u + i, N - 1u) + off;
        q[i] = w[row * (K / 8u) + g * 4u + wi];
        sc[i] = scale(row * (K / 32u) + g);
    }
    for (var j = 0u; j < 8u; j++) {
        let v = vec4(nibble(q.x, j), nibble(q.y, j), nibble(q.z, j), nibble(q.w, j)) * sc;
        let idx = (wi * 8u + j) * 8u + rg;
        if (up) {
            us[idx] = v;
        } else {
            ws[idx] = v;
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n0 = wg.x * TILE;
    let t0 = wg.y * TILE;
    let tokens = step.tokens;
    let tt = li % 8u; // token group: tokens tt*4 .. tt*4+3
    let nn = li / 8u; // row group: rows nn*4 .. nn*4+3

    var acc: array<vec4<f32>, 4>; // acc[j][i]: row j, token i
    var acc2: array<vec4<f32>, 4>;

    let groups = K / 32u;
    let per = (groups + SPLITS - 1u) / SPLITS;
    let g_end = min(groups, (wg.z + 1u) * per);
    for (var g = wg.z * per; g < g_end; g++) {
        // Inputs: 32 k values x 8 token groups, 4 tasks per thread.
        for (var e = li; e < 256u; e += 64u) {
            let k = e % 32u;
            let tg = e / 32u;
            var v = vec4<f32>();
            for (var i = 0u; i < 4u; i++) {
                let t = t0 + tg * 4u + i;
                if (t < tokens) {
                    v[i] = x[t * K + g * 32u + k];
                }
            }
            xs[k * 8u + tg] = v;
        }
        // Weights: 8 row groups x 4 words. Threads 0-31 load W rows,
        // threads 32-63 load the up rows for SwiGLU.
        if (li < 32u) {
            load_rows(n0, li / 4u, 0u, g, li % 4u, false);
        } else if (MODE == 2u) {
            load_rows(n0, (li - 32u) / 4u, N, g, li % 4u, true);
        }
        workgroupBarrier();
        for (var k = 0u; k < 32u; k++) {
            let a = xs[k * 8u + tt];
            let b = ws[k * 8u + nn];
            acc[0] += a * b.x;
            acc[1] += a * b.y;
            acc[2] += a * b.z;
            acc[3] += a * b.w;
            if (MODE == 2u) {
                let u = us[k * 8u + nn];
                acc2[0] += a * u.x;
                acc2[1] += a * u.y;
                acc2[2] += a * u.z;
                acc2[3] += a * u.w;
            }
        }
        workgroupBarrier();
    }

    let width = select(N, 2u * N, MODE == 2u);
    for (var j = 0u; j < 4u; j++) {
        let n = n0 + nn * 4u + j;
        for (var i = 0u; i < 4u; i++) {
            let t = t0 + tt * 4u + i;
            if (n < N && t < tokens) {
                let o = (wg.z * tokens + t) * width + n;
                part[o] = acc[j][i];
                if (MODE == 2u) {
                    part[o + N] = acc2[j][i];
                }
            }
        }
    }
}
