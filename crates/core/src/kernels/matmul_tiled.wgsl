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

// One thread's share of a K step, held in registers so that the loads
// for the next step can overlap with the multiply of the current one.
struct Fetch {
    // Inputs for tasks e = li + 64 m: k = e % 32, token group e / 32.
    xv: array<vec4<f32>, 4>,
    // Weights for 4 rows of one word, and their scales.
    q: vec4<u32>,
    sc: vec4<f32>,
}

fn fetch(g: u32, li: u32, n0: u32, t0: u32, tokens: u32) -> Fetch {
    var f: Fetch;
    for (var m = 0u; m < 4u; m++) {
        let e = li + 64u * m;
        let k = e % 32u;
        let tg = e / 32u;
        let t = t0 + tg * 4u;
        let b = t * K + g * 32u + k;
        f.xv[m] = vec4(
            select(0.0, x[b], t < tokens),
            select(0.0, x[b + K], t + 1u < tokens),
            select(0.0, x[b + 2u * K], t + 2u < tokens),
            select(0.0, x[b + 3u * K], t + 3u < tokens),
        );
    }
    // Threads 0-31 load W rows; threads 32-63 load the up rows (SwiGLU).
    if (li < 32u || MODE == 2u) {
        let rg = (li % 32u) / 4u;
        let off = select(0u, N, li >= 32u);
        let r = vec4(
            min(n0 + rg * 4u, N - 1u),
            min(n0 + rg * 4u + 1u, N - 1u),
            min(n0 + rg * 4u + 2u, N - 1u),
            min(n0 + rg * 4u + 3u, N - 1u),
        ) + vec4(off);
        let wo = g * 4u + li % 4u;
        f.q = vec4(w[r.x * (K / 8u) + wo], w[r.y * (K / 8u) + wo], w[r.z * (K / 8u) + wo], w[r.w * (K / 8u) + wo]);
        f.sc = vec4(
            scale(r.x * (K / 32u) + g),
            scale(r.y * (K / 32u) + g),
            scale(r.z * (K / 32u) + g),
            scale(r.w * (K / 32u) + g),
        );
    }
    return f;
}

fn store(f: Fetch, li: u32) {
    xs[(li % 32u) * 8u + (li + 0u) / 32u] = f.xv[0];
    xs[(li % 32u) * 8u + (li + 64u) / 32u] = f.xv[1];
    xs[(li % 32u) * 8u + (li + 128u) / 32u] = f.xv[2];
    xs[(li % 32u) * 8u + (li + 192u) / 32u] = f.xv[3];
    if (li < 32u || MODE == 2u) {
        let rg = (li % 32u) / 4u;
        let wi = li % 4u;
        // Low nibbles hold values 0, 2, 4, 6 and high nibbles 1, 3, 5, 7.
        let lo = mat4x4(
            unpack4x8unorm(f.q.x & 0x0F0F0F0Fu),
            unpack4x8unorm(f.q.y & 0x0F0F0F0Fu),
            unpack4x8unorm(f.q.z & 0x0F0F0F0Fu),
            unpack4x8unorm(f.q.w & 0x0F0F0F0Fu),
        );
        let hi = mat4x4(
            unpack4x8unorm((f.q.x >> 4u) & 0x0F0F0F0Fu),
            unpack4x8unorm((f.q.y >> 4u) & 0x0F0F0F0Fu),
            unpack4x8unorm((f.q.z >> 4u) & 0x0F0F0F0Fu),
            unpack4x8unorm((f.q.w >> 4u) & 0x0F0F0F0Fu),
        );
        // Column r of lo/hi is row r; transpose to get one vec4 per value.
        let lt = transpose(lo);
        let ht = transpose(hi);
        let base = wi * 64u + rg;
        let sc = f.sc;
        if (li < 32u) {
            ws[base + 0u] = (lt[0] * 255.0 - 7.0) * sc;
            ws[base + 8u] = (ht[0] * 255.0 - 7.0) * sc;
            ws[base + 16u] = (lt[1] * 255.0 - 7.0) * sc;
            ws[base + 24u] = (ht[1] * 255.0 - 7.0) * sc;
            ws[base + 32u] = (lt[2] * 255.0 - 7.0) * sc;
            ws[base + 40u] = (ht[2] * 255.0 - 7.0) * sc;
            ws[base + 48u] = (lt[3] * 255.0 - 7.0) * sc;
            ws[base + 56u] = (ht[3] * 255.0 - 7.0) * sc;
        } else {
            us[base + 0u] = (lt[0] * 255.0 - 7.0) * sc;
            us[base + 8u] = (ht[0] * 255.0 - 7.0) * sc;
            us[base + 16u] = (lt[1] * 255.0 - 7.0) * sc;
            us[base + 24u] = (ht[1] * 255.0 - 7.0) * sc;
            us[base + 32u] = (lt[2] * 255.0 - 7.0) * sc;
            us[base + 40u] = (ht[2] * 255.0 - 7.0) * sc;
            us[base + 48u] = (lt[3] * 255.0 - 7.0) * sc;
            us[base + 56u] = (ht[3] * 255.0 - 7.0) * sc;
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
    let g_start = wg.z * per;
    let g_end = min(groups, g_start + per);
    var next = fetch(min(g_start, groups - 1u), li, n0, t0, tokens);
    for (var g = g_start; g < g_end; g++) {
        store(next, li);
        workgroupBarrier();
        if (g + 1u < g_end) {
            next = fetch(g + 1u, li, n0, t0, tokens);
        }
        // Unrolled by hand: naga emits loops that Metal may not unroll.
        for (var k = 0u; k < 32u; k += 8u) {
            {
                let a = xs[(k + 0u) * 8u + tt];
                let b = ws[(k + 0u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 0u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
            }
            {
                let a = xs[(k + 1u) * 8u + tt];
                let b = ws[(k + 1u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 1u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
            }
            {
                let a = xs[(k + 2u) * 8u + tt];
                let b = ws[(k + 2u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 2u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
            }
            {
                let a = xs[(k + 3u) * 8u + tt];
                let b = ws[(k + 3u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 3u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
            }
            {
                let a = xs[(k + 4u) * 8u + tt];
                let b = ws[(k + 4u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 4u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
            }
            {
                let a = xs[(k + 5u) * 8u + tt];
                let b = ws[(k + 5u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 5u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
            }
            {
                let a = xs[(k + 6u) * 8u + tt];
                let b = ws[(k + 6u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 6u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
            }
            {
                let a = xs[(k + 7u) * 8u + tt];
                let b = ws[(k + 7u) * 8u + nn];
                acc[0] += a * b.x;
                acc[1] += a * b.y;
                acc[2] += a * b.z;
                acc[3] += a * b.w;
                if (MODE == 2u) {
                    let u = us[(k + 7u) * 8u + nn];
                    acc2[0] += a * u.x;
                    acc2[1] += a * u.y;
                    acc2[2] += a * u.z;
                    acc2[3] += a * u.w;
                }
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
