// y[t, n] = sum_k x[t, k] * W[n, k] with W in q4f16_1 format.
// 32 rows per workgroup, 8 lanes per row. Each lane reads one group of
// 32 weights (4 words, one scale) at a time.
// MODE 0 stores, MODE 1 adds to y (residual), MODE 2 applies SwiGLU:
// y[t, n] = silu(row n) * (row n + N).
// With NORM, x is first RMS-normalized with gain g; this replaces a
// separate RMSNorm dispatch. Without NORM, g is not read.

override N: u32;
override K: u32;
override MODE: u32 = 0u;
override NORM: bool = false;
override EPS: f32 = 0.0;

const LANES: u32 = 8u;
const ROWS: u32 = 32u;
// Input words staged at a time; longer rows run in chunks.
const MAX_WORDS: u32 = 320u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> s: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> g: array<u32>;

// For word i: x values 8i+0,2,4,6 and 8i+1,3,5,7, and their sum.
var<workgroup> xe: array<vec4<f32>, MAX_WORDS>;
var<workgroup> xo: array<vec4<f32>, MAX_WORDS>;
var<workgroup> xsum: array<f32, MAX_WORDS>;
var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;

fn scale(i: u32) -> f32 {
    let p = unpack2x16float(s[i >> 1u]);
    return select(p.x, p.y, (i & 1u) == 1u);
}

// sum over the 8 values of one word, without the -7 offset and the scale.
fn word_dot(q: u32, i: u32) -> f32 {
    let lo = unpack4x8unorm(q & 0x0F0F0F0Fu);
    let hi = unpack4x8unorm((q >> 4u) & 0x0F0F0F0Fu);
    return dot(lo, xe[i]) + dot(hi, xo[i]);
}

// Groups of this row whose words c0..c1 are staged.
fn dot_row(row: u32, lane: u32, c0: u32, c1: u32) -> f32 {
    let groups = K / 32u;
    var acc = 0.0;
    for (var g = c0 / 4u + lane; g < c1 / 4u; g += LANES) {
        let q = w[row * groups + g];
        let i = g * 4u - c0;
        // unpack4x8unorm returns byte / 255.
        let d = 255.0 * (word_dot(q.x, i) + word_dot(q.y, i + 1u) + word_dot(q.z, i + 2u) + word_dot(q.w, i + 3u));
        let off = 7.0 * (xsum[i] + xsum[i + 1u] + xsum[i + 2u] + xsum[i + 3u]);
        acc += (d - off) * scale(row * groups + g);
    }
    return acc;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let words = K / 8u;
    var inv = 1.0;
    if (NORM) {
        var ss = 0.0;
        for (var i = li; i < K; i += 256u) {
            let v = x[t * K + i];
            ss += v * v;
        }
        red[li] = ss;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st >>= 1u) {
            if (li < st) {
                red[li] += red[li + st];
            }
            workgroupBarrier();
        }
        inv = inverseSqrt(red[0] / f32(K) + EPS);
        workgroupBarrier();
    }
    let lane = li % LANES;
    let row = wg.x * ROWS + li / LANES;
    let r = min(row, N - 1u);
    var acc = 0.0;
    var acc2 = 0.0;
    for (var c0 = 0u; c0 < words; c0 += MAX_WORDS) {
        let c1 = min(words, c0 + MAX_WORDS);
        for (var i = c0 + li; i < c1; i += 256u) {
            let b = t * K + i * 8u;
            var e = vec4(x[b], x[b + 2u], x[b + 4u], x[b + 6u]);
            var o = vec4(x[b + 1u], x[b + 3u], x[b + 5u], x[b + 7u]);
            if (NORM) {
                let p0 = unpack2x16float(g[4u * i]);
                let p1 = unpack2x16float(g[4u * i + 1u]);
                let p2 = unpack2x16float(g[4u * i + 2u]);
                let p3 = unpack2x16float(g[4u * i + 3u]);
                e *= inv * vec4(p0.x, p1.x, p2.x, p3.x);
                o *= inv * vec4(p0.y, p1.y, p2.y, p3.y);
            }
            xe[i - c0] = e;
            xo[i - c0] = o;
            xsum[i - c0] = dot(e, vec4(1.0)) + dot(o, vec4(1.0));
        }
        workgroupBarrier();
        acc += dot_row(r, lane, c0, c1);
        if (MODE == 2u) {
            acc2 += dot_row(r + N, lane, c0, c1);
        }
        workgroupBarrier();
    }
    red[li] = acc;
    if (MODE == 2u) {
        red2[li] = acc2;
    }
    workgroupBarrier();
    for (var st = LANES / 2u; st > 0u; st >>= 1u) {
        if (lane < st) {
            red[li] += red[li + st];
            if (MODE == 2u) {
                red2[li] += red2[li + st];
            }
        }
        workgroupBarrier();
    }
    if (lane == 0u && row < N) {
        let o = t * N + row;
        if (MODE == 0u) {
            y[o] = red[li];
        } else if (MODE == 1u) {
            y[o] = y[o] + red[li];
        } else {
            let g = red[li];
            y[o] = g / (1.0 + exp(-g)) * red2[li];
        }
    }
}
