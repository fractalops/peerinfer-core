// y[t, n] = sum_k x[t, k] * W[n, k] with W in q4f16_1 format.
// 32 rows per workgroup, 8 lanes per row. Each lane reads one group of
// 32 weights (4 words, one scale) at a time.
// MODE 0 stores, MODE 1 adds to y (residual), MODE 2 applies SwiGLU:
// y[t, n] = silu(row n) * (row n + N).

override N: u32;
override K: u32;
override MODE: u32 = 0u;

const LANES: u32 = 8u;
const ROWS: u32 = 32u;
const MAX_WORDS: u32 = 320u; // K <= 2560

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> s: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

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

fn dot_row(row: u32, lane: u32) -> f32 {
    let groups = K / 32u;
    var acc = 0.0;
    for (var g = lane; g < groups; g += LANES) {
        let q = w[row * groups + g];
        let i = g * 4u;
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
    for (var i = li; i < words; i += 256u) {
        let b = t * K + i * 8u;
        let e = vec4(x[b], x[b + 2u], x[b + 4u], x[b + 6u]);
        let o = vec4(x[b + 1u], x[b + 3u], x[b + 5u], x[b + 7u]);
        xe[i] = e;
        xo[i] = o;
        xsum[i] = dot(e, vec4(1.0)) + dot(o, vec4(1.0));
    }
    workgroupBarrier();

    let lane = li % LANES;
    let row = wg.x * ROWS + li / LANES;
    let r = min(row, N - 1u);
    red[li] = dot_row(r, lane);
    if (MODE == 2u) {
        red2[li] = dot_row(r + N, lane);
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
