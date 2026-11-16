// y[t, n] = sum_k x[t, k] * W[n, k] with W in q4f16_1 format.
// Eight rows per workgroup, 32 lanes per row.
// MODE 0 stores, MODE 1 adds to y (residual), MODE 2 applies SwiGLU:
// y[t, n] = silu(row n) * (row n + N).

override N: u32;
override K: u32;
override MODE: u32 = 0u;

const LANES: u32 = 32u;
const ROWS: u32 = 8u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> s: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;

fn scale(i: u32) -> f32 {
    let p = unpack2x16float(s[i >> 1u]);
    return select(p.x, p.y, (i & 1u) == 1u);
}

fn dot_row(row: u32, t: u32, lane: u32) -> f32 {
    let words = K / 8u;
    var acc = 0.0;
    for (var wi = lane; wi < words; wi += LANES) {
        let q = w[row * words + wi];
        let xb = t * K + wi * 8u;
        var part = 0.0;
        for (var j = 0u; j < 8u; j++) {
            part += (f32((q >> (4u * j)) & 15u) - 7.0) * x[xb + j];
        }
        acc += part * scale(row * (K / 32u) + wi / 4u);
    }
    return acc;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li % LANES;
    let row = wg.x * ROWS + li / LANES;
    let t = wg.y;
    let r = min(row, N - 1u);
    red[li] = dot_row(r, t, lane);
    if (MODE == 2u) {
        red2[li] = dot_row(r + N, t, lane);
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
