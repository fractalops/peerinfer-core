// y[r] = x[src] / rms(x[src]) * g. One workgroup per token row (workgroup y).
// With LAST, reads only the last token row and writes row 0.

override K: u32;
override EPS: f32;
override LAST: bool = false;

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> step: Step;

var<workgroup> red: array<f32, 256>;

fn gain(i: u32) -> f32 {
    let p = unpack2x16float(g[i >> 1u]);
    return select(p.x, p.y, (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let src = select(wg.y, step.tokens - 1u, LAST);
    let base = src * K;
    var acc = 0.0;
    for (var k = li; k < K; k += 256u) {
        let v = x[base + k];
        acc += v * v;
    }
    red[li] = acc;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) {
            red[li] += red[li + st];
        }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(K) + EPS);
    let out = wg.y * K;
    for (var k = li; k < K; k += 256u) {
        y[out + k] = x[base + k] * inv * gain(k);
    }
}
