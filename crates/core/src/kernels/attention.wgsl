// Causal attention with grouped key-value heads over the cache.
// One workgroup of 64 threads per (head, token). D must be at most 64.

override HQ: u32;
override HKV: u32;
override D: u32;

const MAX_CTX: u32 = 2048u;

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> step: Step;

var<workgroup> sc: array<f32, MAX_CTX>;
var<workgroup> q: array<f32, 64>;
var<workgroup> red: array<f32, 64>;

fn reduce_max(v: f32, i: u32) -> f32 {
    red[i] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
        if (i < st) {
            red[i] = max(red[i], red[i + st]);
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn reduce_sum(v: f32, i: u32) -> f32 {
    red[i] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
        if (i < st) {
            red[i] += red[i + st];
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let h = wg.x;
    let t = wg.y;
    let len = step.pos0 + t + 1u;
    let kh = h / (HQ / HKV);
    let kvd = HKV * D;
    if (i < D) {
        q[i] = qkv[t * (HQ + 2u * HKV) * D + h * D + i];
    }
    workgroupBarrier();

    let inv = inverseSqrt(f32(D));
    var m = -3.4e38;
    for (var j = i; j < len; j += 64u) {
        let kb = j * kvd + kh * D;
        var d = 0.0;
        for (var e = 0u; e < D; e++) {
            d += q[e] * kc[kb + e];
        }
        d *= inv;
        sc[j] = d;
        m = max(m, d);
    }
    m = reduce_max(m, i);

    var total = 0.0;
    for (var j = i; j < len; j += 64u) {
        let p = exp(sc[j] - m);
        sc[j] = p;
        total += p;
    }
    total = reduce_sum(total, i);

    if (i < D) {
        var acc = 0.0;
        for (var j = 0u; j < len; j++) {
            acc += sc[j] * vc[j * kvd + kh * D + i];
        }
        out[t * HQ * D + h * D + i] = acc / total;
    }
}
