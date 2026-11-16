// Applies rotary position embedding to q and k in place (rotate-half
// layout), then writes k and v to the cache at position pos0 + t.
// One workgroup per (head, token); heads HQ.. are key-value heads.
// Thread i handles the pair (i, i + D/2). With QKNORM, each query and key
// head is first RMS-normalized with gains qn and kn (Qwen3). Without it,
// qn and kn are not read.

override HQ: u32;
override HKV: u32;
override D: u32;
override BASE: f32;
override QKNORM: bool = false;
override EPS: f32 = 0.0;

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> vc: array<f32>;
@group(0) @binding(3) var<uniform> step: Step;
@group(0) @binding(4) var<storage, read> qn: array<u32>;
@group(0) @binding(5) var<storage, read> kn: array<u32>;

var<workgroup> red: array<f32, 64>;

fn gain(i: u32, key: bool) -> f32 {
    let w = select(qn[i >> 1u], kn[i >> 1u], key);
    let p = unpack2x16float(w);
    return select(p.x, p.y, (i & 1u) == 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let half = D / 2u;
    let owns_pair = i < half;
    let h = wg.x;
    let t = wg.y;
    let pos = step.pos0 + t;
    let row = t * (HQ + 2u * HKV) * D;
    let key = h >= HQ;
    let kh = h - min(h, HQ);
    // Start of this head's query or key values.
    let b = select(row + h * D, row + HQ * D + kh * D, key);

    var a0 = 0.0;
    var a1 = 0.0;
    if (owns_pair) {
        a0 = qkv[b + i];
        a1 = qkv[b + i + half];
    }
    if (QKNORM) {
        red[i] = a0 * a0 + a1 * a1;
        workgroupBarrier();
        for (var st = 32u; st > 0u; st >>= 1u) {
            if (i < st) {
                red[i] += red[i + st];
            }
            workgroupBarrier();
        }
        let inv = inverseSqrt(red[0] / f32(D) + EPS);
        if (owns_pair) {
            a0 *= inv * gain(i, key);
            a1 *= inv * gain(i + half, key);
        }
    }
    if (!owns_pair) {
        return;
    }

    let ang = f32(pos) * pow(BASE, -2.0 * f32(i) / f32(D));
    let c = cos(ang);
    let sn = sin(ang);
    let r0 = a0 * c - a1 * sn;
    let r1 = a1 * c + a0 * sn;
    if (!key) {
        qkv[b + i] = r0;
        qkv[b + i + half] = r1;
    } else {
        let vb = row + (HQ + HKV) * D + kh * D;
        let cb = pos * HKV * D + kh * D;
        kc[cb + i] = r0;
        kc[cb + i + half] = r1;
        vc[cb + i] = qkv[vb + i];
        vc[cb + i + half] = qkv[vb + i + half];
    }
}
