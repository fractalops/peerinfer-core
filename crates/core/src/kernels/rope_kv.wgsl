// Applies rotary position embedding to q and k in place (rotate-half
// layout), then writes k and v to the cache at position pos0 + t.
// One workgroup per (head, token); heads HQ.. are key-value heads.

override HQ: u32;
override HKV: u32;
override D: u32;
override BASE: f32;

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> vc: array<f32>;
@group(0) @binding(3) var<uniform> step: Step;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let half = D / 2u;
    if (i >= half) {
        return;
    }
    let h = wg.x;
    let t = wg.y;
    let pos = step.pos0 + t;
    let row = t * (HQ + 2u * HKV) * D;
    let ang = f32(pos) * pow(BASE, -2.0 * f32(i) / f32(D));
    let c = cos(ang);
    let sn = sin(ang);
    if (h < HQ) {
        let b = row + h * D;
        let a0 = qkv[b + i];
        let a1 = qkv[b + i + half];
        qkv[b + i] = a0 * c - a1 * sn;
        qkv[b + i + half] = a1 * c + a0 * sn;
    } else {
        let kh = h - HQ;
        let kb = row + HQ * D + kh * D;
        let vb = row + (HQ + HKV) * D + kh * D;
        let cb = pos * HKV * D + kh * D;
        let a0 = qkv[kb + i];
        let a1 = qkv[kb + i + half];
        kc[cb + i] = a0 * c - a1 * sn;
        kc[cb + i + half] = a1 * c + a0 * sn;
        vc[cb + i] = qkv[vb + i];
        vc[cb + i + half] = qkv[vb + i + half];
    }
}
