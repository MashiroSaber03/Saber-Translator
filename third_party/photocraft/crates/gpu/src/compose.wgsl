// Photocraft GPU compositor kernels.
//
// Every pass renders one chunk of the canvas into a straight-alpha RGBA32F accumulator, reading
// its inputs with `textureLoad` at the same pixel. Effect-map kernels (`fs_m*`) render
// single-channel maps over a layer's effect region instead (`chunk` = the region). The functions mirror `photocraft-compose`
// (the CPU reference) operation for operation; keep them in sync.

struct Chunk {
    origin: vec2<i32>,
    size: vec2<i32>,
};

struct Op {
    mode: i32,
    kind: i32,
    flags: u32,
    _pad0: u32,
    opacity: f32,
    mask_density: f32,
    mask_default: f32,
    fill: f32,       // layer Fill for fs_blend (psblend::composite_fill); 1 elsewhere
    tex_origin: vec2<i32>,
    tex_size: vec2<i32>,
    mask_origin: vec2<i32>,
    mask_size: vec2<i32>,
    color: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    map_origin: vec2<i32>,
    map_size: vec2<i32>,
    p4: vec4<f32>,
};

const F_MASK: u32 = 1u;          // layer has an enabled mask
const F_MASK_TEX: u32 = 2u;      // mask pixels live in `mask_tex`
const F_TEX: u32 = 4u;           // layer pixels live in `layer_tex`
const F_GRADIENT: u32 = 8u;      // gradient fill (stops in `lut_tex`)
const F_KNOCKOUT: u32 = 16u;     // effect paint: the layer knocks out the coverage (drop shadow)
const F_NO_LAYER: u32 = 32u;    // effect merge: A already holds the layer (only the opacity mix)
const F_OUTLINE: u32 = 131072u;      // outside strokes: filled shape outline (effects::outline_share)
const F_VECTOR: u32 = 64u;       // effect paint: shape layer (outside stroke never inside)
const F_ATOP: u32 = 128u;        // effect merge: clipped layer over an opaque base
const F_GATE: u32 = 256u;        // effect paint: coverage only inside the layer's shape
const F_REL: u32 = 512u;         // effect paint: coverage relative to the layer's alpha
const F_STROKE_OUT: u32 = 1024u; // effect paint: outside stroke band
const F_FIRST: u32 = 2048u;      // outside strokes: nothing accumulated yet
const F_CHANNELS: u32 = 4096u;   // lerp: per-channel weights in p0 (channel restrictions)
const F_LAB: u32 = 65536u;      // Lab document: Normal mixes in CIELAB
const F_HDR: u32 = 262144u;     // 32-bit float document: HDR blends and Exposure retain values above 1
const F_QUANT: u32 = 32768u;    // lerp: A rounded to p0.x steps (adjustment results, integer docs)
const F_ADD_DIFF: u32 = 16384u;  // lerp: A + (B - C) premultiplied (clips on pass-through groups)
const F_BLEND_IF: u32 = 524288u; // lerp: B back towards A by the Blend If ranges in p0..p3
const F_TEXT_GAMMA: u32 = 8192u; // blend / atop / fx merge: type layer, mix coverage at gamma p4.w

@group(0) @binding(0) var<uniform> chunk: Chunk;
@group(0) @binding(1) var<uniform> op: Op;
@group(1) @binding(0) var tex_a: texture_2d<f32>;
@group(1) @binding(1) var tex_b: texture_2d<f32>;
@group(1) @binding(2) var layer_tex: texture_2d<f32>;
@group(1) @binding(3) var mask_tex: texture_2d<f32>;
@group(1) @binding(4) var lut_tex: texture_2d<f32>;
@group(1) @binding(5) var tex_c: texture_2d<f32>;
@group(1) @binding(6) var map_tex: texture_2d<f32>;
@group(1) @binding(7) var pat_tex: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    let uv = vec2(f32((vi << 1u) & 2u), f32(vi & 2u));
    return VOut(vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0));
}

fn local(pos: vec4<f32>) -> vec2<i32> { return vec2<i32>(floor(pos.xy)); }
fn doc_px(p: vec2<i32>) -> vec2<i32> { return chunk.origin + p; }

fn inside(p: vec2<i32>, size: vec2<i32>) -> bool {
    return p.x >= 0 && p.y >= 0 && p.x < size.x && p.y < size.y;
}

// Rust's f32::round (half away from zero) for non-negative inputs.
fn round_half_up(x: f32) -> f32 { return floor(x + 0.5); }
fn rem_euclid(x: f32, m: f32) -> f32 { return x - floor(x / m) * m; }

fn mask_value(d: vec2<i32>) -> f32 {
    if ((op.flags & F_MASK) == 0u) {
        return 1.0;
    }
    var v = op.mask_default;
    if ((op.flags & F_MASK_TEX) != 0u) {
        let m = d - op.mask_origin;
        if (inside(m, op.mask_size)) {
            v = textureLoad(mask_tex, m, 0).r;
        }
    }
    return 1.0 - op.mask_density * (1.0 - v);
}

fn lut(row: i32, v: f32) -> f32 {
    let x = clamp(v, 0.0, 1.0) * 4095.0;
    let i = i32(floor(x));
    let j = min(i + 1, 4095);
    let f = x - f32(i);
    return textureLoad(lut_tex, vec2(i, row), 0).r * (1.0 - f) + textureLoad(lut_tex, vec2(j, row), 0).r * f;
}

fn gray(c: vec3<f32>) -> f32 { return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b; }

// ---- blend modes (photocraft_color::blend + compose::psblend) ------------------------------

const M_NORMAL: i32 = 1;
const M_DISSOLVE: i32 = 2;

// Backdrop extremes within EDGE of 0 / 1 count as exact (psblend::EDGE): composited values that
// should be 1 come out a rounding step below it, which would flip Color Burn's corner case.
const EDGE: f32 = 1e-4;
fn color_burn(cb: f32, cs: f32) -> f32 {
    if (cb >= 1.0 - EDGE) { return 1.0; }
    if (cs <= 0.0) { return 0.0; }
    return 1.0 - min((1.0 - cb) / cs, 1.0);
}
fn color_dodge(cb: f32, cs: f32) -> f32 {
    if (cb <= EDGE) { return 0.0; }
    if (cs >= 1.0) { return 1.0; }
    return min(cb / (1.0 - cs), 1.0);
}
fn hard_light(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) { return cb * 2.0 * cs; }
    let s = 2.0 * cs - 1.0;
    return cb + s - cb * s;
}
fn soft_light_ps(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) { return 2.0 * cb * cs + cb * cb * (1.0 - 2.0 * cs); }
    return 2.0 * cb * (1.0 - cs) + sqrt(max(cb, 0.0)) * (2.0 * cs - 1.0);
}
fn vivid_light_ps(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) {
        if (cs <= 0.0) { return 0.0; }
        return 1.0 - min((1.0 - cb) / (2.0 * cs), 1.0);
    }
    if (cs >= 1.0) { return 1.0; }
    return min(cb / (2.0 * (1.0 - cs)), 1.0);
}
fn vivid_light_generic(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) {
        if (cb >= 1.0 - EDGE) { return 1.0; }
        if (cs <= 0.0) { return 0.0; }
        return 1.0 - min((1.0 - cb) / (2.0 * cs), 1.0);
    }
    if (cb <= EDGE) { return 0.0; }
    if (cs >= 1.0) { return 1.0; }
    return min(cb / (2.0 * (1.0 - cs)), 1.0);
}

// f32::MAX exactly. The decimal 3.40282347e38 is just above it: naga rounds it down, but browsers'
// WebGPU compiler rejects it, which left the web app without a GPU compositor.
const F32_MAX: f32 = 0x1.fffffep+127f;

fn blend_channel(mode: i32, cb: f32, cs: f32) -> f32 {
    // Linear Dodge / Divide clip at 1 at integer depths, at f32::MAX in 32-bit documents.
    let hi = select(1.0, F32_MAX, (op.flags & F_HDR) != 0u);
    switch mode {
        case 3: { return min(cb, cs); }                         // Darken
        case 4: { return cb * cs; }                             // Multiply
        case 5: { return color_burn(cb, cs); }                  // ColorBurn
        case 6: { return max(cb + cs - 1.0, 0.0); }             // LinearBurn
        case 8: { return max(cb, cs); }                         // Lighten
        case 9: { return cb + cs - cb * cs; }                   // Screen
        case 10: { return color_dodge(cb, cs); }                // ColorDodge
        case 11: { return min(cb + cs, hi); }                   // LinearDodge
        case 13: { return hard_light(cs, cb); }                 // Overlay
        case 14: { return soft_light_ps(cb, cs); }              // SoftLight
        case 15: { return hard_light(cb, cs); }                 // HardLight
        case 16: { return vivid_light_ps(cb, cs); }             // VividLight (Photoshop)
        case 17: { return clamp(cb + 2.0 * cs - 1.0, 0.0, 1.0); } // LinearLight
        case 18: {                                              // PinLight
            if (cs <= 0.5) { return min(cb, 2.0 * cs); }
            return max(cb, 2.0 * cs - 1.0);
        }
        case 19: {                                              // HardMix (Photoshop)
            return select(0.0, 1.0, vivid_light_generic(cb, cs) >= 0.5 - 1e-6);
        }
        case 20: { return abs(cb - cs); }                       // Difference
        case 21: { return cb + cs - 2.0 * cb * cs; }            // Exclusion
        case 22: { return max(cb - cs, 0.0); }                  // Subtract
        case 23: {                                              // Divide
            if (cs <= 0.0) { return select(1.0, 0.0, cb <= 0.0); }
            return min(cb / cs, hi);
        }
        case 28: {                                             // Reflect
            if (cs >= 1.0) { return 1.0; }
            return min(cb * cb / (1.0 - cs), 1.0);
        }
        case 29: {                                             // Glow
            if (cb >= 1.0) { return 1.0; }
            return min(cs * cs / (1.0 - cb), 1.0);
        }
        case 30: { return 1.0 - abs(1.0 - cb - cs); }            // Negation
        case 31: {                                             // XOR (8-bit channel operation)
            let b = u32(floor(clamp(cb, 0.0, 1.0) * 255.0 + 0.5));
            let s = u32(floor(clamp(cs, 0.0, 1.0) * 255.0 + 0.5));
            return f32(b ^ s) / 255.0;
        }
        case 32: {                                             // Paint.NET Color Burn
            if (cs <= 0.0) { return 0.0; }
            return max(1.0 - (1.0 - cb) / cs, 0.0);
        }
        case 33: {                                             // Paint.NET Color Dodge
            if (cs >= 1.0) { return 1.0; }
            return min(cb / (1.0 - cs), 1.0);
        }
        default: { return cs; }                                 // Normal, Dissolve, PassThrough
    }
}

fn lum(c: vec3<f32>) -> f32 { return 0.3 * c.r + 0.59 * c.g + 0.11 * c.b; }

// photocraft_color::blend::clip_color: towards grey until within 0..top; a luminosity outside
// that range has no in-gamut colour (and max − lum is rounding noise), so it gives its grey.
fn clip_color(c: vec3<f32>, top: f32) -> vec3<f32> {
    let l = lum(c);
    let n = min(min(c.r, c.g), c.b);
    let x = max(max(c.r, c.g), c.b);
    var out = c;
    if (n < 0.0) {
        let d = l - n;
        if (l > 0.0 && d > 0.0) { out = l + (out - l) * l / d; } else { out = vec3(l); }
    }
    if (x > top) {
        let d = x - l;
        if (l < top && d > 0.0) { out = l + (out - l) * (top - l) / d; } else { out = vec3(l); }
    }
    return out;
}
fn set_lum(c: vec3<f32>, l: f32, top: f32) -> vec3<f32> { return clip_color(c + (l - lum(c)), top); }
fn sat(c: vec3<f32>) -> f32 { return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b); }
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    if (mx > mn) { return (c - mn) * s / (mx - mn); }
    return vec3(0.0);
}

fn blend_rgb(mode: i32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    // blend_rgb_within: 32-bit documents keep values above 1 (f32::MAX).
    let top = select(1.0, F32_MAX, (op.flags & F_HDR) != 0u);
    switch mode {
        case 24: { return set_lum(set_sat(cs, sat(cb)), lum(cb), top); } // Hue
        case 25: { return set_lum(set_sat(cb, sat(cs)), lum(cb), top); } // Saturation
        case 26: { return set_lum(cs, lum(cb), top); }                  // Color
        case 27: { return set_lum(cb, lum(cs), top); }                  // Luminosity
        case 7: { return select(cb, cs, lum(cs) < lum(cb)); }      // DarkerColor
        case 12: { return select(cb, cs, lum(cs) > lum(cb)); }     // LighterColor
        default: {
            return vec3(blend_channel(mode, cb.r, cs.r), blend_channel(mode, cb.g, cs.g), blend_channel(mode, cb.b, cs.b));
        }
    }
}

fn composite(mode: i32, b: vec4<f32>, s: vec4<f32>, opacity: f32) -> vec4<f32> {
    let ab = b.a;
    let as_ = s.a * opacity;
    if (as_ <= 0.0) { return b; }
    if ((op.flags & F_LAB) != 0u && mode == M_NORMAL && ab > 0.0) {
        // psblend::composite on Lab documents: Normal mixes in CIELAB.
        let ao = as_ + ab * (1.0 - as_);
        let m = srgb_to_lab(b.rgb) * (ab * (1.0 - as_) / ao) + srgb_to_lab(s.rgb) * (as_ / ao);
        return vec4(lab_to_srgb(m), ao);
    }
    let bl = blend_rgb(mode, b.rgb, s.rgb);
    let ao = as_ + ab * (1.0 - as_);
    if (ao <= 0.0) { return vec4(0.0); }
    let rgb = ((1.0 - as_) * ab * b.rgb + (1.0 - ab) * as_ * s.rgb + as_ * ab * bl) / ao;
    return vec4(rgb, ao);
}

// `psblend::text_encode` / `text_decode`: linear light raised to 1 / gamma.
fn text_enc(c: vec3<f32>, g: f32) -> vec3<f32> {
    let l = vec3(srgb_to_linear(max(c.r, 0.0)), srgb_to_linear(max(c.g, 0.0)), srgb_to_linear(max(c.b, 0.0)));
    return pow(l, vec3(1.0 / g));
}
fn text_dec(v: vec3<f32>, g: f32) -> vec3<f32> {
    let l = pow(max(v, vec3(0.0)), vec3(g));
    return vec3(linear_to_srgb(l.r), linear_to_srgb(l.g), linear_to_srgb(l.b));
}

// `psblend::composite_gamma`: coverage mixed in the text blending space (type layers, 1.45).
fn composite_g(mode: i32, b: vec4<f32>, s: vec4<f32>, opacity: f32, gamma_on: bool) -> vec4<f32> {
    if (!gamma_on) { return composite(mode, b, s, opacity); }
    let ab = b.a;
    let as_ = s.a * opacity;
    if (as_ <= 0.0) { return b; }
    let bl = blend_rgb(mode, b.rgb, s.rgb);
    let ao = as_ + ab * (1.0 - as_);
    if (ao <= 0.0) { return vec4(0.0); }
    let g = op.p4.w; // psblend::text_gamma
    let pw = (1.0 - as_) * ab * text_enc(b.rgb, g) + (1.0 - ab) * as_ * text_enc(s.rgb, g) + as_ * ab * text_enc(bl, g);
    return vec4(text_dec(pw / ao, g), ao);
}

// psblend::fill_is_special: Color Burn, Linear Burn, Color Dodge, Linear Dodge, Vivid Light,
// Linear Light, Hard Mix, Difference (mode_index 5, 6, 10, 11, 16, 17, 19, 20).
fn fill_is_special(mode: i32) -> bool {
    return mode == 5 || mode == 6 || mode == 10 || mode == 11 || mode == 16 || mode == 17 || mode == 19 || mode == 20;
}

// psblend::composite_fill, with a single blend_rgb call (fs_blend's only composite: every
// inlined blend_rgb copy grows the shader, and WARP compiles it slowly). Normal modes take Fill
// as coverage; below 100% Fill the special eight blend a source pulled toward the mode's neutral
// colour (Hard Mix: (cb - f(1 - cs)) / (1 - f)) at `opacity`, Fill staying coverage over
// transparency. Lab documents mix Normal in CIELAB; type layers mix in the text gamma space.
fn composite_fill(mode: i32, b: vec4<f32>, s: vec4<f32>, opacity: f32, fill: f32, gamma_on: bool) -> vec4<f32> {
    let f = clamp(fill, 0.0, 1.0);
    let special = f < 1.0 && fill_is_special(mode);
    let ab = b.a;
    var over = s.a * opacity;
    var alone = over * f;
    if (!special) { over = alone; }
    if (over <= 0.0) { return b; }
    let ao = ab + (1.0 - ab) * alone;
    if (ao <= 0.0) { return vec4(0.0); }
    if (!special && !gamma_on && (op.flags & F_LAB) != 0u && mode == M_NORMAL && ab > 0.0) {
        let m = srgb_to_lab(b.rgb) * (ab * (1.0 - over) / ao) + srgb_to_lab(s.rgb) * (over / ao);
        return vec4(lab_to_srgb(m), ao);
    }
    var neutral = 0.0;
    if (mode == 5 || mode == 6) { neutral = 1.0; }
    if (mode == 16 || mode == 17) { neutral = 0.5; }
    var cs = s.rgb;
    if (special) { cs = vec3(neutral) + (s.rgb - vec3(neutral)) * f; }
    var bl = blend_rgb(mode, b.rgb, cs);
    if (special && mode == 19) { bl = clamp((b.rgb - f * (vec3(1.0) - s.rgb)) / (1.0 - f), vec3(0.0), vec3(1.0)); }
    let wb = ab * (1.0 - over) / ao;
    let wbs = ab * over / ao;
    let ws = (1.0 - ab) * alone / ao;
    if (!gamma_on) { return vec4(wb * b.rgb + wbs * bl + ws * s.rgb, ao); }
    let g = op.p4.w;
    return vec4(text_dec(wb * text_enc(b.rgb, g) + wbs * text_enc(bl, g) + ws * text_enc(s.rgb, g), g), ao);
}

// photocraft_color::convert::{srgb_to_lab, lab_to_srgb} (D50, Bradford to sRGB).
const D50: vec3<f32> = vec3(0.96422, 1.0, 0.82521);
fn lab_f(t: f32) -> f32 {
    if (t > 0.008856452) { return pow(t, 1.0 / 3.0); }   // (6/29)^3
    return t / 0.12841855 + 0.13793103;                    // 3 (6/29)^2, 4/29
}
fn lab_finv(t: f32) -> f32 {
    if (t > 0.20689656) { return t * t * t; }             // 6/29
    return 0.12841855 * (t - 0.13793103);
}
fn srgb_to_lab(c: vec3<f32>) -> vec3<f32> {
    let lin = vec3(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));
    let x = (0.436074 * lin.r + 0.385064 * lin.g + 0.143080 * lin.b) / D50.x;
    let y = (0.222504 * lin.r + 0.716878 * lin.g + 0.060618 * lin.b) / D50.y;
    let z = (0.013932 * lin.r + 0.097104 * lin.g + 0.714173 * lin.b) / D50.z;
    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);
    return vec3(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}
fn lab_to_srgb(lab: vec3<f32>) -> vec3<f32> {
    let fy = (lab.x + 16.0) / 116.0;
    let fx = fy + lab.y / 500.0;
    let fz = fy - lab.z / 200.0;
    let xyz = vec3(lab_finv(fx) * D50.x, lab_finv(fy) * D50.y, lab_finv(fz) * D50.z);
    let r = 3.133856 * xyz.x - 1.616867 * xyz.y - 0.490615 * xyz.z;
    let g = -0.978768 * xyz.x + 1.916142 * xyz.y + 0.033454 * xyz.z;
    let b = 0.071945 * xyz.x - 0.228991 * xyz.y + 1.405243 * xyz.z;
    return vec3(linear_to_srgb(clamp(r, 0.0, 1.0)), linear_to_srgb(clamp(g, 0.0, 1.0)), linear_to_srgb(clamp(b, 0.0, 1.0)));
}

fn dissolve_noise(d: vec2<i32>) -> f32 {
    var h = (bitcast<u32>(d.x) * 0x8da6b343u) ^ (bitcast<u32>(d.y) * 0xd8163841u) ^ 0x9e3779b9u;
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    return f32(h & 0xffffu) / 65536.0;
}

// ---- adjustments (compose::adjust) ----------------------------------------------------------

fn srgb_to_linear(v: f32) -> f32 {
    if (v <= 0.04045) { return v / 12.92; }
    return pow((v + 0.055) / 1.055, 2.4);
}
fn linear_to_srgb(v: f32) -> f32 {
    if (v <= 0.0031308) { return v * 12.92; }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}
fn t_decode(v: f32, g: f32) -> f32 {
    if (g <= 0.0) { return srgb_to_linear(max(v, 0.0)); }
    return pow(max(v, 0.0), g);
}
fn t_encode(v: f32, g: f32) -> f32 {
    if (g <= 0.0) { return linear_to_srgb(max(v, 0.0)); }
    return pow(max(v, 0.0), 1.0 / g);
}

// compose::adjust::vibrance_lin: Photoshop's Vibrance slider (v in [-1, 1]) as an HSV change in
// linear light that keeps the hue; b0 = boost fit (c0, c1, p, q), b1 = (r0, ra, rt).
fn vibrance_lin(c: vec3<f32>, v: f32, b0: vec4<f32>, b1: vec3<f32>) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    if (mx <= 0.0 || mx - mn <= 1e-9) { return c; }
    let t = 1.0 - mn / mx;
    let shape = t * (1.0 - t) * (2.0 - t + t * t) * mx * max(1.0 - mx, 0.0);
    var t2: f32;
    var v2: f32;
    if (v < 0.0) {
        let a = -v;
        t2 = t * (1.0 - a / 4.0) * (1.0 - a + a * t * (1.0 + t) / 2.0);
        v2 = mx - a * shape;
    } else {
        let d = mx - mn;
        var h: f32;
        if (mx == c.r) { h = rem_euclid((c.g - c.b) / d, 6.0); }
        else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
        else { h = (c.r - c.g) / d + 4.0; }
        h = h * 60.0;
        var w = 0.0;
        if (h <= 30.0) { w = 1.0; }
        else if (h < 45.0) { w = (45.0 - h) / 15.0; }
        else if (h >= 300.0) { w = (h - 300.0) / 60.0; }
        let a = v * (1.0 - w) + 0.64 * pow(v, 1.6) * w;
        let boost = (b0.x * a + b0.y * a * a) * pow(t, b0.z) * pow(1.0 - t, b0.w)
            * exp((b1.x + b1.y * a + b1.z * (1.0 - t)) * max(1.0 - mx, 0.0));
        t2 = min(t * (1.0 + boost), 1.0);
        v2 = mx + v / 4.0 * shape;
    }
    let m2 = v2 * (1.0 - t2);
    let k = (v2 - m2) / (mx - mn);
    return vec3(m2) + (c - vec3(mn)) * k;
}

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    let l = (mx + mn) / 2.0;
    if (abs(mx - mn) < 1e-7) { return vec3(0.0, 0.0, l); }
    let d = mx - mn;
    var s: f32;
    if (l > 0.5) { s = d / (2.0 - mx - mn); } else { s = d / (mx + mn); }
    var h: f32;
    if (mx == c.r) {
        h = rem_euclid((c.g - c.b) / d, 6.0);
    } else if (mx == c.g) {
        h = (c.b - c.r) / d + 2.0;
    } else {
        h = (c.r - c.g) / d + 4.0;
    }
    return vec3(h / 6.0, s, l);
}
fn hue_ch(p: f32, q: f32, t0: f32) -> f32 {
    let t = rem_euclid(t0, 1.0);
    if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
    if (t < 0.5) { return q; }
    if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if (s <= 0.0) { return vec3(l); }
    var q: f32;
    if (l < 0.5) { q = l * (1.0 + s); } else { q = l + s - l * s; }
    let p = 2.0 * l - q;
    return vec3(hue_ch(p, q, h + 1.0 / 3.0), hue_ch(p, q, h), hue_ch(p, q, h - 1.0 / 3.0));
}

fn posterize(v: f32, n: f32) -> f32 {
    let x = round_half_up(clamp(v, 0.0, 1.0) * 255.0);
    let bin = min(floor(x * n / 256.0), n - 1.0);
    return floor(bin * 255.0 / (n - 1.0)) / 255.0;
}

// Document pixel being adjusted (for ordered dither).
var<private> adj_px: vec2<i32>;

fn lut_at(i: i32) -> f32 {
    return textureLoad(lut_tex, vec2(i % 4096, i / 4096), 0).r;
}

fn lut3(n: i32, r: i32, g: i32, b: i32) -> vec3<f32> {
    let i = ((b * n + g) * n + r) * 3;
    return vec3(lut_at(i), lut_at(i + 1), lut_at(i + 2));
}

fn bayer4(p: vec2<i32>) -> f32 {
    var m = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    return (m[(p.y & 3) * 4 + (p.x & 3)] + 0.5) / 16.0 - 0.5;
}

// compose::adjust::lut3d_sample
fn color_lookup(c: vec3<f32>, n: i32, tetra: bool) -> vec3<f32> {
    let m = f32(n - 1);
    let pos = clamp(c, vec3(0.0), vec3(1.0)) * m;
    let i0 = min(vec3<i32>(floor(pos)), vec3(n - 2));
    let f = pos - vec3<f32>(i0);
    let r = i0.x;
    let g = i0.y;
    let b = i0.z;
    if (tetra) {
        let c000 = lut3(n, r, g, b);
        let c111 = lut3(n, r + 1, g + 1, b + 1);
        if (f.x > f.y) {
            if (f.y > f.z) {
                return (1.0 - f.x) * c000 + (f.x - f.y) * lut3(n, r + 1, g, b) + (f.y - f.z) * lut3(n, r + 1, g + 1, b) + f.z * c111;
            } else if (f.x > f.z) {
                return (1.0 - f.x) * c000 + (f.x - f.z) * lut3(n, r + 1, g, b) + (f.z - f.y) * lut3(n, r + 1, g, b + 1) + f.y * c111;
            }
            return (1.0 - f.z) * c000 + (f.z - f.x) * lut3(n, r, g, b + 1) + (f.x - f.y) * lut3(n, r + 1, g, b + 1) + f.y * c111;
        }
        if (f.z > f.y) {
            return (1.0 - f.z) * c000 + (f.z - f.y) * lut3(n, r, g, b + 1) + (f.y - f.x) * lut3(n, r, g + 1, b + 1) + f.x * c111;
        } else if (f.z > f.x) {
            return (1.0 - f.y) * c000 + (f.y - f.z) * lut3(n, r, g + 1, b) + (f.z - f.x) * lut3(n, r, g + 1, b + 1) + f.x * c111;
        }
        return (1.0 - f.y) * c000 + (f.y - f.x) * lut3(n, r, g + 1, b) + (f.x - f.z) * lut3(n, r + 1, g + 1, b) + f.z * c111;
    }
    let c00 = mix(lut3(n, r, g, b), lut3(n, r + 1, g, b), f.x);
    let c10 = mix(lut3(n, r, g + 1, b), lut3(n, r + 1, g + 1, b), f.x);
    let c01 = mix(lut3(n, r, g, b + 1), lut3(n, r + 1, g, b + 1), f.x);
    let c11 = mix(lut3(n, r, g + 1, b + 1), lut3(n, r + 1, g + 1, b + 1), f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

// One Selective Color range: its CMYK percentages (LUT row 0, 4 per range) weighted by `w`.
fn selective_row(r: i32, w: f32, c: vec3<f32>, relative: bool) -> vec3<f32> {
    let k = lut_at(r * 4 + 3) / 100.0;
    var d = vec3(lut_at(r * 4), lut_at(r * 4 + 1), lut_at(r * 4 + 2)) / 100.0 + k;
    if (relative) {
        d = d * (1.0 - c);
    } else {
        // Absolute: scaled by the room the ink has (adding: 2 × (1 − ink), removing: 2 × ink, capped at 1).
        d = d * select(min(2.0 * (1.0 - c), vec3(1.0)), min(2.0 * c, vec3(1.0)), d > vec3(0.0));
    }
    return select(vec3(0.0), d * w, w > 0.0);
}

// compose::adjust::selective_color (ranges × CMYK percentages in LUT row 0). Written without
// loops: FXC (D3D12) aborts compiling loops with dynamic indexing inside `adjust`'s switch.
fn selective_color(c: vec3<f32>, relative: bool) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    let md = c.r + c.g + c.b - mx - mn;
    var delta = selective_row(0, select(0.0, mx - md, c.r >= mx), c, relative);
    delta += selective_row(1, select(0.0, md - mn, c.b <= mn), c, relative);
    delta += selective_row(2, select(0.0, mx - md, c.g >= mx), c, relative);
    delta += selective_row(3, select(0.0, md - mn, c.r <= mn), c, relative);
    delta += selective_row(4, select(0.0, mx - md, c.b >= mx), c, relative);
    delta += selective_row(5, select(0.0, md - mn, c.g <= mn), c, relative);
    delta += selective_row(6, max((mn - 0.5) * 2.0, 0.0), c, relative);
    delta += selective_row(7, clamp(1.0 - abs(mx - 0.5) - abs(mn - 0.5), 0.0, 1.0), c, relative);
    delta += selective_row(8, max((0.5 - mx) * 2.0, 0.0), c, relative);
    return clamp(c - delta, vec3(0.0), vec3(1.0));
}

// Exposure on one channel (p = exposure scale, offset, gamma, transfer gamma). Per channel, not a
// loop over `c[i]`: FXC aborts on that inside `adjust`'s switch.
fn exposure(v: f32, p: vec4<f32>) -> f32 {
    if ((op.flags & F_HDR) != 0u) {
        let lin = v * p.x + p.y;
        if (p.z == 1.0) { return lin; }
        return sign(lin) * pow(abs(lin), 1.0 / p.z);
    }
    let lin = pow(max(t_decode(v, p.w) * p.x + p.y, 0.0), 1.0 / p.z);
    return clamp(t_encode(lin, p.w), 0.0, 1.0);
}

// Mirrors compose::adjust::levels_float; coefficients are master, R, G, B in LUT row 0.
fn levels_float(record: i32, v: f32) -> f32 {
    let base = record * 5;
    let t = (v - lut_at(base)) / max(lut_at(base + 1) - lut_at(base), 0.000001);
    let mapped = sign(t) * pow(abs(t), 1.0 / lut_at(base + 2));
    return lut_at(base + 3) + mapped * (lut_at(base + 4) - lut_at(base + 3));
}

fn tone_channel(row: i32, v: f32, identity: f32) -> f32 {
    if (identity > 0.5) { return v; }
    return lut(row, v);
}

// Modern Brightness curve: line of slope 1.375^(b/50) rolled off to (1,1) by a v^P white anchor.
// Mirrors compose::adjust::modern_brightness.
fn mbright(v: f32, b: f32) -> f32 {
    if (b == 0.0) { return v; }
    let s = pow(1.375, b / 50.0);
    var p: f32;
    if (b >= 0.0) { p = max(4.5 - 0.013 * b, 2.0); } else { p = 5.0 - 0.072 * b; }
    let vv = clamp(v, 0.0, 1.0);
    return clamp(s * vv + (1.0 - s) * pow(vv, p), 0.0, 1.0);
}

// Modern Contrast curve: symmetric cubic-Hermite S pivoting at 0.5.
// Mirrors compose::adjust::modern_contrast.
fn mcontrast(v: f32, ct: f32) -> f32 {
    if (ct == 0.0) { return v; }
    let k = ct / 128.0;
    let e = 1.0 - k; let m = 1.0 + k;
    let vv = clamp(v, 0.0, 1.0);
    var t: f32;
    if (vv <= 0.5) { t = vv / 0.5; } else { t = (1.0 - vv) / 0.5; }
    let h = e * 0.5 * (t * t * t - 2.0 * t * t + t) + (-2.0 * t * t * t + 3.0 * t * t) * 0.5 + m * 0.5 * (t * t * t - t * t);
    if (vv <= 0.5) { return h; }
    return 1.0 - h;
}

fn adjust(c: vec3<f32>) -> vec3<f32> {
    let p0 = op.p0;
    let p1 = op.p1;
    let p2 = op.p2;
    let p3 = op.p3;
    switch op.kind {
        case 1: { return 1.0 - c; }                                            // Invert
        case 2: {                                                              // Threshold (p0.x = 8-bit level)
            return vec3(select(0.0, 1.0, round_half_up(gray(c) * 255.0) >= p0.x));
        }
        case 3: { return vec3(posterize(c.r, p0.x), posterize(c.g, p0.x), posterize(c.b, p0.x)); }
        case 4: { return clamp((c + p0.x - p0.z) * p0.y + p0.z + p0.w, vec3(0.0), vec3(1.0)); }   // B/C legacy: brightness before (p0.x) or after (p0.w) contrast around compose::adjust::LEGACY_PIVOT (p0.z)
        case 5: {                                                              // B/C modern
            let b = p0.x; let ct = p0.y;
            return vec3(mcontrast(mbright(c.r, b), ct), mcontrast(mbright(c.g, b), ct), mcontrast(mbright(c.b, b), ct));
        }
        case 6: { return vec3(exposure(c.r, p0), exposure(c.g, p0), exposure(c.b, p0)); }  // Exposure
        case 7: { return vec3(tone_channel(0, c.r, p0.x), tone_channel(1, c.g, p0.y), tone_channel(2, c.b, p0.z)); } // Levels / Curves
        case 17: { return vec3(levels_float(0, levels_float(1, c.r)), levels_float(0, levels_float(2, c.g)), levels_float(0, levels_float(3, c.b))); }
        case 8: {                                                              // Hue/Saturation
            let hsl = rgb_to_hsl(c);
            var hh: f32;
            var ss: f32;
            // Range edits (LUT rows indexed by the original hue), in full however dull the colour.
            var dh = 0.0;
            var ds = 0.0;
            var dl = 0.0;
            if (p1.x > 0.5) {
                dh = lut(0, hsl.x);
                ds = lut(1, hsl.x);
                dl = lut(2, hsl.x);
            }
            let sat = clamp(p0.y + ds, -1.0, 1.0);
            let light = clamp(p0.z, -1.0, 1.0);
            if (p0.w > 0.5) {
                // compose::adjust::hue_saturation: Colorize at the lightened / darkened lightness.
                var ll = hsl.z * (1.0 + light);
                if (light > 0.0) { ll = hsl.z + (1.0 - hsl.z) * light; }
                return hsl_to_rgb(rem_euclid(p0.x, 360.0) / 360.0, min(abs(sat), 1.0), ll);
            }
            // Lightness first, then hue and saturation of the result.
            var lc = c * (1.0 + light);
            if (light > 0.0) { lc = c + (1.0 - c) * light; }
            let lh = rgb_to_hsl(lc);
            hh = rem_euclid(lh.x + (p0.x + dh) / 360.0, 1.0);
            // Raising saturation divides by 1 - s; lowering it multiplies by 1 + s.
            if (sat > 0.0) {
                ss = clamp(lh.y / max(1.0 - sat, 1.0 / 512.0), 0.0, 1.0);
            } else {
                ss = clamp(lh.y * (1.0 + sat), 0.0, 1.0);
            }
            var rgb = hsl_to_rgb(hh, ss, lh.z);
            // compose::adjust::range_lightness: towards the brightest channel when raised, the darkest when lowered.
            let rl = clamp(dl, -1.0, 1.0);
            if (rl > 0.0) {
                rgb = rgb + (vec3(max(rgb.r, max(rgb.g, rgb.b))) - rgb) * rl;
            } else if (rl < 0.0) {
                rgb = rgb + (vec3(min(rgb.r, min(rgb.g, rgb.b))) - rgb) * -rl;
            }
            return rgb;
        }
        case 9: {                                                              // Vibrance
            // compose::adjust::vibrance_px: linear light (transfer p0.z), Vibrance (p0.x, boost
            // fit p1, p2.xyz) then Saturation (p0.y) towards 0.288 R + 0.712 G; p0.w the ceiling.
            var l = vec3(t_decode(c.r, p0.z), t_decode(c.g, p0.z), t_decode(c.b, p0.z));
            if (p0.x != 0.0) { l = vibrance_lin(l, p0.x, p1, p2.xyz); }
            if (p0.y != 0.0) {
                let g = 0.288 * l.r + 0.712 * l.g;
                l = clamp(vec3(g) + (1.0 + p0.y) * (l - vec3(g)), vec3(0.0), vec3(p0.w));
            }
            l = clamp(l, vec3(0.0), vec3(p0.w));
            return vec3(t_encode(l.r, p0.z), t_encode(l.g, p0.z), t_encode(l.b, p0.z));
        }
        case 10: {                                                             // Channel mixer
            let r = clamp(dot(p0.xyz, c) + p0.w, 0.0, 1.0);
            if (p3.x > 0.5) { return vec3(r); }
            return vec3(r, clamp(dot(p1.xyz, c) + p1.w, 0.0, 1.0), clamp(dot(p2.xyz, c) + p2.w, 0.0, 1.0));
        }
        case 11: {                                                             // Photo filter
            // compose::adjust::photo_filter_matrix (rows p0..p2) in linear light, then SetLum.
            let g = p3.y;
            let lin = vec3(t_decode(c.r, g), t_decode(c.g, g), t_decode(c.b, g));
            var f = vec3(t_encode(dot(p0.xyz, lin), g), t_encode(dot(p1.xyz, lin), g), t_encode(dot(p2.xyz, lin), g));
            if (p3.x > 0.5) { f = set_lum(f, lum(c), 1.0); }
            return clamp(f, vec3(0.0), vec3(1.0));
        }
        case 12: {                                                             // Black & White
            // compose::adjust::black_white_gray: grey + secondary + primary parts, weighted.
            var w = array<f32, 6>(p0.x, p0.y, p0.z, p0.w, p1.x, p1.y);
            let mx = max(c.r, max(c.g, c.b));
            let mn = min(c.r, min(c.g, c.b));
            let mid = c.r + c.g + c.b - mx - mn;
            var primary = 4;
            if (c.r >= c.g && c.r >= c.b) { primary = 0; } else if (c.g >= c.b) { primary = 2; }
            var secondary = 5;
            if (c.b <= c.r && c.b <= c.g) { secondary = 1; } else if (c.r <= c.g) { secondary = 3; }
            let g = clamp(mn + (mid - mn) * w[secondary] / 100.0 + (mx - mid) * w[primary] / 100.0, 0.0, 1.0);
            if (p1.z > 0.5) {
                return clamp(set_lum(p2.rgb, g, 1.0), vec3(0.0), vec3(1.0));   // Color blend of the tint
            }
            return vec3(g);
        }
        case 13: {                                                             // Gradient map
            var t = gray(c);
            if (p0.x > 0.5) { t = 1.0 - t; }
            var o = vec3(lut(0, t), lut(1, t), lut(2, t));
            if (p0.y > 0.5) { o = clamp(o + bayer4(adj_px) / 255.0, vec3(0.0), vec3(1.0)); }
            return o;
        }
        case 15: { return selective_color(c, p0.x > 0.5); }                       // Selective color
        case 16: {                                                             // Color lookup (3D LUT)
            var o = color_lookup(c, i32(p0.x), p0.y > 0.5);
            if (p0.z > 0.5) { o = clamp(o + bayer4(adj_px) / 255.0, vec3(0.0), vec3(1.0)); }
            return o;
        }
        default: { return c; }
    }
}

// ---- fills ----------------------------------------------------------------------------------

fn gradient_t(d: vec2<i32>) -> f32 {
    // p0 = (angle°, scale, reverse, style), p1 = frame (x0, y0, w, h)
    let w = max(op.p1.z, 1.0);
    let h = max(op.p1.w, 1.0);
    // p2.xy: centre offset as a fraction of the bounds (gradient overlays; 0 for fills).
    let cx = op.p1.x + w / 2.0 + op.p2.x * w;
    let cy = op.p1.y + h / 2.0 + op.p2.y * h;
    let a = radians(op.p0.x);
    let s = sin(a);
    let c = cos(a);
    let dx = f32(d.x) + 0.5 - cx;
    let dy = f32(d.y) + 0.5 - cy;
    let along = dx * c - dy * s;
    let across = dx * s + dy * c;
    let len = max(sqrt((c * w) * (c * w) + (s * h) * (s * h)), 1.0) * max(op.p0.y, 1e-3);
    // effects::gradient_t: Linear / Reflected span the bounds' chord along the angle.
    let chord = max(min(w / max(abs(c), 1e-6), h / max(abs(s), 1e-6)), 1.0) * max(op.p0.y, 1e-3);
    // Linear / Reflected sample the pixel's top-left corner (effects::gradient_t).
    let corner = 0.5 * (c - s);
    var t: f32;
    switch i32(op.p0.w) {
        case 1: { t = sqrt(dx * dx + dy * dy) / (len / 2.0); }                  // Radial
        case 2: { t = rem_euclid((a - atan2(-dy, dx)) / 6.28318530718, 1.0); }   // Angle
        case 3: { t = abs((along - corner) / (chord / 2.0)); }                  // Reflected
        case 4: { t = (abs(along) + abs(across)) / (len / 2.0); }               // Diamond
        default: { t = (along - corner) / chord + 0.5; }                        // Linear
    }
    t = clamp(t, 0.0, 1.0);
    if (op.p0.z > 0.5) { t = 1.0 - t; }
    return t;
}

// photocraft_color::dither_noise: the gradient dither's position hash, in [0, 1).
fn dither_noise(d: vec2<i32>) -> f32 {
    var h = (bitcast<u32>(d.x) * 0x9E3779B1u) ^ (bitcast<u32>(d.y) * 0x85EBCA77u);
    h = h ^ (h >> 15u);
    h = h * 0x2C1B3C6Du;
    h = h ^ (h >> 12u);
    return f32(h & 0xFFFFu) / 65535.0;
}

fn layer_texel(d: vec2<i32>) -> vec4<f32> {
    if ((op.flags & F_GRADIENT) != 0u) {
        let t = gradient_t(d);
        var c = vec4(lut(0, t), lut(1, t), lut(2, t), lut(3, t));
        // p2.w: dither (gradient fills), rounded to p2.z levels on integer documents.
        if (op.p2.w > 0.5) {
            var rgb = clamp(c.rgb + (dither_noise(d) - 0.5) / 255.0, vec3(0.0), vec3(1.0));
            if (op.p2.z > 0.0) { rgb = floor(rgb * op.p2.z + 0.5) / op.p2.z; }
            c = vec4(rgb, c.a);
        }
        return c;
    }
    if ((op.flags & F_TEX) != 0u) {
        let q = d - op.tex_origin;
        if (inside(q, op.tex_size)) {
            return textureLoad(layer_tex, q, 0);
        }
    }
    return op.color;
}

// ---- passes ---------------------------------------------------------------------------------

// A layer's own pixels (raster / fill), times its mask.
@fragment
fn fs_content(in: VOut) -> @location(0) vec4<f32> {
    let d = doc_px(local(in.pos));
    var c = layer_texel(d);
    c.a = c.a * mask_value(d);
    return c;
}

// Multiply alpha of A by the mask (isolated group content).
@fragment
fn fs_mask(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    var c = textureLoad(tex_a, p, 0);
    c.a = c.a * mask_value(doc_px(p));
    return c;
}

// blend_into(backdrop = A, src = B, mode, opacity), including Dissolve.
@fragment
fn fs_blend(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let b = textureLoad(tex_a, p, 0);
    var s = textureLoad(tex_b, p, 0);
    if (s.a <= 0.0) { return b; }
    if (op.mode == M_DISSOLVE) {
        s.a = select(0.0, 1.0, dissolve_noise(doc_px(p)) < s.a * op.opacity * op.fill);
        return composite(M_NORMAL, b, s, 1.0);
    }
    return composite_fill(op.mode, b, s, op.opacity, op.fill, (op.flags & F_TEXT_GAMMA) != 0u);
}

// composite_atop(base = A, src = B): blend as if the base were opaque, keep its alpha.
@fragment
fn fs_atop(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let base = textureLoad(tex_a, p, 0);
    if (base.a <= 0.0) { return base; }
    let s = textureLoad(tex_b, p, 0);
    let r = composite_g(op.mode, vec4(base.rgb, 1.0), s, op.opacity, (op.flags & F_TEXT_GAMMA) != 0u);
    return vec4(r.rgb, base.a);
}

// Adjustment applied to A (transparent pixels untouched).
@fragment
fn fs_adjust(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let c = textureLoad(tex_a, p, 0);
    if (c.a <= 0.0) { return c; }
    adj_px = doc_px(p);
    return vec4(adjust(c.rgb), c.a);
}

// Mix an adjusted result B back over the original A with blend mode, opacity and mask.
@fragment
fn fs_adjmix(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let b = textureLoad(tex_a, p, 0);
    let k = op.opacity * mask_value(doc_px(p));
    if (k <= 0.0) { return b; }
    let a = textureLoad(tex_b, p, 0);
    let bl = blend_rgb(op.mode, b.rgb, a.rgb);
    return vec4(b.rgb + (bl - b.rgb) * k, b.a);
}

// Pass-through group coverage mixes premultiplied colour, then returns straight alpha.
@fragment
fn fs_lerp(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let a = textureLoad(tex_a, p, 0);
    let b = textureLoad(tex_b, p, 0);
    if ((op.flags & F_QUANT) != 0u) {
        return floor(a * op.p0.x + 0.5) / op.p0.x;
    }
    if ((op.flags & F_CHANNELS) != 0u) {
        return a + (b - a) * op.p0;
    }
    if ((op.flags & F_ADD_DIFF) != 0u) {
        // compose: pass-through group + (with clipped − without), premultiplied.
        let c = textureLoad(tex_c, p, 0);
        var pm = vec4(a.rgb * a.a, a.a) + vec4(b.rgb * b.a, b.a) - vec4(c.rgb * c.a, c.a);
        let al = clamp(pm.a, 0.0, 1.0);
        if (al <= 0.0) { return vec4(0.0); }
        return vec4(clamp(pm.rgb / al, vec3(0.0), vec3(1.0)), al);
    }
    if ((op.flags & F_BLEND_IF) != 0u) {
        // compose::apply_blend_if: "This Layer" is C (the layer's own content with its masks),
        // or B itself for an adjustment layer; "Underlying Layer" is A.
        var this_px = b;
        if (op.p4.y < 0.5) {
            this_px = textureLoad(tex_c, p, 0);
        }
        let k = blend_if_weight(this_px, a);
        if (k >= 1.0) {
            return b;
        }
        let alpha = a.a + (b.a - a.a) * k;
        if (alpha <= 0.0) { return vec4(0.0); }
        return vec4(clamp((a.rgb * a.a + (b.rgb * b.a - a.rgb * a.a) * k) / alpha, vec3(0.0), vec3(1.0)), alpha);
    }
    let k = op.opacity * mask_value(doc_px(p));
    let wa = a.a * (1.0 - k);
    let wb = b.a * k;
    let alpha = wa + wb;
    if (alpha <= 0.0) { return vec4(0.0); }
    return vec4((a.rgb * wa + b.rgb * wb) / alpha, alpha);
}

// ---- Blend If (photocraft_doc::BlendRange, compose::blend_if_weight) ---------------------------

// 0 at or below `lo`, 1 at or above `hi`; a hard step at `lo` when unsplit.
fn bi_ramp(lo: f32, hi: f32, x: f32) -> f32 {
    if (hi <= lo) {
        return select(0.0, 1.0, x >= lo);
    }
    return clamp((x - lo) / (hi - lo), 0.0, 1.0);
}

// How much of value `v` (0..255) shows through range `e` (0 = Gray, then the channels) on side
// `s` (0 = this layer, 1 = underlying). The ranges are packed by `plan::blend_if_params`.
fn bi_weight(e: u32, s: u32, v: f32) -> f32 {
    var f = array<vec4<f32>, 4>(op.p0, op.p1, op.p2, op.p3);
    let i = (e * 2u + s) * 2u;
    let black = f[i / 4u][i % 4u];
    let white = f[(i + 1u) / 4u][(i + 1u) % 4u];
    let b_lo = floor(black / 256.0);
    let b_hi = black - b_lo * 256.0;
    let w_lo = floor(white / 256.0);
    let w_hi = white - w_lo * 256.0;
    return bi_ramp(b_lo, b_hi, v) * bi_ramp(255.0 - w_hi, 255.0 - w_lo, 255.0 - v);
}

// The product of every range over the layer's own colour and the colour beneath it (a side
// without content, alpha 0, doesn't count). `p4.x` marks a one-channel document.
fn blend_if_weight(this_px: vec4<f32>, under: vec4<f32>) -> f32 {
    var k = 1.0;
    for (var s = 0u; s < 2u; s = s + 1u) {
        let px = select(this_px, under, s == 1u);
        if (px.a <= 0.0) {
            continue;
        }
        // The CPU gets 8- and 16-bit levels exactly (k / 255 × 255 = k), but GPU arithmetic can
        // land a few millionths below, which flips a hard threshold at that level: values that
        // close to a level are that level. Real 8/16-bit values off a level are ≥ 1/257 away.
        let raw = clamp(px.rgb, vec3(0.0), vec3(1.0)) * 255.0;
        let v = select(raw, round(raw), abs(raw - round(raw)) < vec3(1e-4));
        if (op.p4.x > 0.5) {
            k = k * bi_weight(0u, s, v.r) * bi_weight(1u, s, v.r);
        } else {
            let gray = 0.299 * v.r + 0.587 * v.g + 0.114 * v.b;
            k = k * bi_weight(0u, s, gray) * bi_weight(1u, s, v.r) * bi_weight(2u, s, v.g) * bi_weight(3u, s, v.b);
        }
        if (k <= 0.0) {
            return 0.0;
        }
    }
    return k;
}

// ---- layer effects (compose::effects::composite_with_effects) -------------------------------

const INSIDE_EPS: f32 = 0.5 / 255.0;

// Effect map value at document pixel `d` (`outside` beyond the map's region, like
// `FxMaps::crop`).
fn map_value(d: vec2<i32>, outside: f32) -> f32 {
    let q = d - op.map_origin;
    if (inside(q, op.map_size)) {
        return textureLoad(map_tex, q, 0).r;
    }
    return outside;
}

fn wrap(i: i32, n: i32) -> i32 {
    return ((i % n) + n) % n;
}

// compose::pattern::Tile::sample through Placement::map. p3 = (origin, cos, sin),
// p4 = (1 / scale, tile w, tile h); `pat_tex` holds premultiplied RGBA.
fn pattern_sample(d: vec2<i32>) -> vec4<f32> {
    let dx = f32(d.x) + 0.5 - op.p3.x;
    let dy = f32(d.y) + 0.5 - op.p3.y;
    let c = op.p3.z;
    let s = op.p3.w;
    let u = (dx * c - dy * s) * op.p4.x - 0.5;
    let v = (dx * s + dy * c) * op.p4.x - 0.5;
    let x0 = floor(u);
    let y0 = floor(v);
    let fx = u - x0;
    let fy = v - y0;
    let tw = i32(op.p4.y);
    let th = i32(op.p4.z);
    let ix = i32(x0);
    let iy = i32(y0);
    var acc = vec4(0.0);
    let wy = array<f32, 2>(1.0 - fy, fy);
    let wx = array<f32, 2>(1.0 - fx, fx);
    for (var j = 0; j < 2; j++) {
        if (wy[j] == 0.0) { continue; }
        for (var i = 0; i < 2; i++) {
            if (wx[i] == 0.0) { continue; }
            acc += textureLoad(pat_tex, vec2(wrap(ix + i, tw), wrap(iy + j, th)), 0) * (wx[i] * wy[j]);
        }
    }
    if (acc.a <= 0.0) { return vec4(0.0); }
    return vec4(acc.rgb / acc.a, min(acc.a, 1.0));
}

// Effect paint colour at `d`: p2.z = 0 solid (`color`), 1 gradient (stops in `lut_tex`),
// 2 pattern (4, a glow's gradient, is resolved in `fs_fxpaint`).
fn fx_color(d: vec2<i32>) -> vec4<f32> {
    let kind = i32(op.p2.z);
    if (kind == 1) {
        let t = gradient_t(d);
        return vec4(lut(0, t), lut(1, t), lut(2, t), lut(3, t));
    }
    if (kind == 2) {
        return pattern_sample(d);
    }
    return op.color;
}

// Effect chain steps (`kind`): 0 A at `opacity` × alpha; 1 an opaque copy of A (a clipping base
// for clipped layers' effects); 2 A's colour with alpha `opacity` inside B's shape (the layer
// before its interior effects); 3 A with its alpha × B's alpha (the shape's own alpha); 4 the
// vector stroke C over A, relative to the shape B, at `opacity`; 5 A's colour with alpha
// `opacity` × A's alpha relative to B's (the layer within a filled shape's outline); 6 the
// outside-stroke coverage seed (0, A's alpha); 7 A with alpha joined with the effect shape (map).
@fragment
fn fs_fxinit(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let c = textureLoad(tex_a, p, 0);
    switch op.kind {
        case 1: { return vec4(c.rgb, 1.0); }
        case 2: { return vec4(c.rgb, select(0.0, op.opacity, c.a > INSIDE_EPS)); }
        case 3: { return vec4(c.rgb, c.a * min(textureLoad(tex_b, p, 0).a, 1.0)); }
        case 4: {
            let a = textureLoad(tex_b, p, 0).a;
            let s = textureLoad(tex_c, p, 0);
            if (s.a <= 0.0 || a <= INSIDE_EPS) { return c; }
            return composite(op.mode, c, vec4(s.rgb, op.opacity * min(s.a / a, 1.0)), 1.0);
        }
        case 5: {
            let a = textureLoad(tex_b, p, 0).a;
            if (a <= INSIDE_EPS) { return vec4(c.rgb, 0.0); }
            return vec4(c.rgb, op.opacity * min(c.a / a, 1.0));
        }
        case 6: { return vec4(0.0, c.a, 0.0, 0.0); }
        case 7: { return vec4(c.rgb, max(c.a, map_value(doc_px(p), 0.0))); }
        default: { return vec4(c.rgb, c.a * op.opacity); }
    }
}

// effects::outline_share: an outside stroke's share of an edge pixel the outline covers `cov` of,
// beneath a layer of alpha `l` (disjoint areas).
fn outline_share(cov: f32, l: f32) -> f32 {
    let outside = max(1.0 - clamp(cov, 0.0, 1.0), 0.0);
    let rest = 1.0 - clamp(l, 0.0, 1.0);
    if (rest <= 1e-6) { return 1.0; }
    return min(outside / rest, 1.0);
}

// effects::paint: composite `colour × coverage × opacity` into A. B is the layer (its alpha `a`
// is the effect shape). Coverage `kind`: 0 the effect map, 1 full; then F_KNOCKOUT m × (1 − a k)
// (k = `p4.w`),
// F_GATE inside ? m : 0, F_REL inside ? min(m / a, 1) : 0, F_STROKE_OUT inside ? (vector ? 0 : 1)
// : m.
@fragment
fn fs_fxpaint(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let d = doc_px(p);
    let dst = textureLoad(tex_a, p, 0);
    let a = textureLoad(tex_b, p, 0).a;
    let inside_shape = a > INSIDE_EPS;
    var m = 1.0;
    if (op.kind == 0) { m = map_value(d, op.p2.w); }
    if ((op.flags & F_KNOCKOUT) != 0u) { m = m * (1.0 - a * op.p4.w); }
    if ((op.flags & F_GATE) != 0u && !inside_shape) { m = 0.0; }
    if ((op.flags & F_REL) != 0u) {
        if (inside_shape) { m = min(m / a, 1.0); } else { m = 0.0; }
    }
    if ((op.flags & F_STROKE_OUT) != 0u && inside_shape) {
        m = select(1.0, 0.0, (op.flags & F_VECTOR) != 0u);
    }
    if (i32(op.p2.z) == 4) {
        // effects::paint_glow: the gradient at 1 - strength, opaque from strength 1 / gain (p0.x).
        let k = min(m * op.p0.x, 1.0) * op.opacity;
        if (k <= 0.0) { return dst; }
        let t = 1.0 - clamp(m, 0.0, 1.0);
        return composite(op.mode, dst, vec4(lut(0, t), lut(1, t), lut(2, t), lut(3, t) * k), 1.0);
    }
    let k = m * op.opacity;
    if (k <= 0.0) { return dst; }
    let c = fx_color(d);
    return composite(op.mode, dst, vec4(c.rgb, c.a * k), 1.0);
}

// Outside strokes (A = the exterior result before them, B = the layer, C = coverage so far in
// `.r` and, along a filled shape's outline (F_OUTLINE), the layer's alpha in `.g`, D
// (`layer_tex`) = accumulated premultiplied colour). Each stroke takes its band × opacity not yet
// covered above it; `kind` 0 accumulates that share of the stroke blended over A at full
// coverage, `kind` 1 the coverage.
@fragment
fn fs_fxstroke(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let d = doc_px(p);
    let a = textureLoad(tex_b, p, 0).a;
    let outline = (op.flags & F_OUTLINE) != 0u;
    let first = (op.flags & F_FIRST) != 0u;
    var cover = 0.0;
    var lay_a = 0.0;
    let vector = (op.flags & F_VECTOR) != 0u;
    if (!first || outline || !vector) {
        let cv = textureLoad(tex_c, p, 0);
        cover = cv.r;
        lay_a = cv.g;
    }
    var k = map_value(d, 0.0);
    if (outline) {
        k = k * outline_share(a, lay_a);
    } else if (a > INSIDE_EPS) {
        // Pixel layers: the stroke shows beneath only through the part the shape leaves
        // (`effects.rs`): (1 - a) / (1 - layer alpha), 1 under an opaque layer.
        if (vector) {
            k = 0.0;
        } else if (lay_a >= 1.0) {
            k = 1.0;
        } else {
            k = clamp((1.0 - clamp(a, 0.0, 1.0)) / (1.0 - lay_a), 0.0, 1.0);
        }
    }
    let share = k * op.opacity * (1.0 - cover);
    if (op.kind == 1) { return vec4(cover + share, lay_a, 0.0, 0.0); }
    var acc = vec4(0.0);
    if (!first) { acc = textureLoad(layer_tex, p, 0); }
    if (share > 0.0) {
        let base = textureLoad(tex_a, p, 0);
        var bl = base;
        if (i32(op.p2.z) != 3) {
            let c = fx_color(d);
            bl = composite(op.mode, base, vec4(c.rgb, c.a), 1.0);
        }
        acc += share * vec4(bl.rgb * bl.a, bl.a);
    }
    return acc;
}

// Resolve outside strokes: A = exterior result, B = accumulated colour, C = coverage.
@fragment
fn fs_fxstrokeend(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let w = textureLoad(tex_a, p, 0);
    let cover = textureLoad(tex_c, p, 0).r;
    if (cover <= 0.0) { return w; }
    let acc = textureLoad(tex_b, p, 0);
    let k = 1.0 - min(cover, 1.0);
    let alpha = w.a * k + acc.a;
    if (alpha <= 0.0) { return w; }
    return vec4((w.rgb * w.a * k + acc.rgb) / alpha, min(alpha, 1.0));
}

fn mix_premul(a: vec4<f32>, b: vec4<f32>, k: f32) -> vec4<f32> {
    let alpha = a.a + (b.a - a.a) * k;
    if (alpha <= 0.0) { return vec4(0.0); }
    return vec4((a.rgb * a.a + (b.rgb * b.a - a.rgb * a.a) * k) / alpha, alpha);
}

// The layer with its interior effects (B) over the exterior result (A) in the layer's mode, then
// layer opacity against the backdrop C. F_ATOP: C is a clipping base: effects were painted over
// it as if opaque and the base keeps its alpha.
@fragment
fn fs_fxmerge(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    var w = textureLoad(tex_a, p, 0);
    let l = textureLoad(tex_b, p, 0);
    let c = textureLoad(tex_c, p, 0);
    if (l.a > 0.0 && (op.flags & F_NO_LAYER) == 0u) { w = composite_g(op.mode, w, l, 1.0, (op.flags & F_TEXT_GAMMA) != 0u); }
    let atop = (op.flags & F_ATOP) != 0u;
    var before = c;
    if (atop) { before = vec4(c.rgb, 1.0); }
    let o = mix_premul(before, w, op.opacity);
    if (atop) {
        if (c.a > 0.0) { return vec4(o.rgb, c.a); }
        return c;
    }
    return o;
}

// ---- effect maps (compose::effects::build_maps) ----------------------------------------------
//
// Targets cover the effect region (`chunk` = region origin and size); reads outside it return 0,
// like `Map::get`. Each kernel mirrors one step of the CPU map builders. The shape and its
// distance fields come from the CPU (`compose::layer_shape`, `compose::effects::distance_field`).

fn ra(p: vec2<i32>) -> f32 {
    if (inside(p, chunk.size)) { return textureLoad(tex_a, p, 0).r; }
    return 0.0;
}
fn mout(v: f32) -> vec4<f32> { return vec4(v, 0.0, 0.0, 1.0); }

// Map::shifted by whole pixels: p0 = (dx, dy, outside, invert (1 - a before shifting)).
@fragment
fn fs_mshift(in: VOut) -> @location(0) vec4<f32> {
    let s = local(in.pos) - vec2<i32>(i32(op.p0.x), i32(op.p0.y));
    if (!inside(s, chunk.size)) { return mout(op.p0.z); }
    var v = textureLoad(tex_a, s, 0).r;
    if (op.p0.w > 0.5) { v = 1.0 - v; }
    return mout(v);
}

// effects::dilate: inside (distance < 0) min(A + r, 1), else max(A, clamp(r + 0.5 - dist_outside));
// B = distances, p0.x = r.
@fragment
fn fs_mdilate(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let a = textureLoad(tex_a, p, 0).r;
    let d = textureLoad(tex_b, p, 0).r;
    if (d < 0.0) { return mout(min(a + op.p0.x, 1.0)); }
    return mout(max(a, clamp(op.p0.x + 0.5 - d, 0.0, 1.0)));
}

// effects::blur, one separable pass: p0 = (vertical, radius); weights in LUT row 0; zero outside
// the region.
@fragment
fn fs_mblur(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let r = i32(op.p0.y);
    var step = vec2(1, 0);
    var n = chunk.size.x;
    var c = p.x;
    if (op.p0.x > 0.5) {
        step = vec2(0, 1);
        n = chunk.size.y;
        c = p.y;
    }
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k++) {
        let i = c + k - r;
        if (i >= 0 && i < n) {
            acc += textureLoad(tex_a, p + step * (k - r), 0).r * textureLoad(lut_tex, vec2(k, 0), 0).r;
        }
    }
    return mout(acc);
}

// Precise glow from a distance map (`effects::glow_map`): p0 = (solid, soft). Coverage is 0.5 at
// the edge falling to 0 at `size + 1` px centre to centre (the field is 0.5 px short of that),
// solid within the spread / choke.
@fragment
fn fs_mglow(in: VOut) -> @location(0) vec4<f32> {
    let d = textureLoad(tex_a, local(in.pos), 0).r + 0.5;
    var m = 0.5 * clamp(1.0 - (d - op.p0.x) / op.p0.y, 0.0, 1.0);
    if (d <= op.p0.x) { m = 1.0; }
    return mout(m);
}

// Last step of a coverage map: p0 = (|A - B| (satin), 1 - v (1: before the contour, 2: after it,
// as a centre-source inner glow), contour (LUT row 0), × shape (`layer_tex`)).
@fragment
fn fs_mfinish(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    var v = textureLoad(tex_a, p, 0).r;
    if (op.p0.x > 0.5) { v = abs(v - textureLoad(tex_b, p, 0).r); }
    if (op.p0.y > 0.5 && op.p0.y < 1.5) { v = 1.0 - v; }
    if (op.p0.z > 0.5) { v = lut(0, v); }
    if (op.p0.y > 1.5) { v = 1.0 - v; }
    if (op.p0.w > 0.5) { v = v * textureLoad(layer_tex, p, 0).r; }
    return mout(v);
}

// Chiselled bevel height from A = dist_inside, B = dist_outside (`effects::bevel_chisel_h`):
// p0 = (paint (0 outer, 1 both / emboss, 2 inner), size, Chisel Soft).
@fragment
fn fs_mbevelh(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let din = textureLoad(tex_a, p, 0).r;
    let dout = textureLoad(tex_b, p, 0).r;
    let size = op.p0.y;
    var h: f32;
    // The ramp spans size + 1 px, from half a pixel beyond the edge (`effects::bevel_chisel_h`).
    let ramp = size + 1.0;
    let soft = op.p0.z > 0.5;
    switch i32(op.p0.x) {
        case 0: {
            if (soft) {
                h = select((size + 2.0) / ramp, 1.0 - clamp((dout + 0.5) / ramp, 0.0, 1.0), dout > 0.0);
            } else {
                let top = (size + 2.0) / ramp;
                h = select(top, clamp((size + 1.5 - dout) / ramp, 0.0, top), dout > 0.0);
            }
        }
        case 1: {
            let half = size / 2.0 + 0.5;
            if (din > 0.0) {
                h = 0.5 + 0.5 * clamp(din / half, 0.0, 1.0);
            } else {
                h = 0.5 - 0.5 * clamp(dout / half, 0.0, 1.0);
            }
        }
        default: {
            if (soft) {
                h = select(0.0, clamp((din + 1.5) / ramp, 0.0, (size + 2.0) / ramp), din > 0.0);
            } else {
                h = select(0.0, clamp((din + 0.5) / ramp, 0.0, 1.0), din > 0.0);
            }
        }
    }
    return mout(h);
}

// Bevel shading of the blurred height map A over shape S (`layer_tex`): p0 = light vector,
// sin(altitude); p1 = (depth, outer, shadow (else highlight), contour (LUT row 0)).
@fragment
fn fs_mbevelshade(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let s = textureLoad(layer_tex, p, 0).r;
    // effects::bevel_maps' on_bevel: the height above BEVEL_H_EPS (emboss styles: or a
    // 4-neighbour's).
    var hmax = ra(p);
    // p3.x: an outer chisel bevel, which also reaches one pixel past its ramp.
    if (op.p1.y > 1.5 || op.p3.x > 0.5) {
        hmax = max(max(hmax, max(ra(p + vec2(1, 0)), ra(p - vec2(1, 0)))), max(ra(p + vec2(0, 1)), ra(p - vec2(0, 1))));
    }
    let outside = select(0.0, 1.0, s < 1.0 - INSIDE_EPS && hmax > 1e-5);
    // p1.y: 0 inside (× alpha), 1 under the edge and outside (outer bevel), 2 / 3 emboss / pillow
    // emboss (inside where the shape is, plus the outside half, flipped for pillow, × 1 − alpha).
    if (op.p1.y < 0.5) { return mout(bevel_part(p, op.p1.x, s)); }
    if (op.p1.y < 1.5) { return mout(bevel_part(p, op.p1.x, outside)); }
    var inner = 0.0;
    if (s > INSIDE_EPS) { inner = bevel_part(p, op.p1.x, s); }
    let od = select(op.p1.x, -op.p1.x, op.p1.y > 2.5);
    return mout(inner + bevel_part(p, od, outside) * (1.0 - min(s, 1.0)));
}

// One bevel shading value (`bevel_maps`' shade_into): highlight (p1.z = 0) or shadow amount for
// height-gradient scale `depth`, times `region`, through the gloss contour.
fn bevel_part(p: vec2<i32>, depth: f32, region: f32) -> f32 {
    let gx = (ra(p + vec2(1, 0)) - ra(p - vec2(1, 0))) * 0.5 * depth;
    let gy = (ra(p + vec2(0, 1)) - ra(p - vec2(0, 1))) * 0.5 * depth;
    let n = vec3(-gx, -gy, 1.0);
    let len = sqrt(n.x * n.x + n.y * n.y + 1.0);
    let shade = (n.x * op.p0.x + n.y * op.p0.y + n.z * op.p0.z) / len;
    let se = op.p0.w;
    let k = shade - se;
    var v = 0.0;
    if (op.p1.z > 0.5) {
        if (k <= 0.0) { v = clamp(-k / max(se, 1e-3), 0.0, 1.0) * region; }
    } else if (k > 0.0) {
        v = clamp(k / max(1.0 - se, 1e-3), 0.0, 1.0) * region;
    }
    if (op.p1.w > 0.5) { v = lut(0, v); }
    return v;
}

// Bevel texture (`effects::bevel_height`): A + k × luminance of the pattern (× its alpha; 1 −
// that when inverted); p0 = (k, invert), placement in p3 / p4 as for pattern paints.
@fragment
fn fs_mbeveltex(in: VOut) -> @location(0) vec4<f32> {
    let p = local(in.pos);
    let c = pattern_sample(doc_px(p));
    var l = (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) * c.a;
    if (op.p0.y > 0.5) { l = 1.0 - l; }
    return mout(textureLoad(tex_a, p, 0).r + op.p0.x * l);
}

// Stroke band from a distance map: clamp(width + 0.5 - d), p0.x = width.
@fragment
fn fs_mstroke(in: VOut) -> @location(0) vec4<f32> {
    let d = textureLoad(tex_a, local(in.pos), 0).r;
    return mout(clamp(op.p0.x + 0.5 - d, 0.0, 1.0));
}
