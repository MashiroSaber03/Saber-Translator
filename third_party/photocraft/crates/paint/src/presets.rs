//! Built-in brush presets (our own designs; sampled tips and textures are generated procedurally).
//!
//! Grouped like Photoshop's default Brushes panel folders: General, Dry Media, Wet Media and
//! Special Effects. Every preset is plain [`BrushSettings`] data, so it serialises, journals and
//! can be edited like any user preset.

use crate::brush::*;
use crate::procedural::{bristle_tip, chalk_tip, charcoal_tip, grass_tip, leaf_tip, rake_tip, spatter_tip, sponge_tip, star_tip};

/// Preset groups, in panel order.
pub const GROUPS: [&str; 4] = ["General", "Dry Media", "Wet Media", "Special Effects"];

fn preset(group: &str, name: &str, brush: BrushSettings) -> BrushPreset {
    BrushPreset { name: name.to_string(), brush, builtin: true, group: group.to_string(), folder: Vec::new(), captures: Default::default() }
}

fn pressure() -> Dynamic {
    Dynamic::controlled(Control::PenPressure)
}

fn pressure_min(minimum: f32) -> Dynamic {
    Dynamic { minimum, ..pressure() }
}

fn size_dyn(size: Dynamic) -> ShapeDynamics {
    ShapeDynamics { enabled: true, size, ..Default::default() }
}

fn transfer(opacity: Dynamic, flow: Dynamic) -> Transfer {
    Transfer { enabled: true, opacity, flow, ..Default::default() }
}

fn paper(seed: u32, depth: f32) -> Texture {
    Texture { enabled: true, pattern: Pattern::Procedural { style: PatternStyle::Paper, size: 128, seed }, depth, ..Default::default() }
}

fn general(base: &BrushSettings) -> Vec<BrushPreset> {
    let g = |n: &str, b: BrushSettings| preset("General", n, b);
    vec![
        g("Hard Round", BrushSettings { size: 30.0, hardness: 1.0, ..base.clone() }),
        g("Soft Round", BrushSettings { size: 45.0, hardness: 0.0, ..base.clone() }),
        g("Hard Round Fine", BrushSettings { size: 5.0, hardness: 1.0, spacing: 0.1, ..base.clone() }),
        g("Soft Round Large", BrushSettings { size: 300.0, hardness: 0.0, spacing: 0.15, ..base.clone() }),
        g("Hard Round Pressure Size", BrushSettings { size: 30.0, hardness: 1.0, shape_dynamics: size_dyn(pressure()), ..base.clone() }),
        g("Soft Round Pressure Size", BrushSettings { size: 45.0, hardness: 0.0, shape_dynamics: size_dyn(pressure()), ..base.clone() }),
        g("Hard Round Pressure Opacity", BrushSettings { size: 30.0, hardness: 1.0, transfer: transfer(pressure(), Dynamic::default()), ..base.clone() }),
        g("Soft Round Pressure Opacity", BrushSettings { size: 45.0, hardness: 0.0, transfer: transfer(pressure(), Dynamic::default()), ..base.clone() }),
        g(
            "Hard Round Pressure Opacity and Flow",
            BrushSettings { size: 30.0, hardness: 0.9, transfer: transfer(pressure(), pressure_min(0.2)), ..base.clone() },
        ),
        g("Airbrush Soft", BrushSettings { size: 80.0, hardness: 0.0, flow: 0.1, spacing: 0.1, build_up: true, build_up_rate: 25.0, ..base.clone() }),
        g(
            "Airbrush Pressure Flow",
            BrushSettings {
                size: 120.0,
                hardness: 0.0,
                flow: 0.15,
                spacing: 0.1,
                build_up: true,
                build_up_rate: 30.0,
                transfer: transfer(Dynamic::default(), pressure()),
                ..base.clone()
            },
        ),
        g("Calligraphy Flat", BrushSettings { size: 28.0, hardness: 1.0, roundness: 0.2, angle: 45.0, spacing: 0.05, ..base.clone() }),
    ]
}

fn dry_media(base: &BrushSettings) -> Vec<BrushPreset> {
    let g = |n: &str, b: BrushSettings| preset("Dry Media", n, b);
    vec![
        g("Hard Pencil", BrushSettings { size: 3.0, hardness: 1.0, aliased: true, spacing: 0.1, ..base.clone() }),
        g(
            "Soft Pencil",
            BrushSettings {
                size: 6.0,
                hardness: 0.6,
                spacing: 0.08,
                shape_dynamics: size_dyn(pressure_min(0.4)),
                transfer: transfer(pressure_min(0.15), Dynamic::default()),
                texture: Texture { scale: 0.5, ..paper(21, 0.8) },
                ..base.clone()
            },
        ),
        g(
            "Chalk",
            BrushSettings {
                size: 40.0,
                tip: TipShape::Sampled(chalk_tip(64, 7)),
                spacing: 0.2,
                shape_dynamics: ShapeDynamics {
                    enabled: true,
                    angle: Dynamic::jitter(1.0),
                    size: Dynamic { jitter: 0.15, ..Default::default() },
                    ..Default::default()
                },
                texture: paper(3, 0.6),
                ..base.clone()
            },
        ),
        g(
            "Charcoal",
            BrushSettings {
                size: 45.0,
                tip: TipShape::Sampled(charcoal_tip(64, 4)),
                spacing: 0.12,
                shape_dynamics: ShapeDynamics { enabled: true, size: pressure_min(0.5), angle: Dynamic::controlled(Control::Direction), ..Default::default() },
                transfer: transfer(pressure_min(0.3), Dynamic::default()),
                // Subtract takes paint away where the pattern is light (#3340); inverting keeps
                // these presets as they were designed, with the dark parts taking it away.
                texture: Texture { mode: MaskMode::Subtract, invert: true, ..paper(5, 0.7) },
                ..base.clone()
            },
        ),
        g(
            "Pastel",
            BrushSettings {
                size: 50.0,
                tip: TipShape::Sampled(chalk_tip(64, 31)),
                spacing: 0.15,
                shape_dynamics: ShapeDynamics {
                    enabled: true,
                    angle: Dynamic::jitter(1.0),
                    roundness: Dynamic { jitter: 0.3, minimum: 0.5, ..Default::default() },
                    ..Default::default()
                },
                scattering: Scattering { enabled: true, scatter: Dynamic::jitter(0.25), ..Default::default() },
                texture: Texture {
                    enabled: true,
                    pattern: Pattern::Procedural { style: PatternStyle::Canvas, size: 64, seed: 2 },
                    depth: 0.55,
                    mode: MaskMode::Subtract,
                    invert: true,
                    ..Default::default()
                },
                ..base.clone()
            },
        ),
        g(
            "Wax Crayon",
            BrushSettings {
                size: 22.0,
                hardness: 0.85,
                roundness: 0.7,
                spacing: 0.08,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::controlled(Control::Direction), ..Default::default() },
                texture: Texture { each_tip: true, mode: MaskMode::HardMix, depth_jitter: Dynamic::jitter(0.3), ..paper(17, 0.9) },
                ..base.clone()
            },
        ),
        g(
            "Dry Bristle",
            BrushSettings {
                size: 50.0,
                tip: TipShape::Sampled(bristle_tip(64, 5)),
                spacing: 0.04,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::controlled(Control::Direction), ..Default::default() },
                transfer: Transfer {
                    enabled: true,
                    flow: Dynamic { control: Control::Fade, fade_steps: 400, minimum: 0.1, ..Default::default() },
                    ..Default::default()
                },
                ..base.clone()
            },
        ),
        g(
            "Rake",
            BrushSettings {
                size: 60.0,
                tip: TipShape::Sampled(rake_tip(64, 3, 7)),
                spacing: 0.03,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::controlled(Control::Direction), ..Default::default() },
                transfer: transfer(pressure_min(0.25), Dynamic::default()),
                ..base.clone()
            },
        ),
    ]
}

fn wet_media(base: &BrushSettings) -> Vec<BrushPreset> {
    let g = |n: &str, b: BrushSettings| preset("Wet Media", n, b);
    vec![
        g(
            "Watercolor Wet Edges",
            BrushSettings {
                size: 60.0,
                hardness: 0.3,
                wet_edges: true,
                texture: Texture { scale: 1.5, ..paper(9, 0.35) }.with_pattern_size(256),
                ..base.clone()
            },
        ),
        g(
            "Watercolor Wash",
            BrushSettings {
                size: 140.0,
                hardness: 0.0,
                flow: 0.35,
                spacing: 0.12,
                wet_edges: true,
                shape_dynamics: size_dyn(pressure_min(0.35)),
                transfer: transfer(Dynamic::default(), Dynamic { jitter: 0.2, ..pressure() }),
                texture: Texture { scale: 2.0, ..paper(13, 0.25) }.with_pattern_size(256),
                ..base.clone()
            },
        ),
        g(
            "Ink Pen",
            BrushSettings {
                size: 8.0,
                hardness: 1.0,
                spacing: 0.05,
                shape_dynamics: size_dyn(pressure_min(0.15)),
                smoothing: Smoothing { amount: 0.35, ..Default::default() },
                ..base.clone()
            },
        ),
        g(
            "Oil Bristle",
            BrushSettings {
                size: 70.0,
                tip: TipShape::Sampled(bristle_tip(64, 41)),
                spacing: 0.03,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::controlled(Control::Direction), size: pressure_min(0.6), ..Default::default() },
                color_dynamics: ColorDynamics { enabled: true, per_tip: false, fg_bg: Dynamic::jitter(0.15), brightness_jitter: 0.05, ..Default::default() },
                texture: Texture {
                    enabled: true,
                    pattern: Pattern::Procedural { style: PatternStyle::Canvas, size: 64, seed: 4 },
                    depth: 0.3,
                    ..Default::default()
                },
                ..base.clone()
            },
        ),
        g(
            "Gouache Flat",
            BrushSettings {
                size: 40.0,
                hardness: 0.9,
                roundness: 0.35,
                spacing: 0.05,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::controlled(Control::Direction), ..Default::default() },
                transfer: transfer(Dynamic::default(), Dynamic { control: Control::Fade, fade_steps: 600, minimum: 0.35, ..Default::default() }),
                ..base.clone()
            },
        ),
        g(
            "Ink Wash Sponge",
            BrushSettings {
                size: 70.0,
                tip: TipShape::Sampled(sponge_tip(96, 8)),
                spacing: 0.3,
                flow: 0.6,
                wet_edges: true,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::jitter(1.0), size: Dynamic::jitter(0.2), ..Default::default() },
                ..base.clone()
            },
        ),
    ]
}

fn special_effects(base: &BrushSettings) -> Vec<BrushPreset> {
    let g = |n: &str, b: BrushSettings| preset("Special Effects", n, b);
    vec![
        g(
            "Spatter",
            BrushSettings {
                size: 60.0,
                tip: TipShape::Sampled(spatter_tip(64, 11, 14)),
                spacing: 0.6,
                shape_dynamics: ShapeDynamics {
                    enabled: true,
                    size: Dynamic { jitter: 0.6, minimum: 0.2, ..Default::default() },
                    angle: Dynamic::jitter(1.0),
                    flip_x_jitter: true,
                    flip_y_jitter: true,
                    ..Default::default()
                },
                scattering: Scattering { enabled: true, scatter: Dynamic::jitter(1.5), both_axes: true, count: 2, count_jitter: Dynamic::jitter(0.5) },
                ..base.clone()
            },
        ),
        g(
            "Stipple",
            BrushSettings {
                size: 40.0,
                tip: TipShape::Sampled(spatter_tip(64, 77, 40)),
                spacing: 0.5,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::jitter(1.0), flip_x_jitter: true, ..Default::default() },
                scattering: Scattering { enabled: true, scatter: Dynamic::jitter(0.6), both_axes: true, ..Default::default() },
                transfer: transfer(pressure(), Dynamic::default()),
                ..base.clone()
            },
        ),
        g(
            "Confetti",
            BrushSettings {
                size: 24.0,
                hardness: 1.0,
                roundness: 0.5,
                spacing: 1.2,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::jitter(1.0), size: Dynamic::jitter(0.5), ..Default::default() },
                scattering: Scattering { enabled: true, scatter: Dynamic::jitter(3.0), both_axes: true, count: 3, ..Default::default() },
                color_dynamics: ColorDynamics { enabled: true, per_tip: true, hue_jitter: 1.0, saturation_jitter: 0.2, purity: 0.5, ..Default::default() },
                ..base.clone()
            },
        ),
        g(
            "Foliage",
            BrushSettings {
                size: 50.0,
                tip: TipShape::Sampled(leaf_tip(64)),
                spacing: 0.35,
                shape_dynamics: ShapeDynamics {
                    enabled: true,
                    size: Dynamic { jitter: 0.5, minimum: 0.3, ..Default::default() },
                    angle: Dynamic::jitter(1.0),
                    roundness: Dynamic { jitter: 0.4, minimum: 0.4, ..Default::default() },
                    flip_y_jitter: true,
                    ..Default::default()
                },
                scattering: Scattering { enabled: true, scatter: Dynamic::jitter(1.2), both_axes: true, count: 2, count_jitter: Dynamic::jitter(0.5) },
                color_dynamics: ColorDynamics {
                    enabled: true,
                    per_tip: true,
                    fg_bg: Dynamic::jitter(0.6),
                    hue_jitter: 0.04,
                    brightness_jitter: 0.12,
                    ..Default::default()
                },
                ..base.clone()
            },
        ),
        g(
            "Grass",
            BrushSettings {
                size: 70.0,
                tip: TipShape::Sampled(grass_tip(96, 6, 11)),
                spacing: 0.25,
                shape_dynamics: ShapeDynamics {
                    enabled: true,
                    size: Dynamic { jitter: 0.4, minimum: 0.4, ..pressure() },
                    angle: Dynamic::jitter(0.06),
                    flip_x_jitter: true,
                    ..Default::default()
                },
                scattering: Scattering { enabled: true, scatter: Dynamic::jitter(0.3), ..Default::default() },
                color_dynamics: ColorDynamics { enabled: true, per_tip: true, fg_bg: Dynamic::jitter(0.7), hue_jitter: 0.03, ..Default::default() },
                ..base.clone()
            },
        ),
        g(
            "Sparkle",
            BrushSettings {
                size: 40.0,
                tip: TipShape::Sampled(star_tip(64)),
                spacing: 1.5,
                shape_dynamics: ShapeDynamics { enabled: true, size: Dynamic { jitter: 0.8, minimum: 0.15, ..Default::default() }, ..Default::default() },
                scattering: Scattering { enabled: true, scatter: Dynamic::jitter(2.5), both_axes: true, ..Default::default() },
                transfer: transfer(Dynamic::jitter(0.5), Dynamic::default()),
                ..base.clone()
            },
        ),
        g(
            "Sponge Texture",
            BrushSettings {
                size: 60.0,
                tip: TipShape::Sampled(sponge_tip(64, 19)),
                spacing: 0.45,
                shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::jitter(1.0), ..Default::default() },
                transfer: transfer(Dynamic::jitter(0.3), Dynamic::default()),
                ..base.clone()
            },
        ),
        g(
            "Canvas Texture",
            BrushSettings {
                size: 50.0,
                hardness: 0.8,
                texture: Texture {
                    enabled: true,
                    pattern: Pattern::Procedural { style: PatternStyle::Canvas, size: 64, seed: 1 },
                    mode: MaskMode::Subtract,
                    invert: true,
                    depth: 0.7,
                    ..Default::default()
                },
                ..base.clone()
            },
        ),
        g(
            "Halftone Dots",
            BrushSettings {
                size: 80.0,
                hardness: 0.5,
                texture: Texture {
                    enabled: true,
                    pattern: Pattern::Procedural { style: PatternStyle::Dots, size: 64, seed: 1 },
                    mode: MaskMode::HardMix,
                    depth: 1.0,
                    ..Default::default()
                },
                ..base.clone()
            },
        ),
        g(
            "Dual Grain",
            BrushSettings {
                size: 50.0,
                hardness: 0.6,
                noise: true,
                dual_brush: DualBrush {
                    enabled: true,
                    tip: TipShape::Sampled(spatter_tip(48, 23, 20)),
                    size: 30.0,
                    spacing: 0.3,
                    scatter: 1.0,
                    both_axes: true,
                    count: 2,
                    flip: true,
                    ..Default::default()
                },
                ..base.clone()
            },
        ),
    ]
}

impl Texture {
    fn with_pattern_size(mut self, n: u32) -> Self {
        if let Pattern::Procedural { size, .. } = &mut self.pattern {
            *size = n;
        }
        self
    }
}

/// The built-in preset set, in display order (grouped by [`GROUPS`]).
pub fn builtin() -> Vec<BrushPreset> {
    let base = BrushSettings { pressure_size: false, spacing: 0.25, ..Default::default() };
    let mut v = general(&base);
    v.extend(dry_media(&base));
    v.extend(wet_media(&base));
    v.extend(special_effects(&base));
    v
}

/// Find a preset by name (case-insensitive).
pub fn find<'a>(presets: &'a [BrushPreset], name: &str) -> Option<&'a BrushPreset> {
    presets.iter().find(|p| p.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_grouped_and_unique() {
        let v = builtin();
        assert!((30..=40).contains(&v.len()), "{}", v.len());
        let mut names: Vec<&str> = v.iter().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), v.len(), "duplicate preset names");
        for g in GROUPS {
            assert!(v.iter().filter(|p| p.group == g).count() >= 6, "{g}");
        }
        assert!(v.iter().all(|p| p.builtin && GROUPS.contains(&p.group.as_str())));
        // Groups are contiguous, in panel order.
        let order: Vec<usize> = v.iter().filter_map(|p| GROUPS.iter().position(|g| *g == p.group)).collect();
        assert!(order.windows(2).all(|w| w[0] <= w[1]));
        // Pressure variants respond to pressure.
        let ps = find(&v, "Hard Round Pressure Size").map(|p| p.brush.shape_dynamics.size.control);
        assert_eq!(ps, Some(Control::PenPressure));
    }
}
