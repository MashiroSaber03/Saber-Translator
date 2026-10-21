//! Icon rendering. SVGs are embedded (see `icon_data.rs`), recoloured to white, rasterized by the
//! egui_extras SVG loader at the exact on-screen pixel size (crisp at any DPI), and tinted per use.
//!
//! (History: the first shell used Unicode glyphs; half of them rendered as tofu boxes because the
//! bundled fonts lacked them. Vector icons fix that for good.)

use std::sync::OnceLock;

use craft_ui::buttons::{ButtonVisuals, IconButton, IconButtonStyle};
use craft_ui::icons::{SvgIconSet, SvgStyle};
use egui::{Color32, Rect, Response, Sense, Vec2};

use crate::icon_data::ICONS;
use crate::state::Tool;
use crate::theme::Tokens;

fn white_icons() -> &'static SvgIconSet {
    static MAP: OnceLock<SvgIconSet> = OnceLock::new();
    MAP.get_or_init(|| SvgIconSet::new(ICONS, SvgStyle { current_color: "#ffffff", stroke_width: Some("1.75") }, "bytes://icons/", "square"))
}

pub fn exists(name: &str) -> bool {
    white_icons().contains(name)
}

/// An egui image for an icon, tinted.
pub fn image(name: &str, size: f32, tint: Color32) -> egui::Image<'static> {
    egui::Image::new(white_icons().source(name)).fit_to_exact_size(Vec2::splat(size)).tint(tint)
}

/// Paint an icon centred in `rect`.
pub fn paint(ui: &egui::Ui, rect: Rect, name: &str, size: f32, tint: Color32) {
    let r = Rect::from_center_size(rect.center(), Vec2::splat(size));
    image(name, size, tint).paint_at(ui, r);
}

/// An icon as the pointer, above every window: `name` with its hotspot `hot` (a fraction of the
/// icon box) on `p`, white with a dark outline so it reads on any image. The caller hides the OS
/// cursor (`CursorIcon::None`).
pub fn cursor(ctx: &egui::Context, name: &str, p: egui::Pos2, hot: Vec2, size: f32) {
    let rect = Rect::from_min_size(p - hot * size, Vec2::splat(size));
    let area = egui::Area::new(egui::Id::new("pc-icon-cursor")).order(egui::Order::Tooltip).fixed_pos(rect.min).constrain(false).interactable(false);
    area.show(ctx, |ui| {
        let outline = image(name, size, Color32::from_black_alpha(200));
        for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)] {
            outline.paint_at(ui, rect.translate(egui::vec2(dx, dy)));
        }
        image(name, size, Color32::WHITE).paint_at(ui, rect);
    });
}

/// A small icon beside the OS arrow pointer at `p` (Photoshop's cursor badges: scissors for
/// cutting the selected pixels). Below-right of the arrow, white with a dark outline. The caller
/// keeps the arrow (`CursorIcon::Default`).
pub fn cursor_badge(ctx: &egui::Context, name: &str, p: egui::Pos2) {
    badge(ctx, name, p, false);
}

/// Photoshop's duplicate badge: a second, dark arrowhead behind a white one.
pub fn cursor_copy_badge(ctx: &egui::Context, p: egui::Pos2) {
    badge(ctx, "mouse-pointer-2", p, true);
}

fn badge(ctx: &egui::Context, name: &str, p: egui::Pos2, doubled: bool) {
    const SIZE: f32 = 12.0;
    let rect = Rect::from_center_size(p + egui::vec2(20.0, 19.0), Vec2::splat(SIZE));
    let area = egui::Area::new(egui::Id::new("pc-cursor-badge")).order(egui::Order::Tooltip).fixed_pos(rect.min).constrain(false).interactable(false);
    area.show(ctx, |ui| {
        let outline = image(name, SIZE, Color32::from_black_alpha(220));
        let halo = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)];
        if doubled {
            let behind = rect.translate(egui::vec2(4.0, 4.0));
            for (dx, dy) in halo {
                image(name, SIZE, Color32::WHITE).paint_at(ui, behind.translate(egui::vec2(dx, dy)));
            }
            image(name, SIZE, Color32::BLACK).paint_at(ui, behind);
        }
        for (dx, dy) in halo {
            outline.paint_at(ui, rect.translate(egui::vec2(dx, dy)));
        }
        image(name, SIZE, Color32::WHITE).paint_at(ui, rect);
    });
}

pub fn tool_icon(t: Tool) -> &'static str {
    match t {
        Tool::Move => "move",
        Tool::Artboard => "artboard",
        Tool::RectMarquee => "square-dashed",
        Tool::EllipseMarquee => "circle-dashed",
        Tool::SingleRowMarquee => "marquee-row",
        Tool::SingleColumnMarquee => "marquee-column",
        Tool::Brush => "brush",
        Tool::Pencil => "pencil",
        Tool::MixerBrush => "palette",
        Tool::ColorReplacement => "brush-replace",
        Tool::Eraser => "eraser",
        Tool::BackgroundEraser => "eraser-background",
        Tool::MagicEraser => "eraser-magic",
        Tool::Eyedropper => "pipette",
        Tool::ColorSampler => "sampler",
        Tool::Ruler => "ruler",
        Tool::Note => "message-square",
        Tool::Count => "circle-dot",
        Tool::Lasso => "lasso",
        Tool::PolygonLasso => "pentagon",
        Tool::MagneticLasso => "lasso-magnetic",
        Tool::MagicWand => "wand-sparkles",
        Tool::Crop => "crop",
        Tool::PerspectiveCrop => "perspective-crop",
        Tool::Slice => "slice-knife",
        Tool::SliceSelect => "square-dashed-mouse-pointer",
        Tool::Frame => "frame-placeholder",
        Tool::Gradient => "gradient",
        Tool::PaintBucket => "paint-bucket",
        Tool::Type | Tool::VerticalType => "type",
        Tool::HorizontalTypeMask => "type-mask",
        Tool::VerticalTypeMask => "type-mask-vertical",
        Tool::Hand => "hand",
        Tool::RotateView => "compass",
        Tool::Zoom => "zoom-in",
        Tool::SpotHealing | Tool::Healing => "bandage",
        Tool::Remove => "bandage-sparkle",
        Tool::Patch => "lasso-select",
        Tool::ContentAwareMove => "arrow-left-right",
        Tool::RedEye => "eye",
        Tool::CloneStamp => "stamp",
        Tool::PatternStamp => "grid-3x3",
        Tool::HistoryBrush => "clock",
        Tool::Blur => "droplet",
        Tool::Sharpen => "triangle",
        Tool::Smudge => "pointer",
        Tool::Dodge => "lollipop",
        Tool::Burn => "flame",
        Tool::Sponge => "cloud",
        Tool::QuickSelection => "circle-dashed",
        Tool::ObjectSelection => "square-dashed-mouse-pointer",
        Tool::Pen => "pen-tool",
        Tool::FreeformPen => "pen-freeform",
        Tool::CurvaturePen => "pen-curvature",
        Tool::AddAnchorPoint => "pen-add-anchor",
        Tool::DeleteAnchorPoint => "pen-delete-anchor",
        Tool::ConvertPoint => "convert-point",
        Tool::PathSelection => "mouse-pointer-2",
        Tool::DirectSelection => "direct-select",
        Tool::Rectangle => "rectangle-horizontal",
        Tool::EllipseShape => "circle",
        Tool::Triangle => "triangle",
        Tool::Polygon => "pentagon",
        Tool::Line => "slash",
        Tool::CustomShape => "cloud",
    }
}

/// Selected and hover fill shared by square icon buttons. Returns the icon tint.
pub fn button_chrome(ui: &egui::Ui, rect: Rect, selected: bool, hovered: bool) -> Color32 {
    let t = Tokens::get(ui.ctx());
    if selected {
        ui.painter().rect_filled(rect, t.radius_sm, t.accent_soft);
        ui.painter().rect_stroke(rect, t.radius_sm, egui::Stroke::new(1.0, t.accent_border), egui::StrokeKind::Inside);
    } else if hovered {
        ui.painter().rect_filled(rect, t.radius_sm, t.hover);
    }
    if selected {
        t.accent_text
    } else if hovered {
        t.text
    } else {
        t.icon
    }
}

/// Square icon button: transparent until hovered; `selected` gets the accent treatment.
pub fn button(ui: &mut egui::Ui, name: &str, box_size: f32, selected: bool, tooltip: &str) -> Response {
    button_with_icon_size(ui, name, box_size, (box_size * 0.52).round(), selected, tooltip)
}

/// Square icon button with an explicit glyph size, independent of its hit area.
pub fn button_with_icon_size(ui: &mut egui::Ui, name: &str, box_size: f32, icon_size: f32, selected: bool, tooltip: &str) -> Response {
    button_with_icon_size_tip(ui, name, box_size, icon_size, selected, tooltip, true)
}

/// [`button_with_icon_size`] whose tooltip shows only when `show_tip`; the tooltip text still
/// names the button (for screen readers and tests) when it is hidden.
pub fn button_with_icon_size_tip(ui: &mut egui::Ui, name: &str, box_size: f32, icon_size: f32, selected: bool, tooltip: &str, show_tip: bool) -> Response {
    let t = Tokens::get(ui.ctx());
    let style = IconButtonStyle {
        normal: ButtonVisuals { fill: Color32::TRANSPARENT, stroke: egui::Stroke::NONE, icon: t.icon },
        hovered: ButtonVisuals { fill: t.hover, stroke: egui::Stroke::NONE, icon: t.text },
        pressed: None,
        selected: ButtonVisuals { fill: t.accent_soft, stroke: egui::Stroke::new(1.0, t.accent_border), icon: t.accent_text },
        corner_radius: t.radius_sm.into(),
        focus_stroke: egui::Stroke::new(1.0, t.accent),
    };
    let label = if tooltip.is_empty() { name } else { tl!(tooltip) };
    let resp = IconButton::new(label, Vec2::splat(box_size), &style).selected(selected).show(ui, |ui, rect, tint| {
        paint(ui, rect, name, icon_size, tint);
    });
    if tooltip.is_empty() || !show_tip { resp } else { resp.on_hover_text(label) }
}

/// Rail toggle: "on" gets a quiet filled background and full-strength icon (no accent).
pub fn rail_button(ui: &mut egui::Ui, name: &str, box_size: f32, on: bool, tooltip: &str) -> Response {
    let t = Tokens::get(ui.ctx());
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(box_size), Sense::click());
    if on {
        ui.painter().rect_filled(rect, t.radius_sm, t.card);
        ui.painter().rect_stroke(rect, t.radius_sm, egui::Stroke::new(1.0, t.card_border), egui::StrokeKind::Inside);
    } else if resp.hovered() {
        ui.painter().rect_filled(rect, t.radius_sm, t.hover);
    }
    let tint = if on || resp.hovered() { t.text } else { t.text_faint };
    paint(ui, rect, name, (box_size * 0.52).round(), tint);
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), on, tl!(tooltip)));
    resp.on_hover_text(tl!(tooltip))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_an_icon() {
        for t in Tool::ALL {
            assert!(exists(tool_icon(t)), "{t:?}");
        }
        assert_eq!(tool_icon(Tool::Gradient), "gradient");
        assert_ne!(tool_icon(Tool::Gradient), tool_icon(Tool::PaintBucket));
    }

    #[test]
    fn gradient_tool_icon_rasterizes_left_to_right_ramp() {
        let egui::ImageSource::Bytes { bytes, .. } = white_icons().source(tool_icon(Tool::Gradient)) else {
            panic!("icon source must be embedded bytes");
        };
        let image = egui_extras::image::load_svg_bytes_with_size(
            bytes.as_ref(),
            egui::load::SizeHint::Size { width: 24, height: 24, maintain_aspect_ratio: true },
            &Default::default(),
        )
        .unwrap();
        let left_alpha = image[(6, 12)].a();
        let mid_alpha = image[(12, 12)].a();
        let right_alpha = image[(18, 12)].a();
        assert!(
            left_alpha > mid_alpha && mid_alpha > right_alpha,
            "interior ramp should fade from left ({left_alpha}) through middle ({mid_alpha}) to right ({right_alpha})"
        );
    }

    #[test]
    fn shared_catalog_preserves_existing_icons() {
        // Characterize the pre-extraction loader across every real asset. The UI's tint,
        // stroke weight and cache keys must survive shared-crate upgrades unchanged.
        assert!(ICONS.len() >= 60);
        for (name, original) in ICONS {
            let expected = String::from_utf8_lossy(original).replace("currentColor", "#ffffff").replace("stroke-width=\"2\"", "stroke-width=\"1.75\"");
            let egui::ImageSource::Bytes { uri, bytes } = white_icons().source(name) else { panic!("expected SVG bytes") };
            assert_eq!(uri.as_ref(), format!("bytes://icons/{name}.svg"));
            assert_eq!(bytes.as_ref(), expected.as_bytes(), "{name}");
            assert!(!std::str::from_utf8(&bytes).unwrap().contains("currentColor"), "{name}");
        }
    }

    #[test]
    fn shared_catalog_preserves_fallback() {
        let egui::ImageSource::Bytes { uri, bytes } = white_icons().source("missing-icon") else { panic!("expected SVG bytes") };
        let egui::ImageSource::Bytes { bytes: fallback, .. } = white_icons().source("square") else { panic!("expected SVG bytes") };
        assert_eq!(uri.as_ref(), "bytes://icons/missing-icon.svg");
        assert_eq!(bytes.as_ref(), fallback.as_ref());
        assert!(!exists("missing-icon"));
    }

    #[test]
    fn shared_button_preserves_theme_painting_and_pointer_responses() {
        // The unchanged chrome helper is the pre-extraction paint oracle. Keyboard focus
        // and accessibility metadata are deliberate additions, covered in craft-ui.
        fn frame(
            ctx: &egui::Context,
            events: Vec<egui::Event>,
            selected: bool,
            enabled: bool,
            legacy: bool,
        ) -> (egui::Response, Vec<egui::epaint::ClippedShape>) {
            let mut response = None;
            let mut output = ctx.run_ui(egui::RawInput { events, ..Default::default() }, |ui| {
                response = Some(
                    ui.add_enabled_ui(enabled, |ui| {
                        if legacy {
                            let (rect, response) = ui.allocate_exact_size(Vec2::splat(28.0), Sense::click());
                            let tint = button_chrome(ui, rect, selected, response.hovered());
                            paint(ui, rect, "square", (28.0_f32 * 0.52).round(), tint);
                            response
                        } else {
                            button(ui, "square", 28.0, selected, "")
                        }
                    })
                    .inner,
                );
            });
            output.textures_delta.clear();
            (response.unwrap(), output.shapes)
        }
        for theme in crate::theme::ThemeKind::ALL {
            for selected in [false, true] {
                for enabled in [false, true] {
                    let old = egui::Context::default();
                    let new = egui::Context::default();
                    for ctx in [&old, &new] {
                        crate::theme::apply(ctx, theme);
                        egui_extras::install_image_loaders(ctx);
                    }
                    let mut center = egui::Pos2::ZERO;
                    for step in 0..7 {
                        let events = match step {
                            2 => vec![egui::Event::PointerMoved(center)],
                            3 | 4 => vec![egui::Event::PointerButton {
                                pos: center,
                                button: egui::PointerButton::Primary,
                                pressed: step == 3,
                                modifiers: egui::Modifiers::NONE,
                            }],
                            5 => vec![egui::Event::PointerMoved(egui::pos2(300.0, 200.0))],
                            _ => vec![],
                        };
                        let (a, before) = frame(&old, events.clone(), selected, enabled, true);
                        let (b, after) = frame(&new, events, selected, enabled, false);
                        center = a.rect.center();
                        assert_eq!(a.rect, b.rect);
                        assert_eq!((a.clicked(), a.hovered(), a.is_pointer_button_down_on()), (b.clicked(), b.hovered(), b.is_pointer_button_down_on()));
                        assert_eq!(before, after, "{theme:?}, selected={selected}, enabled={enabled}, step={step}");
                    }
                }
            }
        }
    }

    #[test]
    fn symmetry_axes_rasterize_centred_with_matching_line_weight_at_both_dpis() {
        for size in [14_u32, 28] {
            let mut coverage = Vec::new();
            for name in ["symmetry-vertical", "symmetry-horizontal", "symmetry-diagonal", "symmetry-dual"] {
                assert!(white_icons().contains(name));
                let egui::ImageSource::Bytes { bytes, .. } = white_icons().source(name) else { panic!("icon source must be embedded bytes") };
                let image = egui_extras::image::load_svg_bytes_with_size(
                    bytes.as_ref(),
                    egui::load::SizeHint::Size { width: size, height: size, maintain_aspect_ratio: true },
                    &Default::default(),
                )
                .unwrap();
                assert_eq!(image.size, [size as usize; 2]);
                let ink: Vec<_> = image.pixels.iter().enumerate().filter(|(_, p)| p.a() > 16).collect();
                assert!(!ink.is_empty(), "{name}");
                let xs: Vec<_> = ink.iter().map(|(i, _)| i % size as usize).collect();
                let ys: Vec<_> = ink.iter().map(|(i, _)| i / size as usize).collect();
                for coords in [&xs, &ys] {
                    let centre = (*coords.iter().min().unwrap() + *coords.iter().max().unwrap()) as f32 / 2.0;
                    assert!((centre - (size as f32 - 1.0) / 2.0).abs() <= 0.5, "{name}: off-centre");
                }
                assert!(ink.iter().all(|(_, p)| p.r() > 0 && p.r() == p.g() && p.g() == p.b()), "{name}: tinting mask must be white");
                coverage.push(image.pixels.iter().map(|p| f32::from(p.a()) / 255.0).sum::<f32>());
            }
            let vertical = coverage[0];
            // At 14px a diagonal's antialiasing can differ by nearly two covered pixels.
            let tolerance = if size == 14 { 0.15 } else { 0.1 };
            for single in &coverage[1..3] {
                assert!((single - vertical).abs() / vertical < tolerance, "inconsistent axis stroke weight at {size}px: {coverage:?}");
            }
            assert!(coverage[3] > vertical * 1.7 && coverage[3] < vertical * 2.1);
        }
    }
}
