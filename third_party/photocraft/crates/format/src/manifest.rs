//! `manifest.json` schema. This is a *separate* serde model from
//! `photocraft-doc` on purpose: the on-disk schema is versioned and migrated
//! independently of in-memory refactors.
//!
//! Binary payloads never live in JSON. Surfaces reference tiles
//! (`tiles/<blake3>.zst`) and other binary data references blobs
//! (`blobs/<blake3>.zst`), both by the BLAKE3 hash of their uncompressed
//! bytes (little-endian samples for tiles).

use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{
    Adjustment, ClippingPath, Effect, Fill, GlobalLight, Guides, LabelColor, LiveShape, Locks, Path, ShapeStroke, SmartFilter, VectorMask, text,
};
use photocraft_geom::{Affine, Size};
use serde::{Deserialize, Serialize};

/// Current manifest version written by this build.
pub const FORMAT_VERSION: u32 = 1;

/// Hex-encoded BLAKE3 hash (64 chars).
pub type Hash = String;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    /// Free-form writer identification, e.g. `photocraft-format 0.1.0`.
    pub generator: String,
    pub document: DocM,
    /// Optional previews present in the bundle.
    #[serde(default)]
    pub thumbnail: Option<String>,
    #[serde(default)]
    pub composite: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocM {
    pub id: u64,
    pub name: String,
    pub size: Size,
    pub resolution_dpi: f32,
    pub mode: ColorMode,
    pub depth: SampleType,
    pub icc_profile: Option<Hash>,
    /// Bottom-to-top.
    pub layers: Vec<LayerM>,
    pub channels: Vec<ChannelM>,
    pub guides: Guides,
    /// The rulers' zero point in document pixels (absent in older files: the top-left).
    #[serde(default)]
    pub ruler_origin: [f64; 2],
    pub selection: Option<SurfaceM>,
    pub metadata: MetadataM,
    pub global_light: GlobalLight,
    /// Saved paths (Paths panel).
    #[serde(default)]
    pub paths: Vec<NamedPathM>,
    #[serde(default)]
    pub work_path: Option<Path>,
    #[serde(default)]
    pub clipping_path: Option<ClippingPath>,
    /// The temporary Quick Mask channel (saved while Quick Mask mode is on).
    #[serde(default)]
    pub quick_mask: Option<ChannelM>,
    /// Quick Mask was entered with no selection.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub quick_mask_unselected: bool,
    /// Patterns stored with the document.
    #[serde(default)]
    pub patterns: Vec<PatternM>,
    /// Indexed Color palette.
    #[serde(default)]
    pub color_table: Option<photocraft_doc::ColorTable>,
    /// Duotone inks.
    #[serde(default)]
    pub duotone: Option<photocraft_doc::Duotone>,
    /// Window › Layer Comps.
    #[serde(default)]
    pub layer_comps: Vec<LayerCompM>,
    #[serde(default)]
    pub last_applied_comp: Option<u32>,
    #[serde(default)]
    pub last_document_state: Option<LayerCompM>,
    /// Image › Variables and Data Sets.
    #[serde(default)]
    pub variables: photocraft_doc::Variables,
    /// Window › Timeline.
    #[serde(default)]
    pub timeline: Option<photocraft_doc::Timeline>,
    /// The JPEG quality / WebP encoding File › Save reuses.
    #[serde(default, skip_serializing_if = "photocraft_doc::FileEncoding::is_empty")]
    pub file_encoding: photocraft_doc::FileEncoding,
    /// Image › Analysis: measurement scale, count groups, ruler.
    #[serde(default)]
    pub measurement: photocraft_doc::Measurement,
    /// Note tool annotations.
    #[serde(default)]
    pub notes: Vec<photocraft_doc::Note>,
    /// Character and paragraph styles.
    #[serde(default)]
    pub text_styles: photocraft_doc::TextStyles,
    /// Web slices (user and layer-based; layer ids are remapped on load).
    #[serde(default, skip_serializing_if = "photocraft_doc::Slices::is_empty")]
    pub slices: photocraft_doc::Slices,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerCompM {
    pub id: u32,
    pub name: String,
    #[serde(default)]
    pub comment: String,
    pub apply_visibility: bool,
    pub apply_position: bool,
    pub apply_appearance: bool,
    #[serde(default)]
    pub states: Vec<CompStateM>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompStateM {
    /// Layer id (as in `LayerM::id`).
    pub layer: u64,
    #[serde(default)]
    pub visible: Option<bool>,
    #[serde(default)]
    pub position: Option<(i32, i32)>,
    #[serde(default)]
    pub appearance: Option<CompAppearanceM>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompAppearanceM {
    pub blend: BlendMode,
    pub opacity: f32,
    pub fill_opacity: f32,
    pub effects: EffectsM,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternM {
    pub id: String,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub surface: SurfaceM,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedPathM {
    pub name: String,
    pub path: Path,
    #[serde(default)]
    pub psd_raw: Option<Hash>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileRef {
    pub tx: i32,
    pub ty: i32,
    pub hash: Hash,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceM {
    pub format: PixelFormat,
    /// Default ("untouched") pixel as hex of its little-endian encoded bytes.
    pub default: String,
    /// 256×256 tiles, interleaved little-endian samples.
    pub tiles: Vec<TileRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoDataM {
    pub frames: Vec<SurfaceM>,
    /// Optional for backward compatibility with native documents created before original
    /// frame snapshots were stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_frames: Option<Vec<SurfaceM>>,
    pub source: photocraft_doc::VideoSource,
    pub fps: f32,
    #[serde(default)]
    pub show_altered: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskM {
    pub surface: SurfaceM,
    pub enabled: bool,
    pub linked: bool,
    pub density: f32,
    pub feather: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectsM {
    pub enabled: bool,
    pub items: Vec<Effect>,
    pub psd_raw: Option<Hash>,
    #[serde(default)]
    pub reference: Option<(f64, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FillCacheM {
    pub fill: Fill,
    pub surface: SurfaceM,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerM {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub locks: Locks,
    pub blend: BlendMode,
    pub opacity: f32,
    pub fill_opacity: f32,
    pub clipped: bool,
    pub mask: Option<MaskM>,
    #[serde(default)]
    pub vector_mask: Option<VectorMask>,
    pub effects: EffectsM,
    pub label: LabelColor,
    pub content: ContentM,
    /// (4-char key as hex, blob)
    pub psd_blocks: Vec<(String, Hash)>,
    pub psd_id: Option<u32>,
    pub fill_cache: Option<FillCacheM>,
    /// Link Layers group (`None` = not linked).
    #[serde(default)]
    pub link_group: Option<u64>,
    /// Advanced Blending channels left out (bit per colour channel; 0 = all blend).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub excluded_channels: u32,
    /// Blend If ranges (empty = everything blends).
    #[serde(default, skip_serializing_if = "photocraft_doc::BlendIf::is_default")]
    pub blend_if: photocraft_doc::BlendIf,
    /// Advanced Blending switches (knockout, as-group, transparency shapes, masks hide effects);
    /// omitted at Photoshop's defaults, and missing in older files = the defaults.
    #[serde(default, skip_serializing_if = "photocraft_doc::AdvancedBlending::is_default")]
    pub advanced: photocraft_doc::AdvancedBlending,
    #[serde(default)]
    pub video: Option<VideoDataM>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentM {
    Raster {
        surface: SurfaceM,
    },
    Group {
        children: Vec<LayerM>,
        expanded: bool,
        /// Artboard (Layer › New › Artboard).
        #[serde(default)]
        artboard: Option<photocraft_doc::Artboard>,
    },
    Adjustment {
        adjustment: Adjustment,
    },
    Fill {
        fill: Fill,
    },
    Text {
        text: String,
        font_family: String,
        size_pt: f32,
        color: Color,
        transform: Affine,
        cache: Option<SurfaceM>,
        psd_raw: Option<Hash>,
        #[serde(default)]
        runs: Vec<text::TextRun>,
        #[serde(default)]
        paragraphs: Vec<text::ParagraphRun>,
        #[serde(default)]
        shape: text::TextShape,
        #[serde(default)]
        overflow_visible: bool,
        #[serde(default)]
        block_align: text::BlockAlign,
        #[serde(default)]
        orientation: text::Orientation,
        #[serde(default)]
        antialias: text::AntiAlias,
        #[serde(default)]
        warp: Option<text::TextWarp>,
    },
    Shape {
        fill: Option<Fill>,
        cache: Option<SurfaceM>,
        psd_raw: Option<Hash>,
        #[serde(default)]
        path: Path,
        #[serde(default)]
        stroke: Option<ShapeStroke>,
        #[serde(default)]
        live: Option<LiveShape>,
    },
    Smart {
        /// Shared source identity; older bundles did not record it.
        #[serde(default)]
        contents_id: Option<u64>,
        source: SmartSourceM,
        transform: Affine,
        smart_filters: Vec<SmartFilter>,
        cache: Option<SurfaceM>,
        psd_raw: Option<Hash>,
        #[serde(default = "yes")]
        filters_enabled: bool,
        #[serde(default)]
        filter_mask: Option<MaskM>,
        #[serde(default)]
        warp: Option<photocraft_geom::warp::Warp>,
        #[serde(default)]
        stack_mode: Option<photocraft_doc::StackMode>,
        /// Distort / Perspective placement (row-major 3×3); absent for affine placements.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        perspective: Option<[f64; 9]>,
    },
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SmartSourceM {
    Embedded { file_name: String, blob: Hash },
    Linked { path: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelM {
    pub name: String,
    pub surface: SurfaceM,
    pub spot: Option<(Color, f32)>,
    /// Channel Options overlay colour (older files: Photoshop's red).
    #[serde(default = "default_channel_color")]
    pub color: Color,
    #[serde(default = "default_channel_opacity")]
    pub opacity: f32,
    #[serde(default)]
    pub indicates: photocraft_doc::ColorIndicates,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

fn default_channel_color() -> Color {
    photocraft_doc::AlphaChannel::DEFAULT_COLOR
}

fn default_channel_opacity() -> f32 {
    0.5
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MetadataM {
    pub xmp: Option<String>,
    pub exif: Option<Hash>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub text: Vec<(String, String)>,
    /// (id, name, blob)
    pub psd_resources: Vec<(u16, String, Hash)>,
    /// (signature hex, key hex, blob)
    pub psd_global_blocks: Vec<(String, String, Hash)>,
}

#[cfg(test)]
mod channel_tests {
    use super::*;

    #[test]
    fn older_channels_load_with_default_options() {
        let fmt = serde_json::to_value(PixelFormat::GRAY8).unwrap();
        let old = serde_json::json!({
            "name": "Alpha 1",
            "surface": { "format": fmt, "default": "00", "tiles": [] },
            "spot": null
        });
        let c: ChannelM = serde_json::from_value(old).unwrap();
        assert_eq!(c.color, photocraft_doc::AlphaChannel::DEFAULT_COLOR);
        assert_eq!(c.opacity, 0.5);
        assert_eq!(c.indicates, photocraft_doc::ColorIndicates::MaskedAreas);
    }
}
