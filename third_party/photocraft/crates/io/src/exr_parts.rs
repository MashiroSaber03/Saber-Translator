//! Multi-part EXR and channel groups as layers (a Maya/Arnold render writes one part per
//! AOV, or a single part with prefixed channel sets like `diffuse.R`): the entries of a
//! file listed for a chooser ([`entries`]), and the chosen ones imported as the layers of
//! one document ([`import`]), with an optional additive beauty precomp from the AOVs.

use photocraft_cms::Builtin;
use photocraft_codecs::{self as codecs, ChannelLayout, DecodeOptions, Image, SampleType};
use photocraft_color::{ColorMode, SampleType as DocSample};
use photocraft_doc::{Document, Layer, LayerContent, Size};
use photocraft_raster::Surface;

use crate::{ImportResult, IoError, flat::image_surface};

/// One selectable line of the chooser: a whole part, a prefixed channel group inside a
/// part, or a Cryptomatte layer (listed for information; its samples are not image layers).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub kind: EntryKind,
    /// The name as shown and selected: the part name, the channel-group prefix, or the
    /// Cryptomatte layer name.
    pub name: String,
    /// The part's `view` attribute (stereo files), when it has one.
    pub view: Option<String>,
    pub width: u32,
    pub height: u32,
    /// Image channels that would import (4 for RGBA, 3 for RGB); Cryptomatte: stream count.
    pub channels: usize,
    /// Cryptomatte only: objects in the parsed manifest.
    pub objects: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    Part,
    Group,
    Cryptomatte,
}

/// One channel group: where it lives and which exact channels make up its RGB(A).
#[derive(Clone, Debug)]
struct Group {
    /// The prefix before the last dot (`diffuse` of `diffuse.R`).
    prefix: String,
    part: usize,
    channels: Vec<String>,
    alpha: Option<String>,
}

/// The name an AOV normalizes to for the precomp check (`Direct_Diffuse` and
/// `directDiffuse` are the same AOV).
fn normalize(name: &str) -> String {
    name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase()
}

/// The additive AOVs a beauty precomp is built from: the light-path set Arnold writes.
const PRECOMP_REQUIRED: [&str; 5] = ["directdiffuse", "indirectdiffuse", "directspecular", "indirectspecular", "emission"];
const PRECOMP_OPTIONAL: [&str; 3] = ["transmission", "sss", "volume"];

/// The base channel name (`R` of `diffuse.R`, `R` of `R`).
fn channel_base(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// The prefixed RGB(A) channel sets of every flat part (`diffuse.R/G/B` make the group
/// `diffuse`). Unprefixed channels are no group, and a prefix ending in digits is a
/// Cryptomatte stream (`crypto_asset00`), not a group. Each prefix is listed once.
fn groups(bytes: &[u8]) -> Result<Vec<Group>, IoError> {
    let mut out: Vec<Group> = Vec::new();
    for part in codecs::exr_info(bytes, &Default::default())? {
        if part.deep {
            continue;
        }
        let mut found: Vec<Group> = Vec::new();
        for c in &part.channels {
            let Some((prefix, base)) = channel_base_split(&c.name) else { continue };
            if prefix.chars().last().is_some_and(|ch| ch.is_ascii_digit()) {
                continue;
            }
            if found.iter_mut().all(|g| g.prefix != prefix) {
                found.push(Group { prefix: prefix.to_string(), part: part.index, channels: Vec::new(), alpha: None });
            }
            let Some(group) = found.iter_mut().find(|g| g.prefix == prefix) else { continue };
            match base {
                "R" | "G" | "B" => group.channels.push(c.name.clone()),
                "A" => group.alpha = Some(c.name.clone()),
                _ => {}
            }
        }
        found.retain(|g| g.channels.len() == 3);
        out.extend(found);
    }
    Ok(out)
}

/// `(prefix, base)` when the channel name carries a dot (`diffuse.R`); `None` unprefixed.
fn channel_base_split(name: &str) -> Option<(&str, &str)> {
    let (prefix, base) = name.rsplit_once('.')?;
    (!prefix.is_empty()).then_some((prefix, base))
}

/// The name a part is listed, selected and imported as: its `name` attribute, or
/// `part <index>` when it has none (a single-part render usually carries no name).
fn part_display_name(part: &codecs::ExrPartInfo) -> String {
    part.name.clone().unwrap_or_else(|| format!("part {}", part.index))
}

/// Lists the entries of a multi-part or channel-grouped EXR for a chooser: flat parts in
/// file order, then the channel groups, then the Cryptomatte layers as information. A file
/// whose parts are all deep is refused (the deep decoder composites those).
pub fn entries(bytes: &[u8]) -> Result<Vec<Entry>, IoError> {
    let info = codecs::exr_info(bytes, &Default::default())?;
    if info.iter().all(|p| p.deep) {
        return Err(IoError::Unsupported("this EXR holds only deep parts; it opens through the deep decoder".into()));
    }
    let mut out = Vec::new();
    for part in &info {
        if part.deep {
            continue;
        }
        out.push(Entry {
            kind: EntryKind::Part,
            name: part_display_name(part),
            view: part.view.clone(),
            width: part.width,
            height: part.height,
            channels: part.channels.iter().filter(|c| ["R", "G", "B", "A", "Y"].contains(&channel_base(&c.name))).count(),
            objects: None,
        });
    }
    for group in groups(bytes)? {
        out.push(Entry {
            kind: EntryKind::Group,
            name: group.prefix.clone(),
            view: None,
            width: 0,
            height: 0,
            channels: 3 + usize::from(group.alpha.is_some()),
            objects: None,
        });
    }
    for layer in codecs::cryptomatte_layers(bytes, &Default::default())? {
        out.push(Entry {
            kind: EntryKind::Cryptomatte,
            name: layer.name.clone(),
            view: None,
            width: 0,
            height: 0,
            channels: layer.channels.len(),
            objects: Some(layer.manifest.len()),
        });
    }
    Ok(out)
}

/// What [`import`] turns into layers.
pub struct Selection {
    /// Part names to import as whole layers.
    pub parts: Vec<String>,
    /// Channel-group prefixes to import as layers.
    pub groups: Vec<String>,
    /// `"both"`, `"left"` or `"right"`: which stereo views of the parts to consider.
    pub view: String,
    /// Add a `beauty (precomp)` layer summed from the file's AOV groups (see
    /// [`PRECOMP_REQUIRED`]); fails naming the missing AOVs when the set is incomplete.
    pub precomp: bool,
}

/// An image as an importable layer: F16 becomes F32 (lossless, warned once by the caller),
/// the layout matches the document's, the surface is built a band of rows at a time. The
/// returned mode tells the caller whether this layer is grayscale.
fn layer_surface(img: &Image, max: &mut (u32, u32)) -> Result<(Surface, ColorMode), IoError> {
    let (layout, mode) = if matches!(img.layout(), ChannelLayout::Gray | ChannelLayout::GrayA) {
        (ChannelLayout::GrayA, ColorMode::Grayscale)
    } else {
        (ChannelLayout::Rgba, ColorMode::Rgb)
    };
    max.0 = max.0.max(img.width());
    max.1 = max.1.max(img.height());
    let converted = img.convert(layout, SampleType::F32);
    image_surface(&converted, layout, SampleType::F32).map(|s| (s, mode))
}

/// Imports the chosen entries as the layers of one document: parts whole, groups from their
/// own channels, F16 kept losslessly as F32, the linear-light EXR values tagged linear sRGB.
/// `precomp` adds the additive sum of the file's AOV groups as a top layer and fails,
/// naming the missing AOVs, when the light-path set is incomplete.
pub fn import(name: &str, bytes: &[u8], sel: &Selection) -> Result<ImportResult, IoError> {
    let opts = DecodeOptions::default();
    let info = codecs::exr_info(bytes, &opts.limits)?;
    if info.is_empty() {
        return Err(IoError::Unsupported("the EXR has no parts".into()));
    }
    let all_groups = groups(bytes)?;
    let view_passes = |view: &Option<String>| view.as_ref().is_none_or(|v| sel.view.eq_ignore_ascii_case("both") || v.eq_ignore_ascii_case(&sel.view));
    // A named part wins over an unnamed one listed under the same fallback name ("part 0").
    let part_by_name = |want: &str| {
        let flat = || info.iter().filter(|p| !p.deep);
        flat()
            .find(|p| p.name.as_deref() == Some(want))
            .or_else(|| flat().find(|p| p.name.is_none() && part_display_name(p) == want))
            .ok_or_else(|| IoError::Unsupported(format!("no flat part named {want}")))
    };
    let mut chosen_parts: Vec<&codecs::ExrPartInfo> = Vec::new();
    for want in &sel.parts {
        let p = part_by_name(want)?;
        if !view_passes(&p.view) {
            return Err(IoError::Unsupported(format!("the part {want} is of the {:?} view, the filter is {}", p.view, sel.view)));
        }
        if !chosen_parts.iter().any(|c| c.index == p.index) {
            chosen_parts.push(p);
        }
    }
    let mut chosen_groups: Vec<&Group> = Vec::new();
    for want in &sel.groups {
        let g = all_groups.iter().find(|g| g.prefix == *want).ok_or_else(|| IoError::Unsupported(format!("no channel group named {want}")))?;
        let part_view = info.get(g.part).and_then(|p| p.view.clone());
        if !view_passes(&part_view) {
            return Err(IoError::Unsupported(format!("the group {} is of the {:?} view, the filter is {}", want, part_view, sel.view)));
        }
        if !chosen_groups.iter().any(|c| c.prefix == g.prefix) {
            chosen_groups.push(g);
        }
    }
    if chosen_parts.is_empty() && chosen_groups.is_empty() && !sel.precomp {
        return Err(IoError::Unsupported("no parts or groups selected".into()));
    }

    // Decode each needed whole part once; groups come from their part's named channels.
    let mut part_pixels: Vec<Option<Image>> = vec![None; info.len()];
    let mut layers: Vec<Layer> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut gray_only = true;
    let mut max = (1u32, 1u32);
    let mut f16_seen = false;

    for part in &chosen_parts {
        let img = match &part_pixels[part.index] {
            Some(img) => img.clone(),
            None => {
                let img = codecs::decode_exr_part(bytes, part.index, &opts)?;
                part_pixels[part.index] = Some(img.clone());
                img
            }
        };
        if img.sample_type() == SampleType::F16 {
            f16_seen = true;
        }
        let (surface, mode) = layer_surface(&img, &mut max)?;
        gray_only &= mode == ColorMode::Grayscale;
        layers.push(Layer::new(part_display_name(part), LayerContent::Raster(surface)));
    }
    for group in &chosen_groups {
        let mut names = group.channels.clone();
        names.extend(group.alpha.clone());
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let img = codecs::decode_exr_channels(bytes, group.part, &refs, &opts)?;
        if img.sample_type() == SampleType::F16 {
            f16_seen = true;
        }
        let (surface, mode) = layer_surface(&img, &mut max)?;
        gray_only &= mode == ColorMode::Grayscale;
        layers.push(Layer::new(group.prefix.clone(), LayerContent::Raster(surface)));
    }

    if sel.precomp {
        let layer = precomp_layer(bytes, &all_groups, &opts, &mut max, &mut warnings)?;
        gray_only = false; // the precomp is an RGB layer by construction
        layers.push(layer);
    }

    if layers.is_empty() {
        return Err(IoError::Unsupported("nothing imported".into()));
    }
    if f16_seen {
        warnings.push("16-bit float samples are stored as 32-bit float".to_string());
    }
    let mode = if gray_only { ColorMode::Grayscale } else { ColorMode::Rgb };
    let mut doc = Document::new(name, Size::new(max.0, max.1), mode, DocSample::F32);
    doc.layers = layers;
    if mode == ColorMode::Rgb {
        // OpenEXR holds linear, scene-referred values: tag them linear sRGB, as File › Open does.
        doc.icc_profile = Some(Builtin::LinearSrgb.profile().to_bytes());
    }
    Ok(ImportResult { document: doc, warnings, source_read_only: false, preview_only: false })
}

/// The additive beauty from the file's AOV groups: premultiplied RGB summed over the
/// light-path set, alpha the maximum coverage, straightened back. Fails naming the missing
/// AOVs when the required set is incomplete; AOVs beyond the set are ignored.
fn precomp_layer(bytes: &[u8], all_groups: &[Group], opts: &DecodeOptions, max: &mut (u32, u32), warnings: &mut Vec<String>) -> Result<Layer, IoError> {
    let have: Vec<String> = all_groups.iter().map(|g| normalize(&g.prefix)).collect();
    let missing: Vec<&str> = PRECOMP_REQUIRED.iter().copied().filter(|r| !have.iter().any(|h| h == *r)).collect();
    if !missing.is_empty() {
        return Err(IoError::Unsupported(format!("the AOV set is incomplete for a beauty precomp; missing: {}", missing.join(", "))));
    }
    let mut sum_rgb: Vec<f32> = Vec::new();
    let mut alpha: Vec<f32> = Vec::new();
    let mut size = (0u32, 0u32);
    for group in all_groups {
        let n = normalize(&group.prefix);
        if !PRECOMP_REQUIRED.contains(&n.as_str()) && !PRECOMP_OPTIONAL.contains(&n.as_str()) {
            continue;
        }
        let mut names = group.channels.clone();
        names.extend(group.alpha.clone());
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let img = codecs::decode_exr_channels(bytes, group.part, &refs, opts)?;
        let rgba = img.convert(ChannelLayout::Rgba, SampleType::F32);
        let (w, h) = rgba.dimensions();
        if size == (0, 0) {
            size = (w, h);
            sum_rgb = vec![0.0; w as usize * h as usize * 3];
            alpha = vec![0.0; w as usize * h as usize];
        } else if size != (w, h) {
            return Err(IoError::Unsupported("the AOV groups differ in size; a beauty precomp needs one common extent".into()));
        }
        let px = rgba.to_f32_samples().unwrap_or_default();
        for p in 0..(w as usize * h as usize) {
            // AOV channels are premultiplied by their coverage, as renderers write them:
            // the sum adds the stored RGB as-is, and alpha is the maximum coverage.
            let (r, g, b, a) = (px[p * 4], px[p * 4 + 1], px[p * 4 + 2], px[p * 4 + 3]);
            sum_rgb[p * 3] += r;
            sum_rgb[p * 3 + 1] += g;
            sum_rgb[p * 3 + 2] += b;
            alpha[p] = alpha[p].max(a);
        }
    }
    let (w, h) = size;
    if w == 0 || h == 0 {
        return Err(IoError::Unsupported("the AOV groups are empty".into()));
    }
    let mut out = vec![0f32; w as usize * h as usize * 4];
    for p in 0..(w as usize * h as usize) {
        let a = alpha[p];
        let (r, g, b) = if a > 0.0 { (sum_rgb[p * 3] / a, sum_rgb[p * 3 + 1] / a, sum_rgb[p * 3 + 2] / a) } else { (0.0, 0.0, 0.0) };
        out[p * 4] = r;
        out[p * 4 + 1] = g;
        out[p * 4 + 2] = b;
        out[p * 4 + 3] = a;
    }
    let img = Image::from_f32(w, h, ChannelLayout::Rgba, &out)?;
    max.0 = max.0.max(w);
    max.1 = max.1.max(h);
    let (surface, _) = layer_surface(&img, max)?;
    warnings.push("beauty (precomp) is the additive sum of the file's AOVs; alpha is the maximum coverage".to_string());
    Ok(Layer::new("beauty (precomp)", LayerContent::Raster(surface)))
}
