//! Cryptomatte (specification 1.2): per-pixel object/material IDs with coverage in EXR
//! channels, as written by Arnold, V-Ray, Redshift, Mantra/Karma and Cycles. Layers are
//! announced by `cryptomatte/<key>/name|hash|conversion|manifest` header attributes; the
//! samples live in `<name>NN` RGBA channel sets, each holding two (ID, coverage) ranks:
//! `.red`/`.green` (rank 2·NN) and `.blue`/`.alpha` (rank 2·NN+1).
//!
//! IDs are the float *values*: MurmurHash3 x86-32 (seed 0) of the object name, its
//! exponent fixed away from 0/255 so no ID is denormal, infinite or NaN, read as an f32
//! bit pattern. Everything here compares IDs by value, never by bit pattern.

use std::io::Cursor;

use exr::meta::MetaData;
use exr::meta::attribute::AttributeValue;

use crate::Format;
use crate::error::CodecError;
use crate::image::{CryptomatteBuffer, CryptomatteLayer};
use crate::options::Limits;

const F: Format = Format::OpenExr;

/// A Cryptomatte stream channel (`crypto_asset00.red`) splits into its stream name
/// (`crypto_asset00`), the stream index, whether it is a coverage channel (green/alpha)
/// instead of an ID (red/blue), and its rank within the set (0 for red/green, 1 for
/// blue/alpha).
pub(crate) struct StreamChannel {
    pub stream: String,
    pub index: u32,
    pub coverage: bool,
    pub rank: usize,
}

/// `<name>NN` where NN is one or more digits (the reference readers match `\d+`), with a
/// `.red`/`.r` or `.blue`/`.b` (ID) or `.green`/`.g` or `.alpha`/`.a` (coverage) suffix,
/// in any case (Blender writes `.R`/`.G`/`.B`/`.A`).
pub(crate) fn split_stream(channel: &str) -> Option<StreamChannel> {
    let (stream, suffix) = channel.rsplit_once('.')?;
    let (coverage, rank) = match suffix.to_ascii_lowercase().as_str() {
        "red" | "r" => (false, 0),
        "green" | "g" => (true, 0),
        "blue" | "b" => (false, 1),
        "alpha" | "a" => (true, 1),
        _ => return None,
    };
    let digits = stream.chars().rev().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 || digits == stream.len() {
        return None;
    }
    let split = stream.len() - digits;
    let index = stream[split..].parse().ok()?;
    Some(StreamChannel { stream: stream.to_string(), index, coverage, rank })
}

/// MurmurHash3, x86-32 bit variant, seed 0 (public-domain algorithm, Austin Appleby).
fn murmur3_32(data: &[u8]) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h = 0u32;
    let (chunks, rest) = data.as_chunks::<4>();
    for c in chunks {
        let mut k = u32::from_le_bytes(*c);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    if !rest.is_empty() {
        let mut k = [0u8; 4];
        k[..rest.len()].copy_from_slice(rest);
        let mut k = u32::from_le_bytes(k);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }
    h ^= data.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// The Cryptomatte ID of a name: the MurmurHash3 bits with the exponent fixed away from
/// 0 and 255, as an f32 value.
pub fn cryptomatte_id(name: &str) -> f32 {
    let bits = murmur3_32(name.as_bytes());
    let exp = (bits >> 23) & 0xff;
    let bits = if exp == 0 || exp == 255 { bits ^ (1 << 23) } else { bits };
    f32::from_bits(bits)
}

/// The 7-hex-digit metadata key of a layer name (the ID bits without their last digit).
pub fn cryptomatte_key(name: &str) -> String {
    format!("{:08x}", cryptomatte_id(name).to_bits())[..7].to_string()
}

/// The preview colour of an ID (the reference formula): green and blue channels shifted
/// out of the ID's bit pattern, red unused.
pub fn cryptomatte_preview_color(id: f32) -> [f32; 3] {
    let bits = id.to_bits();
    [0.0, f32::from_bits(bits << 8), f32::from_bits(bits << 16)]
}

/// The `cryptomatte/<key>/<field>` text attribute of a header, if present.
fn attribute(other: &std::collections::HashMap<exr::meta::attribute::Text, AttributeValue>, key: &str, field: &str) -> Option<String> {
    let full = exr::meta::attribute::Text::new_or_none(format!("cryptomatte/{key}/{field}").as_str())?;
    match other.get(&full)? {
        AttributeValue::Text(t) => Some(t.to_string()),
        _ => None,
    }
}

/// Parse a manifest JSON (`{"object name": "<8-hex ID bits>"}`) into (ID value, name)
/// pairs. Entries that are not strings of up to eight hex digits are skipped, like the
/// reference readers tolerate a partially broken manifest; a broken document yields none.
fn parse_manifest(json: &str) -> Vec<(f32, String)> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else { return Vec::new() };
    let Some(map) = value.as_object() else { return Vec::new() };
    let mut out = Vec::with_capacity(map.len().min(1 << 20));
    for (name, hex) in map {
        let Some(hex) = hex.as_str() else { continue };
        let Some(bits) = u32::from_str_radix(hex.trim_start_matches("0x"), 16).ok() else { continue };
        out.push((f32::from_bits(bits), name.clone()));
    }
    out
}

/// The stream channel names of a layer (`crypto_asset00`, `crypto_asset01`, …, exactly as
/// the file spells them), matched leniently: the channel prefix and the `name` attribute
/// differ in case or `_`/`-` between renderers (`CryptoAsset` vs `crypto_asset`).
fn layer_streams(header: &exr::meta::header::Header, name: &str) -> Vec<String> {
    let normalize = |s: &str| s.to_ascii_lowercase().replace(['_', '-'], "");
    let want = normalize(name);
    let mut found: Vec<(u32, String)> = header
        .channels
        .list
        .iter()
        .filter_map(|c| {
            let s = split_stream(&c.name.to_string())?;
            (!s.coverage).then_some((s.index, s.stream))
        })
        .filter(|(_, stream)| normalize(stream.trim_end_matches(|c: char| c.is_ascii_digit())) == want)
        .collect();
    found.sort();
    found.dedup();
    found.into_iter().map(|(_, stream)| stream).collect()
}

/// Lists every Cryptomatte layer of the file from the headers alone: what a Cryptomatte
/// picker offers. The embedded manifest (capped, JSON) is parsed into (ID, name) pairs.
pub(crate) fn layers(bytes: &[u8], limits: &Limits) -> Result<Vec<CryptomatteLayer>, CodecError> {
    let meta = MetaData::read_from_buffered(Cursor::new(bytes), false).map_err(|e| CodecError::malformed(F, e))?;
    // Manifests can be large; bound the embedded text before parsing anything.
    const MANIFEST_CAP: usize = 16 << 20;
    let mut out = Vec::new();
    for (part, header) in meta.headers.iter().enumerate() {
        let size = header.layer_size;
        let (w, h) = (u32::try_from(size.0).unwrap_or(u32::MAX), u32::try_from(size.1).unwrap_or(u32::MAX));
        limits.check_bytes(w, h, 8)?;
        for key in cryptomatte_keys(&header.own_attributes.other) {
            let Some(name) = attribute(&header.own_attributes.other, &key, "name") else { continue };
            let channels = layer_streams(header, name.as_str());
            if channels.is_empty() {
                continue;
            }
            let manifest = attribute(&header.own_attributes.other, &key, "manifest")
                .filter(|m| m.len() <= MANIFEST_CAP)
                .map(|m| parse_manifest(m.as_str()))
                .unwrap_or_default();
            out.push(CryptomatteLayer {
                part,
                name,
                key: key.clone(),
                conversion: attribute(&header.own_attributes.other, &key, "conversion"),
                manifest,
                sidecar_manifest: attribute(&header.own_attributes.other, &key, "manif_file"),
                channels,
            });
        }
    }
    out.sort_by(|a, b| (&a.part, &a.name).cmp(&(&b.part, &b.name)));
    Ok(out)
}

/// Every `cryptomatte/<key>/` key of a header, in file order, without duplicates.
fn cryptomatte_keys(other: &std::collections::HashMap<exr::meta::attribute::Text, exr::meta::attribute::AttributeValue>) -> Vec<String> {
    let mut keys = Vec::new();
    for name in other.keys() {
        let full = name.to_string();
        let Some(rest) = full.strip_prefix("cryptomatte/") else { continue };
        let Some((key, _field)) = rest.split_once('/') else { continue };
        if !key.is_empty() && !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
        }
    }
    keys
}

/// Decodes the samples of the Cryptomatte layer called `layer_name` (a
/// [`CryptomatteLayer::name`]): per pixel its (ID, coverage) pairs, coverage above zero,
/// sorted by descending coverage. Refuses files that also contain deep parts, like the
/// other flat part decoders.
pub(crate) fn decode(bytes: &[u8], layer_name: &str, limits: &Limits) -> Result<CryptomatteBuffer, CodecError> {
    let found = layers(bytes, limits)?;
    let layer = found
        .iter()
        .find(|l| l.name.eq_ignore_ascii_case(layer_name))
        .ok_or_else(|| CodecError::unsupported(F, format!("no Cryptomatte layer named {layer_name}")))?;
    // Also caps all parts' declared sizes together, like every other flat decode.
    let meta = super::exr::read_meta(bytes, limits)?;
    if meta.requirements.has_deep_data {
        return Err(CodecError::unsupported(F, "this file has deep parts; Cryptomatte decoding stays with flat files"));
    }
    let header = meta.headers.get(layer.part).ok_or_else(|| CodecError::malformed(F, "the Cryptomatte part is missing"))?;
    let size = header.layer_size;
    let (w, h) = (size.0, size.1);
    let npx = w.checked_mul(h).ok_or_else(|| CodecError::malformed(F, "the layer is too large"))?;
    let too_large = || CodecError::malformed(F, "the layer is too large");
    let (w32, h32) = (u32::try_from(w).map_err(|_| too_large())?, u32::try_from(h).map_err(|_| too_large())?);
    // Each pixel also gets a pair list, so count its header with the channel samples (four
    // f32 channels per set: two ranks of ID and coverage).
    let per_pixel = (layer.channels.len().max(1) as u64).saturating_mul(16).saturating_add(std::mem::size_of::<Vec<(f32, f32)>>() as u64);
    limits.check_bytes(w32, h32, per_pixel)?;

    let ex_layer = super::exr::read_layer(bytes, layer.part)?;

    // Two (ID, coverage) ranks per set: rank `slot * 2 + stream.rank`.
    let ranks = layer.channels.len().saturating_mul(2);
    let mut ids: Vec<Vec<f32>> = vec![Vec::new(); ranks];
    let mut coverages: Vec<Vec<f32>> = vec![Vec::new(); ranks];
    for c in &ex_layer.channel_data.list {
        let Some(stream) = split_stream(&c.name.to_string()) else { continue };
        let Some(slot) = layer.channels.iter().position(|s| *s == stream.stream) else { continue };
        let values: Vec<f32> = c.sample_data.values_as_f32().collect();
        if values.len() != npx {
            return Err(CodecError::malformed(F, "a Cryptomatte channel is smaller than its layer"));
        }
        let bucket = if stream.coverage { &mut coverages } else { &mut ids };
        let Some(dst) = slot.checked_mul(2).and_then(|r| r.checked_add(stream.rank)).and_then(|r| bucket.get_mut(r)) else { continue };
        *dst = values;
    }

    let mut pixels = vec![Vec::new(); npx];
    for (p, out) in pixels.iter_mut().enumerate() {
        for (id, cov) in ids.iter().zip(&coverages) {
            let (id, cov) = (id.get(p).copied().unwrap_or(0.0), cov.get(p).copied().unwrap_or(0.0));
            if cov > 0.0 && id.is_finite() {
                out.push((id, cov));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    Ok(CryptomatteBuffer { width: w32, height: h32, pixels })
}
