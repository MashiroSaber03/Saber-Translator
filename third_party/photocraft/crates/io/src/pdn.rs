//! Native, import-only PDN3 reader. Layout reference: addisonElliott/pypdn's reader;
//! NRBF records follow Microsoft's public MS-NRBF specification (see `nrbf`).

mod nrbf;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::io::Read;

use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_raster::{Interrupt, Surface};

use crate::{ImportResult, IoError};
use nrbf::{Graph, Reader, Value};

type Result<T> = std::result::Result<T, IoError>;

// shortcut: imports cap decoded pixels at 1 GiB, use streamed tile decoding for larger PDNs.
const MAX_PIXEL_BYTES: usize = 1 << 30;
const MAX_LAYERS: usize = 1024;
const MODES: [BlendMode; 14] = [
    BlendMode::Normal,
    BlendMode::Multiply,
    BlendMode::LinearDodge,
    BlendMode::PaintNetColorBurn,
    BlendMode::PaintNetColorDodge,
    BlendMode::Reflect,
    BlendMode::Glow,
    BlendMode::Overlay,
    BlendMode::Difference,
    BlendMode::Negation,
    BlendMode::Lighten,
    BlendMode::Darken,
    BlendMode::Screen,
    BlendMode::Xor,
];
const OLD_MODES: [&str; 14] =
    ["Normal", "Multiply", "Additive", "ColorBurn", "ColorDodge", "Reflect", "Glow", "Overlay", "Difference", "Negation", "Lighten", "Darken", "Screen", "Xor"];

fn invalid(message: impl Into<String>) -> IoError {
    IoError::Pdn(message.into())
}

fn count(graph: &Graph, object: &Value, field: &str) -> Result<usize> {
    usize::try_from(graph.integer(graph.field(object, field)?)?).map_err(|_| invalid(format!("invalid {field}")))
}

fn blend(graph: &Graph, layer: &Value, props: &Value) -> Result<BlendMode> {
    if let Some(mode) = graph.optional_field(props, "blendMode")? {
        let mode = match graph.resolve(mode)? {
            Value::Int(n) => *n,
            _ => graph.integer(graph.field(mode, "value__")?)?,
        };
        let mode = usize::try_from(mode).map_err(|_| invalid("invalid blend mode"))?;
        return MODES.get(mode).copied().ok_or_else(|| invalid(format!("unsupported blend mode {mode}")));
    }
    let props = graph.field(layer, "properties")?;
    let mode = graph.class_name(graph.field(props, "blendOp")?)?;
    let mode = mode
        .strip_prefix("PaintDotNet.UserBlendOps+")
        .and_then(|s| s.strip_suffix("BlendOp"))
        .ok_or_else(|| invalid(format!("unsupported blend operation {mode}")))?;
    OLD_MODES.iter().position(|m| *m == mode).and_then(|i| MODES.get(i)).copied().ok_or_else(|| invalid(format!("unsupported blend operation {mode}")))
}

/// Reads a memory block's numbered chunks in their on-disk order; each chunk has its own gzip
/// stream and checksum. Inflate directly into the bounded destination, checking its exact size.
fn pixels(reader: &mut Reader<'_>, length: usize, ctl: &Interrupt<'_>) -> Result<Vec<u8>> {
    let version = reader.byte()?;
    if version > 1 {
        return Err(invalid(format!("unsupported pixel compression {version}")));
    }
    let chunk_size = reader.be_u32()?;
    if chunk_size == 0 {
        return Err(invalid("zero pixel chunk size"));
    }
    let chunks = length.div_ceil(chunk_size);
    if chunks > 1_000_000 {
        return Err(invalid("too many pixel chunks"));
    }
    // Uncompressed chunks hold every pixel byte: a file shorter than that is truncated, and is
    // rejected before the buffer is allocated.
    if version == 1 && reader.bytes.len().saturating_sub(reader.pos) < length {
        return Err(invalid("uncompressed pixel data is truncated"));
    }
    let mut seen = vec![false; chunks];
    let mut data = vec![0; length];
    for _ in 0..chunks {
        ctl.check().map_err(|_| IoError::Cancelled)?;
        let index = reader.be_u32()?;
        let found = seen.get_mut(index).ok_or_else(|| invalid("pixel chunk index out of bounds"))?;
        if *found {
            return Err(invalid("duplicate pixel chunk"));
        }
        *found = true;
        let size = reader.be_u32()?;
        let raw = reader.take(size)?;
        let start = index.checked_mul(chunk_size).ok_or_else(|| invalid("pixel offset overflow"))?;
        let end = start.checked_add(chunk_size.min(length.saturating_sub(start))).ok_or_else(|| invalid("pixel offset overflow"))?;
        let dst = data.get_mut(start..end).ok_or_else(|| invalid("invalid pixel chunk range"))?;
        if version == 0 {
            let mut decoder = flate2::bufread::GzDecoder::new(raw);
            for band in dst.chunks_mut(1 << 20) {
                ctl.check().map_err(|_| IoError::Cancelled)?;
                decoder.read_exact(band).map_err(|e| invalid(format!("invalid gzip pixel chunk: {e}")))?;
            }
            let mut extra = [0];
            if decoder.read(&mut extra).map_err(|e| invalid(format!("invalid gzip pixel checksum: {e}")))? != 0 || !decoder.into_inner().is_empty() {
                return Err(invalid("pixel chunk has excess data"));
            }
        } else {
            if raw.len() != dst.len() {
                return Err(invalid("incorrect uncompressed pixel chunk size"));
            }
            dst.copy_from_slice(raw);
        }
    }
    Ok(data)
}

pub(super) fn import(name: &str, bytes: &[u8], ctl: &Interrupt<'_>) -> Result<ImportResult> {
    let mut reader = Reader { bytes, pos: 0 };
    if reader.take(4)? != b"PDN3" {
        return Err(invalid("unsupported document signature (expected PDN3)"));
    }
    let mut header_size = [0; 4];
    header_size.get_mut(..3).ok_or_else(|| invalid("invalid header"))?.copy_from_slice(reader.take(3)?);
    reader.take(u32::from_le_bytes(header_size) as usize)?;
    if reader.take(2)? != [0, 1] {
        return Err(invalid("unsupported PDN3 serialization version"));
    }
    let graph = nrbf::parse(&mut reader, ctl)?;
    let root = &graph.root;
    if graph.class_name(root)? != "PaintDotNet.Document" {
        return Err(invalid("root object is not a Paint.NET document"));
    }
    let width = count(&graph, root, "width")?;
    let height = count(&graph, root, "height")?;
    let row_bytes = width.checked_mul(4).ok_or_else(|| invalid("canvas size overflow"))?;
    let canvas_bytes = row_bytes.checked_mul(height).ok_or_else(|| invalid("canvas size overflow"))?;
    if width == 0 || height == 0 || width > i32::MAX as usize || height > i32::MAX as usize || canvas_bytes > MAX_PIXEL_BYTES {
        return Err(invalid("canvas is empty or exceeds the 1 GiB pixel limit"));
    }
    let list = graph.field(root, "layers")?;
    let layers = graph.array(graph.field(list, "ArrayList+_items")?)?;
    let len = count(&graph, list, "ArrayList+_size")?;
    if len == 0 || len > MAX_LAYERS {
        return Err(invalid("document must have 1..1024 bitmap layers"));
    }
    let layers = layers.get(..len).ok_or_else(|| invalid("layer list count exceeds its array"))?;
    let bounds = Rect::from_xywh(0, 0, width as u32, height as u32);
    let mut layouts = HashMap::new();
    let mut output = Vec::with_capacity(len);
    for layer in layers {
        ctl.check().map_err(|_| IoError::Cancelled)?;
        if graph.class_name(layer)? != "PaintDotNet.BitmapLayer" {
            return Err(invalid("only PDN3 bitmap layers are supported"));
        }
        if count(&graph, layer, "Layer+width")? != width || count(&graph, layer, "Layer+height")? != height {
            return Err(invalid("layer dimensions do not match the canvas"));
        }
        let surface = graph.field(layer, "surface")?;
        if graph.class_name(surface)? != "PaintDotNet.Surface" || count(&graph, surface, "width")? != width || count(&graph, surface, "height")? != height {
            return Err(invalid("invalid bitmap surface"));
        }
        let stride = count(&graph, surface, "stride")?;
        if stride < row_bytes {
            return Err(invalid("surface stride is too short for BGRA32 pixels"));
        }
        let memory = graph.field(surface, "scan0")?;
        let Value::Ref(id) = memory else { return Err(invalid("missing pixel memory reference")) };
        if let Some(old) = layouts.insert(*id, stride)
            && old != stride
        {
            return Err(invalid("shared pixel memory has inconsistent strides"));
        }
        let props = graph.field(layer, "Layer+properties")?;
        let opacity = count(&graph, props, "opacity")?;
        if opacity > 255 {
            return Err(invalid("opacity exceeds 255"));
        }
        let mut l = Layer::raster(graph.string(graph.field(props, "name")?)?, PixelFormat::RGBA8);
        l.visible = graph.boolean(graph.field(props, "visible")?)?;
        l.opacity = opacity as f32 / 255.0;
        l.blend = blend(&graph, layer, props)?;
        output.push((*id, l));
    }
    let blocks: Vec<_> = graph.objects_of_class("PaintDotNet.MemoryBlock").collect();
    let mut total = 0usize;
    // Validate every allocation before decoding any block.
    for (id, block) in &blocks {
        let length = count(&graph, block, "length64")?;
        let stride = *layouts.get(id).ok_or_else(|| invalid("unreferenced pixel memory block"))?;
        if length != stride.checked_mul(height).ok_or_else(|| invalid("pixel length overflow"))? {
            return Err(invalid("pixel memory length does not match the surface"));
        }
        if graph.boolean(graph.field(block, "hasParent")?)? || !graph.boolean(graph.field(block, "deferred")?)? {
            return Err(invalid("unsupported non-deferred or parented pixel memory"));
        }
        total = total.checked_add(length).ok_or_else(|| invalid("pixel length overflow"))?;
        if total > MAX_PIXEL_BYTES {
            return Err(invalid("decoded pixels exceed the 1 GiB limit"));
        }
    }
    let mut surfaces = HashMap::new();
    for (i, (id, block)) in blocks.iter().enumerate() {
        let length = count(&graph, block, "length64")?;
        let mut data = pixels(&mut reader, length, ctl)?;
        let stride = *layouts.get(id).ok_or_else(|| invalid("missing surface layout"))?;
        let mut surface = Surface::new(PixelFormat::RGBA8);
        for (y, row) in data.chunks_exact_mut(stride).enumerate() {
            ctl.check().map_err(|_| IoError::Cancelled)?;
            let row = row.get_mut(..row_bytes).ok_or_else(|| invalid("invalid pixel row"))?;
            for pixel in row.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
            surface.write_interleaved(Rect::from_xywh(0, y as i32, bounds.width(), 1), row);
        }
        surfaces.insert(*id, surface);
        ctl.progress(0.05 + 0.95 * (i + 1) as f32 / blocks.len() as f32);
    }
    if reader.pos != bytes.len() {
        return Err(invalid("unexpected trailing data"));
    }
    let mut document = Document::new(name, Size::new(width as u32, height as u32), ColorMode::Rgb, SampleType::U8);
    for (id, mut layer) in output {
        layer.content = LayerContent::Raster(surfaces.get(&id).cloned().ok_or_else(|| invalid("missing pixel memory block"))?);
        document.layers.push(layer);
    }
    let mut warnings = Vec::new();
    if let Some(meta) = graph.optional_field(root, "userMetadataItems")?
        && !matches!(graph.resolve(meta)?, Value::Null)
        && !graph.array(meta)?.is_empty()
    {
        warnings.push("Paint.NET document metadata was not imported".into());
    }
    Ok(ImportResult { document, warnings, source_read_only: true, preview_only: false })
}
