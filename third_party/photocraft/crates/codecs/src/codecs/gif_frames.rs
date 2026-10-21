//! Every frame of an animated GIF. [`crate::decode`] keeps the first frame only; this walks the
//! whole animation for callers that can hold it (video layers).
//!
//! The `image` crate composites each frame onto the logical screen and applies the disposal
//! methods, so every returned frame is a full, self-contained picture.

use std::io::Cursor;

use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, ImageDecoder};

use crate::Format;
use crate::error::CodecError;
use crate::image::{ChannelLayout, DecodeWarning, Image};
use crate::options::DecodeOptions;

/// Most frames one GIF may yield. Frames are held in memory, and a tiny canvas could otherwise
/// ask for millions of them while staying inside the byte budget.
pub const MAX_GIF_FRAMES: usize = 100_000;

/// One composited frame of an animation and how long it stays on screen.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationFrame {
    pub image: Image,
    /// On-screen time in milliseconds. GIF counts in hundredths of a second, so this is a
    /// multiple of 10 (0 when the file gives none).
    pub delay_ms: u32,
}

/// Decodes every frame of a GIF, each as a full RGBA8 image of the logical screen size.
///
/// The first frame must decode. If a later frame is damaged the frames before it are returned
/// and the first one carries [`DecodeWarning::Truncated`]. Dimensions, the frame count
/// ([`MAX_GIF_FRAMES`]) and the total decoded bytes are checked against `opts.limits` before
/// each frame is kept.
pub fn decode(bytes: &[u8], opts: &DecodeOptions) -> Result<Vec<AnimationFrame>, CodecError> {
    let limits = &opts.limits;
    let map = |e: image::ImageError| match e {
        image::ImageError::Limits(l) => CodecError::LimitExceeded(l.to_string()),
        image::ImageError::Unsupported(u) => CodecError::unsupported(Format::Gif, u.to_string()),
        e => CodecError::malformed(Format::Gif, e),
    };
    let mut decoder = GifDecoder::new(Cursor::new(bytes)).map_err(map)?;
    let mut il = image::Limits::default();
    il.max_image_width = Some(limits.max_width);
    il.max_image_height = Some(limits.max_height);
    il.max_alloc = Some(limits.max_alloc);
    decoder.set_limits(il).map_err(map)?;
    let (w, h) = decoder.dimensions();
    limits.check_bytes(w, h, 4)?;
    let frame_bytes = u64::from(w) * u64::from(h) * 4;

    let mut out: Vec<AnimationFrame> = Vec::new();
    let mut damaged = false;
    for frame in decoder.into_frames() {
        let frame = match frame {
            Ok(f) => f,
            Err(e) if out.is_empty() => return Err(map(e)),
            Err(_) => {
                damaged = true;
                break;
            }
        };
        if out.len() >= MAX_GIF_FRAMES {
            return Err(CodecError::LimitExceeded(format!("more than {MAX_GIF_FRAMES} frames")));
        }
        if frame_bytes.saturating_mul(out.len() as u64 + 1) > limits.max_alloc {
            return Err(CodecError::LimitExceeded(format!("{} frames of {w}x{h} exceed max_alloc {}", out.len() + 1, limits.max_alloc)));
        }
        let (numer, denom) = frame.delay().numer_denom_ms();
        let delay_ms = numer.checked_div(denom).unwrap_or(0);
        let rgba = frame.into_buffer();
        out.push(AnimationFrame { image: Image::from_u8(w, h, ChannelLayout::Rgba, rgba.into_raw())?, delay_ms });
    }
    if damaged && let Some(first) = out.first_mut() {
        first.image.warnings.push(DecodeWarning::Truncated { format: Format::Gif });
    }
    Ok(out)
}
